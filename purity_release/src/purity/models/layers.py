"""Graph construction and neural layers shared by the two PURITY models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def physics_edge_index_batch(x, batch):
    """
    Fully-connected intra-slice edges per subsystem.

    radius_graph (via torch_cluster's `radius` op) silently requires the
    `batch` argument to be non-decreasing. The natural input order
    (ATAR slices 1..N then LYSO slices 1..M) makes a cluster_id of the
    form  batch*10000 + slice*10 + subsys  jump DOWN at the subsystem
    boundary (e.g. ...40, 11, 21, ...), and the op responds by silently
    producing phantom cross-cluster edges AND dropping legitimate
    intra-cluster ones. Calling the op once per subsystem with the
    per-call batch explicitly sorted satisfies the invariant by
    construction and structurally guarantees ATAR-LYSO disjoint graphs.
    """
    is_atar = (x[:, 5] > 0.5) | (x[:, 6] > 0.5)
    is_lyso = (x[:, 7] > 0.5)
    slice_id = x[:, 8]

    from torch_geometric.nn import radius_graph

    edges_list = []
    for sub_mask in (is_atar, is_lyso):
        if not sub_mask.any():
            continue
        idx_global = sub_mask.nonzero(as_tuple=False).squeeze(1)
        cluster = (batch[idx_global] * 10000 + slice_id[idx_global] * 10).long()
        order = torch.argsort(cluster)
        cluster_sorted = cluster[order]
        idx_global_sorted = idx_global[order]
        dummy = torch.zeros((idx_global.size(0), 1), device=x.device)
        with torch.no_grad():
            edge_local = radius_graph(
                dummy, r=1.0, batch=cluster_sorted,
                loop=False, max_num_neighbors=3000,
            )
        edges_list.append(idx_global_sorted[edge_local])

    if not edges_list:
        return torch.zeros((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(edges_list, dim=1)

class JointAttentionBlock(nn.Module):
    """
    Graph Transformer block mimicking Particle Transformer structure.
    Uses continuous geometric edge features to guide attention.
    """
    def __init__(self, hidden_dim=128, heads=4, edge_dim=11, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        from torch_geometric.nn import TransformerConv
        self.conv = TransformerConv(
            in_channels=hidden_dim, 
            out_channels=hidden_dim // heads, 
            heads=heads, 
            concat=True, 
            dropout=dropout, 
            edge_dim=edge_dim,
            beta=True
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        norm_x = self.ln1(x)
        conv_out = self.conv(norm_x, edge_index, edge_attr)
        x = x + conv_out # Residual
        
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout_ffn(ffn_out) # Residual
        
        return x

class VectorHead(nn.Module):
    """Predicts a 3D unit vector"""
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )
    def forward(self, x):
        raw = self.mlp(x)
        return F.normalize(raw, p=2, dim=-1)

class QuantileOutputHead(nn.Module):
    def __init__(self, input_dim, num_points=2, coords=3, quantiles=[0.16, 0.50, 0.84]):
        super().__init__()
        self.quantiles = sorted(quantiles)
        self.mid_index = self.quantiles.index(0.50) 
        self.num_points = num_points
        self.coords = coords
        self.num_quantiles = len(quantiles)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_points * coords * self.num_quantiles)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Predict absolute coordinates from latent features
        raw = self.projection(x)
        raw = raw.view(batch_size, self.num_points, self.coords, self.num_quantiles)
        
        median = raw[..., self.mid_index]
        
        upper_offsets = torch.nn.functional.softplus(raw[..., self.mid_index+1:])
        upper_vals = median.unsqueeze(-1) + torch.cumsum(upper_offsets, dim=-1)
        
        lower_offsets = torch.nn.functional.softplus(raw[..., :self.mid_index])
        lower_offsets_flipped = torch.flip(lower_offsets, dims=[-1]) 
        lower_vals = median.unsqueeze(-1) - torch.cumsum(lower_offsets_flipped, dim=-1)
        lower_vals = torch.flip(lower_vals, dims=[-1])
        
        return torch.cat([lower_vals, median.unsqueeze(-1), upper_vals], dim=-1)

def build_atar_edge_attr(x, edge_index):
    """
    4D ATAR-only intra-slice edge features: [d_transverse, dz, dE, is_cross_view]

    - d_transverse = dx for XZ-XZ edges, dy for YZ-YZ edges, 0 for XZ-YZ
      (XZ and YZ strips share no transverse axis, so the diff is undefined/unphysical)
    - dz: Z is shared across both views, always meaningful
    - dE: energy deposit difference
    - is_cross_view: 1 for XZ-YZ edges, 0 for same-view edges
    - NO time: per-hit time within a slice is detector jitter noise, not track geometry
    """
    if edge_index.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float, device=x.device)

    src, dst = edge_index
    u, v = x[src], x[dst]

    is_yz_u = u[:, 6] > 0.5
    is_yz_v = v[:, 6] > 0.5

    m_xz_xz = (~is_yz_u) & (~is_yz_v)
    m_yz_yz  = is_yz_u & is_yz_v
    m_xz_yz  = (~is_yz_u & is_yz_v) | (is_yz_u & ~is_yz_v)

    out = torch.zeros((u.size(0), 4), dtype=torch.float, device=x.device)

    # d_transverse: dx for XZ-XZ, dy for YZ-YZ, 0 for cross-view
    out[m_xz_xz, 0] = v[m_xz_xz, 0] - u[m_xz_xz, 0]  # dx
    out[m_yz_yz,  0] = v[m_yz_yz,  1] - u[m_yz_yz,  1]  # dy

    # dz (col 2) — always defined
    out[:, 1] = v[:, 2] - u[:, 2]

    # dE (col 3)
    out[:, 2] = v[:, 3] - u[:, 3]

    # is_cross_view
    out[m_xz_yz, 3] = 1.0

    return out

def build_lyso_edge_attr(x, edge_index):
    """
    5D LYSO-only intra-slice edge features: [dx, dy, dz, dE, |dt|]

    Genuine 3D crystal positions — no projection ambiguity.  Time enters ONLY as the
    MAGNITUDE of the pairwise difference |dt|: shower components have measurable timing
    SEPARATIONS (the clustering-relevant quantity), but the absolute time and the sign/
    ordering must never reach the model.  If they do, it learns a time-dependent acceptance
    of old muons — rejecting positrons whose matched calo cluster sits at negative absolute
    time — which depletes the negative-time sideband and biases R_e/mu.  |dt| is invariant
    under a global time shift AND under time reversal, so it carries clustering proximity
    with no absolute-time leak.
    """
    if edge_index.numel() == 0:
        return torch.zeros((0, 5), dtype=torch.float, device=x.device)

    src, dst = edge_index
    u, v = x[src], x[dst]

    out = v[:, :5] - u[:, :5]      # [dx, dy, dz, dE, dt]
    out[:, 4] = out[:, 4].abs()    # time -> |dt| magnitude only (no absolute time, no ordering)
    return out
