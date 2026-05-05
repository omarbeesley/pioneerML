"""
PURITY Architecture (PIONEER Unified Reconstruction via Interactive Transformer TopologY)
Fuses ATAR (x-z, y-z) and LYSO (3D) hits using a Joint Self-Attention Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_add_pool, AttentionalAggregation

from unified_reco.constants import (
    NORM_T_LYSO, NORM_E_LYSO, NORM_E_ATAR, NORM_POS_ATAR,
    SIGMA_COINC_NS, TOF_NS,
    ACCEPT_Z_MIN_MM, ACCEPT_Z_MAX_MM, ACCEPT_XY_MAX_MM, ACCEPT_ANGLE_MAX_DEG,
)

import math
_ACCEPT_ANGLE_COS = math.cos(math.radians(ACCEPT_ANGLE_MAX_DEG))

# Define Modality IDs
MODALITY_ATAR_XZ = 0
MODALITY_ATAR_YZ = 1
MODALITY_LYSO = 2

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
    5D LYSO-only intra-slice edge features: [dx, dy, dz, dE, dt]

    Genuine 3D crystal positions — no projection ambiguity.
    Time is retained because shower components have measurable timing differences.
    """
    if edge_index.numel() == 0:
        return torch.zeros((0, 5), dtype=torch.float, device=x.device)

    src, dst = edge_index
    u, v = x[src], x[dst]

    out = v[:, :5] - u[:, :5]  # [dx, dy, dz, dE, dt]
    return out

class PURITYHybridModel(nn.Module):
    """
    The master unified model combining ATAR tracking and LYSO object condensation.
    """
    def __init__(self, hidden_dim=150, num_blocks=3, heads=5,
                 dropout=0.05, num_pdg_classes=3):
        super().__init__()

        # --- Subsystem-aware encoders with clean, physically-motivated feature sets ---
        #
        # ATAR: [transverse_coord, z, E, is_yz] = 4D
        #   - transverse_coord = x for XZ strips, y for YZ strips (avoids zero-padding a physical coord)
        #   - is_yz = 0/1 distinguishes XZ from YZ projection
        #   - NO time: per-hit time within a slice is detector jitter, not geometry
        #   - No modality embedding needed: is_yz already encodes view identity; separate encoder
        #     from LYSO already encodes subsystem identity
        #
        # LYSO: [x, y, z, E, t] = 5D
        #   - Genuine 3D crystal positions, no projection ambiguity
        #   - Time retained: shower timing carries real calorimeter structure
        #   - No modality embedding needed: separate encoder from ATAR encodes subsystem identity
        # ATAR View-Aware Encoder
        # Physics features: [transverse_coord, z, E] = 3D
        # View identity: learnable embedding for XZ(0) vs YZ(1), added to projection
        self.atar_feature_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.atar_view_embedding = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.atar_view_embedding.weight, std=0.02)
        self.lyso_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # --- Separate message passing stacks per subsystem ---
        # No shared edge_dim mismatch; each subsystem's geometry is self-consistent.
        # ATAR edges (4D): [d_transverse, dz, dE, is_cross_view]
        #   - d_transverse = dx for XZ-XZ, dy for YZ-YZ, 0 for XZ-YZ (no shared transverse axis)
        #   - is_cross_view = 1 for XZ-YZ, 0 otherwise
        #   - NO time (same reasoning as node features)
        # LYSO edges (5D): [dx, dy, dz, dE, dt]
        self.atar_blocks = nn.ModuleList([
            JointAttentionBlock(hidden_dim=hidden_dim, heads=heads, edge_dim=4, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.lyso_blocks = nn.ModuleList([
            JointAttentionBlock(hidden_dim=hidden_dim, heads=heads, edge_dim=5, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        from torch_geometric.nn import JumpingKnowledge, AttentionalAggregation

        # Separate JK per subsystem — ATAR and LYSO representations never share a joint JK
        self.atar_jk = JumpingKnowledge(mode="cat")
        self.lyso_jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden_dim * num_blocks

        # Early Fusion via Cross-Attention Bridge
        self.cross_attention = nn.MultiheadAttention(embed_dim=jk_dim, num_heads=5, batch_first=True)

        self.num_pdg_classes = num_pdg_classes

        # --- ATAR Specific Heads ---
        # 1. Time-Slice PDG (Group Classifier)
        # Body compresses pool_all (jk_dim) to a bottleneck; total slice energy (1 scalar)
        # is injected at the final layer so it isn't swamped by the high-dim pooled vector.
        _pdg_hidden = hidden_dim
        self.atar_slice_pdg_body = nn.Sequential(
            nn.Linear(jk_dim, _pdg_hidden), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(_pdg_hidden, _pdg_hidden // 2), nn.GELU(), nn.Dropout(0.05),
        )
        self.atar_slice_pdg_norm = nn.LayerNorm(_pdg_hidden // 2 + 1)
        self.atar_slice_pdg_final = nn.Linear(_pdg_hidden // 2 + 1, num_pdg_classes)

        # 2. Time-Slice Multi-Event Flag (Binary Classifier)
        # Input Dimensions: 4 * jk_dim (XZ/YZ Mean/Max) + 4 (Counts/Sums)
        self.atar_slice_multi_head = nn.Sequential(
            nn.Linear(jk_dim * 4 + 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 1)
        )
        
        # 2.5 Multi-Particle Context Bridges (Isolated X and Y)
        # 8D each view-specific context bridge (binary pileup signal, compact)
        self.multi_x_context_head = nn.Sequential(
            nn.Linear(jk_dim, 16), nn.GELU(), nn.Linear(16, 8)
        )
        self.multi_y_context_head = nn.Sequential(
            nn.Linear(jk_dim, 16), nn.GELU(), nn.Linear(16, 8)
        )

        # 2.6 Global Event Context Bridges (Isolated X and Y)
        # 32D each view-specific global context (richer event-level structure)
        self.global_x_context_head = nn.Sequential(
            nn.Linear(jk_dim, 48), nn.GELU(), nn.Linear(48, 32)
        )
        self.global_y_context_head = nn.Sequential(
            nn.Linear(jk_dim, 48), nn.GELU(), nn.Linear(48, 32)
        )

        # 3. Splitter (Node PDG) - Input: Node JK
        # Body compresses per-hit jk representation; total slice energy injected at final layer.
        self.atar_pdg_body = nn.Sequential(
            nn.Linear(jk_dim, _pdg_hidden), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(_pdg_hidden, _pdg_hidden // 2), nn.GELU(), nn.Dropout(0.05),
        )
        self.atar_pdg_norm = nn.LayerNorm(_pdg_hidden // 2 + 1)
        self.atar_pdg_final = nn.Linear(_pdg_hidden // 2 + 1, 3)

        # --- Unified Event Synthesis (Object-Level Transformer) ---
        D_A = 256
        self.D_A = D_A  # stored so forward pass can reference it without redefining

        # Per-view ATAR Kinematics MLPs
        # Each view pools its own hit information with its own kinematic info:
        #   XZ: x-endpoint + z-endpoint (+ shared slice PDG), 12 + 3 = 15 -> 64
        #   YZ: y-endpoint + z-endpoint (+ shared slice PDG), 12 + 3 = 15 -> 64
        # Z is stereo so it appears in both views; PDG is slice-level and shared.
        self.atar_kinematics_mlp_xz = nn.Sequential(
            nn.Linear(12 + 3, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        self.atar_kinematics_mlp_yz = nn.Sequential(
            nn.Linear(12 + 3, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

        # Stereo pool projections: jk_dim -> 128D each (compress for balanced event token)
        self.pool_x_event_proj = nn.Sequential(nn.Linear(jk_dim, 128), nn.GELU())
        self.pool_y_event_proj = nn.Sequential(nn.Linear(jk_dim, 128), nn.GELU())

        # Per-view ATAR Event MLPs: fuse view-specific pool + view-specific kinematics
        # within each view, producing a D_A//2 half-token per view. The full slice token is
        # [token_xz || token_yz] so that downstream self-attention sees both pools of both
        # slices simultaneously when comparing pairs.
        # No time feature: absolute slice time and anchor-relative slice ordering are
        # both omitted to keep the role-head time-blind. False positives on out-of-window
        # pimu events should then be flat in time, preserving the assumption used in the
        # late-tail background subtraction.
        _D_HALF = D_A // 2
        self.atar_event_mlp_xz = nn.Sequential(
            nn.Linear(128 + 64, _D_HALF),
            nn.GELU(),
            nn.Linear(_D_HALF, _D_HALF)
        )
        self.atar_event_mlp_yz = nn.Sequential(
            nn.Linear(128 + 64, _D_HALF),
            nn.GELU(),
            nn.Linear(_D_HALF, _D_HALF)
        )

        # Anchor flag embedding: 2 entries (not-anchor / anchor) × D_A added to slice
        # tokens. Matches the standard additive positional-encoding pattern used
        # elsewhere. Carries NO ordering information — only "is the triggering pion
        # slice or not." Required because the transformer otherwise treats all
        # slices symmetrically once time-feat and position-embedding are gone.
        self.anchor_flag_embedding = nn.Embedding(2, D_A)

        # Dead-material energy regression head (small MLP, detached inputs).
        # 8 inputs: cos θ pos, cos φ pos, cos θ exit, cos φ exit, total
        # positron-tagged energy normalized by NORM_E_LYSO, energy-weighted
        # mean LYSO position (x, y, z) — already normalized by NORM_POS_LYSO.
        # Output: log(1 + dead_E_pred) in MeV.
        self.dead_energy_head = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

        # --- Slim Event Builder (D_EVENT=32) ---
        # Each LYSO cluster token is built from 12 raw physics features
        # projected to 32D. ATAR tokens are projected down from D_A.
        # A small transformer lets clusters see each other and the ATAR
        # track, then the classifier concatenates raw [coinc_feat, cos_sep]
        # for the final decision.
        D_EVENT = 32
        self.D_EVENT = D_EVENT

        # LYSO: 12 raw physics features → 32D token
        self.lyso_event_proj = nn.Linear(9, D_EVENT)

        # ATAR: project detached 256D tokens down to 32D
        self.atar_event_down = nn.Linear(D_A, D_EVENT)

        # Modality + position embeddings at event builder dim
        self.event_modality_emb = nn.Embedding(2, D_EVENT)
        self.event_slice_emb = nn.Embedding(64, D_EVENT)

        # Slim transformer: D=32, 2 heads, 2 layers
        slim_layer = nn.TransformerEncoderLayer(
            d_model=D_EVENT, nhead=2, dim_feedforward=D_EVENT * 4,
            batch_first=True, dropout=0.1
        )
        self.slim_event_transformer = nn.TransformerEncoder(slim_layer, num_layers=2)

        # Decision head: project transformer 32D → 8D, concat with 2 physics features → 10D → 1 logit
        # Balances transformer context (8 dims) with physics features (2 dims)
        self.event_head_proj = nn.Linear(D_EVENT, 8)
        self.event_head = nn.Linear(8 + 2, 1)

        # Graph Level Pooling (Independent Splitting)
        def make_pool():
            return AttentionalAggregation(nn.Sequential(
                nn.Linear(jk_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 1)
            ))

        # Joint pool over all ATAR hits (both XZ + YZ): used for PDG and Z endpoint head.
        # View-agnostic tasks benefit from attending over the full hit set rather than
        # separately pooling each view and taking a fixed equal-weight mean.
        self.pool_all = make_pool()

        # Per-view pools: used for X and Y endpoint heads (view-specific tasks)
        self.pool_x_shared = make_pool()
        self.pool_y_shared = make_pool()
        
        # Multi-particle head remains independent and un-masked
        self.pool_x_multi = make_pool()
        self.pool_y_multi = make_pool()
        
        # Global context pools (Event-level aggregation)
        self.pool_x_global = make_pool()
        self.pool_y_global = make_pool()

        # Dedicated stereo pools for event token construction
        self.pool_x_event = make_pool()
        self.pool_y_event = make_pool()

        # --- Phase 9: ATAR-Only Event Building ---
        # Stacked self-attention for multi-hop chain reasoning across slices.
        # Two layers cover the longest real chain (π→μ→e → 2 hops from
        # positron back to pion); JK concatenation of the pre-attention
        # input plus each layer's output lets the classifier pick its own
        # effective depth per slice.
        self.L_TRIG = 2
        self.atar_event_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_A, nhead=4, dim_feedforward=2 * D_A,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(self.L_TRIG)
        ])
        # Role-based attachment head.
        # Input per slice: [ self_token | anchor_token | self - anchor | Δendpoint (6) ]
        # Output: 3-class logits over {none, muon-in-chain, positron-in-chain}.
        # The triggering pion itself is the anchor; it is not predicted by this head.
        JK_DIM = (self.L_TRIG + 1) * D_A
        ROLE_IN = 3 * JK_DIM + 6
        self.atar_role_head = nn.Sequential(
            nn.Linear(ROLE_IN, D_A), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(D_A, D_A // 2), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(D_A // 2, 3),
        )

        # Per-axis scale for the endpoint-compatibility kernel that biases
        # cross-slice attention. Raw parameter is passed through softplus to
        # keep the axis weights positive; init at -2 so softplus(-2) ≈ 0.13
        # starts the bias weak, letting training adjust its magnitude.
        self.kernel_alpha = nn.Parameter(torch.full((3,), -2.0))

        # --- Phase 10: Pion Stop (stereo softmax pooling + per-axis residual) ---
        # Mirrors the endpoint head's structure: view-specific residuals for
        # the transverse coords (x sees x-branch only, y sees y-branch only)
        # and a stereo residual for z (sees both branches).
        # Score factorization per hit:
        #   log-score = logit_trig (detached) + logit_pion (detached) + logit_endpoint
        # Trigger + pion act as a soft gate (confidently non-pion hits get
        # large-negative log-odds → softmax ignores them). The endpoint
        # scorer is the only learnable path from pion_stop loss to the
        # softmax weights, so it specializes in "which pion hit is the stop?".
        self.pion_endpoint_scorer = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.pion_pool_log_tau = nn.Parameter(torch.tensor(0.0))
        # Bounded residuals: one MLP per coordinate.
        self.pion_stop_residual_x = nn.Sequential(
            nn.Linear(1 + jk_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.pion_stop_residual_y = nn.Sequential(
            nn.Linear(1 + jk_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.pion_stop_residual_z = nn.Sequential(
            nn.Linear(1 + 2 * jk_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Per-axis scale on the tanh residual (same units as x[:, 0:3]).
        self.pion_stop_residual_scale = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))

        # --- Phase 11: Positron Direction (stereo pools, joint head) ---
        # Hard-gated positron hits are mean-pooled per view (x-strip vs y-strip).
        # Both pools + detached pion stop feed one VectorHead that emits the
        # full (dx, dy, dz) jointly, then F.normalize. Stereo separation avoids
        # view-feature pollution; joint output ensures the three components
        # are coherent as a unit vector.
        self.positron_dir_head = VectorHead(2 * jk_dim + 3, hidden_dim)

        # --- Phase 11b: has-trigger-positron ---
        # Replaced by a pure decision rule in forward(): any slice with >=2
        # qualifying (trigger>0.5 AND mip>0.5) hits. Learned head disabled.
        # self.has_trigger_positron_head = nn.Sequential(
        #     nn.Linear(4 + 3, 32), nn.GELU(),
        #     nn.Linear(32, 16), nn.GELU(),
        #     nn.Linear(16, 1),
        # )

        # --- Phase 14: Attention bias parameters (used in slim event builder) ---
        # Per-token temporal σ. ATAR is always a MIP → single constant contribution.
        # LYSO is photon-statistics-limited: σ(E) = σ_lyso_floor + σ_lyso_scale/√E.
        # Pair σ² = σ_i² + σ_j², applied per-token via modality flag.
        self.sigma_t_atar_ns = nn.Parameter(torch.tensor(1.0))
        self.sigma_t_lyso_floor_ns = nn.Parameter(torch.tensor(0.5))
        self.sigma_t_lyso_scale_ns = nn.Parameter(torch.tensor(2.0))
        # Directional bias σ_angle(E_cluster) — MCS-dominated via positron E proxied
        # by cluster E. σ_angle = σ_a_floor + σ_a_scale · √(E_ref/E): higher cluster
        # E → smaller σ → sharper directional bias. Bias is cos_align / σ_angle.
        self.angle_sigma_floor = nn.Parameter(torch.tensor(0.5))
        self.angle_sigma_scale = nn.Parameter(torch.tensor(1.0))
        # Legacy params kept so old checkpoints still load; unused in forward.
        self.direction_bias_temperature = nn.Parameter(torch.tensor(1.0))
        self.temporal_bias_sigma = nn.Parameter(torch.tensor(2.0))

        # Asymmetric regressors for Kinematics with Ortho-Context Injection
        # Input Dimensions:
        # X/Y: jk_dim (AttentionalAgg) + 32 (Global) + 8 (Multi) = jk_dim + 40
        # Z:   jk_dim (Joint AttentionalAgg) + 2*32 (Global X+Y) + 2*8 (Multi X+Y) = jk_dim + 80
        self.atar_endpoint_x = QuantileOutputHead(input_dim=jk_dim + 40, num_points=2, coords=1)
        self.atar_endpoint_y = QuantileOutputHead(input_dim=jk_dim + 40, num_points=2, coords=1)
        self.atar_endpoint_z = QuantileOutputHead(input_dim=jk_dim + 80, num_points=2, coords=1)
        
        # --- LYSO Specific Heads (Object Condensation) ---
        # Node Level
        self.lyso_beta_head = nn.Sequential(
            nn.Linear(jk_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.lyso_cluster_coord_head = nn.Sequential(
            nn.Linear(jk_dim, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.lyso_fraction_head = nn.Sequential(
            nn.Linear(jk_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, batch, task_weights=None, triggering_pion_slice=None):
        """
        x: [N_total_hits, 6] (features + modality_idx)
        batch: [N_total_hits] PyG batch index
        triggering_pion_slice: [B] int tensor giving the LOCAL (within-event, over
            valid slices in ascending slice-id order) index of the anchor pion
            slice for each event. -1 means "no anchor" (background event, or
            pion not visible in ATAR). If None, the anchor is predicted from
            slice-PDG + slice-time.
        """
        if task_weights is None: task_weights = {}

        # 0. Extract modality flags and slice IDs
        # x columns: [pos_x, pos_y, pos_z, energy, time, is_xz, is_yz, is_lyso, slice_id]
        is_atar_x = (x[:, 5] > 0.5)
        is_atar_y = (x[:, 6] > 0.5)
        is_atar = is_atar_x | is_atar_y
        is_lyso = (x[:, 7] > 0.5)
        
        # 1. Encode nodes — subsystem-separated, clean feature sets
        # ATAR: [transverse_coord, z, E, is_yz]
        #   transverse_coord = x for XZ strips, y for YZ strips (no zero-padding a real coordinate)
        #   is_yz encodes view identity; separate encoder from LYSO encodes subsystem identity
        # LYSO: [x, y, z, E, t]  (genuine 3D + time for shower structure)
        hidden_dim = self.atar_feature_proj[0].out_features
        h_atar_in = torch.zeros(x.size(0), hidden_dim, device=x.device)
        h_lyso_in = torch.zeros(x.size(0), hidden_dim, device=x.device)

        if is_atar.any():
            # View-Aware Encoding: project physics [transverse, z, E] + add learnable view token
            transverse = torch.where(is_atar_y[is_atar].unsqueeze(1), x[is_atar, 1:2], x[is_atar, 0:1])
            atar_phys = torch.cat([transverse, x[is_atar, 2:4]], dim=1)  # [transverse, z, E] = 3D
            h_proj = self.atar_feature_proj(atar_phys)
            view_idx = x[is_atar, 6].long()  # is_yz: 0=XZ, 1=YZ
            h_atar_in[is_atar] = h_proj + self.atar_view_embedding(view_idx)

        if is_lyso.any():
            h_lyso_in[is_lyso] = self.lyso_encoder(x[is_lyso, :5])  # [x, y, z, E, t]

        # 2. Physics-Based Edge Construction — intra-slice, no cross-subsystem edges
        edge_index = physics_edge_index_batch(x, batch)

        # 3. Separate edge features and message passing per subsystem
        # ATAR edges (4D): [d_transverse, dz, dE, is_cross_view]  — see build_atar_edge_attr
        # LYSO edges (5D): [dx, dy, dz, dE, dt]
        # We run each subsystem through its own block stack and JK independently,
        # then recombine into h_out for the cross-attention bridge and downstream heads.

        # Split edge index by subsystem (ATAR-only and LYSO-only edges are already separate
        # since physics_edge_index_batch uses cluster_id = batch*10000 + slice_id*10 + subsys_id)
        src, dst = edge_index
        edge_is_atar = is_atar[src] & is_atar[dst]
        edge_is_lyso = is_lyso[src] & is_lyso[dst]

        atar_edge_index = edge_index[:, edge_is_atar]
        lyso_edge_index = edge_index[:, edge_is_lyso]

        atar_edge_attr = build_atar_edge_attr(x, atar_edge_index)  # [E_atar, 4]
        lyso_edge_attr = build_lyso_edge_attr(x, lyso_edge_index)  # [E_lyso, 5]

        # ATAR message passing
        h_atar = h_atar_in
        atar_xs = []
        if is_atar.any():
            for block in self.atar_blocks:
                h_atar = block(h_atar, atar_edge_index, atar_edge_attr)
                atar_xs.append(h_atar[is_atar])
            h_atar_jk = self.atar_jk(atar_xs)  # [N_atar, jk_dim]
        else:
            h_atar_jk = torch.zeros(0, self.atar_feature_proj[0].out_features * len(self.atar_blocks), device=x.device)

        # LYSO message passing
        h_lyso = h_lyso_in
        lyso_xs = []
        if is_lyso.any():
            for block in self.lyso_blocks:
                h_lyso = block(h_lyso, lyso_edge_index, lyso_edge_attr)
                lyso_xs.append(h_lyso[is_lyso])
            h_lyso_jk = self.lyso_jk(lyso_xs)  # [N_lyso, jk_dim]
        else:
            h_lyso_jk = torch.zeros(0, self.lyso_encoder[0].out_features * len(self.lyso_blocks), device=x.device)

        # Recombine into a single [N_total, jk_dim] tensor for cross-attention and downstream heads
        jk_dim = h_atar_jk.shape[1] if is_atar.any() else h_lyso_jk.shape[1]
        h_out = torch.zeros(x.size(0), jk_dim, device=x.device)
        if is_atar.any():
            h_out[is_atar] = h_atar_jk
        if is_lyso.any():
            h_out[is_lyso] = h_lyso_jk

        # Phase 2: The Dense Cross-Attention Bridge
        h_out_new = h_out.clone()
        if is_atar.any() and is_lyso.any():
            h_atar = h_out[is_atar]
            h_calo = h_out[is_lyso]
            batch_atar = batch[is_atar]
            batch_calo = batch[is_lyso]
            
            batch_size = int(batch.max().item() + 1)
            from torch_geometric.utils import to_dense_batch
            
            # Convert to padded dense batches
            h_atar_dense, atar_mask = to_dense_batch(h_atar, batch_atar, batch_size=batch_size)
            h_calo_dense, calo_mask = to_dense_batch(h_calo, batch_calo, batch_size=batch_size)
            
            # Time-Proximity Masking: Only allow cross-attention within 5ns
            # slice_mean_t is stored in column 9 (raw ns)
            if x.shape[1] > 9:
                atar_mean_t = x[is_atar, 9]
                lyso_mean_t = x[is_lyso, 9]
                atar_t_dense, _ = to_dense_batch(atar_mean_t, batch_atar, batch_size=batch_size)  # [B, N_atar_max]
                lyso_t_dense, _ = to_dense_batch(lyso_mean_t, batch_calo, batch_size=batch_size)  # [B, N_lyso_max]
                
                # |t_atar - t_lyso| > 5.0 ns → block attention (additive mask: 0=allow, -inf=block)
                time_diff = torch.abs(atar_t_dense.unsqueeze(2) - lyso_t_dense.unsqueeze(1))  # [B, N_atar, N_lyso]
                time_block = (time_diff > 5.0)
                
                # Track which queries have no valid calo keys (all blocked by time gate + padding)
                combined_block = time_block | (~calo_mask.unsqueeze(1))  # [B, N_atar, N_lyso]
                all_keys_blocked = combined_block.all(dim=2)  # [B, N_atar]

                # Expand for num_heads (nn.MHA expects [B*nhead, L, S] for 3D attn_mask)
                # For fully-blocked queries, unblock the time gate so MHA computes a valid
                # (non-NaN) attention output — we zero those outputs below anyway, so the
                # unblocking only prevents NaN gradients, not actual signal leakage.
                safe_time_block = time_block & (~all_keys_blocked.unsqueeze(2))
                num_heads = self.cross_attention.num_heads
                attn_mask = torch.zeros_like(safe_time_block, dtype=h_atar_dense.dtype)
                attn_mask[safe_time_block] = float('-inf')
                attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)  # [B*nhead, N_atar, N_lyso]
            else:
                attn_mask = None
                all_keys_blocked = None

            # Cross attention: ATAR queries Calo (time-gated)
            # Detach key/value so ATAR-head gradients do not back-propagate into
            # the LYSO backbone. Without this, every ATAR loss (PDG, pion_stop,
            # positron_angle, trigger_slice, …) flows through cross_attention.W_k
            # and W_v into h_calo → h_lyso_jk → lyso_blocks → lyso_encoder,
            # drowning out w_lyso_condensation and causing OC to plateau on an
            # ATAR-optimized representation. ATAR still *reads* LYSO context in
            # the forward pass; it just no longer trains the LYSO stack.
            h_atar_enriched, _ = self.cross_attention(
                query=h_atar_dense,
                key=h_calo_dense.detach(),
                value=h_calo_dense.detach(),
                key_padding_mask=~calo_mask,
                attn_mask=attn_mask
            )

            # For queries with no valid calo keys (stopped/upstream particles), zero out
            # the cross-attention contribution so the residual acts as identity — no noisy
            # LYSO signal is added for particles that have no calorimeter counterpart.
            if all_keys_blocked is not None:
                h_atar_enriched[all_keys_blocked] = 0.0

            # Residual connection
            h_atar_dense = h_atar_dense + h_atar_enriched
            h_atar = h_atar_dense[atar_mask]
            
            # Repopulate
            h_out_new[is_atar] = h_atar
            
        h_out = h_out_new
        
        # --- 5. Routing to Specialized Heads ---
        output = {}
        
        # === Time Slice Grouping ===
        # slice_id is now at column 8
        if x.shape[1] > 8:
            slice_ids_dense = x[:, 8].long()
        else:
            slice_ids_dense = torch.zeros_like(batch)
            
        # We need to map (batch_id, slice_id) into a unique ID so we can pool 
        # distinct slices separately across the batch.
        num_slices_max = slice_ids_dense.max().item() + 1
        num_graphs_in_batch = batch.max().item() + 1
        
        # unique_slice_id = batch_id * num_slices_max + slice_id
        global_slice_ids = batch * num_slices_max + slice_ids_dense
        num_global_slices = num_graphs_in_batch * num_slices_max

        count_x = torch.zeros(num_global_slices, 1, device=x.device)
        count_y = torch.zeros(num_global_slices, 1, device=x.device)
        count_lyso = torch.zeros(num_global_slices, 1, device=x.device)

        # === ATAR PREDICTIONS ===
        if is_atar.any():
            h_atar = h_out[is_atar]
            
            # --- PHASE 1: Time-Slice Group Classification ---
            h_atar_x = h_out[is_atar_x]
            global_slice_idx_x = global_slice_ids[is_atar_x]
            
            h_atar_y = h_out[is_atar_y]
            global_slice_idx_y = global_slice_ids[is_atar_y]

            # Joint index over all ATAR hits (both views) for view-agnostic pooling
            global_slice_idx_all = global_slice_ids[is_atar]

            # Total energy per slice — injected late into PDG heads (col 3 = E)
            energy_per_hit = x[is_atar, 3].unsqueeze(1)  # [N_atar, 1]
            slice_energy = global_add_pool(energy_per_hit, global_slice_idx_all, size=num_global_slices)  # [num_global_slices, 1]

            # Scatter ones to verify hits exist
            count_x.index_add_(0, global_slice_idx_x, torch.ones_like(global_slice_idx_x, dtype=torch.float).unsqueeze(1))
            count_y.index_add_(0, global_slice_idx_y, torch.ones_like(global_slice_idx_y, dtype=torch.float).unsqueeze(1))

            has_x = (count_x > 0).squeeze()
            has_y = (count_y > 0).squeeze()

            # Apply PyG AttentionalAggregation per Time Slice with specific Splitter masks
            def pool_with_soft_mask(pool_layer_x, pool_layer_y, mask_x, mask_y):
                hx = h_atar_x * mask_x if mask_x is not None else h_atar_x
                hy = h_atar_y * mask_y if mask_y is not None else h_atar_y
                
                px = pool_layer_x(hx, global_slice_idx_x, dim_size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=hx.device)
                py = pool_layer_y(hy, global_slice_idx_y, dim_size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=hy.device)
                
                return px, py
                
            def pool_with_hard_mask(pool_layer_x, pool_layer_y, mask_x, mask_y):
                # Pass the explicit mask array directly to the MaskedAttentionalAggregation
                # layer so it can selectively drop the softmax gates for background hits.
                px = pool_layer_x(h_atar_x, global_slice_idx_x, dim_size=num_global_slices, mask=mask_x) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_x.device)
                py = pool_layer_y(h_atar_y, global_slice_idx_y, dim_size=num_global_slices, mask=mask_y) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_y.device)
                
                return px, py
                
            # Bring back has_x to unsqueezed for math
            has_x_f = has_x.float().unsqueeze(1)
            has_y_f = has_y.float().unsqueeze(1)
            
            # Combine X and Y 
            valid_slice_mask = ((has_x_f + has_y_f) > 0).squeeze() # Slices that actually have ATAR hits
            output['valid_slice_mask'] = valid_slice_mask
            
            def safe_mean(px, py):
                return ((px * has_x_f) + (py * has_y_f)) / (has_x_f + has_y_f).clamp(min=1.0)
            
            # --- PHASE 1: Evaluate Node-Level Predictions First! ---
            # Node PDG uses the new body+final architecture with late energy injection.
            # Each hit gets the total energy of its enclosing time-slice appended at the
            # final layer so the high-dim JK features aren't swamped by a single scalar.
            if task_weights.get('w_node_pdg', 1.0) > 0.0:
                node_pdg_hidden = self.atar_pdg_body(h_atar)  # [N_atar, hidden//2]
                # Broadcast per-slice energy to every hit in that slice
                node_slice_energy = slice_energy[global_slice_idx_all]  # [N_atar, 1]
                node_pdg_input = torch.cat([node_pdg_hidden, node_slice_energy], dim=1)  # [N_atar, hidden//2 + 1]
                node_pdg_input = self.atar_pdg_norm(node_pdg_input)
                output['atar_node_pdg'] = self.atar_pdg_final(node_pdg_input)
            else:
                output['atar_node_pdg'] = torch.zeros(h_atar.size(0), self.num_pdg_classes, device=x.device)

            atar_node_probs = torch.sigmoid(output['atar_node_pdg'])

            # --- PHASE 2: View-Isolated Context Extraction ---
            
            # 2.1 Global Context Bridges (Event Level)
            pool_x_global = self.pool_x_global(h_atar_x, batch[is_atar_x], dim_size=num_graphs_in_batch) if is_atar_x.any() else torch.zeros(num_graphs_in_batch, jk_dim, device=x.device)
            pool_y_global = self.pool_y_global(h_atar_y, batch[is_atar_y], dim_size=num_graphs_in_batch) if is_atar_y.any() else torch.zeros(num_graphs_in_batch, jk_dim, device=x.device)
            
            # Project Global View contexts (Broadcast back to all valid slices in event)
            # Each global_slice_idx has a unique 1:1 mapping to batch ID via integer division.
            slice_to_batch = torch.arange(num_global_slices, device=x.device) // num_slices_max
            valid_slice_batch_ids = slice_to_batch[valid_slice_mask]
            
            global_x_32d = self.global_x_context_head(pool_x_global)[valid_slice_batch_ids]
            global_y_32d = self.global_y_context_head(pool_y_global)[valid_slice_batch_ids]
            
            # 2.2 Multi-Particle Context Bridges (Slice Level)
            pool_x_multi_attn = self.pool_x_multi(h_atar_x, global_slice_idx_x, dim_size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_x.device)
            pool_y_multi_attn = self.pool_y_multi(h_atar_y, global_slice_idx_y, dim_size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_y.device)
            
            # --- PHASE 2.5: Cardinality-Aware Multi-Event Prediction ---
            # Explicitly compute Hit Counts and Energy Sums (Raw Features)
            # Row 3 and 4 in x are energy features (already normalized)
            energy_x = x[is_atar_x, 3]
            energy_y = x[is_atar_y, 3]
            
            sum_x = global_add_pool(energy_x.unsqueeze(1), global_slice_idx_x, size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, 1, device=x.device)
            sum_y = global_add_pool(energy_y.unsqueeze(1), global_slice_idx_y, size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, 1, device=x.device)
            
            ones_x = torch.ones(is_atar_x.sum(), 1, device=x.device)
            ones_y = torch.ones(is_atar_y.sum(), 1, device=x.device)
            count_x = global_add_pool(ones_x, global_slice_idx_x, size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, 1, device=x.device)
            count_y = global_add_pool(ones_y, global_slice_idx_y, size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, 1, device=x.device)
            
            # Feature Max (XZ and YZ)
            max_x_multi = global_max_pool(h_atar_x, global_slice_idx_x, size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=x.device)
            max_y_multi = global_max_pool(h_atar_y, global_slice_idx_y, size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=x.device)
            
            # Assemble Multi-Vector (Only for valid slices)
            # Features: [X_Mean, Y_Mean, X_Max, Y_Max, counts, sums]
            valid_slice_counts = torch.cat([count_x, count_y], dim=-1)[valid_slice_mask] / 100.0 # Normalize roughly
            valid_slice_sums = torch.cat([sum_x, sum_y], dim=-1)[valid_slice_mask] / 1.0 # Normalize roughly
            
            valid_slice_multi_input = torch.cat([
                pool_x_multi_attn[valid_slice_mask], 
                pool_y_multi_attn[valid_slice_mask],
                max_x_multi[valid_slice_mask], 
                max_y_multi[valid_slice_mask],
                valid_slice_counts, 
                valid_slice_sums
            ], dim=-1)
            
            # Predict Multi-Event Slice Flag
            output['atar_slice_multi'] = self.atar_slice_multi_head(valid_slice_multi_input).squeeze(-1)
            
            # Ortho-Context stereo bridge used for PDG and Downstream (uses weighted mean of views)
            stereo_multi = safe_mean(pool_x_multi_attn, pool_y_multi_attn)[valid_slice_mask]
            
            valid_pool_x_multi = pool_x_multi_attn[valid_slice_mask]
            valid_pool_y_multi = pool_y_multi_attn[valid_slice_mask]
            
            # Project Multi contexts (For Endpoint Regressors)
            multi_x_8d = self.multi_x_context_head(valid_pool_x_multi)
            multi_y_8d = self.multi_y_context_head(valid_pool_y_multi)

            # --- PHASE 4: Final Weighted Pooling ---
            # Joint pool over all ATAR hits (both views): used for view-agnostic tasks
            pool_all = self.pool_all(h_atar, global_slice_idx_all, dim_size=num_global_slices)
            pool_all_valid = pool_all[valid_slice_mask]

            # Per-view pools for X and Y endpoint heads
            pool_x_shared = self.pool_x_shared(h_atar_x, global_slice_idx_x, dim_size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_x.device)
            pool_y_shared = self.pool_y_shared(h_atar_y, global_slice_idx_y, dim_size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_y.device)

            # Predict Time-Slice PDG from joint pool (all ATAR hits, both views)
            # Uses the new body+final architecture with late energy injection:
            # the body compresses pool_all (jk_dim) to a bottleneck, then total
            # slice energy is concatenated before the final classification layer.
            slice_pdg_hidden = self.atar_slice_pdg_body(pool_all_valid)  # [N_valid_slices, hidden//2]
            valid_slice_energy = slice_energy[valid_slice_mask]  # [N_valid_slices, 1]
            slice_pdg_input = torch.cat([slice_pdg_hidden, valid_slice_energy], dim=1)  # [N_valid_slices, hidden//2 + 1]
            slice_pdg_input = self.atar_slice_pdg_norm(slice_pdg_input)
            slice_logits = self.atar_slice_pdg_final(slice_pdg_input)
            output['atar_slice_pdg'] = slice_logits
            
            # --- PHASE 4: Evaluate Sub-Heads ---
            valid_x_shared = pool_x_shared[valid_slice_mask]
            valid_y_shared = pool_y_shared[valid_slice_mask]

            # Stage 1: Expert Projections with Ortho-Context Fusion
            # X Context: [Local Attn Pool (jk), Global X (32), Multi X (8)]
            valid_x_concat = torch.cat([valid_x_shared, global_x_32d, multi_x_8d], dim=-1)

            # Y Context: [Local Attn Pool (jk), Global Y (32), Multi Y (8)]
            valid_y_concat = torch.cat([valid_y_shared, global_y_32d, multi_y_8d], dim=-1)

            # Z Context: [Joint Attn Pool (jk), Global X+Y (48), Multi X+Y (32)]
            z_context_stereo = torch.cat([global_x_32d, global_y_32d, multi_x_8d, multi_y_8d], dim=-1)
            stereo_concat = torch.cat([pool_all_valid, z_context_stereo], dim=-1)
            
            x_pred_expert = self.atar_endpoint_x(valid_x_concat)
            y_pred_expert = self.atar_endpoint_y(valid_y_concat)
            z_pred_expert = self.atar_endpoint_z(stereo_concat)

            # Final output: Independent specialists aligned by Z-progressions
            output['atar_endpoints'] = torch.cat([x_pred_expert, y_pred_expert, z_pred_expert], dim=2)


            # Export expert predictions for optional auxiliary loss monitoring
            output['atar_endpoints_expert_x'] = x_pred_expert
            output['atar_endpoints_expert_y'] = y_pred_expert
            output['atar_endpoints_expert_z'] = z_pred_expert
            
            # [Pion stop and positron angle heads moved to Phase 10/11 after ATAR event building]

            # Return meta information to unroll predictions downstream
            output['valid_slice_mask'] = valid_slice_mask
            output['valid_slice_indices'] = torch.nonzero(valid_slice_mask).squeeze(1)
            output['num_graphs_in_batch'] = num_graphs_in_batch
            output['num_slices_max'] = num_slices_max
            
            # --- ATAR Event Builder Early Fusion (stereo) ---
            # Endpoints: [N_slices, 2 (points), 3 (coords=x,y,z), 3 (quantiles)].
            # Split into per-view kinematic bundles so that each view's hit pool is fused with
            # that view's endpoint info (xz pool ↔ x-endpoint + z-endpoint, same for yz).
            endpoints_all = output['atar_endpoints']
            # xz and yz each take 2 coords × 2 points × 3 quantiles = 12 features.
            endpoints_xz_flat = endpoints_all[:, :, [0, 2], :].reshape(endpoints_all.size(0), -1)
            endpoints_yz_flat = endpoints_all[:, :, [1, 2], :].reshape(endpoints_all.size(0), -1)
            slice_pdg_det = output['atar_slice_pdg']

            atar_kin_xz = self.atar_kinematics_mlp_xz(torch.cat([endpoints_xz_flat, slice_pdg_det], dim=1))
            atar_kin_yz = self.atar_kinematics_mlp_yz(torch.cat([endpoints_yz_flat, slice_pdg_det], dim=1))

            # Per-slice energy-weighted mean time (col 4 = normalized time)
            hit_times = x[is_atar, 4]
            hit_energies = x[is_atar, 3].clamp(min=1e-6)
            slice_time_wsum = torch.zeros(num_global_slices, device=x.device)
            slice_energy_sum_t = torch.zeros(num_global_slices, device=x.device)
            slice_time_wsum.index_add_(0, global_slice_idx_all, hit_times * hit_energies)
            slice_energy_sum_t.index_add_(0, global_slice_idx_all, hit_energies)
            slice_mean_time = (slice_time_wsum / slice_energy_sum_t.clamp(min=1e-6))[valid_slice_mask]  # [N_valid_slices]

            # Stereo pooling for event tokens (dedicated pools, view-separated)
            pool_x_ev = self.pool_x_event(h_atar_x, global_slice_idx_x, dim_size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=x.device)
            pool_y_ev = self.pool_y_event(h_atar_y, global_slice_idx_y, dim_size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=x.device)
            proj_x_ev = self.pool_x_event_proj(pool_x_ev[valid_slice_mask])  # [N_valid_slices, 128]
            proj_y_ev = self.pool_y_event_proj(pool_y_ev[valid_slice_mask])  # [N_valid_slices, 128]

            # Per-view slice half-tokens: fuse view-specific pool + view-specific kinematics.
            # Final slice token is [token_xz || token_yz], so the subsequent self-attention sees
            # both pools of both slices simultaneously when comparing pairs.
            # No time/position features here — see __init__ comment for rationale (FP-time flatness).
            token_xz = self.atar_event_mlp_xz(torch.cat([proj_x_ev, atar_kin_xz], dim=1))
            token_yz = self.atar_event_mlp_yz(torch.cat([proj_y_ev, atar_kin_yz], dim=1))
            atar_event_tokens = torch.cat([token_xz, token_yz], dim=1)

            # Anchor flag: add a learned D_A-vector to slice tokens marking which slice
            # is the triggering pion. No ordering information — purely "is anchor or not."
            valid_slice_indices = torch.nonzero(valid_slice_mask).squeeze(1)
            atar_event_batch_pe = valid_slice_indices // num_slices_max
            slices_per_event_pe = torch.zeros(num_graphs_in_batch, dtype=torch.long, device=x.device)
            slices_per_event_pe.index_add_(
                0, atar_event_batch_pe, torch.ones_like(atar_event_batch_pe, dtype=torch.long)
            )
            event_offset_pe = torch.zeros_like(slices_per_event_pe)
            if num_graphs_in_batch > 1:
                event_offset_pe[1:] = torch.cumsum(slices_per_event_pe, dim=0)[:-1]

            is_anchor_per_slice = torch.zeros(
                valid_slice_indices.size(0), dtype=torch.bool, device=x.device
            )
            if triggering_pion_slice is not None and valid_slice_indices.numel() > 0:
                anchor_local = triggering_pion_slice.to(x.device).long().view(-1)
                anchor_valid = anchor_local >= 0
                anchor_local_safe = anchor_local.clamp(min=0)
                slice_budget = (slices_per_event_pe - 1).clamp(min=0)
                anchor_local_clamped = torch.minimum(anchor_local_safe, slice_budget)
                anchor_global_idx = (event_offset_pe + anchor_local_clamped).clamp(
                    max=valid_slice_indices.size(0) - 1
                )
                # Only mark events whose anchor index was non-negative.
                if anchor_valid.any():
                    is_anchor_per_slice[anchor_global_idx[anchor_valid]] = True

            atar_event_tokens = atar_event_tokens + self.anchor_flag_embedding(
                is_anchor_per_slice.long()
            )

            output['atar_event_tokens'] = atar_event_tokens

            # === PHASE 9: ATAR-Only Event Building ===
            # Self-attention over ATAR event tokens for cross-slice temporal reasoning,
            # then per-token binary trigger classification.
            if task_weights.get('w_atar_trigger_slice', 0.0) > 0.0 or task_weights.get('w_pion_kinematics', 0.0) > 0.0 or task_weights.get('w_event_builder', 0.0) > 0.0 or task_weights.get('w_has_trigger_positron', 0.0) > 0.0:
                B_atar_idx = valid_slice_mask.nonzero().squeeze(1) // num_slices_max

                sort_idx_atar9 = torch.argsort(B_atar_idx)
                sorted_tokens9 = atar_event_tokens[sort_idx_atar9]
                sorted_batch9 = B_atar_idx[sort_idx_atar9]

                from torch_geometric.utils import to_dense_batch
                dense_atar9, pad_mask9 = to_dense_batch(sorted_tokens9, sorted_batch9)

                # --- Endpoint-compatibility kernel as attention bias ---
                # For each pair (i, j) of slices in the same event, compute a
                # Mahalanobis-like Gaussian log-overlap between their endpoints:
                #     -(μ_i - μ_j)² / (σ_i² + σ_j²)  per axis, weighted per-axis.
                # Take logsumexp over all 4 (point_i, point_j) combinations so
                # the kernel is agnostic to which endpoint is entry vs exit.
                # Kernel is detached from the endpoint regressors — pure physics
                # prior into the attention logits, no gradient from trigger loss
                # into upstream kinematics.
                endpoint_mean = endpoints_all[..., 1]  # [N_valid, 2, 3] median quantile
                endpoint_sigma = 0.5 * (endpoints_all[..., 2] - endpoints_all[..., 0]).abs().clamp(min=1e-3)

                mean_sorted = endpoint_mean[sort_idx_atar9].reshape(-1, 6)
                sigma_sorted = endpoint_sigma[sort_idx_atar9].reshape(-1, 6)
                dense_mean_flat, _ = to_dense_batch(mean_sorted, sorted_batch9)
                dense_sigma_flat, _ = to_dense_batch(sigma_sorted, sorted_batch9)
                B_size, N_max = dense_mean_flat.shape[0], dense_mean_flat.shape[1]
                dense_mean = dense_mean_flat.view(B_size, N_max, 2, 3)
                dense_sigma = dense_sigma_flat.view(B_size, N_max, 2, 3)

                # Broadcast to [B, N, N, 2(point_i), 2(point_j), 3(axis)]
                m_i = dense_mean.unsqueeze(2).unsqueeze(4)
                m_j = dense_mean.unsqueeze(1).unsqueeze(3)
                s_i = dense_sigma.unsqueeze(2).unsqueeze(4)
                s_j = dense_sigma.unsqueeze(1).unsqueeze(3)
                delta_sq = (m_i - m_j) ** 2
                var_sum = s_i ** 2 + s_j ** 2 + 1e-6

                alpha = F.softplus(self.kernel_alpha).view(1, 1, 1, 1, 1, 3)
                neg_chi2 = -(delta_sq / var_sum) * alpha
                g_per_pair = neg_chi2.sum(dim=-1)  # [B, N, N, 2, 2]
                kernel_bias = torch.logsumexp(g_per_pair.flatten(-2, -1), dim=-1)  # [B, N, N]
                kernel_bias = kernel_bias.clamp(min=-50.0)  # guard against numerically extreme pairs

                num_heads_sa = self.atar_event_layers[0].self_attn.num_heads
                kernel_bias_heads = (
                    kernel_bias.unsqueeze(1)
                    .expand(-1, num_heads_sa, -1, -1)
                    .reshape(B_size * num_heads_sa, N_max, N_max)
                )

                # Stacked self-attention with pre-LN (inside TransformerEncoderLayer).
                # Collect each layer's output alongside the pre-attention input so
                # the classifier can read whichever depth a given slice needs:
                #   h^(0) = raw slice token               (0 hops — e.g. the pion itself)
                #   h^(1) = 1-hop chain neighbor info     (e.g. direct-decay positron)
                #   h^(2) = 2-hop chain neighbor info     (e.g. michel positron via muon)
                jk_stack = [dense_atar9]
                h_jk = dense_atar9
                for layer in self.atar_event_layers:
                    h_next = layer(h_jk, src_mask=kernel_bias_heads, src_key_padding_mask=~pad_mask9)
                    # Safety net: if a layer produced NaN, fall back to its input
                    # rather than propagate garbage forward.
                    if torch.isnan(h_next).any():
                        h_next = h_jk
                    h_jk = h_next
                    jk_stack.append(h_jk)

                # JK concat along feature dim → [B, N_max, (L+1) * D_A]
                atar_tokens_jk = torch.cat(jk_stack, dim=-1)

                # Flatten back and un-sort both the JK concat (for the trigger
                # classifier) and the last-layer output (for downstream consumers
                # that still want D_A-dim tokens).
                inverse_sort9 = torch.argsort(sort_idx_atar9)
                jk_flat = atar_tokens_jk[pad_mask9][inverse_sort9]
                last_flat = jk_stack[-1][pad_mask9][inverse_sort9]

                # --- Role-based attachment head ---
                # Per valid slice, classify role in the triggering chain:
                #   0 = none, 1 = muon-in-chain, 2 = positron-in-chain.
                # The anchor pion slice is fixed (not predicted) — its role output
                # is ignored by the loss and its in-chain prob is clamped to 1.
                B_size_evt = num_graphs_in_batch
                N_valid = jk_flat.size(0)

                # Event id per valid slice (in jk_flat order).
                slice_event_idx = B_atar_idx  # [N_valid]

                # Count valid slices per event and derive cumulative offsets so we
                # can resolve each event's anchor into its global jk_flat index.
                slices_per_event = torch.zeros(B_size_evt, dtype=torch.long, device=x.device)
                slices_per_event.index_add_(
                    0, slice_event_idx,
                    torch.ones_like(slice_event_idx, dtype=torch.long),
                )
                event_offset = torch.zeros_like(slices_per_event)
                if B_size_evt > 1:
                    event_offset[1:] = torch.cumsum(slices_per_event, dim=0)[:-1]

                # Resolve the per-event anchor. Training: read from batch; Eval /
                # no-truth: fall back to the earliest pion-class slice in the event.
                if triggering_pion_slice is not None:
                    anchor_local = triggering_pion_slice.to(device=x.device, dtype=torch.long)
                else:
                    # Predicted anchor: per event, pick the pion-class valid slice
                    # (argmax over the 3-class slice_pdg head) with the smallest
                    # slice mean time. slice_pdg_det here is the per-slice PDG
                    # soft prediction, shape [N_valid, 3].
                    # slice_pdg_det holds LOGITS (sigmoid not yet applied), so
                    # threshold > 0 is equivalent to sigmoid > 0.5.
                    pion_score = slice_pdg_det[:, 0]  # [N_valid]
                    is_pion_pred = (pion_score > 0.0)
                    t_mask = torch.where(
                        is_pion_pred, slice_mean_time, torch.full_like(slice_mean_time, 1e9)
                    )
                    # Per-event argmin over time_masked. Use scatter_min.
                    anchor_local = torch.full((B_size_evt,), -1, dtype=torch.long, device=x.device)
                    best_time = torch.full((B_size_evt,), 1e9, device=x.device)
                    # Simple per-event sequential resolution via index_reduce-style.
                    # Valid slices are in event-sorted order already (B_atar_idx
                    # is sorted thanks to sort_idx_atar9 semantics) — use a loop
                    # over events if batch is small; else fall back to a scatter.
                    # Vectorised argmin per event:
                    sort_t_perm = torch.argsort(t_mask, stable=True)
                    ev_sorted = slice_event_idx[sort_t_perm]
                    # First occurrence of each event id in ev_sorted has smallest t.
                    seen = torch.zeros(B_size_evt, dtype=torch.bool, device=x.device)
                    for pos, ev in zip(sort_t_perm.tolist(), ev_sorted.tolist()):
                        if not seen[ev] and t_mask[pos] < 5e8:
                            # Local idx = global pos - event_offset[ev]
                            anchor_local[ev] = pos - int(event_offset[ev].item())
                            seen[ev] = True

                anchor_valid = anchor_local >= 0  # [B]
                # Clamp to 0 so indexing into jk_flat is always in-bounds; we
                # mask the result below for events where anchor_valid=False.
                anchor_local_safe = anchor_local.clamp(min=0)
                anchor_global_in_jk = event_offset + anchor_local_safe  # [B]
                anchor_global_in_jk = anchor_global_in_jk.clamp(max=N_valid - 1)

                # Anchor token per event, broadcast to per-slice.
                anchor_token_per_event = jk_flat[anchor_global_in_jk]  # [B, JK_DIM]
                # Zero out anchor token for events with no valid anchor — self-anchor
                # dynamics still work (self - 0 = self) and the head can learn to
                # behave gracefully on background events.
                anchor_token_per_event = anchor_token_per_event * anchor_valid.float().unsqueeze(-1)
                anchor_token_per_slice = anchor_token_per_event[slice_event_idx]  # [N_valid, JK_DIM]

                # Relative endpoint feature: flattened Δendpoint to anchor (6 dims).
                endpoint_mean_flat = endpoint_mean.reshape(N_valid, 6)  # median quantile endpoints
                endpoint_anchor_per_event = endpoint_mean_flat[anchor_global_in_jk]
                endpoint_anchor_per_event = endpoint_anchor_per_event * anchor_valid.float().unsqueeze(-1)
                endpoint_delta = endpoint_mean_flat - endpoint_anchor_per_event[slice_event_idx]

                role_input = torch.cat([
                    jk_flat,
                    anchor_token_per_slice,
                    jk_flat - anchor_token_per_slice,
                    endpoint_delta,
                ], dim=-1)
                role_logits = self.atar_role_head(role_input)  # [N_valid, 3]

                # In-chain logit: log p(role ∈ {μ, e}) / p(none) = logsumexp(l_μ, l_e) - l_none.
                in_chain_logits = torch.logsumexp(role_logits[:, 1:], dim=-1) - role_logits[:, 0]

                # Build anchor-slice boolean mask for downstream use and override.
                is_anchor_slice = torch.zeros(N_valid, dtype=torch.bool, device=x.device)
                if anchor_valid.any():
                    is_anchor_slice[anchor_global_in_jk[anchor_valid]] = True

                # Anchor is the pion — clamp its in-chain logit very high so all
                # downstream hit-gating / softmax consumers treat it as triggered.
                in_chain_logits = torch.where(
                    is_anchor_slice,
                    torch.full_like(in_chain_logits, 20.0),
                    in_chain_logits,
                )

                output['atar_role_logits'] = role_logits
                output['atar_anchor_slice_mask'] = is_anchor_slice
                output['atar_slice_event_idx'] = slice_event_idx
                # Back-compat: downstream heads still read 'atar_trigger_logits'
                # as a per-slice "in-chain" score.
                atar_trigger_logits = in_chain_logits
                output['atar_trigger_logits'] = atar_trigger_logits
                atar_trigger_probs = torch.sigmoid(atar_trigger_logits).detach()

                # Downstream event-builder / pion-stop heads consume D_A tokens,
                # so hand them the last-layer output (not the JK concat).
                atar_event_tokens = last_flat
                output['atar_event_tokens'] = atar_event_tokens

            # === PHASE 10: Pion Stop Extraction (stereo softmax pooling) ===
            # Separate x-branch (pools x-hits → x, z_from_x) and y-branch
            # (pools y-hits → y, z_from_y). Per-hit score combines trigger
            # logit and pion-class logit under a learned temperature.
            # Bounded residual from joint features captures sub-strip depth
            # encoded in energy (Bragg peak).
            if task_weights.get('w_pion_kinematics', 0.0) > 0.0 and 'atar_trigger_logits' in output:
                from torch_geometric.utils import softmax as pyg_softmax

                # Broadcast trigger LOGITS (not probs) to hit level. Slices
                # outside valid_slice_mask get a very negative logit so they
                # contribute ~0 weight under softmax.
                trigger_logit_full = torch.full((num_global_slices,), -50.0, device=x.device)
                trigger_logit_full[valid_slice_mask] = atar_trigger_logits.detach()
                hit_trigger_logit = trigger_logit_full[global_slice_idx_all]  # [N_atar]

                # Pion class logit per hit (DETACHED — used only as soft gate)
                hit_pion_logit = output['atar_node_pdg'][:, 0].detach()  # [N_atar]

                # Learnable endpoint score: the ONLY trainable path from
                # pion_stop loss to the softmax weights. Discriminates
                # *within* the pion track (trig+pion logits gate out
                # non-pion hits before it competes).
                endpoint_logit = self.pion_endpoint_scorer(h_atar).squeeze(-1)  # [N_atar]

                tau = torch.exp(self.pion_pool_log_tau).clamp(min=0.1, max=10.0)
                hit_score = (hit_trigger_logit + hit_pion_logit + endpoint_logit) / tau

                batch_atar = batch[is_atar]
                pos_atar = x[is_atar, 0:3]  # [N_atar, 3]
                is_x_local = is_atar_x[is_atar]  # [N_atar] bool
                is_y_local = is_atar_y[is_atar]

                # Per-event view presence flags
                ones_atar = torch.ones(is_x_local.shape[0], 1, device=x.device)
                n_x_per_event = global_add_pool(
                    is_x_local.float().unsqueeze(-1) * ones_atar, batch_atar,
                    size=num_graphs_in_batch).squeeze(-1)
                n_y_per_event = global_add_pool(
                    is_y_local.float().unsqueeze(-1) * ones_atar, batch_atar,
                    size=num_graphs_in_batch).squeeze(-1)
                has_x_event = (n_x_per_event > 0).float().unsqueeze(-1)  # [B,1]
                has_y_event = (n_y_per_event > 0).float().unsqueeze(-1)

                # X-branch: softmax over x-hits within each event
                if is_x_local.any():
                    score_x = hit_score[is_x_local]
                    batch_x = batch_atar[is_x_local]
                    pos_x_hits = pos_atar[is_x_local]
                    h_x_hits = h_atar[is_x_local]
                    w_x = pyg_softmax(score_x, batch_x, num_nodes=num_graphs_in_batch).unsqueeze(-1)
                    pooled_x_coord = global_add_pool(w_x * pos_x_hits[:, 0:1], batch_x, size=num_graphs_in_batch)
                    pooled_z_from_x = global_add_pool(w_x * pos_x_hits[:, 2:3], batch_x, size=num_graphs_in_batch)
                    pooled_feat_x = global_add_pool(w_x * h_x_hits, batch_x, size=num_graphs_in_batch)
                else:
                    pooled_x_coord = torch.zeros(num_graphs_in_batch, 1, device=x.device)
                    pooled_z_from_x = torch.zeros(num_graphs_in_batch, 1, device=x.device)
                    pooled_feat_x = torch.zeros(num_graphs_in_batch, jk_dim, device=x.device)

                # Y-branch: softmax over y-hits within each event
                if is_y_local.any():
                    score_y = hit_score[is_y_local]
                    batch_y = batch_atar[is_y_local]
                    pos_y_hits = pos_atar[is_y_local]
                    h_y_hits = h_atar[is_y_local]
                    w_y = pyg_softmax(score_y, batch_y, num_nodes=num_graphs_in_batch).unsqueeze(-1)
                    pooled_y_coord = global_add_pool(w_y * pos_y_hits[:, 1:2], batch_y, size=num_graphs_in_batch)
                    pooled_z_from_y = global_add_pool(w_y * pos_y_hits[:, 2:3], batch_y, size=num_graphs_in_batch)
                    pooled_feat_y = global_add_pool(w_y * h_y_hits, batch_y, size=num_graphs_in_batch)
                else:
                    pooled_y_coord = torch.zeros(num_graphs_in_batch, 1, device=x.device)
                    pooled_z_from_y = torch.zeros(num_graphs_in_batch, 1, device=x.device)
                    pooled_feat_y = torch.zeros(num_graphs_in_batch, jk_dim, device=x.device)

                # Combine z: average of view-contributed estimates (denom ≥ 1 avoids /0)
                pooled_z = (pooled_z_from_x * has_x_event + pooled_z_from_y * has_y_event) \
                    / (has_x_event + has_y_event).clamp(min=1.0)
                # If an event is missing a view, clone the available transverse
                # coord so downstream math is well-defined (the residual will
                # refine it, though an event with only one view won't be well
                # constrained in the missing axis).
                pooled_x_coord = pooled_x_coord * has_x_event
                pooled_y_coord = pooled_y_coord * has_y_event
                pooled_pos = torch.cat([pooled_x_coord, pooled_y_coord, pooled_z], dim=-1)  # [B, 3]

                # Per-axis bounded residuals (mirrors endpoint head structure):
                #   x residual sees x-branch only; y residual sees y-branch only;
                #   z residual is stereo (both branches).
                res_input_x = torch.cat([pooled_x_coord, pooled_feat_x], dim=-1)
                res_input_y = torch.cat([pooled_y_coord, pooled_feat_y], dim=-1)
                res_input_z = torch.cat([pooled_z, pooled_feat_x, pooled_feat_y], dim=-1)
                delta_x = torch.tanh(self.pion_stop_residual_x(res_input_x))  # [B, 1]
                delta_y = torch.tanh(self.pion_stop_residual_y(res_input_y))  # [B, 1]
                delta_z = torch.tanh(self.pion_stop_residual_z(res_input_z))  # [B, 1]
                # Suppress residuals for axes that have no hits (sanity).
                delta_x = delta_x * has_x_event
                delta_y = delta_y * has_y_event
                delta = torch.cat([delta_x, delta_y, delta_z], dim=-1) * self.pion_stop_residual_scale
                pion_stop_pred = pooled_pos + delta
                output['atar_pion_stop'] = pion_stop_pred
            else:
                pion_stop_pred = torch.zeros(num_graphs_in_batch, 3, device=x.device)

            # === PHASE 11: Positron Direction (per-graph, stereo hard-gated) ===
            # (exit_dir_per_graph removed — pion direction carries no physical
            # information about positron direction for a stopped pion.)

            # Per-graph positron reference time: unweighted mean of normalized
            # hit times over ATAR hits that are both in the triggering slice
            # AND classified as MIP/positron. Both masks are detached so a
            # hard threshold is safe and avoids soft weights dragging the
            # mean toward non-positron hits. Used as the origin of the LYSO
            # time feature (dt = cluster_time - positron_time), the dominant
            # physical signal for trigger/background discrimination.
            #
            # Sentinel -500 (normalized → -250 µs raw) for events with no
            # trigger positron: pushes every cluster far outside the σ=2 ns
            # coincidence window, so coinc_feat → 0 and the event classifier
            # receives no spurious "coincident" signal on background-only events.
            NO_POSITRON_TIME = -500.0
            positron_time_per_graph = torch.full(
                (num_graphs_in_batch,), NO_POSITRON_TIME, device=x.device)
            if 'atar_trigger_logits' in output:
                trigger_prob_full_t = torch.zeros(num_global_slices, device=x.device)
                trigger_prob_full_t[valid_slice_mask] = atar_trigger_probs
                hit_trigger_prob_t = trigger_prob_full_t[global_slice_idx_all]  # [N_atar]
                mip_class_prob_t = torch.sigmoid(output['atar_node_pdg'][:, 2]).detach()  # [N_atar]
                output['atar_hit_trigger_prob'] = hit_trigger_prob_t.detach()
                output['atar_hit_mip_prob'] = mip_class_prob_t
                positron_hit_mask = (hit_trigger_prob_t > 0.5) & (mip_class_prob_t > 0.5)

                batch_atar_t = batch[is_atar]
                hit_w = positron_hit_mask.float()
                t_num = torch.zeros(num_graphs_in_batch, device=x.device)
                t_den = torch.zeros(num_graphs_in_batch, device=x.device)
                t_num.index_add_(0, batch_atar_t, x[is_atar, 4] * hit_w)
                t_den.index_add_(0, batch_atar_t, hit_w)
                positron_time_per_graph = torch.where(
                    t_den > 0.5,
                    t_num / t_den.clamp(min=1.0),
                    torch.full_like(t_num, NO_POSITRON_TIME),
                )  # [B] normalized (col 4 is already /NORM_T_ATAR=500)

            if task_weights.get('w_positron_angle', 0.0) > 0.0 and 'atar_trigger_logits' in output:
                # Hard positron-hit mask: trigger-slice AND MIP-like. Both upstream
                # heads are pretrained and detached here, so the boolean mask does
                # not break gradients to h_atar — only determines which hits enter
                # each branch's mean pool. On hits passing the mask, full gradient
                # flows through h_atar into the backbone as usual.
                mip_class_prob = torch.sigmoid(output['atar_node_pdg'][:, 2]).detach()  # [N_atar]
                trigger_prob_full = torch.zeros(num_global_slices, device=x.device)
                trigger_prob_full[valid_slice_mask] = atar_trigger_probs
                hit_trigger_prob = trigger_prob_full[global_slice_idx_all]

                positron_mask = (hit_trigger_prob > 0.5) & (mip_class_prob > 0.5)  # [N_atar]

                batch_atar = batch[is_atar]
                is_x_local = is_atar_x[is_atar]
                is_y_local = is_atar_y[is_atar]
                mask_x = positron_mask & is_x_local
                mask_y = positron_mask & is_y_local

                # Per-view mean pool of positron hits.
                def _mean_pool_view(mask_view):
                    if mask_view.any():
                        h_v = h_atar[mask_view]
                        b_v = batch_atar[mask_view]
                        ones_v = torch.ones(h_v.shape[0], 1, device=x.device)
                        sum_feat = global_add_pool(h_v, b_v, size=num_graphs_in_batch)
                        n_v = global_add_pool(ones_v, b_v, size=num_graphs_in_batch)
                        pool = sum_feat / n_v.clamp(min=1.0)
                        has_v = (n_v > 0).float()
                    else:
                        pool = torch.zeros(num_graphs_in_batch, jk_dim, device=x.device)
                        has_v = torch.zeros(num_graphs_in_batch, 1, device=x.device)
                    return pool, has_v

                pool_x_pos, _ = _mean_pool_view(mask_x)
                pool_y_pos, _ = _mean_pool_view(mask_y)

                dir_input = torch.cat([pool_x_pos, pool_y_pos, pion_stop_pred.detach()], dim=-1)
                positron_dir = self.positron_dir_head(dir_input)  # [B, 3] unit vector
                output['atar_positron_dir'] = positron_dir
            else:
                positron_dir = torch.tensor([[0.0, 0.0, 1.0]], device=x.device).expand(num_graphs_in_batch, -1)

            # === Phase 11b: has-trigger-positron (pure decision rule) ===
            # Per-slice count of hits passing (trigger>0.5 AND mip>0.5).
            # Event is flagged htp=1 iff ANY slice has >= 2 qualifying hits.
            # No parameters, no loss — consumed only at inference.
            if ('atar_hit_trigger_prob' in output
                    and 'atar_hit_mip_prob' in output):
                hit_trig = output['atar_hit_trigger_prob']      # [N_atar]
                hit_mip  = output['atar_hit_mip_prob']          # [N_atar]
                qualifies = ((hit_trig > 0.5) & (hit_mip > 0.5)).float()
                slice_count = torch.zeros(num_global_slices, device=x.device)
                slice_count.index_add_(0, global_slice_idx_all, qualifies)
                slice_count = slice_count.view(num_graphs_in_batch, num_slices_max)
                output['has_trigger_positron_rule'] = (slice_count >= 2.0).any(dim=1).float()
        else:
            # No ATAR hits — provide default direction/position for LYSO token construction
            pion_stop_pred = torch.zeros(num_graphs_in_batch, 3, device=x.device)
            positron_dir = torch.tensor([[0.0, 0.0, 1.0]], device=x.device).expand(num_graphs_in_batch, -1)
            # No ATAR → no positron reference: use the same -500 sentinel as above.
            positron_time_per_graph = torch.full(
                (num_graphs_in_batch,), -500.0, device=x.device)

        # === LYSO PREDICTIONS (Object Condensation) ===
        if is_lyso.any():
            h_lyso = h_out[is_lyso]

            if task_weights.get('w_lyso_condensation', 1.0) > 0.0:
                output['lyso_beta'] = self.lyso_beta_head(h_lyso)
                output['lyso_cluster_coords'] = self.lyso_cluster_coord_head(h_lyso)
                output['lyso_fractions'] = self.lyso_fraction_head(h_lyso)
            output['lyso_embedding'] = h_lyso
            
            # Pool LYSO hits per Time Slice for global feature combinations
            lyso_slice_idx = global_slice_ids[is_lyso]
            pool_lyso_sum = torch.zeros(num_global_slices, h_lyso.size(1), device=h_lyso.device)
            pool_lyso_sum.index_add_(0, lyso_slice_idx, h_lyso)
            count_lyso.index_add_(0, lyso_slice_idx, torch.ones_like(lyso_slice_idx, dtype=torch.float).unsqueeze(1))
            lyso_stereo = pool_lyso_sum / count_lyso.clamp(min=1.0)
            
        else:
            lyso_stereo = torch.zeros(num_global_slices, h_out.size(1), device=x.device)
            
        # === UNIFIED HEADS (Event Builder) ===
        K_LYSO = 5 # Top-K soft clustering parameter
        
        # 1. LYSO Top-K Soft Clustering
        if (hasattr(self, 'lyso_event_mlp') or hasattr(self, 'lyso_event_proj')) and is_lyso.any() and task_weights.get('w_lyso_condensation', 1.0) > 0.0:
            pred_coords = output['lyso_cluster_coords'] # [N_lyso, 3]
            pred_beta = output['lyso_beta'].squeeze(-1) # [N_lyso]
            lyso_batch = batch[is_lyso]                 # [N_lyso]

            B = num_graphs_in_batch if is_atar.any() else lyso_batch.max().item() + 1

            from torch_geometric.utils import to_dense_batch

            # --- Vectorized Object Condensation ---
            g_coords, mask = to_dense_batch(pred_coords, lyso_batch, batch_size=B) # [B, N_max, 3]
            g_beta, _ = to_dense_batch(pred_beta, lyso_batch, batch_size=B)
            g_feats, _ = to_dense_batch(h_lyso, lyso_batch, batch_size=B)
            
            x_lyso = x[is_lyso]
            g_slice_id, _ = to_dense_batch(x_lyso[:, 8], lyso_batch, batch_size=B)
            g_energy, _ = to_dense_batch(x_lyso[:, 3], lyso_batch, batch_size=B)
            g_phys_pos, _ = to_dense_batch(x_lyso[:, :3], lyso_batch, batch_size=B)  # [B, N_max, 3] physical /100mm
            
            g_beta[~mask] = 0.0 # Padded nodes exert 0 priority and 0 affinity
            
            energy_threshold_norm = 2.0 / NORM_E_LYSO  # 2 MeV threshold
            is_valid_seed = (g_slice_id > 0) | (g_energy > energy_threshold_norm)
            g_beta_seeds = g_beta * is_valid_seed.float()
            
            g_beta_seeds[~mask] = -1e9 # Prevent padded nodes from being selected
            
            actual_k = min(K_LYSO, g_coords.size(1))
            actual_k = max(actual_k, 1) # Prevent topk with k=0
            
            topk_vals, topk_idx = torch.topk(g_beta_seeds, k=actual_k, dim=1) # [B, actual_k]

            topk_idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, 3)
            seed_coords = torch.gather(g_coords, dim=1, index=topk_idx_expand) # [B, actual_k, 3]
            seed_beta = torch.gather(g_beta, dim=1, index=topk_idx)            # [B, actual_k]

            # Threshold filter: residual sibling seeds (β stuck around 0.25 after BCE
            # highlander) are not real cluster centers. Real pileup gives β≈1 on each
            # genuine seed, so 0.5 is a clean cut between residual and legitimate.
            # Zeroing seed_β here propagates through affinity/pool/cluster_energy_sum
            # naturally — the existing seed_has_weight check (w_sum > 1e-4) will then
            # mark these seeds invalid downstream.
            seed_beta_threshold = 0.5
            seed_beta = seed_beta * (seed_beta > seed_beta_threshold).float()
            
            dists = torch.cdist(g_coords, seed_coords) # [B, N_max, actual_k]
            
            tau = 0.8
            affinity = torch.exp(-dists / tau) * seed_beta.unsqueeze(1) # [B, N_max, actual_k]
            affinity[~mask] = 0.0 # Nullify out-bound affinity from padded matching nodes

            # Slice-ID gate: hits from a different physics slice than the seed cannot
            # belong to the same cluster. Without this, OC freely merges radiation
            # hits from disjoint time windows (slice_ids) into one cluster, which
            # smears cluster_energy_time across 100+ ns and breaks coinc with pt.
            seed_slice_id = torch.gather(g_slice_id, 1, topk_idx).long()        # [B, actual_k]
            slice_match = g_slice_id.long().unsqueeze(-1) == seed_slice_id.unsqueeze(1)  # [B, N_max, actual_k]
            affinity = affinity * slice_match.float()

            # Adaptive dustbin: the floor fades to 0 when a hit has a strong real match
            # (max affinity → 1) so signal hits can reach w=1, but grows to 0.05 for orphan
            # hits (max affinity → 0) so radioactivity is still absorbed by the virtual slot.
            max_affinity = affinity.max(dim=2, keepdim=True).values.clamp(min=0.0, max=1.0)  # [B, N_max, 1]
            adaptive_floor = 0.05 * (1.0 - max_affinity)                                     # [B, N_max, 1]
            w_norm = affinity / (affinity.sum(dim=2, keepdim=True) + adaptive_floor)         # [B, N_max, actual_k]
            
            g_pool = torch.bmm(w_norm.transpose(1, 2), g_feats)  # [B, actual_k, D]
            # Restored β² pool attenuation: w_norm already carries one β through
            # affinity = exp(-d/τ)·β, and multiplying again by seed_beta shrinks
            # low-β (non-seed-like) cluster tokens toward zero. Low-β clusters
            # thus contribute almost nothing to the 8D detached pool residual
            # or downstream consumers — an inference-time suppressor of the
            # "too many clusters" symptom. Safe now that lyso_pool_proj is
            # detached and dim-reduced, so heterogeneous pool magnitudes can
            # no longer dominate the event classifier.
            g_pool = g_pool * seed_beta.unsqueeze(-1)

            # Per-cluster temporal info for positional encoding
            # Use seed's slice_id directly (each cluster is anchored by its seed hit)
            seed_slice_id = torch.gather(g_slice_id, dim=1, index=topk_idx).long()  # [B, actual_k]
            # Also compute soft-weighted mean time for temporal attention bias
            g_time, _ = to_dense_batch(x_lyso[:, 4], lyso_batch, batch_size=B)  # [B, N_max] normalized (col 4 = t/500)
            g_time[~mask] = 0.0
            cluster_mean_time = torch.bmm(w_norm.transpose(1, 2), g_time.unsqueeze(-1)).squeeze(-1)  # [B, actual_k]

            # Per-cluster energy sum: critical for radioactivity rejection
            # g_energy [B, N_max] already dense-batched above; nullified for padded hits
            g_energy[~mask] = 0.0
            cluster_energy_sum = torch.bmm(w_norm.transpose(1, 2), g_energy.unsqueeze(-1)).squeeze(-1)  # [B, actual_k] Σ(w·E), used only as energy-time denominator

            # Compute seed validity. Two failure modes must be caught:
            #   (a) structurally padded: topk pulled an index past the real hit count
            #   (b) valid-but-empty: a real seed that lost all weight to competitors.
            # With sharp trained affinities and the adaptive dustbin (floor → 0 for
            # confident hits), (b) is common: one cluster can absorb every hit and
            # the rest end up with w_sum ≈ 0, turning /w_sum into a backward NaN.
            lengths = mask.sum(dim=1)                                # [B]
            seed_structural = topk_idx < lengths.unsqueeze(-1)       # [B, actual_k]
            w_sum_per_k = w_norm.sum(dim=1)                          # [B, actual_k] unclamped
            seed_has_weight = w_sum_per_k > 1e-4                     # [B, actual_k]
            # A seed can be structurally real and weight-valid yet still have
            # cluster_energy_sum ≈ 0 (e.g. if every assigned hit has E rounded
            # to zero, which the upstream smearing fix in root_to_parquet should
            # prevent — but defensive checks are cheap and protect against
            # similar regressions). Without this, cluster_energy_time below
            # divides by ~0 and produces NaN that propagates everywhere.
            seed_has_energy = cluster_energy_sum > 1e-9              # [B, actual_k]
            seed_is_valid = seed_structural & seed_has_weight & seed_has_energy
            seed_invalid = ~seed_is_valid

            # Energy-weighted mean time (more physical than affinity-weighted).
            # Mask denominator via masked_fill (not clamp) so backward through
            # empty seeds returns zero instead of hitting the clamp boundary.
            g_e_time = g_energy * g_time  # [B, N_max]
            cluster_etime = torch.bmm(w_norm.transpose(1, 2), g_e_time.unsqueeze(-1)).squeeze(-1)  # [B, actual_k]
            e_sum_safe = cluster_energy_sum.masked_fill(seed_invalid, 1.0)
            cluster_energy_time = cluster_etime / e_sum_safe         # [B, actual_k]
            cluster_energy_time = cluster_energy_time.masked_fill(seed_invalid, 0.0)

            # Per-cluster position centroid (weighted mean in /100mm space). Kept for
            # downstream consumers (pion-stop-relative geometry) that need an absolute
            # position, not just a direction.
            g_phys_pos[~mask] = 0.0
            w_sum_safe = w_sum_per_k.masked_fill(seed_invalid, 1.0)  # safe denom for invalid/empty
            cluster_phys_pos = torch.bmm(
                w_norm.transpose(1, 2), g_phys_pos                    # [B, actual_k, 3]
            ) / w_sum_safe.unsqueeze(-1)                              # normalize to centroid
            cluster_phys_pos = cluster_phys_pos.masked_fill(seed_invalid.unsqueeze(-1), 0.0)

            # Per-cluster *direction*: weighted mean of unit hit directions, then renormalize.
            # Independent of position-magnitude accumulation, so a low-weight hit at one
            # angle can't be confused with a high-weight hit at another. Used for the
            # angular features (sin/cos θ,φ, cos_sep_positron, cos_sep_exit).
            hit_r = g_phys_pos.norm(dim=-1, keepdim=True).clamp(min=1e-6)       # [B, N_max, 1]
            hit_dir = g_phys_pos / hit_r                                         # [B, N_max, 3]
            hit_dir = hit_dir * mask.unsqueeze(-1).float()
            cluster_dir_raw = torch.bmm(
                w_norm.transpose(1, 2), hit_dir                                  # [B, actual_k, 3]
            ) / w_sum_safe.unsqueeze(-1)
            cluster_dir_norm = cluster_dir_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            cluster_dir = cluster_dir_raw / cluster_dir_norm                     # unit vector
            cluster_dir = cluster_dir.masked_fill(seed_invalid.unsqueeze(-1), 0.0)

            lyso_assignments = w_norm[mask] # [N_lyso, actual_k] exactly aligns to PyG layout
            
            pad_dim = K_LYSO - actual_k
            if pad_dim > 0:
                lyso_assignments = torch.nn.functional.pad(lyso_assignments, (0, pad_dim))
                pad_zero = torch.zeros((B, pad_dim, g_pool.size(2)), device=x.device)
                g_pool = torch.cat([g_pool, pad_zero], dim=1)
                seed_beta_padded = torch.cat([seed_beta, torch.zeros(B, pad_dim, device=x.device)], dim=1)
                cluster_phys_pos_padded = torch.cat([cluster_phys_pos, torch.zeros(B, pad_dim, 3, device=x.device)], dim=1)
                cluster_dir_padded = torch.cat([cluster_dir, torch.zeros(B, pad_dim, 3, device=x.device)], dim=1)
                seed_latent_coords_padded = torch.cat([seed_coords, torch.zeros(B, pad_dim, 3, device=x.device)], dim=1)

                valid_mask = torch.cat([seed_is_valid, torch.zeros(B, pad_dim, dtype=torch.bool, device=x.device)], dim=1)
            else:
                valid_mask = seed_is_valid
                seed_beta_padded = seed_beta
                cluster_phys_pos_padded = cluster_phys_pos
                cluster_dir_padded = cluster_dir
                seed_latent_coords_padded = seed_coords

            has_lyso = lengths > 0 # [B]

            # Down-select only graphs that had LYSO hits to match original variable sizes
            g_pool = g_pool[has_lyso]
            valid_mask = valid_mask[has_lyso]

            # Unroll variables
            if g_pool.numel() > 0:
                lyso_pool_all = g_pool.view(-1, g_pool.size(-1)) # [B_valid * K_LYSO, D]
                lyso_valid_tensor = valid_mask.view(-1)          # [B_valid * K_LYSO]

                valid_b_indices = torch.nonzero(has_lyso).squeeze(-1) # [B_valid]
                lyso_event_batch_tensor = valid_b_indices.unsqueeze(-1).expand(-1, K_LYSO).reshape(-1)

                output['lyso_soft_assignments'] = lyso_assignments.detach()
                output['lyso_seed_beta'] = seed_beta_padded[has_lyso].view(-1).detach()

                # === Detach OC features from event builder gradient path ===
                # The event builder (transformer + classifier) trains only its
                # own layers. Gradients must NOT flow back into the OC coord/beta/
                # GNN heads — the trigger classification objective conflicts with
                # the clustering objective and degrades OC quality.
                # Note: cluster_energy_sum/time/mean_time are padded later (Phase 14
                # temporal block) — those are detached inline where they're created.
                cluster_phys_pos_padded = cluster_phys_pos_padded.detach()
                cluster_dir_padded = cluster_dir_padded.detach()
                seed_beta_padded = seed_beta_padded.detach()

                # === PHASE 14: Angular LYSO Token Construction (projective geometry) ===
                # Angles come from cluster_dir (unit-vector weighted mean of hit directions)
                # so weight magnitude cannot leak into direction: a low-weight hit at one
                # angle is cleanly distinguishable from a high-weight hit at another.
                # LYSO is a spherical shell so radius carries no information and is dropped.
                dir_to_cluster = cluster_dir_padded  # [B, K_LYSO, 3] already unit

                # Clamp the sqrt argument *before* sqrt so backward never hits
                # 1/(2·sqrt(0)) = inf when a cluster sits on the z-axis.
                cos_theta = dir_to_cluster[..., 2:3]  # z-component = cos(polar angle)
                sin_theta_sq = (1.0 - cos_theta ** 2).clamp(min=1e-6)
                sin_theta = sin_theta_sq.sqrt()
                axial = sin_theta < 2e-3
                sin_phi = dir_to_cluster[..., 1:2] / sin_theta
                cos_phi = dir_to_cluster[..., 0:1] / sin_theta
                sin_phi = sin_phi.masked_fill(axial, 0.0)
                cos_phi = cos_phi.masked_fill(axial, 1.0)

                # Cosine alignment with positron direction
                pos_dir = positron_dir.detach().unsqueeze(1)   # [B, 1, 3]
                cos_sep_positron = (dir_to_cluster * F.normalize(pos_dir, dim=-1)).sum(-1, keepdim=True)

                # Temporal + energy info: carry through K-padding and has_lyso filtering
                if pad_dim > 0:
                    seed_slice_id_padded = torch.cat([seed_slice_id, torch.zeros(B, pad_dim, dtype=torch.long, device=x.device)], dim=1)
                    cluster_mean_time_padded = torch.cat([cluster_mean_time, torch.zeros(B, pad_dim, device=x.device)], dim=1)
                    cluster_energy_sum_padded = torch.cat([cluster_energy_sum, torch.zeros(B, pad_dim, device=x.device)], dim=1)
                    cluster_energy_time_padded = torch.cat([cluster_energy_time, torch.zeros(B, pad_dim, device=x.device)], dim=1)
                else:
                    seed_slice_id_padded = seed_slice_id
                    cluster_mean_time_padded = cluster_mean_time
                    cluster_energy_sum_padded = cluster_energy_sum
                    cluster_energy_time_padded = cluster_energy_time

                cluster_energy_sum_padded = cluster_energy_sum_padded.detach()
                cluster_energy_time_padded = cluster_energy_time_padded.detach()
                cluster_mean_time_padded = cluster_mean_time_padded.detach()

                # Per-cluster angular + calorimetric features: [B, K_LYSO, 9]
                # Energy sum and energy-weighted time are critical for radioactivity rejection
                # NOTE: cluster_energy_time is in RAW ns; normalize by NORM_T_LYSO=500 before
                # feeding into the angular projection so it sits at the same O(1) scale as the
                # other features (sin/cos, inv_r, normalized energy, beta). Without this the
                # raw-ns time dominates the linear layer and collapses the per-cluster tokens
                # to nearly identical values, killing the event_classifier discrimination.
                cluster_e_flat = cluster_energy_sum_padded[has_lyso].view(-1, 1)   # [Total_K, 1]

                # === Positron-referenced cluster time (physical coincidence window) ===
                # positron_time_per_graph is in normalized units (col 4 = t/NORM_T_ATAR=500).
                # Both NORM_T_ATAR and NORM_T_LYSO are 500, so we can subtract directly after
                # normalizing the raw-ns cluster time. dt=0 → coincident with trigger positron.
                if positron_time_per_graph.size(0) >= B:
                    pt_per_graph = positron_time_per_graph[:B]
                else:
                    pt_per_graph = torch.zeros(B, device=x.device)
                    pt_per_graph[:positron_time_per_graph.size(0)] = positron_time_per_graph
                pt_per_graph_valid = pt_per_graph[has_lyso]                        # [B_valid]
                pt_flat = pt_per_graph_valid.unsqueeze(-1).expand(-1, K_LYSO).reshape(-1, 1)  # [Total_K, 1]
                cluster_et_norm = cluster_energy_time_padded[has_lyso].view(-1, 1)  # [Total_K, 1] already normalized (col 4)
                dt_from_pos = cluster_et_norm - pt_flat                            # [Total_K, 1] normalized

                # === Time-of-flight correction ===
                # The positron emits at the pion stop inside the ATAR and the signal travels
                # to the LYSO crystal at ~c. Physical coincidence means
                #     cluster_time ≈ positron_time + |r_crystal - r_stop| / c
                # The ATAR footprint (~10 mm) is tiny compared to the LYSO radius (~100 mm),
                # so approximating the source at the detector origin introduces <0.05 ns of
                # error — well below the σ=2 ns coincidence window — and avoids coupling to
                # the pion_stop_head during event-builder training.
                # LYSO inner radius ≈ 15 cm → TOF ≈ 150 mm / 300 mm·ns⁻¹ = 0.5 ns.
                # Cluster-to-cluster r variation is <<σ=2 ns so a constant TOF is fine
                # and avoids coupling to cluster_phys_pos magnitude.
                tof_norm_scalar = TOF_NS / NORM_T_LYSO
                tof_flat = torch.full_like(dt_from_pos, tof_norm_scalar)             # [Total_K, 1]
                dt_corr = dt_from_pos - tof_flat                                    # [Total_K, 1] normalized
                dt_corr_ns = dt_corr * NORM_T_LYSO                                  # [Total_K, 1] ns

                # === Gaussian coincidence window ===
                # Peaked at dt_corr=0, σ=2 ns. Physically motivated: the LYSO time resolution
                # is O(ns) so clusters within ±2 ns of the TOF-corrected positron time are
                # consistent with the trigger, and any hit >>σ away carries no coincidence
                # information. This gives the classifier a feature that is 1 at coincidence
                # and ≈0 everywhere else, matching the "evidence for trigger" semantics,
                # instead of a linear dt that grows with the *wrong* clusters.
                coinc_feat = torch.exp(
                    -(dt_corr_ns ** 2) / (2.0 * SIGMA_COINC_NS ** 2)
                )                                                                   # [Total_K, 1]
                cluster_et_flat = coinc_feat                                        # coincidence window as time feature

                angular_feat = torch.cat([
                    sin_theta, cos_theta, sin_phi, cos_phi,
                    cos_sep_positron,
                ], dim=-1)

                # Filter to valid LYSO graphs and flatten
                angular_feat_flat = angular_feat[has_lyso].view(-1, 5)  # [Total_K, 5]

                # === Per-cluster discriminative features for trigger classification ===
                # soft_n_hits = Σw per cluster; also used below to turn all weighted
                # sums into proper weighted means so weight magnitude can't leak into
                # the scalar features.
                w_norm_d = w_norm.detach()
                soft_n_hits = w_norm_d.sum(dim=1)                        # [B, actual_k] = Σw

                if pad_dim > 0:
                    soft_n_hits = torch.cat([soft_n_hits, torch.zeros(B, pad_dim, device=x.device)], dim=1)

                soft_n_hits_flat = soft_n_hits[has_lyso].view(-1, 1)
                beta_flat = seed_beta_padded[has_lyso].view(-1, 1)

                # Energy MEAN (not sum): Σ(w·E) / Σw. soft_n_hits carries the cluster-size
                # information separately, so switching to mean preserves info while making
                # this scalar weight-magnitude invariant.
                sn_safe = soft_n_hits_flat.clamp(min=1e-4)
                cluster_e_mean = cluster_e_flat / sn_safe                # [Total_K, 1]

                angular_feat_full = torch.cat([
                    angular_feat_flat,    # 5  (sin/cos theta/phi, cos_sep_positron)
                    cluster_e_mean,       # 1  (weighted-mean hit energy)
                    cluster_et_flat,      # 1  (coincidence window)
                    beta_flat,            # 1  (OC seed confidence)
                    soft_n_hits_flat,     # 1  (cluster occupancy = Σw)
                ], dim=-1)  # [Total_K, 9]
                lyso_cluster_times = cluster_mean_time_padded[has_lyso].view(-1) * NORM_T_LYSO  # [Total_K] raw ns (back from normalized for temporal bias)
                lyso_cluster_etimes = cluster_energy_time_padded[has_lyso].view(-1)  # [Total_K]
                lyso_cluster_energies = cluster_e_mean.view(-1).detach()  # [Total_K] normalized cluster E (for attention bias scaling)
                output['lyso_cluster_energies'] = lyso_cluster_energies

                # Export diagnostics
                output['lyso_dt_from_pos'] = dt_from_pos.view(-1)  # [Total_K] normalized
                output['lyso_dt_corr_ns'] = dt_corr_ns.view(-1)     # [Total_K] ns, TOF-corrected
                output['lyso_coinc_feat'] = coinc_feat.view(-1)     # [Total_K] Gaussian window
                output['positron_time_per_graph'] = positron_time_per_graph  # [B] normalized

                # Raw physics features for classifier head bypass
                cos_sep_pos_flat = cos_sep_positron[has_lyso].view(-1, 1)  # [Total_K, 1]
                output['lyso_coinc_skip_input'] = torch.cat(
                    [coinc_feat, cos_sep_pos_flat], dim=-1)  # [Total_K, 2]

                # Full 9-feature vector for per-cluster probing / feature-importance studies.
                # Columns (matched to angular_feat_full): sin_theta, cos_theta, sin_phi,
                # cos_phi, cos_sep_positron, energy_mean, coinc_feat, beta, soft_n_hits.
                output['lyso_cluster_features'] = angular_feat_full.detach()  # [Total_K, 9]

                # --- Slim LYSO token: 9 raw features → 32D ---
                # angular_feat_full is [Total_K, 9] containing:
                # [sin_theta, cos_theta, sin_phi, cos_phi, cos_sep_positron,
                #  energy_mean, coinc_feat, beta, soft_n_hits]
                lyso_event_tokens = self.lyso_event_proj(angular_feat_full)  # [Total_K, 32]

                # Slice positional embedding
                lyso_slice_ids = seed_slice_id_padded[has_lyso].view(-1).clamp(max=63)  # [Total_K]
                lyso_event_tokens = lyso_event_tokens + self.event_slice_emb(lyso_slice_ids)

                # Store for Phase 15 temporal + directional attention bias
                output['lyso_cluster_times'] = lyso_cluster_times
                output['lyso_seed_coords'] = cluster_phys_pos_padded  # [B, K_LYSO, 3] physical /100mm
                output['lyso_seed_latent_coords'] = seed_latent_coords_padded  # [B, K_LYSO, 3] OC latent space
                output['lyso_seed_has_lyso'] = has_lyso    # [B] bool
            else:
                lyso_pool_all = torch.empty((0, h_lyso.size(1)), device=x.device)
                output['lyso_soft_assignments'] = torch.empty((0, K_LYSO), device=x.device)
                lyso_event_batch_tensor = torch.empty((0,), dtype=torch.long, device=x.device)
                lyso_valid_tensor = torch.empty((0,), dtype=torch.bool, device=x.device)
                lyso_event_tokens = torch.empty((0, self.D_EVENT), device=x.device)

        # 2. Transformer Event Synthesis
        B = num_graphs_in_batch if is_atar.any() else len(torch.unique(batch))

        all_tokens = []
        all_batch = []
        all_valid = []
        all_times = []    # Per-token mean times (raw ns) for temporal attention bias
        all_lyso_E = []   # Per-token LYSO cluster E (normalized); ATAR slots stored as 0

        if is_atar.any() and 'atar_event_tokens' in output:
            B_atar_idx = valid_slice_mask.nonzero().squeeze(1) // num_slices_max
            # DETACH ATAR tokens and project down to D_EVENT=32
            atar_detached = output['atar_event_tokens'].detach()
            atar_down = self.atar_event_down(atar_detached)  # [N_slices, 32]

            atar_tokens_with_mod = atar_down + self.event_modality_emb.weight[0]
            all_tokens.append(atar_tokens_with_mod)

            # Per-slice mean time for ATAR tokens
            atar_hit_times = x[is_atar, 9]  # [N_atar] raw ns
            slice_time_sum = torch.zeros(num_global_slices, device=x.device)
            slice_time_cnt = torch.zeros(num_global_slices, device=x.device)
            slice_time_sum.index_add_(0, global_slice_idx_all, atar_hit_times)
            slice_time_cnt.index_add_(0, global_slice_idx_all, torch.ones_like(atar_hit_times))
            slice_mean_times_all = slice_time_sum / slice_time_cnt.clamp(min=1.0)
            atar_token_times = slice_mean_times_all[valid_slice_mask]  # [N_valid_slices]
            all_times.append(atar_token_times)
            all_lyso_E.append(torch.zeros_like(atar_token_times))   # ATAR placeholder
            all_batch.append(B_atar_idx)
            all_valid.append(torch.ones(atar_tokens_with_mod.size(0), dtype=torch.bool, device=x.device))
            output['unified_num_atar_tokens'] = all_tokens[0].size(0)

        if is_lyso.any() and 'lyso_soft_assignments' in output:
            lyso_tokens_with_mod = lyso_event_tokens + self.event_modality_emb.weight[1]
            all_tokens.append(lyso_tokens_with_mod)
            all_batch.append(lyso_event_batch_tensor)
            all_valid.append(lyso_valid_tensor)
            if 'lyso_cluster_times' in output:
                all_times.append(output['lyso_cluster_times'])
            if 'lyso_cluster_energies' in output:
                all_lyso_E.append(output['lyso_cluster_energies'])

        if len(all_tokens) > 0:
            unified_tokens = torch.cat(all_tokens, dim=0)
            unified_batch = torch.cat(all_batch, dim=0)
            unified_valid = torch.cat(all_valid, dim=0)

            # Track original ordering to un-shuffle after to_dense_batch
            original_idx = torch.arange(unified_tokens.size(0), device=unified_tokens.device)

            # CRITICAL FIX: PyG `to_dense_batch` deletes items via memory collisions if batch isn't sorted!
            # stable=True preserves (ATAR-then-LYSO, cluster 0..K-1) order within each graph so
            # the vectorized spatial-bias block below can map dense LYSO slots → cluster indices
            # via cumcount.
            sort_idx = torch.argsort(unified_batch, stable=True)
            unified_tokens = unified_tokens[sort_idx]
            unified_valid = unified_valid[sort_idx]
            unified_batch = unified_batch[sort_idx]
            original_idx = original_idx[sort_idx]

            from torch_geometric.utils import to_dense_batch
            # Pass batch_size=num_graphs_in_batch explicitly so the dense slot index matches
            # the original graph index even when the last graph in the batch contributes no
            # tokens (without it, B_tf would shrink to batch.max()+1, and downstream indexing
            # into lyso_seed_coords / pion_stop_pred — which are sized [num_graphs_in_batch] —
            # could silently desync for tail-empty graphs).
            dense_tokens, pad_mask = to_dense_batch(unified_tokens, unified_batch, batch_size=num_graphs_in_batch) # [B, max_tokens, D_A]
            dense_idx, _ = to_dense_batch(original_idx, unified_batch, batch_size=num_graphs_in_batch)

            # Use to_dense_batch to naturally align our custom manual validity mask!
            # Any element explicitly marked False (our manual LYSO padding) or
            # padded by the batch itself (PyG padding) evaluates to False.
            dense_valid, _ = to_dense_batch(unified_valid, unified_batch, batch_size=num_graphs_in_batch)

            # Transformer Forward
            padding_mask = ~dense_valid

            # === PHASE 15: Directional + Temporal Attention Bias ===
            src_mask = None
            n_atar_flat = output.get('unified_num_atar_tokens', 0)
            has_lyso_seeds = 'lyso_seed_coords' in output and is_lyso.any()

            # Build per-token modality flags (needed for temporal + directional bias)
            modality_flag = torch.zeros(unified_tokens.size(0), dtype=torch.long, device=x.device)
            modality_flag[n_atar_flat:] = 1
            sorted_modality = modality_flag[sort_idx]
            dense_modality, _ = to_dense_batch(sorted_modality, unified_batch, batch_size=num_graphs_in_batch)  # [B, T]

            # Build per-token trigger probability (ATAR tokens get their trigger prob, LYSO get 0)
            trigger_per_token = torch.zeros(unified_tokens.size(0), device=x.device)
            if n_atar_flat > 0 and 'atar_trigger_logits' in output:
                trigger_per_token[:n_atar_flat] = torch.sigmoid(output['atar_trigger_logits']).detach()
            sorted_trigger = trigger_per_token[sort_idx]
            dense_trigger, _ = to_dense_batch(sorted_trigger, unified_batch, batch_size=num_graphs_in_batch)  # [B, T]

            # Temporal bias: Gaussian with physics-aware time-of-flight offset
            # Same modality (ATAR↔ATAR, LYSO↔LYSO): expect dt ≈ 0
            # Cross modality: LYSO arrives ~TOF_NS after ATAR
            if len(all_times) > 0:
                unified_times = torch.cat(all_times, dim=0)
                sorted_times = unified_times[sort_idx]
                dense_times, _ = to_dense_batch(sorted_times, unified_batch, batch_size=num_graphs_in_batch)  # [B, T]

                unified_lyso_E = torch.cat(all_lyso_E, dim=0)
                sorted_lyso_E = unified_lyso_E[sort_idx]
                dense_lyso_E, _ = to_dense_batch(sorted_lyso_E, unified_batch, batch_size=num_graphs_in_batch)  # [B, T]

                B_tf = dense_tokens.size(0)
                dt = dense_times.unsqueeze(2) - dense_times.unsqueeze(1)  # [B, T, T] = t_i - t_j

                # Cross-modality offset: (mod_j - mod_i) = +1 when i=ATAR,j=LYSO; -1 when reverse; 0 same
                mod_i = dense_modality.unsqueeze(2)  # [B, T, 1]
                mod_j = dense_modality.unsqueeze(1)  # [B, 1, T]
                expected_dt = (mod_j - mod_i).float() * TOF_NS  # [B, T, T]

                # Per-token σ: ATAR = constant MIP-timing scale; LYSO = floor + scale/√E.
                # Pair σ² = σ_i² + σ_j² (independent errors add in quadrature).
                is_lyso_tok = (dense_modality == 1).float()                                   # [B, T]
                sigma_lyso_tok = (
                    self.sigma_t_lyso_floor_ns.abs()
                    + self.sigma_t_lyso_scale_ns.abs() / torch.sqrt(dense_lyso_E.clamp(min=1e-2))
                )                                                                              # [B, T]
                sigma_per_tok = is_lyso_tok * sigma_lyso_tok + (1.0 - is_lyso_tok) * self.sigma_t_atar_ns.abs()
                sigma_sq_pair = (sigma_per_tok.unsqueeze(1) ** 2 + sigma_per_tok.unsqueeze(2) ** 2).clamp(min=0.1)

                temporal_bias = torch.exp(-(dt - expected_dt) ** 2 / (2.0 * sigma_sq_pair))
                temporal_bias = torch.log(temporal_bias.clamp(min=1e-6))  # Additive log-space
                temporal_bias = temporal_bias * dense_valid.unsqueeze(2).float() * dense_valid.unsqueeze(1).float()
                # Clone before the spatial-bias loop below performs in-place += on src_mask.
                src_mask = temporal_bias.clone()

            if has_lyso_seeds:

                lyso_seed_coords = output['lyso_seed_coords']   # [B, K_LYSO, 3] physical /100mm
                has_lyso_flag = output['lyso_seed_has_lyso']    # [B] bool
                T = dense_tokens.size(1)
                B_tf = dense_tokens.size(0)
                K_LYSO_dim = lyso_seed_coords.size(1)

                if src_mask is None:
                    src_mask = torch.zeros(B_tf, T, T, device=x.device)

                is_atar_tok_mat = (dense_modality == 0) & dense_valid   # [B, T]
                is_lyso_tok_mat = (dense_modality == 1) & dense_valid   # [B, T]

                # Map each dense LYSO slot → cluster index k via cumulative count (requires
                # stable sort above so cluster order 0..K-1 is preserved).
                lyso_cumcount = is_lyso_tok_mat.long().cumsum(dim=1) - 1               # [B, T]
                k_range = torch.arange(K_LYSO_dim, device=x.device)                    # [K]
                slot_match = (
                    (lyso_cumcount.unsqueeze(1) == k_range.view(1, -1, 1))
                    & is_lyso_tok_mat.unsqueeze(1)
                )                                                                       # [B, K, T]

                K_actual_per_event = is_lyso_tok_mat.sum(dim=1)                        # [B]
                k_valid = (
                    (k_range.unsqueeze(0) < K_actual_per_event.unsqueeze(1))
                    & has_lyso_flag.unsqueeze(1)
                )                                                                       # [B, K]
                slot_match = slot_match & k_valid.unsqueeze(-1)
                sm = slot_match.float()                                                 # [B, K, T]

                # Direction from pion stop to each cluster (pion-stop-centered frame)
                # Both in physical /100mm space (pion_stop converted from ATAR /10mm)
                # Detached: this is a physics prior, gradients should not flow back
                # into lyso_seed_coords through normalization. Manual norm with a
                # generous eps avoids F.normalize's 1e-12 divide that blows up
                # (forward AND backward) when pion_stop and a cluster coincide or
                # when padded slots leave delta_cluster exactly zero.
                pion_stop_phys = pion_stop_pred.detach() * 0.1                         # [B, 3]
                delta_cluster = (lyso_seed_coords - pion_stop_phys.unsqueeze(1)).detach()  # [B, K, 3]
                delta_norm = delta_cluster.norm(dim=-1, keepdim=True).clamp(min=1e-3)  # [B, K, 1]
                cluster_dir = delta_cluster / delta_norm                               # [B, K, 3]

                # ATAR↔LYSO directional bias: cos(positron_dir, dir_to_cluster) / σ_angle(E).
                # σ_angle grows with 1/√E (MCS dilutes directional info for low-E clusters),
                # so high-E clusters get sharper bias, low-E clusters get weaker bias.
                if n_atar_flat > 0:
                    pd = positron_dir.detach()
                    pos_dir_norm = pd / pd.norm(dim=-1, keepdim=True).clamp(min=1e-3)  # [B, 3]
                    cos_align = (cluster_dir * pos_dir_norm.unsqueeze(1)).sum(dim=-1)  # [B, K]

                    # Per-cluster E via slot-match from dense_lyso_E [B, T] → [B, K].
                    # Padded slots stay at 0 and get σ pinned to floor+scale/√eps (harmless
                    # since k_valid gates bias scatter below).
                    cluster_E_BK = (sm * dense_lyso_E.unsqueeze(1)).sum(dim=-1)                # [B, K]
                    sigma_angle = (
                        self.angle_sigma_floor.abs()
                        + self.angle_sigma_scale.abs() / torch.sqrt(cluster_E_BK.clamp(min=1e-2))
                    )                                                                           # [B, K]
                    bias_values = cos_align / sigma_angle.clamp(min=1e-3)                       # [B, K]

                    # Per-LYSO-token bias: scatter cluster bias onto its dense slot
                    lyso_bias_per_tok = (sm * bias_values.unsqueeze(-1)).sum(dim=1)    # [B, T]
                    atar_f = is_atar_tok_mat.float()                                    # [B, T]
                    atar_lyso = atar_f.unsqueeze(-1) * lyso_bias_per_tok.unsqueeze(1)  # [B, T, T]
                    src_mask = src_mask + atar_lyso + atar_lyso.transpose(1, 2)

                # LYSO↔LYSO back-to-back bias: disabled. Pure temporal Gaussian
                # (expected_dt=0 for same-modality) already handles LYSO-LYSO
                # coincidence via the temporal_bias block above.
                # if len(all_times) > 0:
                #     cos_pair = torch.bmm(cluster_dir, cluster_dir.transpose(1, 2))
                #     cluster_times = (sm * dense_times.unsqueeze(1)).sum(dim=-1)
                #     dt_pair = cluster_times.unsqueeze(2) - cluster_times.unsqueeze(1)
                #     time_gate = torch.exp(-dt_pair ** 2 / (2.0 * sigma_sq))
                #     pair_bias = self.direction_bias_temperature * cos_pair.abs() * time_gate
                #     kk_valid = k_valid.unsqueeze(2) & k_valid.unsqueeze(1)
                #     pair_bias = pair_bias * kk_valid.float()
                #     eye_mask = torch.eye(K_LYSO_dim, device=x.device, dtype=torch.bool).unsqueeze(0)
                #     pair_bias = pair_bias.masked_fill(eye_mask, 0.0)
                #     lyso_pair_mat = torch.einsum('bki,bkl,blj->bij', sm, pair_bias, sm)
                #     src_mask = src_mask + lyso_pair_mat

            # TransformerEncoder expects mask as [T,T] or [B*nhead, T, T]
            if src_mask is not None:
                nhead = 2  # Must match slim TransformerEncoderLayer nhead
                # Expand [B, T, T] -> [B*nhead, T, T]
                src_mask_expanded = src_mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(-1, src_mask.size(1), src_mask.size(2))
            else:
                src_mask_expanded = None

            transformed_tokens = self.slim_event_transformer(
                dense_tokens,
                mask=src_mask_expanded,
                src_key_padding_mask=padding_mask
            )

            # Flatten back
            flat_transformed = transformed_tokens[pad_mask]
            flat_idx = dense_idx[pad_mask]

            # Un-shuffle back into [ATAR_TOKENS, LYSO_TOKENS] format
            inverse_sort = torch.argsort(flat_idx)
            flat_transformed = flat_transformed[inverse_sort]

            # Classifier: project transformer 32D → 8D, concat [coinc_feat, cos_sep] → Linear(10, 1)
            n_atar_tok = output.get('unified_num_atar_tokens', 0)
            coinc_skip_in = output.get('lyso_coinc_skip_input')  # [Total_K, 2]
            projected = self.event_head_proj(flat_transformed)  # [total_tokens, 8]
            phys_feat = torch.zeros(projected.size(0), 2, device=projected.device)
            if coinc_skip_in is not None and n_atar_tok < projected.size(0):
                phys_feat[n_atar_tok:] = coinc_skip_in

            event_logits = self.event_head(
                torch.cat([projected, phys_feat], dim=-1))  # [total_tokens, 1]

            output['unified_event_logits'] = event_logits
            output['unified_token_batch'] = unified_batch

        # === EVENT SUMMARY (non-trained aggregator) ===
        # Collects existing predictions into a per-event summary. Missing fields
        # use SENTINEL = -999.0. Acceptance mirrors pileup_mixer.py:146-153:
        # fiducial_z in [1.2, 4.8] mm, |x|<8, |y|<8, polar angle < 120°.
        SENTINEL = -999.0
        B = num_graphs_in_batch
        device = x.device

        pion_stop = output.get('atar_pion_stop')
        if pion_stop is None:
            pion_stop = torch.full((B, 3), SENTINEL, device=device)

        positron_dir = output.get('atar_positron_dir')
        if positron_dir is None:
            positron_dir = torch.full((B, 3), SENTINEL, device=device)
            polar_angle = torch.full((B,), SENTINEL, device=device)
        else:
            cos_theta = positron_dir[:, 2].clamp(-1.0, 1.0)
            polar_angle = torch.acos(cos_theta)  # radians

        pos_time = output.get('positron_time_per_graph')
        if pos_time is None:
            pos_time = torch.full((B,), SENTINEL, device=device)

        # Positron energy: ATAR positron-hit E + triggered LYSO cluster-hit E.
        pos_energy = torch.zeros(B, device=device)
        has_any = False

        # ATAR contribution
        trig_p = output.get('atar_hit_trigger_prob')
        mip_p = output.get('atar_hit_mip_prob')
        if trig_p is not None and mip_p is not None and is_atar.any():
            pos_mask = ((trig_p > 0.5) & (mip_p > 0.5)).float()
            batch_atar = batch[is_atar]
            e_atar = x[is_atar, 3] * NORM_E_ATAR  # back to physical MeV
            pos_energy.index_add_(0, batch_atar, e_atar * pos_mask)
            has_any = True

        # LYSO contribution: per-hit trigger prob via assignment-weighted mixture
        # p_hit_i = Σ_k(w_ik · β_k · p_k) / Σ_k(w_ik · β_k); hit counted if p_hit > 0.5
        w_lyso = output.get('lyso_soft_assignments')
        beta_lyso = output.get('lyso_seed_beta')
        ev_logits = output.get('unified_event_logits')
        n_atar_tok = output.get('unified_num_atar_tokens', 0)
        if (is_lyso.any() and w_lyso is not None and beta_lyso is not None
                and ev_logits is not None):
            K = w_lyso.size(1)
            lyso_p = torch.sigmoid(ev_logits[n_atar_tok:].view(-1))  # [Total_K]
            beta_bk = beta_lyso.view(-1, K)
            p_bk = lyso_p.view(-1, K)
            # Rebuild has_lyso -> valid-graph mapping
            hit_graph = batch[is_lyso]
            counts = torch.zeros(B, device=device, dtype=torch.long)
            counts.index_add_(0, hit_graph, torch.ones_like(hit_graph))
            has_lyso_g = counts > 0
            graph_to_valid = torch.full((B,), -1, device=device, dtype=torch.long)
            graph_to_valid[has_lyso_g] = torch.arange(
                int(has_lyso_g.sum().item()), device=device)
            hit_vg = graph_to_valid[hit_graph]  # [N_lyso]
            wb = w_lyso * beta_bk[hit_vg]  # [N_lyso, K]
            wb_sum = wb.sum(dim=1).clamp(min=1e-6)
            p_hit = (wb * p_bk[hit_vg]).sum(dim=1) / wb_sum  # [N_lyso]
            output['lyso_hit_trigger_prob'] = p_hit.detach()
            lyso_mask = (p_hit > 0.5).float()
            e_lyso = x[is_lyso, 3] * NORM_E_LYSO  # back to physical MeV
            pos_energy.index_add_(0, hit_graph, e_lyso * lyso_mask)
            has_any = True

        if not has_any:
            pos_energy = torch.full((B,), SENTINEL, device=device)

        # Acceptance decision based on predicted quantities.
        ps_raw = pion_stop * NORM_POS_ATAR
        fiducial = (
            (ps_raw[:, 2] > ACCEPT_Z_MIN_MM) & (ps_raw[:, 2] < ACCEPT_Z_MAX_MM) &
            (ps_raw[:, 0].abs() < ACCEPT_XY_MAX_MM) & (ps_raw[:, 1].abs() < ACCEPT_XY_MAX_MM)
        )
        angle_ok = positron_dir[:, 2] > _ACCEPT_ANGLE_COS
        htp_rule = output.get('has_trigger_positron_rule')
        if htp_rule is not None:
            accepted = (fiducial & angle_ok & (htp_rule > 0.5)).float()
        else:
            accepted = (fiducial & angle_ok).float()

        # === Dead-material energy head (detached, post-hoc regression) ===
        # All inputs are detached so this head cannot influence upstream
        # predictions (especially the positron-direction head). Output is
        # log(1 + dead_E_pred) in MeV; converted back via expm1 for the
        # event summary. Reported but NOT used in the acceptance rule.
        def _angle_features(direction):
            cos_t = direction[:, 2]
            sin_t = torch.sqrt((1.0 - cos_t.pow(2)).clamp(min=1e-6))
            cos_p = direction[:, 0] / sin_t
            cos_p = torch.where(sin_t < 1e-3, torch.zeros_like(cos_p), cos_p)
            return cos_t, cos_p

        cos_t_pos, cos_p_pos = _angle_features(positron_dir.detach())

        # Exit direction from the predicted-positron slice's endpoints.
        # Pick the slice with the largest p_e per event (excluding anchor).
        exit_unit = torch.zeros(B, 3, device=device)
        role_logits_de = output.get('atar_role_logits')
        slice_event_idx_de = output.get('atar_slice_event_idx')
        endpoints_de = output.get('atar_endpoints')
        anchor_mask_de = output.get('atar_anchor_slice_mask')
        if (role_logits_de is not None and slice_event_idx_de is not None
                and endpoints_de is not None):
            rl_d = role_logits_de.detach()
            ev_idx_d = slice_event_idx_de.detach()
            ep_d = endpoints_de.detach()
            pe_per_slice = F.softmax(rl_d, dim=-1)[:, 2]
            if anchor_mask_de is not None:
                pe_per_slice = pe_per_slice * (~anchor_mask_de).float()
            pred_start_n = ep_d[:, 0, :, 1]   # normalized units (direction unaffected)
            pred_stop_n  = ep_d[:, 1, :, 1]
            for ev in range(B):
                m = (ev_idx_d == ev)
                if m.any():
                    pe_ev = torch.where(m, pe_per_slice, torch.full_like(pe_per_slice, -1.0))
                    if float(pe_ev.max().item()) > 1e-6:
                        i = int(pe_ev.argmax().item())
                        diff = pred_stop_n[i] - pred_start_n[i]
                        n = diff.norm()
                        if n > 1e-6:
                            exit_unit[ev] = diff / n
        cos_t_exit, cos_p_exit = _angle_features(exit_unit)

        # Total positron-tagged energy (MeV), normalized by NORM_E_LYSO.
        pos_E_safe = torch.where(pos_energy > 0,
                                 pos_energy,
                                 torch.zeros_like(pos_energy))
        total_E_norm = (pos_E_safe / NORM_E_LYSO).detach()

        # Energy-weighted mean LYSO position (already normalized in the dataset).
        mean_x_lyso = torch.zeros(B, device=device)
        mean_y_lyso = torch.zeros(B, device=device)
        mean_z_lyso = torch.zeros(B, device=device)
        if 'lyso_hit_trigger_prob' in output and is_lyso.any():
            p_hit_d = output['lyso_hit_trigger_prob'].detach()
            mask_d = (p_hit_d > 0.5).float()
            e_lyso_mev = (x[is_lyso, 3] * NORM_E_LYSO).detach()
            gated = e_lyso_mev * mask_d
            batch_lyso_d = batch[is_lyso].detach()
            sum_E   = torch.zeros(B, device=device)
            sum_xE  = torch.zeros(B, device=device)
            sum_yE  = torch.zeros(B, device=device)
            sum_zE  = torch.zeros(B, device=device)
            sum_E.index_add_(0, batch_lyso_d, gated)
            sum_xE.index_add_(0, batch_lyso_d, x[is_lyso, 0].detach() * gated)
            sum_yE.index_add_(0, batch_lyso_d, x[is_lyso, 1].detach() * gated)
            sum_zE.index_add_(0, batch_lyso_d, x[is_lyso, 2].detach() * gated)
            safe_E = sum_E.clamp(min=1e-6)
            has_lyso = sum_E > 0
            mean_x_lyso = torch.where(has_lyso, sum_xE / safe_E, mean_x_lyso)
            mean_y_lyso = torch.where(has_lyso, sum_yE / safe_E, mean_y_lyso)
            mean_z_lyso = torch.where(has_lyso, sum_zE / safe_E, mean_z_lyso)

        dead_input = torch.stack([
            cos_t_pos, cos_p_pos,
            cos_t_exit, cos_p_exit,
            total_E_norm,
            mean_x_lyso, mean_y_lyso, mean_z_lyso,
        ], dim=1)
        log_dead_pred = self.dead_energy_head(dead_input).squeeze(-1)  # [B]
        dead_energy_pred = torch.expm1(log_dead_pred)                  # MeV
        output['dead_energy_log_pred'] = log_dead_pred                  # for loss
        output['dead_energy_pred']     = dead_energy_pred.detach()      # diagnostic

        summary = {
            'pion_stop': pion_stop.detach(),
            'positron_dir': positron_dir.detach(),
            'positron_polar_angle': polar_angle.detach(),
            'positron_time': pos_time.detach(),
            'positron_energy': pos_energy.detach(),
            'dead_energy': dead_energy_pred.detach(),
            'accepted': accepted.detach(),
        }
        if htp_rule is not None:
            summary['has_trigger_positron'] = htp_rule.detach()
        output['event_summary'] = summary

        return output


