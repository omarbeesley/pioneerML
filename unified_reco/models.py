"""
PURITY Architecture (PIONEER Unified Reconstruction via Interactive Transformer TopologY)
Fuses ATAR (x-z, y-z) and LYSO (3D) hits using a Joint Self-Attention Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

# Define Modality IDs
MODALITY_ATAR_XZ = 0
MODALITY_ATAR_YZ = 1
MODALITY_LYSO = 2

class PURITYHitEncoder(nn.Module):
    """
    Embeds heterogeneous hits into a shared latent space.
    Input format expected: [x, y, z, E, t, is_xz, is_yz, is_lyso, slice_id]
    We use the first 8 columns as numerical features.
    """
    def __init__(self, input_dim=8, hidden_dim=128):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # Learnable modality tokens for XZ, YZ, and LYSO
        self.modality_embedding = nn.Embedding(3, hidden_dim)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

    def forward(self, x):
        # x is expected to be [N_hits, 9]
        features = x[:, :8] # [x, y, z, E, t, is_xz, is_yz, is_lyso]
        
        # Derive modality index from the one-hot columns [5, 6, 7]
        # xz=0, yz=1, lyso=2
        modality_idx = torch.where(x[:, 7] > 0.5, 2, torch.where(x[:, 6] > 0.5, 1, 0))
        
        hit_embed = self.feature_proj(features)
        
        # Inject the modality token
        return hit_embed + self.modality_embedding(modality_idx)

from torch_geometric.utils import dense_to_sparse

def fully_connected_edge_index_batch(batch):
    """Creates a fully connected graph for each element in the batch, dropping self loops."""
    # Mask [N, N] is true where batch matches
    adj = (batch.unsqueeze(1) == batch.unsqueeze(0))
    adj.fill_diagonal_(False) # No self loops
    edge_index, _ = dense_to_sparse(adj)
    return edge_index

def build_purity_edge_attr(x, edge_index):
    """Builds discrete explicit edge features following strict structural rules: [dx, dy, dz, dE, dt, edge_type (one-hot)]"""
    if edge_index.numel() == 0:
        return torch.zeros((0, 11), dtype=torch.float, device=x.device)

    src, dst = edge_index
    u, v = x[src], x[dst]
    
    # 1. Identify modality from one-hot columns [5: is_xz, 6: is_yz, 7: is_lyso]
    # mod == 0 (XZ), 1 (YZ), 2 (LYSO)
    u_mod = torch.where(u[:, 7] > 0.5, 2, torch.where(u[:, 6] > 0.5, 1, 0))
    v_mod = torch.where(v[:, 7] > 0.5, 2, torch.where(v[:, 6] > 0.5, 1, 0))
    
    # 2. Re-scale to physical millimeters!
    # ATAR is normalized by 10.0, LYSO by 100.0
    u_scale = torch.ones_like(u[:, :5])
    u_scale[:, :3] = torch.where(u_mod.unsqueeze(1) < 2, 10.0, 100.0)
    u_scale[:, 3] = torch.where(u_mod < 2, 1.0, 70.0) # E-norms
    u_scale[:, 4] = 500.0 # T-norm is same for both
    
    v_scale = torch.ones_like(v[:, :5])
    v_scale[:, :3] = torch.where(v_mod.unsqueeze(1) < 2, 10.0, 100.0)
    v_scale[:, 3] = torch.where(v_mod < 2, 1.0, 70.0)
    v_scale[:, 4] = 500.0

    physical_u = u[:, :5] * u_scale
    physical_v = v[:, :5] * v_scale
    
    # 3. Calculate Physical Diffs FIRST
    diffs = physical_v - physical_u
    
    # 4. Edge-Type Specific Normalization
    # Identify which edges occur ENTIRELY within the ATAR
    is_pure_atar_edge = (u_mod < 2) & (v_mod < 2)
    
    # Scale ATAR spatial edges by 10.0mm, everything else (LYSO, Mix) by 100.0mm
    spatial_scale = torch.where(is_pure_atar_edge.unsqueeze(1), 10.0, 100.0)
    diffs[:, :3] /= spatial_scale
    
    # Scale ATAR energy edges by 1.0 MeV, everything else by 70.0 MeV
    energy_scale = torch.where(is_pure_atar_edge, 1.0, 70.0)
    diffs[:, 3] /= energy_scale
    
    # Global scaling for T
    diffs[:, 4] /= 500.0
    
    # Logical Interaction Masks
    m_xz_xz = (u_mod == 0) & (v_mod == 0)
    m_yz_yz = (u_mod == 1) & (v_mod == 1)
    m_xz_yz = ((u_mod == 0) & (v_mod == 1)) | ((u_mod == 1) & (v_mod == 0))
    m_calo_calo = (u_mod == 2) & (v_mod == 2)
    m_xz_calo = ((u_mod == 0) & (v_mod == 2)) | ((u_mod == 2) & (v_mod == 0))
    m_yz_calo = ((u_mod == 1) & (v_mod == 2)) | ((u_mod == 2) & (v_mod == 1))
    
    # Target 11D Container (5 geometry features + 6 distinct categorical one-hot flags)
    out = torch.zeros((diffs.size(0), 11), dtype=torch.float, device=x.device)
    out[:, :5] = diffs
    
    # Overwrite/Masking to enforce explicit user rules
    
    # 0. xz-xz: [dx, dz, dE, dt, one_hot=0]
    out[m_xz_xz, 1] = 0.0 # Drop dy
    out[m_xz_xz, 5] = 1.0 # edge_type 0
    
    # 1. yz-yz: [dy, dz, dE, dt, one_hot=1]
    out[m_yz_yz, 0] = 0.0 # Drop dx
    out[m_yz_yz, 6] = 1.0 # edge_type 1
    
    # 2. xz-yz: [dz, dE, dt, one_hot=2]
    out[m_xz_yz, 0] = 0.0 # Drop dx
    out[m_xz_yz, 1] = 0.0 # Drop dy
    out[m_xz_yz, 7] = 1.0 # edge_type 2
    
    # 3. calo-calo: [dx, dy, dz, dE, dt, one_hot=3]
    out[m_calo_calo, 8] = 1.0 # edge_type 3
    
    # 4. xz-calo: [dx, dy, dz, dt, one_hot=4] 
    out[m_xz_calo, 3] = 0.0 # Drop dE
    out[m_xz_calo, 9] = 1.0 # edge_type 4
    
    # 5. yz-calo: [dx, dy, dz, dt, one_hot=5]
    out[m_yz_calo, 3] = 0.0 # Drop dE
    out[m_yz_calo, 10] = 1.0 # edge_type 5
    
    return out


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
    def __init__(self, input_dim, hidden_dim):
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
        
        self.projection = nn.Linear(input_dim, num_points * coords * self.num_quantiles)

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

class PURITYModel(nn.Module):
    """
    The master unified model combining ATAR tracking and LYSO object condensation.
    """
    def __init__(self, input_dim=8, hidden_dim=150, num_blocks=3, heads=5, 
                 dropout=0.05, num_pdg_classes=3):
        super().__init__()
        
        self.encoder = PURITYHitEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        
        self.blocks = nn.ModuleList([
            JointAttentionBlock(hidden_dim=hidden_dim, heads=heads, edge_dim=11, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        from torch_geometric.nn import JumpingKnowledge, AttentionalAggregation
        
        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden_dim * num_blocks
        
        # --- ATAR Specific Heads ---
        # 1. Time-Slice PDG (Group Classifier)
        self.atar_slice_pdg_head = nn.Linear(jk_dim, num_pdg_classes) # Outputs 3
        
        # 2. Time-Slice Multi-Event Flag (Binary Classifier)
        self.atar_slice_multi_head = nn.Linear(jk_dim, 1)
        
        # 3. Splitter (Node PDG) - Input: Node JK
        self.atar_pdg_head = nn.Linear(jk_dim, 3) # Outputs 3

        # 3.5 Global anchor point
        
        
        # 4. Edge-Level Link Prediction Head (MLP on concatenated src+dst features)
        # Memory is controlled by the physics-based sparsification upstream, not here.
        self.atar_edge_head = nn.Sequential(
            nn.Linear(jk_dim * 2, jk_dim),
            nn.GELU(),
            nn.Linear(jk_dim, 1)
        )
        
        # --- Unified Event Synthesis (Object-Level Transformer) ---
        D_A = 256
        
        # ATAR Kinematics MLP: [Endpoints (2 points * 3 coords * 3 quantiles = 18) + Slice PDG (3) = 21] -> 32
        self.atar_kinematics_mlp = nn.Sequential(
            nn.Linear(21, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )
        
        # ATAR Event MLP: [Pooled Hit Features (jk_dim) + Kinematics (32)] -> D_A
        self.atar_event_mlp = nn.Sequential(
            nn.Linear(jk_dim + 32, D_A),
            nn.GELU(),
            nn.Linear(D_A, D_A)
        )
        
        # LYSO Event MLP: [Pooled Top-K Shell Features (jk_dim)] -> D_A
        self.lyso_event_mlp = nn.Sequential(
            nn.Linear(jk_dim, D_A),
            nn.GELU(),
            nn.Linear(D_A, D_A)
        )
        
        # Event Builder Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_A, nhead=4, dim_feedforward=D_A*2, batch_first=True)
        self.event_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 0 = ATAR Token, 1 = LYSO Token
        self.modality_embedding = nn.Embedding(2, D_A)
        
        # Binary Classifier for Triggering Event
        self.event_classifier = nn.Sequential(
            nn.Linear(D_A, D_A // 2),
            nn.GELU(),
            nn.Linear(D_A // 2, 1)
        )
        
        # Graph Level Pooling (General, Pion-Specific, MIP-Specific)
        def make_pool():
            return AttentionalAggregation(nn.Sequential(
                nn.Linear(jk_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
            ))
            
        self.pool_x_gen = make_pool()
        self.pool_y_gen = make_pool()
        
        #self.pool_x_pion = make_pool()
        #self.pool_y_pion = make_pool()
        
        #self.pool_x_mip = make_pool()
        #self.pool_y_mip = make_pool()
        
        # Asymmetric regressors for Kinematics (with +2 dimensions for Hybrid Max/Min pooling)
        # Specialized coordinate regressors with Joint "Stereo-Vision" Boundary Context
        # Input: Specialized Pool [X, Y, or Z] + ALL 6 Bounds [X, Y, Z Min/Max]
        # This prevents coordinate-mixing confusion while solving the "Wrong X to Wrong Z" mismatch.
        # specialized 1D Specialist Heads (Expert Projs) — prevent cross-view confusion
        self.atar_endpoint_x = QuantileOutputHead(input_dim=jk_dim, num_points=2, coords=1)
        self.atar_endpoint_y = QuantileOutputHead(input_dim=jk_dim, num_points=2, coords=1)
        self.atar_endpoint_z = QuantileOutputHead(input_dim=jk_dim, num_points=2, coords=1)
        
        # Stage 2: Permutation Selector — Resolves view-alignment globally for the slice.
        # [COMMENTED OUT: Z-ordering in dataset.py handles this naturally]
        #self.atar_permutation_selector = nn.Sequential(
        #    nn.Linear(jk_dim * 3 + 18 + 6, hidden_dim // 2),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim // 2, 1)
        #)

        #self.atar_endpoints_joint = QuantileOutputHead(jk_dim * 3, num_points=2, coords=3)
        
        def make_coord_head():
            return nn.Sequential(
                nn.Linear(jk_dim, jk_dim // 2),
                nn.GELU(),
                nn.Linear(jk_dim // 2, 1)
            )
            
        #self.atar_pion_stop_x = make_coord_head()
        #self.atar_pion_stop_y = make_coord_head()
        #self.atar_pion_stop_z = make_coord_head()
        
        #self.atar_angle_head = VectorHead(jk_dim * 2, hidden_dim)
        
        # --- LYSO Specific Heads (Object Condensation) ---
        # Node Level
        self.lyso_beta_head = nn.Sequential(
            nn.Linear(jk_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.lyso_cluster_coord_head = nn.Sequential(
            nn.Linear(jk_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3) 
        )
        self.lyso_fraction_head = nn.Sequential(
            nn.Linear(jk_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        #self.lyso_payload_head = nn.Sequential(
        #    nn.Linear(jk_dim, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 4) # x, y, z, E_tot
        #)

    def forward(self, x, batch, task_weights=None):
        """
        x: [N_total_hits, 6] (features + modality_idx)
        batch: [N_total_hits] PyG batch index
        """
        if task_weights is None: task_weights = {}

        # 0. Extract modality flags and slice IDs
        # x columns: [pos_x, pos_y, pos_z, energy, time, is_xz, is_yz, is_lyso, slice_id]
        is_atar_x = (x[:, 5] > 0.5)
        is_atar_y = (x[:, 6] > 0.5)
        is_atar = is_atar_x | is_atar_y
        is_lyso = (x[:, 7] > 0.5)
        
        # 1. Encode all nodes
        h_out = self.encoder(x) # [N_total, hidden_dim]
        
        # 2. Create fully connected graph (within each batch element)
        edge_index = fully_connected_edge_index_batch(batch)
        
        # 3. Build geometric edge features [dx, dy, dz, dE, dt, same_modality]
        edge_attr = build_purity_edge_attr(x, edge_index)
        
        # 4. Dynamic Joint Attention Message Passing
        xs = []
        for block in self.blocks:
            h_out = block(h_out, edge_index, edge_attr)
            xs.append(h_out)
            
        h_out = self.jk(xs)
        
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
            
            # Scatter ones to verify hits exist
            count_x.index_add_(0, global_slice_idx_x, torch.ones_like(global_slice_idx_x, dtype=torch.float).unsqueeze(1))
            count_y.index_add_(0, global_slice_idx_y, torch.ones_like(global_slice_idx_y, dtype=torch.float).unsqueeze(1))
            
            has_x = (count_x > 0).squeeze()
            has_y = (count_y > 0).squeeze()
            
            # Apply PyG AttentionalAggregation per Time Slice with specific Splitter masks
            def pool_with_mask(pool_layer_x, pool_layer_y, mask_x, mask_y):
                hx = h_atar_x * mask_x if mask_x is not None else h_atar_x
                hy = h_atar_y * mask_y if mask_y is not None else h_atar_y
                
                px = pool_layer_x(hx, global_slice_idx_x, dim_size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=hx.device)
                py = pool_layer_y(hy, global_slice_idx_y, dim_size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=hy.device)
                
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
            if task_weights.get('w_node_pdg', 1.0) > 0.0:
                output['atar_node_pdg'] = self.atar_pdg_head(h_atar)
            else:
                output['atar_node_pdg'] = torch.zeros(h_atar.size(0), self.atar_pdg_head.out_features, device=x.device)
            
            #if task_weights.get('w_atar_hit_trigger', 1.0) > 0.0:
            #    output['atar_hit_trigger'] = self.atar_hit_trigger_head(h_atar).squeeze(-1)
            #else:
            #    output['atar_hit_trigger'] = torch.zeros(h_atar.size(0), device=x.device)

            atar_node_probs = torch.sigmoid(output['atar_node_pdg'])

            # --- PHASE 1b: Edge-Level Link Prediction (radius_graph on Z coordinate) ---
            # radius_graph is an optimized CUDA kernel from torch_geometric.
            # It builds all edges within Z_THR = 0.2mm — equivalent to our sliding window,
            # but in a single GPU call with no Python loops and no N² intermediates.
            from torch_geometric.nn import radius_graph
            Z_THR = 0.02  # 0.2mm / NORM_POS_ATAR(10mm)

            atar_global_indices = torch.where(is_atar)[0]  # [N_atar] global node indices

            atar_z_pos = x[is_atar, 2].unsqueeze(1)  # [N_atar, 1] — 1D position
            batch_atar = batch[is_atar]               # [N_atar] batch ID (no cross-event edges)

            with torch.no_grad():
                sparse_local_ei = radius_graph(
                    atar_z_pos,
                    r=Z_THR,
                    batch=batch_atar,
                    loop=False,
                    max_num_neighbors=32,
                )  # [2, N_sparse] ATAR-local indices

            sparse_src_local = sparse_local_ei[0]
            sparse_dst_local = sparse_local_ei[1]

            n_sparse = sparse_src_local.size(0)
            edge_logits = torch.zeros(n_sparse, 1, device=x.device)
            if n_sparse > 0:
                # DETACH the hitting features before the edge head! 
                # This prevents BCE gradients from "distorting" the latent space 
                # that the endpoint regression heads rely on.
                edge_feats = torch.cat([
                    h_atar[sparse_src_local], 
                    h_atar[sparse_dst_local]
                ], dim=1)
                edge_logits = self.atar_edge_head(edge_feats)

            edge_probs = torch.sigmoid(edge_logits).squeeze(-1)

            output['atar_edge_logits'] = edge_logits.squeeze(-1)
            output['atar_local_edge_index'] = torch.stack([sparse_src_local, sparse_dst_local], dim=0)
            # atar_edge_index for label propagation (remap local -> global)
            if n_sparse > 0:
                atar_edge_index = torch.stack([
                    atar_global_indices[sparse_src_local],
                    atar_global_indices[sparse_dst_local]
                ], dim=0)
            else:
                atar_edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            output['atar_edge_index'] = atar_edge_index
            
            # --- PHASE 1c: GPU-Native Label Propagation (No CPU sync!) ---
            # Prune to high-confidence edges — indices are already ATAR-local from radius_graph
            valid_edge_mask = (edge_probs > 0.5)
            pruned_src_local = sparse_src_local[valid_edge_mask]
            pruned_dst_local = sparse_dst_local[valid_edge_mask]

            N_atar = h_atar.size(0)

            with torch.no_grad():
                # Initialize: each node is its own component (unique ID = local index)
                labels = torch.arange(N_atar, dtype=torch.long, device=h_atar.device)

                if pruned_src_local.size(0) > 0:
                    # Label Propagation: N_atar iterations guarantees full convergence
                    # on any connected subgraph, regardless of path length.
                    for _ in range(N_atar):
                        # Broadcast MAX label from src to dst
                        src_labels = labels[pruned_src_local]
                        labels.scatter_reduce_(0, pruned_dst_local, src_labels, reduce='amax', include_self=True)
                        # Also propagate in reverse direction for undirected connectivity
                        dst_labels = labels[pruned_dst_local]
                        labels.scatter_reduce_(0, pruned_src_local, dst_labels, reduce='amax', include_self=True)
            
            # --- PHASE 1d: Trigger Subgraph Identification via Time Proximity ---
            # Triggering hits are, by definition, the first hits at t >= 0.
            # Pileup hits at t < 0 arrived before the trigger and are excluded.
            # The connected component containing the earliest positive-time hit is the trigger subgraph.
            atar_times = x[is_atar, 4]  # Normalized time column (col 4)
            pos_time_mask = (atar_times >= 0.0)
            
            if pos_time_mask.any():
                # Offset local indices of t>=0 hits, then find the earliest
                pos_indices = torch.where(pos_time_mask)[0]
                earliest_local = pos_indices[torch.argmin(atar_times[pos_time_mask])]
            else:
                # Fallback: all times are negative, just take the least negative
                earliest_local = torch.argmin(torch.abs(atar_times))
            
            trigger_label = labels[earliest_local]
            atar_hit_trigger_probs = (labels == trigger_label).float()
            
            output['atar_hit_trigger'] = atar_hit_trigger_probs

            # --- PHASE 2: General Pooling (Raw) for Multi-Event Slice Prediction ---
            pool_x_raw, pool_y_raw = pool_with_mask(self.pool_x_gen, self.pool_y_gen, None, None)
            stereo_gen_raw = safe_mean(pool_x_raw, pool_y_raw)[valid_slice_mask]
            
            # Predict Multi-Event Slice Flag
            output['atar_slice_multi'] = self.atar_slice_multi_head(stereo_gen_raw).squeeze(-1)
            slice_multi_probs = torch.sigmoid(output['atar_slice_multi']) # [N_valid_slices]
            
            # --- PHASE 3: Soft Gating (Apply Trigger Probabilities to Multi-Event Slices) ---
            global_slice_multi_probs = torch.zeros(num_global_slices, device=x.device)
            slice_global_indices = torch.nonzero(valid_slice_mask).squeeze(1)
            global_slice_multi_probs[slice_global_indices] = slice_multi_probs
            
            # Broadcast multi-prob back to individual ATAR hits
            h_atar_slice_idx = global_slice_ids[is_atar]
            node_multi_probs = global_slice_multi_probs[h_atar_slice_idx]
            
            # Confidence sharpening: only activate trigger-gating when very sure about multi-particle overlap.
            # alpha=3 suppresses the gate for P_multi < ~0.7, breaking the circular dependency during early training.
            ALPHA = 1
            sharp_multi_probs = node_multi_probs ** ALPHA
            
            # Soft weight: (1.0 - P_multi^alpha) [Single Event] + (P_multi^alpha * P_trigger) [Pileup]
            hit_weights = (1.0 - sharp_multi_probs) + (sharp_multi_probs * atar_hit_trigger_probs)
            hit_weights = hit_weights.unsqueeze(-1) # [N_atar, 1] for masking
            
            # Reproject to global node space for Splitter Masks
            global_node_probs = torch.zeros(x.size(0), self.atar_pdg_head.out_features, device=x.device)
            global_node_probs[is_atar] = atar_node_probs
            
            global_hit_weights = torch.ones(x.size(0), 1, device=x.device)
            global_hit_weights[is_atar] = hit_weights
            
            prob_x = global_node_probs[is_atar_x]
            prob_y = global_node_probs[is_atar_y]
            
            weight_x = global_hit_weights[is_atar_x]
            weight_y = global_hit_weights[is_atar_y]
            
            # --- PHASE 4: Final Weighted Pooling (For Slice PDG and Downstream) ---
            pool_x_gen, pool_y_gen = pool_with_mask(self.pool_x_gen, self.pool_y_gen, weight_x, weight_y)
            stereo_gen = safe_mean(pool_x_gen, pool_y_gen)[valid_slice_mask]
            
            # Predict Time-Slice PDG from Weighted Features
            slice_logits = self.atar_slice_pdg_head(stereo_gen)
            output['atar_slice_pdg'] = slice_logits
            
            # Task-Specific Pooling (Masked by Splitter Context + Trigger Soft-Gate)
            # DETACH the mask to prevent gradients destroying the Splitter classifier!
            #prob_pion_x = (prob_x[:, 0].unsqueeze(1) * weight_x).detach()
            #prob_pion_y = (prob_y[:, 0].unsqueeze(1) * weight_y).detach()
            
            #pool_x_pion, pool_y_pion = pool_with_mask(self.pool_x_pion, self.pool_y_pion, prob_pion_x, prob_pion_y)
            #stereo_pion = safe_mean(pool_x_pion, pool_y_pion)[valid_slice_mask]
            
            #prob_mip_x = (prob_x[:, 2].unsqueeze(1) * weight_x).detach()
            #prob_mip_y = (prob_y[:, 2].unsqueeze(1) * weight_y).detach()
            
            #pool_x_mip, pool_y_mip = pool_with_mask(self.pool_x_mip, self.pool_y_mip, prob_mip_x, prob_mip_y)
            #stereo_mip = safe_mean(pool_x_mip, pool_y_mip)[valid_slice_mask]
            
            # --- PHASE 4: Evaluate Sub-Heads ---
            
            # Filter away empty slices to save computation for X and Y specific heads
            #valid_pool_x_pion = pool_x_pion[valid_slice_mask]
            #valid_pool_y_pion = pool_y_pion[valid_slice_mask]
            
            # Defer Event Builder and Energy until LYSO features are available
            atar_stereo_gen_full = torch.zeros(num_global_slices, stereo_gen.size(1), device=x.device)
            atar_stereo_gen_full[slice_global_indices] = stereo_gen
            
            #atar_stereo_mip_full = torch.zeros(num_global_slices, stereo_mip.size(1), device=x.device)
            #atar_stereo_mip_full[slice_global_indices] = stereo_mip

            valid_x_gen = pool_x_gen[valid_slice_mask]
            valid_y_gen = pool_y_gen[valid_slice_mask]
            # stereo_gen already defined at line 488

            # Stage 1: Expert Projections (Focused Specialists)
            # Regress directly from the trigger-masked pooled features
            x_pred_expert = self.atar_endpoint_x(valid_x_gen)
            y_pred_expert = self.atar_endpoint_y(valid_y_gen)
            z_pred_expert = self.atar_endpoint_z(stereo_gen)

            # Stage 2: Permutation Selection (Joint Context)
            # [COMMENTED OUT: Z-ordering in dataset.py is sufficient]
            #selector_input = torch.cat([
            #    valid_x_gen, valid_y_gen, stereo_gen,
            #    x_pred_expert.flatten(1), 
            #    y_pred_expert.flatten(1), 
            #    z_pred_expert.flatten(1),
            #    x_bounds, y_bounds, z_bounds
            #], dim=1)
            #prob = torch.sigmoid(self.atar_permutation_selector(selector_input)).unsqueeze(-1).unsqueeze(-1)
            #prob_hard = (prob > 0.5).float()
            #prob_ste = (prob_hard - prob).detach() + prob
            #y_pred_expert_flipped = torch.flip(y_pred_expert, dims=[1])
            #y_pred_final = (1.0 - prob_ste) * y_pred_expert + prob_ste * y_pred_expert_flipped
            
            # Final output: Independent specialists aligned by Z-progressions
            output['atar_endpoints'] = torch.cat([x_pred_expert, y_pred_expert, z_pred_expert], dim=2)
            #output['atar_prob_swap_y'] = prob.squeeze()
            
            # Export expert predictions for optional auxiliary loss monitoring
            output['atar_endpoints_expert_x'] = x_pred_expert
            output['atar_endpoints_expert_y'] = y_pred_expert
            output['atar_endpoints_expert_z'] = z_pred_expert
            
            #if task_weights.get('w_pion_kinematics', 1.0) > 0.0:
            #    output['atar_pion_stop_x'] = self.atar_pion_stop_x(valid_pool_x_pion).squeeze(-1)
            #    output['atar_pion_stop_y'] = self.atar_pion_stop_y(valid_pool_y_pion).squeeze(-1)
            #    output['atar_pion_stop_z'] = self.atar_pion_stop_z(stereo_pion).squeeze(-1)
            
            #if task_weights.get('w_positron_angle', 1.0) > 0.0:
            #    # Concatenate orthogonal representations to prevent angular blurring
            #    stereo_mip_cat = torch.cat([pool_x_mip, pool_y_mip], dim=1)[valid_slice_mask]
            #    output['atar_angle'] = self.atar_angle_head(stereo_mip_cat) # Unit Vector
            
            # Return meta information to unroll predictions downstream
            output['valid_slice_mask'] = valid_slice_mask
            output['valid_slice_indices'] = torch.nonzero(valid_slice_mask).squeeze(1)
            output['num_graphs_in_batch'] = num_graphs_in_batch
            output['num_slices_max'] = num_slices_max
            
            # --- ATAR Event Builder Early Fusion ---
            # Extract endpoints: [N_slices, 2 (points), 3 (coords), 3 (quantiles)] -> 18 features total
            # We preserve all quantiles (lower, median, upper) so the Event Builder sees the uncertainty
            endpoints_all = output['atar_endpoints'] 
            # Flatten to [N_slices, 18]
            endpoints_flat = endpoints_all.view(endpoints_all.size(0), -1)
            
            # Combine kinematics: Endpoints + Slice PDG -> [N_slices, 21]
            atar_kin_input = torch.cat([endpoints_flat, output['atar_slice_pdg']], dim=1)
            
            # Pass through ATAR Kinematics MLP
            atar_kin_feat = self.atar_kinematics_mlp(atar_kin_input)
            
            # Concatenate soft-gated pooled hits (stereo_gen) and kinematics -> ATAR Event Tokens
            atar_event_input = torch.cat([stereo_gen, atar_kin_feat], dim=1)
            atar_event_tokens = self.atar_event_mlp(atar_event_input)
            output['atar_event_tokens'] = atar_event_tokens
        
        # === LYSO PREDICTIONS (Object Condensation) ===
        if is_lyso.any():
            h_lyso = h_out[is_lyso]
            
            if task_weights.get('w_lyso_condensation', 1.0) > 0.0:
                output['lyso_beta'] = self.lyso_beta_head(h_lyso)
                output['lyso_cluster_coords'] = self.lyso_cluster_coord_head(h_lyso)
                output['lyso_fractions'] = self.lyso_fraction_head(h_lyso)
                #output['lyso_payload'] = self.lyso_payload_head(h_lyso)
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
        if hasattr(self, 'lyso_event_mlp') and is_lyso.any() and task_weights.get('w_lyso_condensation', 1.0) > 0.0:
            pred_coords = output['lyso_cluster_coords'] # [N_lyso, 3]
            pred_beta = output['lyso_beta'].squeeze(-1) # [N_lyso]
            lyso_batch = batch[is_lyso]                 # [N_lyso]
            
            B = num_graphs_in_batch if is_atar.any() else len(torch.unique(batch))
            lyso_pool_list = []
            lyso_event_batch = []
            lyso_assign_list = []
            lyso_valid_list = []
            
            for b in range(B):
                g_mask = (lyso_batch == b)
                if not g_mask.any():
                    continue
                
                g_coords = pred_coords[g_mask] # [N_g, 3]
                g_beta = pred_beta[g_mask]     # [N_g]
                g_feats = h_lyso[g_mask]       # [N_g, D]
                
                # --- FILTER INTRINSIC RADIOACTIVITY SEEDS ---
                # A LYSO hit is only allowed to form a shower seed if:
                # 1. It belongs to a coincident time slice (slice_id > 0)
                # 2. Or it has high isolated energy (> 2 MeV)
                g_slice_id = x[is_lyso][g_mask, 6]
                g_energy = x[is_lyso][g_mask, 3] # Normalized Energy
                energy_threshold_norm = 2.0 / 70.0 # 2 MeV / NORM_E_LYSO
                
                is_valid_seed = (g_slice_id > 0) | (g_energy > energy_threshold_norm)
                g_beta_seeds = g_beta * is_valid_seed.float()
                
                k = min(K_LYSO, g_coords.size(0))
                # Protect against edge case where max graph nodes < k
                topk_vals, topk_idx = torch.topk(g_beta_seeds, k=k)
                
                # Extract seeds
                seed_coords = g_coords[topk_idx] # [k, 3]
                seed_beta = g_beta[topk_idx]     # [k]
                
                # Compute distance [N_g, k]
                dists = torch.cdist(g_coords, seed_coords)
                
                # Raw weights: Use absolute Gaussian affinity instead of relative softmax!
                # This ensures that hits far from ALL seeds get absolute affinity near 0.0
                tau = 1.0 
                affinity = torch.exp(-dists / tau) * seed_beta.unsqueeze(0) # [N_g, k]
                
                # Normalize so hits split energy between active showers, 
                # but ADD a background dustbin factor (e.g. 0.05). 
                # If a hit is far from everything, affinity.sum() is tiny, so w_norm drops to ~0.0
                noise_floor = 0.05 
                w_norm = affinity / (affinity.sum(dim=1, keepdim=True) + noise_floor) # [N_g, k]
                
                # Pool Features: w_norm.T [k, N_g] @ g_feats [N_g, D] -> [k, D]
                g_pool = torch.matmul(w_norm.t(), g_feats)
                # This line crushes the vectors of low-confidence seeds -- we need to check on whether this causes problems
                g_pool = g_pool * seed_beta.unsqueeze(1)

                if k < K_LYSO:
                    pad_dim = K_LYSO - k
                    # Pad the last dimension (columns) with zeros
                    w_norm = torch.nn.functional.pad(w_norm, (0, pad_dim)) 
                    
                    # Also pad the pooled features so every graph returns exactly K_LYSO vectors
                    # g_pool is [k, D] -> pad dim 0 (rows)
                    g_pool = torch.nn.functional.pad(g_pool, (0, 0, 0, pad_dim))
                    
                    # Pad the batch tracking
                    lyso_event_batch.append(torch.full((K_LYSO,), b, dtype=torch.long, device=x.device))
                    lyso_valid_list.append(torch.cat([torch.ones(k, dtype=torch.bool, device=x.device), torch.zeros(pad_dim, dtype=torch.bool, device=x.device)]))
                else:
                    lyso_event_batch.append(torch.full((k,), b, dtype=torch.long, device=x.device))
                    lyso_valid_list.append(torch.ones(k, dtype=torch.bool, device=x.device))

                # Hard constraint to trap the 68-token anomaly precisely at the source:
                assert g_pool.size(0) == K_LYSO, f"CRITICAL SHAPE ANOMALY: g_pool has {g_pool.size(0)} rows, strongly proving padding did NOT evaluate for graph {b} (hits: {g_coords.size(0)})!"

                lyso_assign_list.append(w_norm)
                lyso_pool_list.append(g_pool)

                #lyso_pool_list.append(g_pool)
                #lyso_event_batch.append(torch.full((k,), b, dtype=torch.long, device=x.device))
                #lyso_assign_list.append(w_norm)
                
            if len(lyso_pool_list) > 0:
                lyso_pool_all = torch.cat(lyso_pool_list, dim=0) # [Sum(K_g), D]
                lyso_event_batch_tensor = torch.cat(lyso_event_batch, dim=0)
                lyso_valid_tensor = torch.cat(lyso_valid_list, dim=0)
                output['lyso_soft_assignments'] = torch.cat(lyso_assign_list, dim=0) # [N_lyso_total, k] mapping local hits to their top K
                
                lyso_event_tokens = self.lyso_event_mlp(lyso_pool_all)

        # 2. Transformer Event Synthesis
        B = num_graphs_in_batch if is_atar.any() else len(torch.unique(batch))
        
        all_tokens = []
        all_batch = []
        all_valid = []
        
        if is_atar.any() and 'atar_event_tokens' in output:
            B_atar_idx = valid_slice_mask.nonzero().squeeze(1) // num_slices_max
            atar_tokens_with_mod = output['atar_event_tokens'] + self.modality_embedding.weight[0]
            all_tokens.append(atar_tokens_with_mod)
            all_batch.append(B_atar_idx)
            all_valid.append(torch.ones(atar_tokens_with_mod.size(0), dtype=torch.bool, device=x.device))
            output['unified_num_atar_tokens'] = all_tokens[0].size(0)
            
        if is_lyso.any() and 'lyso_soft_assignments' in output:
            lyso_tokens_with_mod = lyso_event_tokens + self.modality_embedding.weight[1]
            all_tokens.append(lyso_tokens_with_mod)
            all_batch.append(lyso_event_batch_tensor)
            all_valid.append(lyso_valid_tensor)
            
        if len(all_tokens) > 0:
            unified_tokens = torch.cat(all_tokens, dim=0)
            unified_batch = torch.cat(all_batch, dim=0)
            unified_valid = torch.cat(all_valid, dim=0)
            
            # Track original ordering to un-shuffle after to_dense_batch
            original_idx = torch.arange(unified_tokens.size(0), device=unified_tokens.device)
            
            # CRITICAL FIX: PyG `to_dense_batch` deletes items via memory collisions if batch isn't sorted!
            sort_idx = torch.argsort(unified_batch)
            unified_tokens = unified_tokens[sort_idx]
            unified_valid = unified_valid[sort_idx]
            unified_batch = unified_batch[sort_idx]
            original_idx = original_idx[sort_idx]
            
            from torch_geometric.utils import to_dense_batch
            dense_tokens, pad_mask = to_dense_batch(unified_tokens, unified_batch) # [B, max_tokens, D_A]
            dense_idx, _ = to_dense_batch(original_idx, unified_batch)
            
            # Use to_dense_batch to naturally align our custom manual validity mask!
            # Any element explicitly marked False (our manual LYSO padding) or 
            # padded by the batch itself (PyG padding) evaluates to False. 
            dense_valid, _ = to_dense_batch(unified_valid, unified_batch)
            
            # Transformer Forward
            # nn.TransformerEncoder expects src_key_padding_mask as True for PADDED elements
            padding_mask = ~dense_valid
            
            transformed_tokens = self.event_transformer(dense_tokens, src_key_padding_mask=padding_mask) # [B, max_tokens, D_A]
            
            # Flatten back (this naturally orders by batch index!)
            # Note: We must slice using pad_mask (PyG layout elements) to recover the identical row sizes 
            # for event_logits so it maps precisely to p_tokens logic downstream.
            flat_transformed = transformed_tokens[pad_mask]
            flat_idx = dense_idx[pad_mask]
            
            # Un-shuffle back into [ATAR_TOKENS, LYSO_TOKENS] format
            inverse_sort = torch.argsort(flat_idx)
            flat_transformed = flat_transformed[inverse_sort]
            
            # Classifier
            event_logits = self.event_classifier(flat_transformed)
            output['unified_event_logits'] = event_logits
            output['unified_token_batch'] = unified_batch

        return output

