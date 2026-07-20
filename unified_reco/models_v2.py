"""
PURITY V2 — Targeted capacity increases over the baseline PURITYHybridModel.

Changes from models.py PURITYHybridModel:
1. ATAR Event Builder: FFN widened from 2×D_A to 4×D_A (richer per-layer capacity
   without adding hops — the 2-hop chain limit is physically motivated and unchanged)
2. Slim Event Builder: D_EVENT 32→64, nhead 2→4, layers 2→3 (more capacity for
   integrating ATAR track + LYSO cluster information for the trigger decision)
3. LYSO Object Condensation heads: intermediate dim 32→128→64 (decompresses the
   jk_dim→output bottleneck that was 14× in one layer)
4. Bidirectional cross-attention bridge: added LYSO-queries-ATAR direction so the
   LYSO backbone can see ATAR track context (ATAR tokens detached, so ATAR losses
   don't flow into LYSO backbone — mirrors the existing forward direction's detach)
5. Role head: added 512D intermediate layer (softens the 2310→256 compression)

All changes are backward-compatible with the same dataset, loss functions, and
training scripts — just replace PURITYHybridModel with PURITYHybridModelV2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_add_pool, AttentionalAggregation

from models import (
    PURITYHybridModel,
    physics_edge_index_batch,
    build_atar_edge_attr,
    build_lyso_edge_attr,
)
from constants import (
    NORM_T_LYSO, NORM_E_LYSO, NORM_E_ATAR, NORM_POS_ATAR,
    SIGMA_COINC_NS, TOF_NS,
    ACCEPT_Z_MIN_MM, ACCEPT_Z_MAX_MM, ACCEPT_XY_MAX_MM, ACCEPT_ANGLE_MAX_DEG,
)

_ACCEPT_ANGLE_COS = math.cos(math.radians(ACCEPT_ANGLE_MAX_DEG))


class PURITYHybridModelV2(PURITYHybridModel):
    """
    Scaled PURITY architecture.  Subclasses PURITYHybridModel and surgically
    replaces the five module groups listed in the module docstring.  The
    forward() method is overridden to add reverse cross-attention and to use
    the updated slim-event-builder head count.
    """

    def __init__(self, hidden_dim=150, num_blocks=3, heads=5,
                 dropout=0.1, num_pdg_classes=3):
        # Build the full V1 model first — all shared components are inherited.
        super().__init__(hidden_dim, num_blocks, heads, dropout, num_pdg_classes)

        jk_dim = hidden_dim * num_blocks
        D_A = self.D_A  # 256

        # === V2 CHANGE 0: TIME-BLIND LYSO node encoder ===
        # The base class encodes LYSO nodes from [x, y, z, E, t] (col 4 = absolute time).
        # That absolute time flows through the ATAR<-LYSO cross-attention into the trigger/
        # MIP classifiers and hence into the accept decision, letting the model gate on the
        # calo cluster's absolute (decay) time — it learns to reject positrons coincident
        # with negative-time calo clusters, so old-muon acceptance is NOT flat in time.  Drop
        # col 4 here: nodes carry only [x, y, z, E].  Relative timing needed for clustering
        # still reaches the model as |dt| on the LYSO edges (build_lyso_edge_attr) and as the
        # TOF-corrected coincidence feature; the slice grouping (col 8) handles coarse time
        # grouping.  Forward must feed x[is_lyso, :4] to match this Linear(4, ...).
        self.lyso_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # === V2 CHANGE 1: ATAR Event Builder FFN 2×D_A → 4×D_A ===
        # Wider FFN gives each layer more capacity to process slice relationships
        # without adding extra hops (L_TRIG stays at 2 — chain is max 2 hops).
        self.atar_event_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_A, nhead=4, dim_feedforward=4 * D_A,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(self.L_TRIG)
        ])

        # === V2 CHANGE 2: Slim Event Builder D_EVENT 32→64 ===
        # The event builder integrates ATAR track tokens with LYSO cluster tokens
        # for the trigger decision.  With richer upstream tokens (from change 1),
        # the decision head benefits from more capacity.
        D_EVENT = 64
        self.D_EVENT = D_EVENT
        self.lyso_event_proj = nn.Linear(9, D_EVENT)
        self.atar_event_down = nn.Linear(D_A, D_EVENT)
        self.event_modality_emb = nn.Embedding(2, D_EVENT)
        self.event_slice_emb = nn.Embedding(64, D_EVENT)
        slim_layer = nn.TransformerEncoderLayer(
            d_model=D_EVENT, nhead=4, dim_feedforward=D_EVENT * 4,
            batch_first=True, dropout=dropout,
        )
        self.slim_event_transformer = nn.TransformerEncoder(slim_layer, num_layers=3)
        self.event_head_proj = nn.Linear(D_EVENT, 16)
        self.event_head = nn.Linear(16 + 2, 1)

        # === V2 CHANGE 3: LYSO OC heads — wider intermediate ===
        # V1 compressed jk_dim (450) → 32 in one layer (14× compression).
        # V2 uses 128 → 64 intermediate, giving the heads more room to learn
        # fine-grained cluster structure.
        self.lyso_beta_head = nn.Sequential(
            nn.Linear(jk_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.lyso_cluster_coord_head = nn.Sequential(
            nn.Linear(jk_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.lyso_fraction_head = nn.Sequential(
            nn.Linear(jk_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

        # Bidirectional cross-attention removed — causes NaN instability
        # during early training when LYSO OC heads see extreme values from
        # randomly initialized reverse attention.  Kept as V1 (ATAR queries
        # LYSO only).

        # === V2 CHANGE 4b: Positron direction with attentional pooling ===
        self.dir_attn_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(jk_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1),
            ),
        )
        dir_input_dim = 2 * jk_dim + 3
        self.positron_dir_head = nn.Sequential(
            nn.Linear(dir_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        # === V2 CHANGE 5: Role head — added 512D intermediate ===
        # V1 compressed ROLE_IN=2310 → 256 in one layer (9× compression).
        # The extra 512D layer gives the head more room to process the
        # self/anchor/delta token combination.
        JK_DIM_ROLE = (self.L_TRIG + 1) * D_A
        ROLE_IN = 3 * JK_DIM_ROLE + 6
        self.atar_role_head = nn.Sequential(
            nn.Linear(ROLE_IN, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, D_A), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D_A, D_A // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D_A // 2, 3),
        )

    # --------------------------------------------------------------------- #
    #  forward() — full override for reverse cross-attention + nhead fix     #
    #  Search for "=== V2 CHANGE ===" to see the two modifications.          #
    # --------------------------------------------------------------------- #

    def forward(self, x, batch, task_weights=None, triggering_pion_slice=None,
                truth_positron_mask=None):
        if task_weights is None:
            task_weights = {}

        # 0. Extract modality flags and slice IDs
        is_atar_x = (x[:, 5] > 0.5)
        is_atar_y = (x[:, 6] > 0.5)
        is_atar = is_atar_x | is_atar_y
        is_lyso = (x[:, 7] > 0.5)

        # 1. Encode nodes
        hidden_dim = self.atar_feature_proj[0].out_features
        h_atar_in = torch.zeros(x.size(0), hidden_dim, device=x.device)
        h_lyso_in = torch.zeros(x.size(0), hidden_dim, device=x.device)

        if is_atar.any():
            transverse = torch.where(is_atar_y[is_atar].unsqueeze(1), x[is_atar, 1:2], x[is_atar, 0:1])
            atar_phys = torch.cat([transverse, x[is_atar, 2:4]], dim=1)
            h_proj = self.atar_feature_proj(atar_phys)
            view_idx = x[is_atar, 6].long()
            h_atar_in[is_atar] = h_proj + self.atar_view_embedding(view_idx)

        if is_lyso.any():
            # Time-blind: nodes = [x, y, z, E] only (col 4 = absolute time dropped, V2 CHANGE 0).
            # Relative timing reaches the model via |dt| on LYSO edges, not the node features.
            h_lyso_in[is_lyso] = self.lyso_encoder(x[is_lyso, :4])

        # 2. Physics-Based Edge Construction
        edge_index = physics_edge_index_batch(x, batch)

        # 3. Separate edge features and message passing per subsystem
        src, dst = edge_index
        edge_is_atar = is_atar[src] & is_atar[dst]
        edge_is_lyso = is_lyso[src] & is_lyso[dst]

        atar_edge_index = edge_index[:, edge_is_atar]
        lyso_edge_index = edge_index[:, edge_is_lyso]

        atar_edge_attr = build_atar_edge_attr(x, atar_edge_index)
        lyso_edge_attr = build_lyso_edge_attr(x, lyso_edge_index)

        # ATAR message passing
        h_atar = h_atar_in
        atar_xs = []
        if is_atar.any():
            for block in self.atar_blocks:
                h_atar = block(h_atar, atar_edge_index, atar_edge_attr)
                atar_xs.append(h_atar[is_atar])
            h_atar_jk = self.atar_jk(atar_xs)
        else:
            h_atar_jk = torch.zeros(0, self.atar_feature_proj[0].out_features * len(self.atar_blocks), device=x.device)

        # LYSO message passing
        h_lyso = h_lyso_in
        lyso_xs = []
        if is_lyso.any():
            for block in self.lyso_blocks:
                h_lyso = block(h_lyso, lyso_edge_index, lyso_edge_attr)
                lyso_xs.append(h_lyso[is_lyso])
            h_lyso_jk = self.lyso_jk(lyso_xs)
        else:
            h_lyso_jk = torch.zeros(0, self.lyso_encoder[0].out_features * len(self.lyso_blocks), device=x.device)

        # Recombine into [N_total, jk_dim]
        jk_dim = h_atar_jk.shape[1] if is_atar.any() else h_lyso_jk.shape[1]
        h_out = torch.zeros(x.size(0), jk_dim, device=x.device)
        if is_atar.any():
            h_out[is_atar] = h_atar_jk
        if is_lyso.any():
            h_out[is_lyso] = h_lyso_jk

        # Phase 2: The Dense Cross-Attention Bridge (ATAR queries LYSO)
        h_out_new = h_out.clone()
        if is_atar.any() and is_lyso.any():
            h_atar = h_out[is_atar]
            h_calo = h_out[is_lyso]
            batch_atar = batch[is_atar]
            batch_calo = batch[is_lyso]

            batch_size = int(batch.max().item() + 1)
            from torch_geometric.utils import to_dense_batch

            h_atar_dense, atar_mask = to_dense_batch(h_atar, batch_atar, batch_size=batch_size)
            h_calo_dense, calo_mask = to_dense_batch(h_calo, batch_calo, batch_size=batch_size)

            # Time-Proximity Masking
            if x.shape[1] > 9:
                atar_mean_t = x[is_atar, 9]
                lyso_mean_t = x[is_lyso, 9]
                atar_t_dense, _ = to_dense_batch(atar_mean_t, batch_atar, batch_size=batch_size)
                lyso_t_dense, _ = to_dense_batch(lyso_mean_t, batch_calo, batch_size=batch_size)

                time_diff = torch.abs(atar_t_dense.unsqueeze(2) - lyso_t_dense.unsqueeze(1))
                time_block = (time_diff > 5.0)
                combined_block = time_block | (~calo_mask.unsqueeze(1))
                all_keys_blocked = combined_block.all(dim=2)

                safe_time_block = time_block & (~all_keys_blocked.unsqueeze(2))
                num_heads = self.cross_attention.num_heads
                attn_mask = torch.zeros_like(safe_time_block, dtype=h_atar_dense.dtype)
                attn_mask[safe_time_block] = float('-inf')
                attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)
            else:
                attn_mask = None
                all_keys_blocked = None

            h_atar_enriched, _ = self.cross_attention(
                query=h_atar_dense,
                key=h_calo_dense.detach(),
                value=h_calo_dense.detach(),
                key_padding_mask=~calo_mask,
                attn_mask=attn_mask
            )

            if all_keys_blocked is not None:
                h_atar_enriched[all_keys_blocked] = 0.0

            h_atar_dense = h_atar_dense + h_atar_enriched
            h_atar = h_atar_dense[atar_mask]
            h_out_new[is_atar] = h_atar

        h_out = h_out_new

        # (V1 architecture: no reverse cross-attention)

        # --- 5. Routing to Specialized Heads ---
        output = {}

        # === Time Slice Grouping ===
        if x.shape[1] > 8:
            slice_ids_dense = x[:, 8].long()
        else:
            slice_ids_dense = torch.zeros_like(batch)

        num_slices_max = slice_ids_dense.max().item() + 1
        num_graphs_in_batch = batch.max().item() + 1

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

            global_slice_idx_all = global_slice_ids[is_atar]

            energy_per_hit = x[is_atar, 3].unsqueeze(1)
            slice_energy = global_add_pool(energy_per_hit, global_slice_idx_all, size=num_global_slices)

            count_x.index_add_(0, global_slice_idx_x, torch.ones_like(global_slice_idx_x, dtype=torch.float).unsqueeze(1))
            count_y.index_add_(0, global_slice_idx_y, torch.ones_like(global_slice_idx_y, dtype=torch.float).unsqueeze(1))

            has_x = (count_x > 0).squeeze()
            has_y = (count_y > 0).squeeze()

            def pool_with_soft_mask(pool_layer_x, pool_layer_y, mask_x, mask_y):
                hx = h_atar_x * mask_x if mask_x is not None else h_atar_x
                hy = h_atar_y * mask_y if mask_y is not None else h_atar_y
                px = pool_layer_x(hx, global_slice_idx_x, dim_size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=hx.device)
                py = pool_layer_y(hy, global_slice_idx_y, dim_size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=hy.device)
                return px, py

            has_x_f = has_x.float().unsqueeze(1)
            has_y_f = has_y.float().unsqueeze(1)

            valid_slice_mask = ((has_x_f + has_y_f) > 0).squeeze()
            output['valid_slice_mask'] = valid_slice_mask

            def safe_mean(px, py):
                return ((px * has_x_f) + (py * has_y_f)) / (has_x_f + has_y_f).clamp(min=1.0)

            # --- PHASE 1: Node-Level Predictions ---
            if task_weights.get('w_node_pdg', 1.0) > 0.0:
                node_pdg_hidden = self.atar_pdg_body(h_atar)
                node_slice_energy = slice_energy[global_slice_idx_all]
                node_pdg_input = torch.cat([node_pdg_hidden, node_slice_energy], dim=1)
                node_pdg_input = self.atar_pdg_norm(node_pdg_input)
                output['atar_node_pdg'] = self.atar_pdg_final(node_pdg_input)
            else:
                output['atar_node_pdg'] = torch.zeros(h_atar.size(0), self.num_pdg_classes, device=x.device)

            atar_node_probs = torch.sigmoid(output['atar_node_pdg'])

            # --- PHASE 2: View-Isolated Context Extraction ---
            pool_x_global = self.pool_x_global(h_atar_x, batch[is_atar_x], dim_size=num_graphs_in_batch) if is_atar_x.any() else torch.zeros(num_graphs_in_batch, jk_dim, device=x.device)
            pool_y_global = self.pool_y_global(h_atar_y, batch[is_atar_y], dim_size=num_graphs_in_batch) if is_atar_y.any() else torch.zeros(num_graphs_in_batch, jk_dim, device=x.device)

            slice_to_batch = torch.arange(num_global_slices, device=x.device) // num_slices_max
            valid_slice_batch_ids = slice_to_batch[valid_slice_mask]

            global_x_32d = self.global_x_context_head(pool_x_global)[valid_slice_batch_ids]
            global_y_32d = self.global_y_context_head(pool_y_global)[valid_slice_batch_ids]

            pool_x_multi_attn = self.pool_x_multi(h_atar_x, global_slice_idx_x, dim_size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_x.device)
            pool_y_multi_attn = self.pool_y_multi(h_atar_y, global_slice_idx_y, dim_size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_y.device)

            # --- PHASE 2.5: Multi-Event Prediction ---
            energy_x = x[is_atar_x, 3]
            energy_y = x[is_atar_y, 3]

            sum_x = global_add_pool(energy_x.unsqueeze(1), global_slice_idx_x, size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, 1, device=x.device)
            sum_y = global_add_pool(energy_y.unsqueeze(1), global_slice_idx_y, size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, 1, device=x.device)

            ones_x = torch.ones(is_atar_x.sum(), 1, device=x.device)
            ones_y = torch.ones(is_atar_y.sum(), 1, device=x.device)
            count_x = global_add_pool(ones_x, global_slice_idx_x, size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, 1, device=x.device)
            count_y = global_add_pool(ones_y, global_slice_idx_y, size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, 1, device=x.device)

            max_x_multi = global_max_pool(h_atar_x, global_slice_idx_x, size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=x.device)
            max_y_multi = global_max_pool(h_atar_y, global_slice_idx_y, size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=x.device)

            valid_slice_counts = torch.cat([count_x, count_y], dim=-1)[valid_slice_mask] / 100.0
            valid_slice_sums = torch.cat([sum_x, sum_y], dim=-1)[valid_slice_mask] / 1.0

            valid_slice_multi_input = torch.cat([
                pool_x_multi_attn[valid_slice_mask],
                pool_y_multi_attn[valid_slice_mask],
                max_x_multi[valid_slice_mask],
                max_y_multi[valid_slice_mask],
                valid_slice_counts,
                valid_slice_sums
            ], dim=-1)

            output['atar_slice_multi'] = self.atar_slice_multi_head(valid_slice_multi_input).squeeze(-1)

            stereo_multi = safe_mean(pool_x_multi_attn, pool_y_multi_attn)[valid_slice_mask]

            valid_pool_x_multi = pool_x_multi_attn[valid_slice_mask]
            valid_pool_y_multi = pool_y_multi_attn[valid_slice_mask]

            multi_x_8d = self.multi_x_context_head(valid_pool_x_multi)
            multi_y_8d = self.multi_y_context_head(valid_pool_y_multi)

            # --- PHASE 4: Final Weighted Pooling ---
            pool_all = self.pool_all(h_atar, global_slice_idx_all, dim_size=num_global_slices)
            pool_all_valid = pool_all[valid_slice_mask]

            pool_x_shared = self.pool_x_shared(h_atar_x, global_slice_idx_x, dim_size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_x.device)
            pool_y_shared = self.pool_y_shared(h_atar_y, global_slice_idx_y, dim_size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=h_atar_y.device)

            slice_pdg_hidden = self.atar_slice_pdg_body(pool_all_valid)
            valid_slice_energy = slice_energy[valid_slice_mask]
            slice_pdg_input = torch.cat([slice_pdg_hidden, valid_slice_energy], dim=1)
            slice_pdg_input = self.atar_slice_pdg_norm(slice_pdg_input)
            slice_logits = self.atar_slice_pdg_final(slice_pdg_input)
            output['atar_slice_pdg'] = slice_logits

            # --- PHASE 4: Sub-Heads ---
            valid_x_shared = pool_x_shared[valid_slice_mask]
            valid_y_shared = pool_y_shared[valid_slice_mask]

            valid_x_concat = torch.cat([valid_x_shared, global_x_32d, multi_x_8d], dim=-1)
            valid_y_concat = torch.cat([valid_y_shared, global_y_32d, multi_y_8d], dim=-1)

            z_context_stereo = torch.cat([global_x_32d, global_y_32d, multi_x_8d, multi_y_8d], dim=-1)
            stereo_concat = torch.cat([pool_all_valid, z_context_stereo], dim=-1)

            x_pred_expert = self.atar_endpoint_x(valid_x_concat)
            y_pred_expert = self.atar_endpoint_y(valid_y_concat)
            z_pred_expert = self.atar_endpoint_z(stereo_concat)

            output['atar_endpoints'] = torch.cat([x_pred_expert, y_pred_expert, z_pred_expert], dim=2)
            output['atar_endpoints_expert_x'] = x_pred_expert
            output['atar_endpoints_expert_y'] = y_pred_expert
            output['atar_endpoints_expert_z'] = z_pred_expert

            output['valid_slice_mask'] = valid_slice_mask
            output['valid_slice_indices'] = torch.nonzero(valid_slice_mask).squeeze(1)
            output['num_graphs_in_batch'] = num_graphs_in_batch
            output['num_slices_max'] = num_slices_max

            # --- ATAR Event Builder Early Fusion (stereo) ---
            endpoints_all = output['atar_endpoints']
            endpoints_xz_flat = endpoints_all[:, :, [0, 2], :].reshape(endpoints_all.size(0), -1)
            endpoints_yz_flat = endpoints_all[:, :, [1, 2], :].reshape(endpoints_all.size(0), -1)
            slice_pdg_det = output['atar_slice_pdg']

            atar_kin_xz = self.atar_kinematics_mlp_xz(torch.cat([endpoints_xz_flat, slice_pdg_det], dim=1))
            atar_kin_yz = self.atar_kinematics_mlp_yz(torch.cat([endpoints_yz_flat, slice_pdg_det], dim=1))

            hit_times = x[is_atar, 4]
            hit_energies = x[is_atar, 3].clamp(min=1e-6)
            slice_time_wsum = torch.zeros(num_global_slices, device=x.device)
            slice_energy_sum_t = torch.zeros(num_global_slices, device=x.device)
            slice_time_wsum.index_add_(0, global_slice_idx_all, hit_times * hit_energies)
            slice_energy_sum_t.index_add_(0, global_slice_idx_all, hit_energies)
            slice_mean_time = (slice_time_wsum / slice_energy_sum_t.clamp(min=1e-6))[valid_slice_mask]

            pool_x_ev = self.pool_x_event(h_atar_x, global_slice_idx_x, dim_size=num_global_slices) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=x.device)
            pool_y_ev = self.pool_y_event(h_atar_y, global_slice_idx_y, dim_size=num_global_slices) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=x.device)
            proj_x_ev = self.pool_x_event_proj(pool_x_ev[valid_slice_mask])
            proj_y_ev = self.pool_y_event_proj(pool_y_ev[valid_slice_mask])

            token_xz = self.atar_event_mlp_xz(torch.cat([proj_x_ev, atar_kin_xz], dim=1))
            token_yz = self.atar_event_mlp_yz(torch.cat([proj_y_ev, atar_kin_yz], dim=1))
            atar_event_tokens = torch.cat([token_xz, token_yz], dim=1)

            # Anchor flag
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
                if anchor_valid.any():
                    is_anchor_per_slice[anchor_global_idx[anchor_valid]] = True

            atar_event_tokens = atar_event_tokens + self.anchor_flag_embedding(
                is_anchor_per_slice.long()
            )

            output['atar_event_tokens'] = atar_event_tokens

            # === PHASE 9: ATAR-Only Event Building ===
            if task_weights.get('w_atar_trigger_slice', 0.0) > 0.0 or task_weights.get('w_pion_kinematics', 0.0) > 0.0 or task_weights.get('w_event_builder', 0.0) > 0.0 or task_weights.get('w_has_trigger_positron', 0.0) > 0.0:
                B_atar_idx = valid_slice_mask.nonzero().squeeze(1) // num_slices_max

                sort_idx_atar9 = torch.argsort(B_atar_idx)
                sorted_tokens9 = atar_event_tokens[sort_idx_atar9]
                sorted_batch9 = B_atar_idx[sort_idx_atar9]

                from torch_geometric.utils import to_dense_batch
                dense_atar9, pad_mask9 = to_dense_batch(sorted_tokens9, sorted_batch9)

                # Endpoint-compatibility kernel as attention bias
                endpoint_mean = endpoints_all[..., 1]
                endpoint_sigma = 0.5 * (endpoints_all[..., 2] - endpoints_all[..., 0]).abs().clamp(min=1e-3)

                mean_sorted = endpoint_mean[sort_idx_atar9].reshape(-1, 6)
                sigma_sorted = endpoint_sigma[sort_idx_atar9].reshape(-1, 6)
                dense_mean_flat, _ = to_dense_batch(mean_sorted, sorted_batch9)
                dense_sigma_flat, _ = to_dense_batch(sigma_sorted, sorted_batch9)
                B_size, N_max = dense_mean_flat.shape[0], dense_mean_flat.shape[1]
                dense_mean = dense_mean_flat.view(B_size, N_max, 2, 3)
                dense_sigma = dense_sigma_flat.view(B_size, N_max, 2, 3)

                m_i = dense_mean.unsqueeze(2).unsqueeze(4)
                m_j = dense_mean.unsqueeze(1).unsqueeze(3)
                s_i = dense_sigma.unsqueeze(2).unsqueeze(4)
                s_j = dense_sigma.unsqueeze(1).unsqueeze(3)
                delta_sq = (m_i - m_j) ** 2
                var_sum = s_i ** 2 + s_j ** 2 + 1e-6

                alpha = F.softplus(self.kernel_alpha).view(1, 1, 1, 1, 1, 3)
                neg_chi2 = -(delta_sq / var_sum) * alpha
                g_per_pair = neg_chi2.sum(dim=-1)
                kernel_bias = torch.logsumexp(g_per_pair.flatten(-2, -1), dim=-1)
                kernel_bias = kernel_bias.clamp(min=-50.0)

                num_heads_sa = self.atar_event_layers[0].self_attn.num_heads
                kernel_bias_heads = (
                    kernel_bias.unsqueeze(1)
                    .expand(-1, num_heads_sa, -1, -1)
                    .reshape(B_size * num_heads_sa, N_max, N_max)
                )

                jk_stack = [dense_atar9]
                h_jk = dense_atar9
                for layer in self.atar_event_layers:
                    h_next = layer(h_jk, src_mask=kernel_bias_heads, src_key_padding_mask=~pad_mask9)
                    if torch.isnan(h_next).any():
                        h_next = h_jk
                    h_jk = h_next
                    jk_stack.append(h_jk)

                atar_tokens_jk = torch.cat(jk_stack, dim=-1)

                inverse_sort9 = torch.argsort(sort_idx_atar9)
                jk_flat = atar_tokens_jk[pad_mask9][inverse_sort9]
                last_flat = jk_stack[-1][pad_mask9][inverse_sort9]

                # --- Role-based attachment head ---
                B_size_evt = num_graphs_in_batch
                N_valid = jk_flat.size(0)

                slice_event_idx = B_atar_idx

                slices_per_event = torch.zeros(B_size_evt, dtype=torch.long, device=x.device)
                slices_per_event.index_add_(
                    0, slice_event_idx,
                    torch.ones_like(slice_event_idx, dtype=torch.long),
                )
                event_offset = torch.zeros_like(slices_per_event)
                if B_size_evt > 1:
                    event_offset[1:] = torch.cumsum(slices_per_event, dim=0)[:-1]

                if triggering_pion_slice is not None:
                    anchor_local = triggering_pion_slice.to(device=x.device, dtype=torch.long)
                else:
                    pion_score = slice_pdg_det[:, 0]
                    is_pion_pred = (pion_score > 0.0)
                    t_mask = torch.where(
                        is_pion_pred, slice_mean_time, torch.full_like(slice_mean_time, 1e9)
                    )
                    anchor_local = torch.full((B_size_evt,), -1, dtype=torch.long, device=x.device)
                    sort_t_perm = torch.argsort(t_mask, stable=True)
                    ev_sorted = slice_event_idx[sort_t_perm]
                    seen = torch.zeros(B_size_evt, dtype=torch.bool, device=x.device)
                    for pos, ev in zip(sort_t_perm.tolist(), ev_sorted.tolist()):
                        if not seen[ev] and t_mask[pos] < 5e8:
                            anchor_local[ev] = pos - int(event_offset[ev].item())
                            seen[ev] = True

                anchor_valid = anchor_local >= 0
                anchor_local_safe = anchor_local.clamp(min=0)
                anchor_global_in_jk = event_offset + anchor_local_safe
                anchor_global_in_jk = anchor_global_in_jk.clamp(max=N_valid - 1)

                anchor_token_per_event = jk_flat[anchor_global_in_jk]
                anchor_token_per_event = anchor_token_per_event * anchor_valid.float().unsqueeze(-1)
                anchor_token_per_slice = anchor_token_per_event[slice_event_idx]

                endpoint_mean_flat = endpoint_mean.reshape(N_valid, 6)
                endpoint_anchor_per_event = endpoint_mean_flat[anchor_global_in_jk]
                endpoint_anchor_per_event = endpoint_anchor_per_event * anchor_valid.float().unsqueeze(-1)
                endpoint_delta = endpoint_mean_flat - endpoint_anchor_per_event[slice_event_idx]

                role_input = torch.cat([
                    jk_flat,
                    anchor_token_per_slice,
                    jk_flat - anchor_token_per_slice,
                    endpoint_delta,
                ], dim=-1)
                role_logits = self.atar_role_head(role_input)

                in_chain_logits = torch.logsumexp(role_logits[:, 1:], dim=-1) - role_logits[:, 0]

                is_anchor_slice = torch.zeros(N_valid, dtype=torch.bool, device=x.device)
                if anchor_valid.any():
                    is_anchor_slice[anchor_global_in_jk[anchor_valid]] = True

                in_chain_logits = torch.where(
                    is_anchor_slice,
                    torch.full_like(in_chain_logits, 20.0),
                    in_chain_logits,
                )

                output['atar_role_logits'] = role_logits
                output['atar_anchor_slice_mask'] = is_anchor_slice
                output['atar_slice_event_idx'] = slice_event_idx
                atar_trigger_logits = in_chain_logits
                output['atar_trigger_logits'] = atar_trigger_logits
                atar_trigger_probs = torch.sigmoid(atar_trigger_logits).detach()

                atar_event_tokens = last_flat
                output['atar_event_tokens'] = atar_event_tokens

            # === PHASE 10: Pion Stop Extraction ===
            if task_weights.get('w_pion_kinematics', 0.0) > 0.0 and 'atar_trigger_logits' in output:
                from torch_geometric.utils import softmax as pyg_softmax

                trigger_logit_full = torch.full((num_global_slices,), -50.0, device=x.device)
                trigger_logit_full[valid_slice_mask] = atar_trigger_logits.detach()
                hit_trigger_logit = trigger_logit_full[global_slice_idx_all]

                hit_pion_logit = output['atar_node_pdg'][:, 0].detach()

                endpoint_logit = self.pion_endpoint_scorer(h_atar).squeeze(-1)

                tau = torch.exp(self.pion_pool_log_tau).clamp(min=0.1, max=10.0)
                hit_score = (hit_trigger_logit + hit_pion_logit + endpoint_logit) / tau

                batch_atar = batch[is_atar]
                pos_atar = x[is_atar, 0:3]
                is_x_local = is_atar_x[is_atar]
                is_y_local = is_atar_y[is_atar]

                ones_atar = torch.ones(is_x_local.shape[0], 1, device=x.device)
                n_x_per_event = global_add_pool(
                    is_x_local.float().unsqueeze(-1) * ones_atar, batch_atar,
                    size=num_graphs_in_batch).squeeze(-1)
                n_y_per_event = global_add_pool(
                    is_y_local.float().unsqueeze(-1) * ones_atar, batch_atar,
                    size=num_graphs_in_batch).squeeze(-1)
                has_x_event = (n_x_per_event > 0).float().unsqueeze(-1)
                has_y_event = (n_y_per_event > 0).float().unsqueeze(-1)

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

                pooled_z = (pooled_z_from_x * has_x_event + pooled_z_from_y * has_y_event) \
                    / (has_x_event + has_y_event).clamp(min=1.0)
                pooled_x_coord = pooled_x_coord * has_x_event
                pooled_y_coord = pooled_y_coord * has_y_event
                pooled_pos = torch.cat([pooled_x_coord, pooled_y_coord, pooled_z], dim=-1)

                res_input_x = torch.cat([pooled_x_coord, pooled_feat_x], dim=-1)
                res_input_y = torch.cat([pooled_y_coord, pooled_feat_y], dim=-1)
                res_input_z = torch.cat([pooled_z, pooled_feat_x, pooled_feat_y], dim=-1)
                delta_x = torch.tanh(self.pion_stop_residual_x(res_input_x))
                delta_y = torch.tanh(self.pion_stop_residual_y(res_input_y))
                delta_z = torch.tanh(self.pion_stop_residual_z(res_input_z))
                delta_x = delta_x * has_x_event
                delta_y = delta_y * has_y_event
                delta = torch.cat([delta_x, delta_y, delta_z], dim=-1) * self.pion_stop_residual_scale
                pion_stop_pred = pooled_pos + delta
                output['atar_pion_stop'] = pion_stop_pred
            else:
                pion_stop_pred = torch.zeros(num_graphs_in_batch, 3, device=x.device)

            # === PHASE 11: Positron Direction ===
            NO_POSITRON_TIME = -500.0
            positron_time_per_graph = torch.full(
                (num_graphs_in_batch,), NO_POSITRON_TIME, device=x.device)
            if 'atar_trigger_logits' in output:
                trigger_prob_full_t = torch.zeros(num_global_slices, device=x.device)
                trigger_prob_full_t[valid_slice_mask] = atar_trigger_probs
                hit_trigger_prob_t = trigger_prob_full_t[global_slice_idx_all]
                mip_class_prob_t = torch.sigmoid(output['atar_node_pdg'][:, 2]).detach()
                pion_class_prob_t = torch.sigmoid(output['atar_node_pdg'][:, 0]).detach()
                muon_class_prob_t = torch.sigmoid(output['atar_node_pdg'][:, 1]).detach()
                output['atar_hit_trigger_prob'] = hit_trigger_prob_t.detach()
                output['atar_hit_mip_prob'] = mip_class_prob_t
                output['atar_hit_pion_prob'] = pion_class_prob_t
                output['atar_hit_muon_prob'] = muon_class_prob_t
                # A positron hit must be a triggering MIP with the non-MIP (pion/muon) classes
                # both suppressed (<0.05).  This rejects merged pixels where a muon's stopping
                # Bragg deposit (or a pion stop) shares the positron's crystal within the merge
                # window: such pixels carry BOTH the muon/pion and positron bits, and their
                # heavy deposit would otherwise leak ~muon-KE into the positron energy/time for
                # early-decay michels (a decay-time-correlated fake-pie bias).
                clean_pos_t = (pion_class_prob_t < 0.05) & (muon_class_prob_t < 0.05)
                positron_hit_mask = ((hit_trigger_prob_t > 0.5) & (mip_class_prob_t > 0.5)
                                     & clean_pos_t)

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
                )

                # --- Robust "consensus" positron time (pileup-hardened readout) ---
                # A merged pileup hit sits in a DIFFERENT time-slice than the real positron,
                # so it drags the plain mean (above).  Instead: sum the soft positron weight
                # (P_trig * P_mip) per time-slice, pick the DOMINANT slice, and use its
                # weighted-mean time -- outlier hits in other slices cannot move it.
                w_pos = hit_trigger_prob_t * mip_class_prob_t                 # [N_atar]
                sl_w  = torch.zeros(num_global_slices, device=x.device)
                sl_wt = torch.zeros(num_global_slices, device=x.device)
                sl_w.index_add_(0, global_slice_idx_all, w_pos)
                sl_wt.index_add_(0, global_slice_idx_all, w_pos * x[is_atar, 4])
                sl_mean = sl_wt / sl_w.clamp(min=1e-6)
                sl_w_bg = sl_w.view(num_graphs_in_batch, num_slices_max)
                sl_m_bg = sl_mean.view(num_graphs_in_batch, num_slices_max)
                best_sl = sl_w_bg.argmax(dim=1)
                consensus_t = sl_m_bg[torch.arange(num_graphs_in_batch, device=x.device), best_sl]
                output['positron_time_consensus'] = torch.where(
                    sl_w_bg.sum(dim=1) > 1e-3, consensus_t,
                    torch.full_like(consensus_t, NO_POSITRON_TIME))

                # --- DIAGNOSTIC: per-slice MEAN confidence readout + tagged-slice count ---
                # (1) MEAN-based dominant slice: highest AVERAGE (P_trig*P_mip) per hit, not the
                #     hit-count-weighted SUM.  A many-hit accidental cannot win on length alone.
                sl_cnt = torch.zeros(num_global_slices, device=x.device)
                sl_cnt.index_add_(0, global_slice_idx_all, torch.ones_like(w_pos))
                sl_meanconf = sl_w / sl_cnt.clamp(min=1.0)                 # avg P_trig*P_mip in slice
                sl_mc_bg = sl_meanconf.view(num_graphs_in_batch, num_slices_max)
                best_sl_mc = sl_mc_bg.argmax(dim=1)
                consensus_mc_t = sl_m_bg[torch.arange(num_graphs_in_batch, device=x.device), best_sl_mc]
                output['positron_time_consensus_meanconf'] = torch.where(
                    sl_mc_bg.max(dim=1).values > 1e-3, consensus_mc_t,
                    torch.full_like(consensus_mc_t, NO_POSITRON_TIME))
                # (2) how many DISTINCT slices contribute a tagged (trig>0.5 & mip>0.5) hit to the
                #     plain-mean average -> >=2 means multiple positrons are being blended.
                hit_tagged = ((hit_trigger_prob_t > 0.5) & (mip_class_prob_t > 0.5)).float()
                sl_tag = torch.zeros(num_global_slices, device=x.device)
                sl_tag.index_add_(0, global_slice_idx_all, hit_tagged)
                sl_tag_bg = (sl_tag.view(num_graphs_in_batch, num_slices_max) > 0.5)
                output['n_tagged_pos_slices'] = sl_tag_bg.sum(dim=1)

                # --- ARGMAX-ROLE single-slice readout (exclusivity-consistent) ---
                # Force exactly ONE positron slice: argmax of the role head's e-in-chain
                # probability -- the quantity the chain-exclusivity loss concentrates onto one
                # slice.  Unlike the trig*MIP consensus (which the longer accidental track can
                # win), this uses the head trained to isolate the chain positron; and when it
                # errs it returns the accidental's OWN (flat) time -> the sideband subtracts it,
                # instead of a prompt blend.  Time value = that slice's MIP-weighted mean time.
                if 'atar_role_logits' in output:
                    role_e = torch.softmax(output['atar_role_logits'].detach(), dim=-1)[:, 2]  # [N_valid]
                    sl_role = torch.zeros(num_global_slices, device=x.device)
                    sl_role[valid_slice_mask] = role_e
                    sl_mw  = torch.zeros(num_global_slices, device=x.device)
                    sl_mwt = torch.zeros(num_global_slices, device=x.device)
                    sl_mw.index_add_(0, global_slice_idx_all, mip_class_prob_t)
                    sl_mwt.index_add_(0, global_slice_idx_all, mip_class_prob_t * x[is_atar, 4])
                    sl_time_mip = sl_mwt / sl_mw.clamp(min=1e-6)
                    sl_role_bg = sl_role.view(num_graphs_in_batch, num_slices_max)
                    sl_tmip_bg = sl_time_mip.view(num_graphs_in_batch, num_slices_max)
                    best_role_sl = sl_role_bg.argmax(dim=1)
                    argmax_role_t = sl_tmip_bg[torch.arange(num_graphs_in_batch, device=x.device), best_role_sl]
                    output['positron_time_argmax_role'] = torch.where(
                        sl_role_bg.max(dim=1).values > 1e-3, argmax_role_t,
                        torch.full_like(argmax_role_t, NO_POSITRON_TIME))

            if task_weights.get('w_positron_angle', 0.0) > 0.0 and 'atar_trigger_logits' in output:
                if truth_positron_mask is not None:
                    positron_mask = truth_positron_mask[is_atar]
                else:
                    mip_class_prob = torch.sigmoid(output['atar_node_pdg'][:, 2]).detach()
                    trigger_prob_full = torch.zeros(num_global_slices, device=x.device)
                    trigger_prob_full[valid_slice_mask] = atar_trigger_probs
                    hit_trigger_prob = trigger_prob_full[global_slice_idx_all]
                    positron_mask = (hit_trigger_prob > 0.5) & (mip_class_prob > 0.5)

                batch_atar = batch[is_atar]
                is_x_local = is_atar_x[is_atar]
                is_y_local = is_atar_y[is_atar]
                mask_x = positron_mask & is_x_local
                mask_y = positron_mask & is_y_local

                def _attn_pool_view(mask_view):
                    if mask_view.any():
                        h_v = h_atar[mask_view]
                        b_v = batch_atar[mask_view]
                        pool = self.dir_attn_pool(h_v, b_v, dim_size=num_graphs_in_batch)
                    else:
                        pool = torch.zeros(num_graphs_in_batch, jk_dim, device=x.device)
                    return pool

                pool_x_pos = _attn_pool_view(mask_x)
                pool_y_pos = _attn_pool_view(mask_y)

                dir_input = torch.cat([pool_x_pos, pool_y_pos, pion_stop_pred.detach()], dim=-1)
                dir_logits = self.positron_dir_head(dir_input)
                positron_dir = F.normalize(dir_logits, p=2, dim=-1, eps=1e-6)
                output['atar_positron_dir'] = positron_dir
            else:
                positron_dir = torch.tensor([[0.0, 0.0, 1.0]], device=x.device).expand(num_graphs_in_batch, -1)

            # === Phase 11b: has-trigger-positron (pure decision rule) ===
            if ('atar_hit_trigger_prob' in output
                    and 'atar_hit_mip_prob' in output):
                hit_trig = output['atar_hit_trigger_prob']
                hit_mip = output['atar_hit_mip_prob']
                qualifies = ((hit_trig > 0.5) & (hit_mip > 0.5)).float()
                slice_count = torch.zeros(num_global_slices, device=x.device)
                slice_count.index_add_(0, global_slice_idx_all, qualifies)
                slice_count = slice_count.view(num_graphs_in_batch, num_slices_max)
                output['has_trigger_positron_rule'] = (slice_count >= 2.0).any(dim=1).float()
        else:
            pion_stop_pred = torch.zeros(num_graphs_in_batch, 3, device=x.device)
            positron_dir = torch.tensor([[0.0, 0.0, 1.0]], device=x.device).expand(num_graphs_in_batch, -1)
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

            lyso_slice_idx = global_slice_ids[is_lyso]
            pool_lyso_sum = torch.zeros(num_global_slices, h_lyso.size(1), device=h_lyso.device)
            pool_lyso_sum.index_add_(0, lyso_slice_idx, h_lyso)
            count_lyso.index_add_(0, lyso_slice_idx, torch.ones_like(lyso_slice_idx, dtype=torch.float).unsqueeze(1))
            lyso_stereo = pool_lyso_sum / count_lyso.clamp(min=1.0)
        else:
            lyso_stereo = torch.zeros(num_global_slices, h_out.size(1), device=x.device)

        # === UNIFIED HEADS (Event Builder) ===
        K_LYSO = 5

        # 1. LYSO Top-K Soft Clustering
        if (hasattr(self, 'lyso_event_proj')) and is_lyso.any() and task_weights.get('w_lyso_condensation', 1.0) > 0.0:
            pred_coords = output['lyso_cluster_coords']
            pred_beta = output['lyso_beta'].squeeze(-1)
            lyso_batch = batch[is_lyso]

            B = num_graphs_in_batch if is_atar.any() else lyso_batch.max().item() + 1

            from torch_geometric.utils import to_dense_batch

            g_coords, mask = to_dense_batch(pred_coords, lyso_batch, batch_size=B)
            g_beta, _ = to_dense_batch(pred_beta, lyso_batch, batch_size=B)
            g_feats, _ = to_dense_batch(h_lyso, lyso_batch, batch_size=B)

            x_lyso = x[is_lyso]
            g_slice_id, _ = to_dense_batch(x_lyso[:, 8], lyso_batch, batch_size=B)
            g_energy, _ = to_dense_batch(x_lyso[:, 3], lyso_batch, batch_size=B)
            g_phys_pos, _ = to_dense_batch(x_lyso[:, :3], lyso_batch, batch_size=B)

            g_beta[~mask] = 0.0

            energy_threshold_norm = 2.0 / NORM_E_LYSO
            is_valid_seed = (g_slice_id > 0) | (g_energy > energy_threshold_norm)
            g_beta_seeds = g_beta * is_valid_seed.float()

            g_beta_seeds[~mask] = -1e9

            actual_k = min(K_LYSO, g_coords.size(1))
            actual_k = max(actual_k, 1)

            topk_vals, topk_idx = torch.topk(g_beta_seeds, k=actual_k, dim=1)

            topk_idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, 3)
            seed_coords = torch.gather(g_coords, dim=1, index=topk_idx_expand)
            seed_beta = torch.gather(g_beta, dim=1, index=topk_idx)

            seed_beta_threshold = 0.25
            seed_beta = seed_beta * (seed_beta > seed_beta_threshold).float()

            dists = torch.cdist(g_coords, seed_coords)

            tau = 0.8
            affinity = torch.exp(-dists / tau) * seed_beta.unsqueeze(1)
            affinity[~mask] = 0.0

            seed_slice_id = torch.gather(g_slice_id, 1, topk_idx).long()
            slice_match = g_slice_id.long().unsqueeze(-1) == seed_slice_id.unsqueeze(1)
            affinity = affinity * slice_match.float()

            max_affinity = affinity.max(dim=2, keepdim=True).values.clamp(min=0.0, max=1.0)
            adaptive_floor = 0.05 * (1.0 - max_affinity)
            w_norm = affinity / (affinity.sum(dim=2, keepdim=True) + adaptive_floor)

            g_pool = torch.bmm(w_norm.transpose(1, 2), g_feats)
            g_pool = g_pool * seed_beta.unsqueeze(-1)

            seed_slice_id = torch.gather(g_slice_id, dim=1, index=topk_idx).long()
            g_time, _ = to_dense_batch(x_lyso[:, 4], lyso_batch, batch_size=B)
            g_time[~mask] = 0.0
            cluster_time_wsum = torch.bmm(w_norm.transpose(1, 2), g_time.unsqueeze(-1)).squeeze(-1)

            g_energy[~mask] = 0.0
            cluster_energy_sum = torch.bmm(w_norm.transpose(1, 2), g_energy.unsqueeze(-1)).squeeze(-1)

            lengths = mask.sum(dim=1)
            seed_structural = topk_idx < lengths.unsqueeze(-1)
            w_sum_per_k = w_norm.sum(dim=1)
            seed_has_weight = w_sum_per_k > 1e-4
            seed_has_energy = cluster_energy_sum > 1e-9
            seed_is_valid = seed_structural & seed_has_weight & seed_has_energy
            seed_invalid = ~seed_is_valid

            g_e_time = g_energy * g_time
            cluster_etime = torch.bmm(w_norm.transpose(1, 2), g_e_time.unsqueeze(-1)).squeeze(-1)
            e_sum_safe = cluster_energy_sum.masked_fill(seed_invalid, 1.0)
            cluster_energy_time = cluster_etime / e_sum_safe
            cluster_energy_time = cluster_energy_time.masked_fill(seed_invalid, 0.0)

            g_phys_pos[~mask] = 0.0
            w_sum_safe = w_sum_per_k.masked_fill(seed_invalid, 1.0)
            cluster_phys_pos = torch.bmm(
                w_norm.transpose(1, 2), g_phys_pos
            ) / w_sum_safe.unsqueeze(-1)
            cluster_phys_pos = cluster_phys_pos.masked_fill(seed_invalid.unsqueeze(-1), 0.0)

            # Weighted MEAN time. The bmm alone is a per-seed weighted SUM
            # (w_norm normalizes over seeds per hit, so its hit-sum is the
            # soft hit count, not 1) — divide by w_sum_safe exactly as
            # cluster_phys_pos does. Checkpoints trained before 2026-07-11
            # saw the un-divided, size-scaled time, which saturated the
            # event-builder temporal attention bias at the clamp constant;
            # their sigma_t_* params adapted to that dead bias, so evals of
            # old checkpoints under this code see a live bias they were not
            # trained with.
            cluster_mean_time = (cluster_time_wsum / w_sum_safe).masked_fill(seed_invalid, 0.0)

            ps_lyso = pion_stop_pred.detach().unsqueeze(1) * 0.1
            hit_delta = g_phys_pos - ps_lyso
            hit_r = hit_delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            hit_dir = hit_delta / hit_r
            hit_dir = hit_dir * mask.unsqueeze(-1).float()
            cluster_dir_raw = torch.bmm(
                w_norm.transpose(1, 2), hit_dir
            ) / w_sum_safe.unsqueeze(-1)
            cluster_dir_norm = cluster_dir_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            cluster_dir = cluster_dir_raw / cluster_dir_norm
            cluster_dir = cluster_dir.masked_fill(seed_invalid.unsqueeze(-1), 0.0)

            lyso_assignments = w_norm[mask]

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

            has_lyso = lengths > 0

            g_pool = g_pool[has_lyso]
            valid_mask = valid_mask[has_lyso]

            if g_pool.numel() > 0:
                lyso_pool_all = g_pool.view(-1, g_pool.size(-1))
                lyso_valid_tensor = valid_mask.view(-1)

                valid_b_indices = torch.nonzero(has_lyso).squeeze(-1)
                lyso_event_batch_tensor = valid_b_indices.unsqueeze(-1).expand(-1, K_LYSO).reshape(-1)

                output['lyso_soft_assignments'] = lyso_assignments.detach()
                output['lyso_seed_beta'] = seed_beta_padded[has_lyso].view(-1).detach()

                cluster_phys_pos_padded = cluster_phys_pos_padded.detach()
                cluster_dir_padded = cluster_dir_padded.detach()
                seed_beta_padded = seed_beta_padded.detach()

                # === PHASE 14: Angular LYSO Token Construction ===
                dir_to_cluster = cluster_dir_padded

                cos_theta = dir_to_cluster[..., 2:3]
                sin_theta_sq = (1.0 - cos_theta ** 2).clamp(min=1e-6)
                sin_theta = sin_theta_sq.sqrt()
                axial = sin_theta < 2e-3
                sin_phi = dir_to_cluster[..., 1:2] / sin_theta
                cos_phi = dir_to_cluster[..., 0:1] / sin_theta
                sin_phi = sin_phi.masked_fill(axial, 0.0)
                cos_phi = cos_phi.masked_fill(axial, 1.0)

                pos_dir = positron_dir.detach().unsqueeze(1)
                cos_sep_positron = (dir_to_cluster * F.normalize(pos_dir, dim=-1)).sum(-1, keepdim=True)

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

                cluster_e_flat = cluster_energy_sum_padded[has_lyso].view(-1, 1)

                # Positron-referenced cluster time
                if positron_time_per_graph.size(0) >= B:
                    pt_per_graph = positron_time_per_graph[:B]
                else:
                    pt_per_graph = torch.zeros(B, device=x.device)
                    pt_per_graph[:positron_time_per_graph.size(0)] = positron_time_per_graph
                pt_per_graph_valid = pt_per_graph[has_lyso]
                pt_flat = pt_per_graph_valid.unsqueeze(-1).expand(-1, K_LYSO).reshape(-1, 1)
                cluster_et_norm = cluster_energy_time_padded[has_lyso].view(-1, 1)
                dt_from_pos = cluster_et_norm - pt_flat

                # Time-of-flight correction
                tof_norm_scalar = TOF_NS / NORM_T_LYSO
                tof_flat = torch.full_like(dt_from_pos, tof_norm_scalar)
                dt_corr = dt_from_pos - tof_flat
                dt_corr_ns = dt_corr * NORM_T_LYSO

                # Gaussian coincidence window
                coinc_feat = torch.exp(
                    -(dt_corr_ns ** 2) / (2.0 * SIGMA_COINC_NS ** 2)
                )
                cluster_et_flat = coinc_feat

                angular_feat = torch.cat([
                    sin_theta, cos_theta, sin_phi, cos_phi,
                    cos_sep_positron,
                ], dim=-1)

                angular_feat_flat = angular_feat[has_lyso].view(-1, 5)

                w_norm_d = w_norm.detach()
                soft_n_hits = w_norm_d.sum(dim=1)

                if pad_dim > 0:
                    soft_n_hits = torch.cat([soft_n_hits, torch.zeros(B, pad_dim, device=x.device)], dim=1)

                soft_n_hits_flat = soft_n_hits[has_lyso].view(-1, 1)
                beta_flat = seed_beta_padded[has_lyso].view(-1, 1)

                sn_safe = soft_n_hits_flat.clamp(min=1e-4)
                cluster_e_mean = cluster_e_flat / sn_safe

                angular_feat_full = torch.cat([
                    angular_feat_flat,
                    cluster_e_mean,
                    cluster_et_flat,
                    beta_flat,
                    soft_n_hits_flat,
                ], dim=-1)
                lyso_cluster_times = cluster_mean_time_padded[has_lyso].view(-1) * NORM_T_LYSO
                lyso_cluster_etimes = cluster_energy_time_padded[has_lyso].view(-1)
                # Feed the cluster TOTAL energy (cluster_e_flat) to the 1/sqrt(E) time/angle
                # resolution laws, NOT the per-crystal mean (cluster_e_mean = total/soft_n_hits).
                # Detector resolution scales as 1/sqrt(E_total); the per-crystal mean inflated the
                # E-term by sqrt(soft_n_hits) (channel-asymmetric: ~2.44x pie vs 2.23x michel) and,
                # since soft_n_hits ~ sqrt(E_total), gave sigma ~ E^-0.25 instead of E^-0.5. The
                # per-crystal mean is kept as a network FEATURE in angular_feat_full above.
                lyso_cluster_energies = cluster_e_flat.view(-1).detach()
                output['lyso_cluster_energies'] = lyso_cluster_energies

                output['lyso_dt_from_pos'] = dt_from_pos.view(-1)
                output['lyso_dt_corr_ns'] = dt_corr_ns.view(-1)
                output['lyso_coinc_feat'] = coinc_feat.view(-1)
                output['positron_time_per_graph'] = positron_time_per_graph

                cos_sep_pos_flat = cos_sep_positron[has_lyso].view(-1, 1)
                output['lyso_coinc_skip_input'] = torch.cat(
                    [coinc_feat, cos_sep_pos_flat], dim=-1)

                output['lyso_cluster_features'] = angular_feat_full.detach()

                # Slim LYSO token: 9 raw features → D_EVENT
                lyso_event_tokens = self.lyso_event_proj(angular_feat_full)

                lyso_slice_ids = seed_slice_id_padded[has_lyso].view(-1).clamp(max=63)
                lyso_event_tokens = lyso_event_tokens + self.event_slice_emb(lyso_slice_ids)

                output['lyso_cluster_times'] = lyso_cluster_times
                output['lyso_seed_coords'] = cluster_phys_pos_padded
                output['lyso_seed_latent_coords'] = seed_latent_coords_padded
                output['lyso_seed_has_lyso'] = has_lyso
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
        all_times = []
        all_lyso_E = []

        if is_atar.any() and 'atar_event_tokens' in output:
            B_atar_idx = valid_slice_mask.nonzero().squeeze(1) // num_slices_max
            # DETACH ATAR tokens and project down to D_EVENT
            atar_detached = output['atar_event_tokens'].detach()
            atar_down = self.atar_event_down(atar_detached)

            atar_tokens_with_mod = atar_down + self.event_modality_emb.weight[0]
            all_tokens.append(atar_tokens_with_mod)

            atar_hit_times = x[is_atar, 9]
            slice_time_sum = torch.zeros(num_global_slices, device=x.device)
            slice_time_cnt = torch.zeros(num_global_slices, device=x.device)
            slice_time_sum.index_add_(0, global_slice_idx_all, atar_hit_times)
            slice_time_cnt.index_add_(0, global_slice_idx_all, torch.ones_like(atar_hit_times))
            slice_mean_times_all = slice_time_sum / slice_time_cnt.clamp(min=1.0)
            atar_token_times = slice_mean_times_all[valid_slice_mask]
            all_times.append(atar_token_times)
            all_lyso_E.append(torch.zeros_like(atar_token_times))
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

            original_idx = torch.arange(unified_tokens.size(0), device=unified_tokens.device)

            sort_idx = torch.argsort(unified_batch, stable=True)
            unified_tokens = unified_tokens[sort_idx]
            unified_valid = unified_valid[sort_idx]
            unified_batch = unified_batch[sort_idx]
            original_idx = original_idx[sort_idx]

            from torch_geometric.utils import to_dense_batch
            dense_tokens, pad_mask = to_dense_batch(unified_tokens, unified_batch, batch_size=num_graphs_in_batch)
            dense_idx, _ = to_dense_batch(original_idx, unified_batch, batch_size=num_graphs_in_batch)
            dense_valid, _ = to_dense_batch(unified_valid, unified_batch, batch_size=num_graphs_in_batch)

            padding_mask = ~dense_valid

            # === PHASE 15: Directional + Temporal Attention Bias ===
            src_mask = None
            n_atar_flat = output.get('unified_num_atar_tokens', 0)
            has_lyso_seeds = 'lyso_seed_coords' in output and is_lyso.any()

            modality_flag = torch.zeros(unified_tokens.size(0), dtype=torch.long, device=x.device)
            modality_flag[n_atar_flat:] = 1
            sorted_modality = modality_flag[sort_idx]
            dense_modality, _ = to_dense_batch(sorted_modality, unified_batch, batch_size=num_graphs_in_batch)

            trigger_per_token = torch.zeros(unified_tokens.size(0), device=x.device)
            if n_atar_flat > 0 and 'atar_trigger_logits' in output:
                trigger_per_token[:n_atar_flat] = torch.sigmoid(output['atar_trigger_logits']).detach()
            sorted_trigger = trigger_per_token[sort_idx]
            dense_trigger, _ = to_dense_batch(sorted_trigger, unified_batch, batch_size=num_graphs_in_batch)

            # Temporal bias
            if len(all_times) > 0:
                unified_times = torch.cat(all_times, dim=0)
                sorted_times = unified_times[sort_idx]
                dense_times, _ = to_dense_batch(sorted_times, unified_batch, batch_size=num_graphs_in_batch)

                unified_lyso_E = torch.cat(all_lyso_E, dim=0)
                sorted_lyso_E = unified_lyso_E[sort_idx]
                dense_lyso_E, _ = to_dense_batch(sorted_lyso_E, unified_batch, batch_size=num_graphs_in_batch)

                B_tf = dense_tokens.size(0)
                dt = dense_times.unsqueeze(2) - dense_times.unsqueeze(1)

                mod_i = dense_modality.unsqueeze(2)
                mod_j = dense_modality.unsqueeze(1)
                # dt[i,j] = t_i - t_j. The positron crosses the ATAR before the
                # calorimeter, so for i=ATAR (mod 0), j=LYSO (mod 1) the peak
                # belongs at t_ATAR - t_LYSO = -TOF, i.e. (mod_i - mod_j)*TOF.
                # Pre-2026-07-11 this was (mod_j - mod_i): sign-flipped, and
                # moot only because the size-scaled cluster time saturated the
                # bias. Matches the dt_corr = dt_from_pos - TOF convention of
                # the coincidence feature.
                expected_dt = (mod_i - mod_j).float() * TOF_NS

                is_lyso_tok = (dense_modality == 1).float()
                sigma_lyso_tok = (
                    self.sigma_t_lyso_floor_ns.abs()
                    + self.sigma_t_lyso_scale_ns.abs() / torch.sqrt(dense_lyso_E.clamp(min=1e-2))
                )
                sigma_per_tok = is_lyso_tok * sigma_lyso_tok + (1.0 - is_lyso_tok) * self.sigma_t_atar_ns.abs()
                sigma_sq_pair = (sigma_per_tok.unsqueeze(1) ** 2 + sigma_per_tok.unsqueeze(2) ** 2).clamp(min=0.1)

                temporal_bias = torch.exp(-(dt - expected_dt) ** 2 / (2.0 * sigma_sq_pair))
                temporal_bias = torch.log(temporal_bias.clamp(min=1e-6))
                temporal_bias = temporal_bias * dense_valid.unsqueeze(2).float() * dense_valid.unsqueeze(1).float()
                src_mask = temporal_bias.clone()

            if has_lyso_seeds:
                lyso_seed_coords = output['lyso_seed_coords']
                has_lyso_flag = output['lyso_seed_has_lyso']
                T = dense_tokens.size(1)
                B_tf = dense_tokens.size(0)
                K_LYSO_dim = lyso_seed_coords.size(1)

                if src_mask is None:
                    src_mask = torch.zeros(B_tf, T, T, device=x.device)

                is_atar_tok_mat = (dense_modality == 0) & dense_valid
                is_lyso_tok_mat = (dense_modality == 1) & dense_valid

                lyso_cumcount = is_lyso_tok_mat.long().cumsum(dim=1) - 1
                k_range = torch.arange(K_LYSO_dim, device=x.device)
                slot_match = (
                    (lyso_cumcount.unsqueeze(1) == k_range.view(1, -1, 1))
                    & is_lyso_tok_mat.unsqueeze(1)
                )

                K_actual_per_event = is_lyso_tok_mat.sum(dim=1)
                k_valid = (
                    (k_range.unsqueeze(0) < K_actual_per_event.unsqueeze(1))
                    & has_lyso_flag.unsqueeze(1)
                )
                slot_match = slot_match & k_valid.unsqueeze(-1)
                sm = slot_match.float()

                pion_stop_phys = pion_stop_pred.detach() * 0.1
                delta_cluster = (lyso_seed_coords - pion_stop_phys.unsqueeze(1)).detach()
                delta_norm = delta_cluster.norm(dim=-1, keepdim=True).clamp(min=1e-3)
                cluster_dir = delta_cluster / delta_norm

                if n_atar_flat > 0:
                    pd = positron_dir.detach()
                    pos_dir_norm = pd / pd.norm(dim=-1, keepdim=True).clamp(min=1e-3)
                    cos_align = (cluster_dir * pos_dir_norm.unsqueeze(1)).sum(dim=-1)

                    cluster_E_BK = (sm * dense_lyso_E.unsqueeze(1)).sum(dim=-1)
                    sigma_angle = (
                        self.angle_sigma_floor.abs()
                        + self.angle_sigma_scale.abs() / torch.sqrt(cluster_E_BK.clamp(min=1e-2))
                    )
                    bias_values = cos_align / sigma_angle.clamp(min=1e-3)

                    lyso_bias_per_tok = (sm * bias_values.unsqueeze(-1)).sum(dim=1)
                    atar_f = is_atar_tok_mat.float()
                    atar_lyso = atar_f.unsqueeze(-1) * lyso_bias_per_tok.unsqueeze(1)
                    src_mask = src_mask + atar_lyso + atar_lyso.transpose(1, 2)

            # ============================================================= #
            # === V2 CHANGE: nhead read from module (4 in V2, was 2)    === #
            # ============================================================= #
            if src_mask is not None:
                nhead = self.slim_event_transformer.layers[0].self_attn.num_heads
                src_mask_expanded = src_mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(-1, src_mask.size(1), src_mask.size(2))
            else:
                src_mask_expanded = None

            transformed_tokens = self.slim_event_transformer(
                dense_tokens,
                mask=src_mask_expanded,
                src_key_padding_mask=padding_mask
            )

            flat_transformed = transformed_tokens[pad_mask]
            flat_idx = dense_idx[pad_mask]

            inverse_sort = torch.argsort(flat_idx)
            flat_transformed = flat_transformed[inverse_sort]

            # Classifier: project transformer D_EVENT → 16D, concat [coinc_feat, cos_sep] → Linear(18, 1)
            n_atar_tok = output.get('unified_num_atar_tokens', 0)
            coinc_skip_in = output.get('lyso_coinc_skip_input')
            projected = self.event_head_proj(flat_transformed)
            phys_feat = torch.zeros(projected.size(0), 2, device=projected.device)
            if coinc_skip_in is not None and n_atar_tok < projected.size(0):
                phys_feat[n_atar_tok:] = coinc_skip_in

            event_logits = self.event_head(
                torch.cat([projected, phys_feat], dim=-1))

            output['unified_event_logits'] = event_logits
            output['unified_token_batch'] = unified_batch

        # === EVENT SUMMARY ===
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
            polar_angle = torch.acos(cos_theta)

        pos_time = output.get('positron_time_per_graph')
        if pos_time is None:
            pos_time = torch.full((B,), SENTINEL, device=device)

        pos_energy = torch.zeros(B, device=device)
        has_any = False

        trig_p = output.get('atar_hit_trigger_prob')
        mip_p = output.get('atar_hit_mip_prob')
        pion_p = output.get('atar_hit_pion_prob')
        muon_p = output.get('atar_hit_muon_prob')
        if trig_p is not None and mip_p is not None and is_atar.any():
            pos_mask = ((trig_p > 0.5) & (mip_p > 0.5)).float()
            # suppress merged muon-Bragg / pion-stop pixels (both non-MIP class probs < 0.05)
            # so their heavy deposit does not leak into the positron energy sum -- see the
            # positron_hit_mask above; keeps clean positron hits, drops muon/pion-contaminated ones.
            if pion_p is not None and muon_p is not None:
                pos_mask = pos_mask * ((pion_p < 0.05) & (muon_p < 0.05)).float()
            batch_atar = batch[is_atar]
            e_atar = x[is_atar, 3] * NORM_E_ATAR
            pos_energy.index_add_(0, batch_atar, e_atar * pos_mask)
            has_any = True

        w_lyso = output.get('lyso_soft_assignments')
        beta_lyso = output.get('lyso_seed_beta')
        ev_logits = output.get('unified_event_logits')
        n_atar_tok = output.get('unified_num_atar_tokens', 0)
        if (is_lyso.any() and w_lyso is not None and beta_lyso is not None
                and ev_logits is not None):
            K = w_lyso.size(1)
            lyso_p = torch.sigmoid(ev_logits[n_atar_tok:].view(-1))
            beta_bk = beta_lyso.view(-1, K)
            p_bk = lyso_p.view(-1, K)
            hit_graph = batch[is_lyso]
            counts = torch.zeros(B, device=device, dtype=torch.long)
            counts.index_add_(0, hit_graph, torch.ones_like(hit_graph))
            has_lyso_g = counts > 0
            graph_to_valid = torch.full((B,), -1, device=device, dtype=torch.long)
            graph_to_valid[has_lyso_g] = torch.arange(
                int(has_lyso_g.sum().item()), device=device)
            hit_vg = graph_to_valid[hit_graph]
            wb = w_lyso * beta_bk[hit_vg]
            wb_sum = wb.sum(dim=1).clamp(min=1e-6)
            p_hit = (wb * p_bk[hit_vg]).sum(dim=1) / wb_sum
            output['lyso_hit_trigger_prob'] = p_hit.detach()
            lyso_mask = (p_hit > 0.5).float()
            e_lyso = x[is_lyso, 3] * NORM_E_LYSO
            pos_energy.index_add_(0, hit_graph, e_lyso * lyso_mask)
            has_any = True

        if not has_any:
            pos_energy = torch.full((B,), SENTINEL, device=device)

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

        # Dead-material energy head
        def _angle_features(direction):
            cos_t = direction[:, 2]
            sin_t = torch.sqrt((1.0 - cos_t.pow(2)).clamp(min=1e-6))
            cos_p = direction[:, 0] / sin_t
            cos_p = torch.where(sin_t < 1e-3, torch.zeros_like(cos_p), cos_p)
            return cos_t, cos_p

        cos_t_pos, cos_p_pos = _angle_features(positron_dir.detach())

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
            pred_start_n = ep_d[:, 0, :, 1]
            pred_stop_n = ep_d[:, 1, :, 1]
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

        pos_E_safe = torch.where(pos_energy > 0,
                                 pos_energy,
                                 torch.zeros_like(pos_energy))
        total_E_norm = (pos_E_safe / NORM_E_LYSO).detach()

        mean_x_lyso = torch.zeros(B, device=device)
        mean_y_lyso = torch.zeros(B, device=device)
        mean_z_lyso = torch.zeros(B, device=device)
        if 'lyso_hit_trigger_prob' in output and is_lyso.any():
            p_hit_d = output['lyso_hit_trigger_prob'].detach()
            mask_d = (p_hit_d > 0.5).float()
            e_lyso_mev = (x[is_lyso, 3] * NORM_E_LYSO).detach()
            gated = e_lyso_mev * mask_d
            batch_lyso_d = batch[is_lyso].detach()
            sum_E = torch.zeros(B, device=device)
            sum_xE = torch.zeros(B, device=device)
            sum_yE = torch.zeros(B, device=device)
            sum_zE = torch.zeros(B, device=device)
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
        log_dead_pred = self.dead_energy_head(dead_input).squeeze(-1)
        dead_energy_pred = torch.expm1(log_dead_pred)
        output['dead_energy_log_pred'] = log_dead_pred
        output['dead_energy_pred'] = dead_energy_pred.detach()

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