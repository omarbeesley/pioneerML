"""
PURITY Tail-Reveal Architecture
================================

ATAR-only variant of PURITY specialized for the sec-1.2 tail-reveal analysis:
isolating clean π → e ν events from background (Michel chains, pileup) without
biasing the surviving energy spectrum across §1.2's low/high energy bins.

Three heads sit downstream of the shared ATAR trunk:
    PieTaggerHead   : positive identifier of clean π → e ν topology (is_pie)
    MuonVetoHead    : aggressive any-muon-evidence detector
    PileupVetoHead  : aggressive any-ATAR-pileup detector

Each emits one logit per graph. Downstream analysis cuts in the 3D plane:
    keep events with high pie_logit AND low muon_logit AND low pileup_logit.

Why is_pie (not is_tail = pie & low_E):
    Training the head against an energy-gated label would make its output
    correlated with positron energy — cuts on it would warp the surviving
    energy spectrum across §1.2 bins, biasing R_{e/μ}. is_pie is energy-blind
    by construction; the downstream PURITY model handles energy binning.

Why ATAR-only:
    The §1.2 worry is bias between low-E pie and high-E pie. ATAR sees both
    as MIPs (unbiased); calorimeter sees them very differently and would leak
    energy-dependent signal into any cut. The trunk drops LYSO entirely; the
    pileup veto's truth label (atar_origin > 0) ignores LYSO pileup so the
    label and feature space stay aligned.

Training: heads are detached from the trunk (Stage 1). All three event-level
classifiers train on the full event population — no clean-event masking, so
each output remains a valid axis of the 3D cut plane. Per-hit aux losses
supervise the dedicated muon and pileup classifiers for recall.

Upstream (DTAR) inputs are not wired in yet; the pion anchor is currently a
stand-in for the single-chain constraint.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import (
    JumpingKnowledge,
    AttentionalAggregation,
    global_add_pool,
    global_max_pool,
    radius_graph,
)

from unified_reco.models import (
    JointAttentionBlock,
    VectorHead,
    QuantileOutputHead,
    build_atar_edge_attr,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
D_TAIL = 128          # transformer width for the tail branch
MUON_CLASS = 1        # column index of atar_node_pdg / atar_slice_pdg
PION_CLASS = 0
MIP_CLASS = 2


# --------------------------------------------------------------------------
# ATAR-only edge construction (no LYSO, no cross-subsystem logic)
# --------------------------------------------------------------------------

def atar_only_edge_index(x, batch):
    """Fully-connected intra-slice ATAR edges only.

    radius_graph runs on ATAR hits ONLY — LYSO rows of x are not seen by
    the call. The previous version ran radius_graph on the whole x (ATAR
    + LYSO) and post-filtered to ATAR-ATAR pairs, but the resulting edge
    set turned out to depend on LYSO data through some interaction in
    torch_cluster's radius operator, leaking calorimeter information
    into the ATAR trunk. Constructing the graph from ATAR hits alone
    closes that path by construction.
    """
    is_atar = (x[:, 5] > 0.5) | (x[:, 6] > 0.5)
    if not is_atar.any():
        return torch.zeros((2, 0), dtype=torch.long, device=x.device)

    atar_global_idx = is_atar.nonzero(as_tuple=False).squeeze(1)
    slice_id_atar = x[atar_global_idx, 8]
    batch_atar = batch[atar_global_idx]
    cluster_atar = (batch_atar * 10000 + slice_id_atar * 10).long()

    # radius_graph silently requires non-decreasing `batch`; sort to satisfy.
    order = torch.argsort(cluster_atar)
    cluster_sorted = cluster_atar[order]
    idx_sorted = atar_global_idx[order]

    dummy_pos = torch.zeros((atar_global_idx.size(0), 1), device=x.device)
    with torch.no_grad():
        edge_local = radius_graph(
            dummy_pos, r=1.0, batch=cluster_sorted,
            loop=False, max_num_neighbors=3000,
        )

    # Map sorted-local indices back to global indices in x.
    return idx_sorted[edge_local]


# ==========================================================================
# PURITYTailBackbone: ATAR-only reconstruction trunk
# --------------------------------------------------------------------------
# Keeps the pieces of PURITY that the tail reveal needs:
#   - ATAR encoder, block stack, JK
#   - slice / node PDG heads, multi head, endpoint heads
#   - ATAR event token construction + self-attention
#   - Pion stop, positron direction, positron time
# Drops:
#   - LYSO encoder / blocks / heads
#   - Cross-attention bridge
#   - Slim event builder (LYSO coincidence logic)
# ==========================================================================

class PURITYTailBackbone(nn.Module):
    def __init__(self, hidden_dim=150, num_blocks=3, heads=5,
                 dropout=0.05, num_pdg_classes=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_pdg_classes = num_pdg_classes

        # ATAR view-aware encoder: [transverse, z, E] + view embedding
        self.atar_feature_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.atar_view_embedding = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.atar_view_embedding.weight, std=0.02)

        # ATAR transformer blocks + JK
        self.atar_blocks = nn.ModuleList([
            JointAttentionBlock(hidden_dim=hidden_dim, heads=heads,
                                edge_dim=4, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.atar_jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden_dim * num_blocks
        self.jk_dim = jk_dim

        # --- Slice PDG: body + late energy injection ---
        _ph = hidden_dim
        self.atar_slice_pdg_body = nn.Sequential(
            nn.Linear(jk_dim, _ph), nn.GELU(),
            nn.Linear(_ph, _ph // 2), nn.GELU(),
        )
        self.atar_slice_pdg_norm = nn.LayerNorm(_ph // 2 + 1)
        self.atar_slice_pdg_final = nn.Linear(_ph // 2 + 1, num_pdg_classes)

        # --- Multi-event flag head ---
        self.atar_slice_multi_head = nn.Sequential(
            nn.Linear(jk_dim * 4 + 4, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.multi_x_context_head = nn.Sequential(
            nn.Linear(jk_dim, 16), nn.GELU(), nn.Linear(16, 8))
        self.multi_y_context_head = nn.Sequential(
            nn.Linear(jk_dim, 16), nn.GELU(), nn.Linear(16, 8))
        self.global_x_context_head = nn.Sequential(
            nn.Linear(jk_dim, 48), nn.GELU(), nn.Linear(48, 32))
        self.global_y_context_head = nn.Sequential(
            nn.Linear(jk_dim, 48), nn.GELU(), nn.Linear(48, 32))

        # --- Node PDG ---
        self.atar_pdg_body = nn.Sequential(
            nn.Linear(jk_dim, _ph), nn.GELU(),
            nn.Linear(_ph, _ph // 2), nn.GELU(),
        )
        self.atar_pdg_norm = nn.LayerNorm(_ph // 2 + 1)
        self.atar_pdg_final = nn.Linear(_ph // 2 + 1, 3)

        # --- ATAR event tokens (D_A=256) ---
        D_A = 256
        self.D_A = D_A
        self.atar_kinematics_mlp = nn.Sequential(
            nn.Linear(21, 64), nn.GELU(), nn.Linear(64, 64),
        )
        self.pool_x_event_proj = nn.Sequential(nn.Linear(jk_dim, 128), nn.GELU())
        self.pool_y_event_proj = nn.Sequential(nn.Linear(jk_dim, 128), nn.GELU())
        self.atar_time_proj = nn.Sequential(nn.Linear(1, 4), nn.GELU())
        self.atar_event_mlp = nn.Sequential(
            nn.Linear(128 * 2 + 64 + 4, D_A), nn.GELU(),
            nn.Linear(D_A, D_A),
        )
        self.slice_position_embedding = nn.Embedding(64, D_A)

        # --- Pools ---
        def make_pool():
            return AttentionalAggregation(nn.Sequential(
                nn.Linear(jk_dim, hidden_dim * 2), nn.GELU(),
                nn.Linear(hidden_dim * 2, 1),
            ))
        self.pool_all = make_pool()
        self.pool_x_shared = make_pool()
        self.pool_y_shared = make_pool()
        self.pool_x_multi = make_pool()
        self.pool_y_multi = make_pool()
        self.pool_x_global = make_pool()
        self.pool_y_global = make_pool()
        self.pool_x_event = make_pool()
        self.pool_y_event = make_pool()
        self.pool_all_pion = make_pool()
        self.pool_all_mip = make_pool()

        # --- Phase 9: ATAR event self-attention + trigger ---
        self.atar_event_self_attn = nn.MultiheadAttention(
            D_A, num_heads=4, batch_first=True, dropout=dropout)
        self.atar_event_self_attn_norm = nn.LayerNorm(D_A)
        self.atar_trigger_classifier = nn.Sequential(
            nn.Linear(D_A, D_A // 2), nn.GELU(), nn.Linear(D_A // 2, 1),
        )

        # --- Phase 10 & 11: pion stop + positron direction ---
        self.pion_stop_head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.positron_dir_head = VectorHead(jk_dim + 6, hidden_dim)

        # --- Endpoint heads ---
        self.atar_endpoint_x = QuantileOutputHead(
            input_dim=jk_dim + 40, num_points=2, coords=1)
        self.atar_endpoint_y = QuantileOutputHead(
            input_dim=jk_dim + 40, num_points=2, coords=1)
        self.atar_endpoint_z = QuantileOutputHead(
            input_dim=jk_dim + 80, num_points=2, coords=1)

    def forward(self, x, batch):
        """
        x: [N_total_hits, 9+] with columns
            [pos_x, pos_y, pos_z, E, t, is_xz, is_yz, is_lyso, slice_id, ...]
        batch: [N_total_hits]

        Returns a dict with the usual PURITY ATAR outputs plus the extras the
        tail-reveal heads need:
            h_atar, pion_event_pool, mip_event_pool, exit_dir_per_graph,
            positron_time, slice_energy, slice_mean_time,
            atar_muon_pool (muon-gated pooled feature).
        """
        is_atar_x = (x[:, 5] > 0.5)
        is_atar_y = (x[:, 6] > 0.5)
        is_atar = is_atar_x | is_atar_y

        device = x.device
        hidden_dim = self.hidden_dim
        jk_dim = self.jk_dim

        output = {}
        output['is_atar'] = is_atar

        # --- Encode ATAR hits ---
        if not is_atar.any():
            return output  # empty batch of ATAR data — caller must handle

        h_atar_in = torch.zeros(x.size(0), hidden_dim, device=device)
        transverse = torch.where(
            is_atar_y[is_atar].unsqueeze(1), x[is_atar, 1:2], x[is_atar, 0:1])
        atar_phys = torch.cat([transverse, x[is_atar, 2:4]], dim=1)
        h_proj = self.atar_feature_proj(atar_phys)
        view_idx = x[is_atar, 6].long()
        h_atar_in[is_atar] = h_proj + self.atar_view_embedding(view_idx)

        # --- ATAR-only edges ---
        edge_index = atar_only_edge_index(x, batch)
        src, dst = edge_index
        atar_edge_mask = is_atar[src] & is_atar[dst]
        atar_edge_index = edge_index[:, atar_edge_mask]
        atar_edge_attr = build_atar_edge_attr(x, atar_edge_index)

        # --- Message passing + JK ---
        h_atar = h_atar_in
        atar_xs = []
        for block in self.atar_blocks:
            h_atar = block(h_atar, atar_edge_index, atar_edge_attr)
            atar_xs.append(h_atar[is_atar])
        h_atar_jk = self.atar_jk(atar_xs)  # [N_atar, jk_dim]

        # Naturally ATAR-only: no cross-attention bridge, no LYSO info.
        h_atar = h_atar_jk
        output['h_atar'] = h_atar

        # --- Slice bookkeeping ---
        slice_ids_dense = x[:, 8].long()
        num_slices_max = int(slice_ids_dense.max().item() + 1)
        num_graphs_in_batch = int(batch.max().item() + 1)
        global_slice_ids = batch * num_slices_max + slice_ids_dense
        num_global_slices = num_graphs_in_batch * num_slices_max

        global_slice_idx_x = global_slice_ids[is_atar_x]
        global_slice_idx_y = global_slice_ids[is_atar_y]
        global_slice_idx_all = global_slice_ids[is_atar]

        # Scatter pools split by view
        h_atar_x = h_atar[is_atar_x[is_atar]]
        h_atar_y = h_atar[is_atar_y[is_atar]]

        count_x = global_add_pool(
            torch.ones(is_atar_x.sum(), 1, device=device),
            global_slice_idx_x, size=num_global_slices,
        ) if is_atar_x.any() else torch.zeros(num_global_slices, 1, device=device)
        count_y = global_add_pool(
            torch.ones(is_atar_y.sum(), 1, device=device),
            global_slice_idx_y, size=num_global_slices,
        ) if is_atar_y.any() else torch.zeros(num_global_slices, 1, device=device)
        has_x = (count_x > 0).squeeze(-1)
        has_y = (count_y > 0).squeeze(-1)
        valid_slice_mask = ((count_x + count_y) > 0).squeeze(-1)
        output['valid_slice_mask'] = valid_slice_mask
        output['num_global_slices'] = num_global_slices
        output['num_graphs_in_batch'] = num_graphs_in_batch
        output['num_slices_max'] = num_slices_max
        output['global_slice_idx_all'] = global_slice_idx_all

        # Slice total energy
        energy_per_hit = x[is_atar, 3].unsqueeze(1)
        slice_energy = global_add_pool(
            energy_per_hit, global_slice_idx_all, size=num_global_slices)
        output['slice_energy'] = slice_energy

        # --- Node PDG ---
        node_pdg_hidden = self.atar_pdg_body(h_atar)
        node_slice_energy = slice_energy[global_slice_idx_all]
        node_pdg_input = self.atar_pdg_norm(
            torch.cat([node_pdg_hidden, node_slice_energy], dim=1))
        output['atar_node_pdg'] = self.atar_pdg_final(node_pdg_input)

        # --- Global & multi pools ---
        batch_atar = batch[is_atar]
        if is_atar_x.any():
            pool_x_global = self.pool_x_global(
                h_atar_x, batch[is_atar_x], dim_size=num_graphs_in_batch)
        else:
            pool_x_global = torch.zeros(num_graphs_in_batch, jk_dim, device=device)
        if is_atar_y.any():
            pool_y_global = self.pool_y_global(
                h_atar_y, batch[is_atar_y], dim_size=num_graphs_in_batch)
        else:
            pool_y_global = torch.zeros(num_graphs_in_batch, jk_dim, device=device)

        slice_to_batch = torch.arange(num_global_slices, device=device) // num_slices_max
        valid_slice_batch_ids = slice_to_batch[valid_slice_mask]
        global_x_32d = self.global_x_context_head(pool_x_global)[valid_slice_batch_ids]
        global_y_32d = self.global_y_context_head(pool_y_global)[valid_slice_batch_ids]

        if has_x.any():
            pool_x_multi = self.pool_x_multi(
                h_atar_x, global_slice_idx_x, dim_size=num_global_slices)
        else:
            pool_x_multi = torch.zeros(num_global_slices, jk_dim, device=device)
        if has_y.any():
            pool_y_multi = self.pool_y_multi(
                h_atar_y, global_slice_idx_y, dim_size=num_global_slices)
        else:
            pool_y_multi = torch.zeros(num_global_slices, jk_dim, device=device)

        # Multi head inputs
        energy_x = x[is_atar_x, 3]
        energy_y = x[is_atar_y, 3]
        sum_x = global_add_pool(
            energy_x.unsqueeze(1), global_slice_idx_x, size=num_global_slices
        ) if is_atar_x.any() else torch.zeros(num_global_slices, 1, device=device)
        sum_y = global_add_pool(
            energy_y.unsqueeze(1), global_slice_idx_y, size=num_global_slices
        ) if is_atar_y.any() else torch.zeros(num_global_slices, 1, device=device)
        max_x_multi = global_max_pool(
            h_atar_x, global_slice_idx_x, size=num_global_slices
        ) if is_atar_x.any() else torch.zeros(num_global_slices, jk_dim, device=device)
        max_y_multi = global_max_pool(
            h_atar_y, global_slice_idx_y, size=num_global_slices
        ) if is_atar_y.any() else torch.zeros(num_global_slices, jk_dim, device=device)

        valid_slice_counts = torch.cat([count_x, count_y], dim=-1)[valid_slice_mask] / 100.0
        valid_slice_sums = torch.cat([sum_x, sum_y], dim=-1)[valid_slice_mask]
        multi_input = torch.cat([
            pool_x_multi[valid_slice_mask], pool_y_multi[valid_slice_mask],
            max_x_multi[valid_slice_mask], max_y_multi[valid_slice_mask],
            valid_slice_counts, valid_slice_sums,
        ], dim=-1)
        output['atar_slice_multi'] = self.atar_slice_multi_head(multi_input).squeeze(-1)

        multi_x_8d = self.multi_x_context_head(pool_x_multi[valid_slice_mask])
        multi_y_8d = self.multi_y_context_head(pool_y_multi[valid_slice_mask])

        # --- Joint pool + per-view pools ---
        pool_all = self.pool_all(h_atar, global_slice_idx_all, dim_size=num_global_slices)
        pool_all_valid = pool_all[valid_slice_mask]
        pool_x_shared = self.pool_x_shared(
            h_atar_x, global_slice_idx_x, dim_size=num_global_slices
        ) if has_x.any() else torch.zeros(num_global_slices, jk_dim, device=device)
        pool_y_shared = self.pool_y_shared(
            h_atar_y, global_slice_idx_y, dim_size=num_global_slices
        ) if has_y.any() else torch.zeros(num_global_slices, jk_dim, device=device)

        # --- Slice PDG ---
        slice_pdg_hidden = self.atar_slice_pdg_body(pool_all_valid)
        valid_slice_energy = slice_energy[valid_slice_mask]
        slice_pdg_input = self.atar_slice_pdg_norm(
            torch.cat([slice_pdg_hidden, valid_slice_energy], dim=1))
        output['atar_slice_pdg'] = self.atar_slice_pdg_final(slice_pdg_input)

        # --- Endpoint heads ---
        valid_x_shared = pool_x_shared[valid_slice_mask]
        valid_y_shared = pool_y_shared[valid_slice_mask]
        valid_x_concat = torch.cat([valid_x_shared, global_x_32d, multi_x_8d], dim=-1)
        valid_y_concat = torch.cat([valid_y_shared, global_y_32d, multi_y_8d], dim=-1)
        z_context_stereo = torch.cat([global_x_32d, global_y_32d, multi_x_8d, multi_y_8d], dim=-1)
        stereo_concat = torch.cat([pool_all_valid, z_context_stereo], dim=-1)
        x_pred = self.atar_endpoint_x(valid_x_concat)
        y_pred = self.atar_endpoint_y(valid_y_concat)
        z_pred = self.atar_endpoint_z(stereo_concat)
        output['atar_endpoints'] = torch.cat([x_pred, y_pred, z_pred], dim=2)

        # --- Slice mean time (energy-weighted) ---
        hit_times = x[is_atar, 4]
        hit_energies = x[is_atar, 3].clamp(min=1e-6)
        slice_time_wsum = torch.zeros(num_global_slices, device=device)
        slice_energy_sum_t = torch.zeros(num_global_slices, device=device)
        slice_time_wsum.index_add_(0, global_slice_idx_all, hit_times * hit_energies)
        slice_energy_sum_t.index_add_(0, global_slice_idx_all, hit_energies)
        slice_mean_time = (slice_time_wsum / slice_energy_sum_t.clamp(min=1e-6))[valid_slice_mask]
        output['slice_mean_time'] = slice_mean_time

        # --- ATAR event tokens ---
        endpoints_flat = output['atar_endpoints'].detach().reshape(
            output['atar_endpoints'].size(0), -1)
        kin_input = torch.cat([endpoints_flat, output['atar_slice_pdg'].detach()], dim=1)
        atar_kin_feat = self.atar_kinematics_mlp(kin_input)

        if has_x.any():
            pool_x_ev = self.pool_x_event(
                h_atar_x, global_slice_idx_x, dim_size=num_global_slices)
        else:
            pool_x_ev = torch.zeros(num_global_slices, jk_dim, device=device)
        if has_y.any():
            pool_y_ev = self.pool_y_event(
                h_atar_y, global_slice_idx_y, dim_size=num_global_slices)
        else:
            pool_y_ev = torch.zeros(num_global_slices, jk_dim, device=device)
        proj_x_ev = self.pool_x_event_proj(pool_x_ev[valid_slice_mask])
        proj_y_ev = self.pool_y_event_proj(pool_y_ev[valid_slice_mask])
        time_feat = self.atar_time_proj(slice_mean_time.unsqueeze(-1))
        event_input = torch.cat([proj_x_ev, proj_y_ev, atar_kin_feat, time_feat], dim=1)
        atar_event_tokens = self.atar_event_mlp(event_input)
        valid_slice_indices = torch.nonzero(valid_slice_mask).squeeze(1)
        local_slice_ids = (valid_slice_indices % num_slices_max).clamp(max=63)
        atar_event_tokens = atar_event_tokens + self.slice_position_embedding(local_slice_ids)

        # --- Phase 9: ATAR event self-attention + trigger ---
        B_atar_idx = valid_slice_indices // num_slices_max
        sort_idx = torch.argsort(B_atar_idx)
        sorted_tokens = atar_event_tokens[sort_idx]
        sorted_batch = B_atar_idx[sort_idx]
        dense_atar, pad_mask = to_dense_batch(sorted_tokens, sorted_batch)
        normed = self.atar_event_self_attn_norm(dense_atar)
        sa_out, _ = self.atar_event_self_attn(
            normed, normed, normed, key_padding_mask=~pad_mask)
        tokens_refined = dense_atar + sa_out
        if torch.isnan(tokens_refined).any():
            tokens_refined = dense_atar
        refined_flat = tokens_refined[pad_mask]
        inverse_sort = torch.argsort(sort_idx)
        refined_flat = refined_flat[inverse_sort]
        output['atar_event_tokens'] = refined_flat

        trigger_logits = self.atar_trigger_classifier(refined_flat).squeeze(-1)
        output['atar_trigger_logits'] = trigger_logits
        trigger_probs = torch.sigmoid(trigger_logits).detach()

        # --- Phase 10: pion stop ---
        trigger_prob_full = torch.zeros(num_global_slices, device=device)
        trigger_prob_full[valid_slice_mask] = trigger_probs
        hit_trigger_prob = trigger_prob_full[global_slice_idx_all]
        pion_prob = torch.sigmoid(output['atar_node_pdg'][:, PION_CLASS]).detach()
        pion_gate = (hit_trigger_prob * pion_prob).unsqueeze(-1)
        h_pion = h_atar * pion_gate
        pion_event_pool = global_add_pool(h_pion, batch_atar, size=num_graphs_in_batch)
        pion_gate_sum = global_add_pool(pion_gate, batch_atar, size=num_graphs_in_batch)
        pion_event_pool = pion_event_pool / pion_gate_sum.clamp(min=1e-6)
        output['pion_event_pool'] = pion_event_pool
        output['atar_pion_stop'] = self.pion_stop_head(pion_event_pool)

        # --- Phase 11: positron direction + time ---
        endpoints_det = output['atar_endpoints'].detach()
        start_median = endpoints_det[:, 0, :, 1]
        stop_median = endpoints_det[:, 1, :, 1]
        slice_exit_dir = F.normalize(stop_median - start_median, p=2, dim=-1)
        slice_trigger_w = trigger_probs.unsqueeze(-1)
        weighted_exit = slice_exit_dir * slice_trigger_w
        exit_dir_sum = torch.zeros(num_graphs_in_batch, 3, device=device)
        exit_weight_sum = torch.zeros(num_graphs_in_batch, 1, device=device)
        exit_dir_sum.index_add_(0, B_atar_idx, weighted_exit)
        exit_weight_sum.index_add_(0, B_atar_idx, slice_trigger_w)
        exit_dir_per_graph = F.normalize(
            exit_dir_sum / exit_weight_sum.clamp(min=1e-6), p=2, dim=-1)
        output['exit_dir_per_graph'] = exit_dir_per_graph

        mip_prob = torch.sigmoid(output['atar_node_pdg'][:, MIP_CLASS]).detach()
        positron_hit_mask = (hit_trigger_prob > 0.5) & (mip_prob > 0.5)
        hit_w = positron_hit_mask.float()
        t_num = torch.zeros(num_graphs_in_batch, device=device)
        t_den = torch.zeros(num_graphs_in_batch, device=device)
        t_num.index_add_(0, batch_atar, x[is_atar, 4] * hit_w)
        t_den.index_add_(0, batch_atar, hit_w)
        NO_POSITRON_TIME = -500.0
        positron_time = torch.where(
            t_den > 0.5,
            t_num / t_den.clamp(min=1.0),
            torch.full_like(t_num, NO_POSITRON_TIME),
        )
        output['positron_time'] = positron_time

        mip_gate = (hit_trigger_prob * mip_prob).unsqueeze(-1)
        h_mip = h_atar * mip_gate
        mip_event_pool = global_add_pool(h_mip, batch_atar, size=num_graphs_in_batch)
        mip_gate_sum = global_add_pool(mip_gate, batch_atar, size=num_graphs_in_batch)
        mip_event_pool = mip_event_pool / mip_gate_sum.clamp(min=1e-6)
        output['mip_event_pool'] = mip_event_pool

        dir_input = torch.cat([
            mip_event_pool,
            output['atar_pion_stop'].detach(),
            exit_dir_per_graph.detach(),
        ], dim=-1)
        output['atar_positron_dir'] = self.positron_dir_head(dir_input)

        # --- Muon-gated pool (for tail reveal head's muon anchor) ---
        muon_prob = torch.sigmoid(output['atar_node_pdg'][:, MUON_CLASS]).detach()
        muon_gate = muon_prob.unsqueeze(-1)
        h_muon = h_atar * muon_gate
        muon_event_pool = global_add_pool(h_muon, batch_atar, size=num_graphs_in_batch)
        muon_gate_sum = global_add_pool(muon_gate, batch_atar, size=num_graphs_in_batch)
        output['muon_event_pool'] = muon_event_pool / muon_gate_sum.clamp(min=1e-6)

        return output


# ==========================================================================
# Feature assembly: pulls what both heads need out of the trunk output,
# detached, normalized, aligned.
# ==========================================================================

def assemble_tail_features(output, x, batch):
    is_atar = output['is_atar']
    B = output['num_graphs_in_batch']
    device = x.device

    # per-hit
    h_atar = output['h_atar'].detach()
    node_pdg = output['atar_node_pdg'].detach()
    node_probs = torch.sigmoid(node_pdg)
    muon_prob = node_probs[:, MUON_CLASS]
    mip_prob = node_probs[:, MIP_CLASS]
    hit_pos = x[is_atar, 0:3]
    hit_energy = x[is_atar, 3]
    hit_time = x[is_atar, 4]
    batch_atar = batch[is_atar]

    # per-slice (valid only)
    slice_pdg = output['atar_slice_pdg'].detach()
    slice_multi = output['atar_slice_multi'].detach().unsqueeze(-1)
    slice_trigger = torch.sigmoid(output['atar_trigger_logits']).detach().unsqueeze(-1)
    slice_energy = output['slice_energy'][output['valid_slice_mask']].detach()
    slice_mean_t = output['slice_mean_time'].detach().unsqueeze(-1)
    event_tokens = output['atar_event_tokens'].detach()
    valid_slice_indices = torch.nonzero(output['valid_slice_mask']).squeeze(1)
    B_slice_idx = (valid_slice_indices // output['num_slices_max']).long()

    # per-graph anchors
    pion_stop = output['atar_pion_stop'].detach()
    positron_dir = output['atar_positron_dir'].detach()
    exit_dir = output['exit_dir_per_graph'].detach()
    positron_time = output['positron_time'].detach().unsqueeze(-1)
    pion_pool = output['pion_event_pool'].detach()
    mip_pool = output['mip_event_pool'].detach()
    muon_pool = output['muon_event_pool'].detach()

    return dict(
        B=B, device=device,
        h_atar=h_atar, muon_prob=muon_prob, mip_prob=mip_prob,
        hit_pos=hit_pos, hit_energy=hit_energy, hit_time=hit_time,
        batch_atar=batch_atar,
        slice_pdg=slice_pdg, slice_multi=slice_multi, slice_trigger=slice_trigger,
        slice_energy=slice_energy, slice_mean_t=slice_mean_t,
        event_tokens=event_tokens, B_slice_idx=B_slice_idx,
        pion_stop=pion_stop, positron_dir=positron_dir, exit_dir=exit_dir,
        positron_time=positron_time,
        pion_pool=pion_pool, mip_pool=mip_pool, muon_pool=muon_pool,
    )


# ==========================================================================
# Scatter helpers (avoid torch_scatter dep by using dense-batch masked ops)
# ==========================================================================

def scatter_max_dense(values, index, dim_size):
    """Per-index max without torch_scatter. values: [N], index: [N] long."""
    N = values.numel()
    dense = torch.full((dim_size,), -float('inf'), device=values.device, dtype=values.dtype)
    # scatter_reduce_ available in torch >= 1.12
    dense = dense.scatter_reduce(0, index, values, reduce="amax", include_self=True)
    dense = torch.where(torch.isinf(dense), torch.zeros_like(dense), dense)
    return dense


def scatter_logsumexp_dense(values, index, dim_size):
    """Per-index logsumexp without torch_scatter."""
    max_per = scatter_max_dense(values, index, dim_size)
    shifted = values - max_per[index]
    exp_sum = torch.zeros(dim_size, device=values.device, dtype=values.dtype)
    exp_sum.index_add_(0, index, shifted.exp())
    return max_per + torch.log(exp_sum.clamp(min=1e-20))


# ==========================================================================
# MuonVetoHead
# --------------------------------------------------------------------------
# Three aggressive aggregations so a single muon-ish hit or slice fires the
# veto. Owns its own per-hit muon classifier so tail-task gradients can push
# muon recall without moving the shared node-PDG head.
# ==========================================================================

class MuonVetoHead(nn.Module):
    def __init__(self, jk_dim, hidden=128):
        super().__init__()

        self.muon_gated_pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(jk_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        )
        self.pool_proj = nn.Linear(jk_dim, hidden)

        # kinematics: [E_mu_total, mu_tspan, |mu_stop - pion_stop|,
        #              |mu_stop - positron_start|, N_mu, gap_score]
        N_KIN = 6
        self.kin_mlp = nn.Sequential(
            nn.Linear(N_KIN, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # max / LSE scalars: [max_hit, lse_hit, max_slice, lse_slice, n_mu_slices]
        N_MAX = 5
        self.max_mlp = nn.Sequential(
            nn.Linear(N_MAX, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, hidden // 2),
        )

        # dedicated per-hit muon classifier (independent gradient path)
        self.muon_node_head = nn.Sequential(
            nn.Linear(jk_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + hidden // 2, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def compute_muon_kinematics(self, f):
        device = f['device']
        w = f['muon_prob'].unsqueeze(-1)
        w_sum = global_add_pool(w, f['batch_atar'], size=f['B']).clamp(min=1e-6)

        # total muon energy
        E_mu = global_add_pool(
            f['hit_energy'].unsqueeze(-1) * w, f['batch_atar'], size=f['B'])

        # muon hit count (normalized by ~32)
        N_mu = global_add_pool(w, f['batch_atar'], size=f['B']) / 32.0

        # energy-weighted muon centroid
        mu_pos = global_add_pool(
            f['hit_pos'] * w, f['batch_atar'], size=f['B']) / w_sum
        d_mu_pion = (mu_pos - f['pion_stop']).norm(dim=-1, keepdim=True)
        positron_start = f['pion_stop']  # placeholder: muons die near pion stop in DAR
        d_mu_pos = (mu_pos - positron_start).norm(dim=-1, keepdim=True)

        # muon temporal spread (energy-weighted std)
        t = f['hit_time'].unsqueeze(-1)
        t_w = global_add_pool(t * w, f['batch_atar'], size=f['B']) / w_sum
        t2_w = global_add_pool((t ** 2) * w, f['batch_atar'], size=f['B']) / w_sum
        mu_tspan = (t2_w - t_w ** 2).clamp(min=0.0).sqrt()

        # gap continuity score: placeholder (0). DTAR hits would fill this in.
        gap_score = torch.zeros_like(E_mu)

        return torch.cat([E_mu, mu_tspan, d_mu_pion, d_mu_pos, N_mu, gap_score], dim=-1)

    def compute_max_features(self, f):
        w_hit = f['muon_prob']
        max_hit = scatter_max_dense(w_hit, f['batch_atar'], f['B'])
        lse_hit = scatter_logsumexp_dense(w_hit, f['batch_atar'], f['B'])

        slice_muon = torch.sigmoid(f['slice_pdg'][:, MUON_CLASS])
        max_slice = scatter_max_dense(slice_muon, f['B_slice_idx'], f['B'])
        lse_slice = scatter_logsumexp_dense(slice_muon, f['B_slice_idx'], f['B'])

        n_mu_slices = global_add_pool(
            (slice_muon > 0.5).float().unsqueeze(-1),
            f['B_slice_idx'], size=f['B'],
        ).squeeze(-1) / 8.0

        return torch.stack([max_hit, lse_hit, max_slice, lse_slice, n_mu_slices], dim=-1)

    def forward(self, features):
        f = dict(features)  # shallow copy so we can override muon_prob

        # own muon classifier — gradients flow here, not into shared node PDG
        muon_logit_tail = self.muon_node_head(f['h_atar']).squeeze(-1)
        muon_prob_tail = torch.sigmoid(muon_logit_tail)
        f['muon_prob'] = 0.5 * f['muon_prob'] + 0.5 * muon_prob_tail

        # (a) learned muon-gated pool
        gated = f['h_atar'] * f['muon_prob'].unsqueeze(-1)
        pool = self.muon_gated_pool(gated, f['batch_atar'], dim_size=f['B'])
        pool_feat = F.gelu(self.pool_proj(pool))

        # (b) kinematics
        muon_kin = self.compute_muon_kinematics(f)
        kin_feat = self.kin_mlp(muon_kin)

        # (c) max / LSE
        max_feat = self.max_mlp(self.compute_max_features(f))

        z = torch.cat([pool_feat, kin_feat, max_feat], dim=-1)
        muon_logit = self.fuse(z).squeeze(-1)

        return muon_logit, muon_logit_tail, muon_kin


# ==========================================================================
# PileupVetoHead
# --------------------------------------------------------------------------
# Mirrors MuonVetoHead's three-pathway redundancy on ATAR pileup signals.
# Reuses the trunk's per-slice multi-event head (already trained against
# multi-origin slices) and the trigger logit's complement as cheap
# per-slice pileup signals; adds its own per-hit pileup classifier with
# its own gradient path.
#
# ATAR-only: no LYSO inputs anywhere. Truth label uses (atar_origin > 0)
# only — cal-only pileup events get labeled "no pileup" because the head
# can't see them anyway, keeping label and feature space aligned.
# ==========================================================================

class PileupVetoHead(nn.Module):
    def __init__(self, jk_dim, hidden=128):
        super().__init__()

        # (a) learned pool over pileup-gated hit features
        self.pileup_gated_pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(jk_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        )
        self.pool_proj = nn.Linear(jk_dim, hidden)

        # (b) ATAR-only structural scalars:
        #   [n_valid_slices_norm, slice_t_span, max_non_trigger_prob,
        #    n_non_trigger_slices_norm, hit_t_span]
        N_KIN = 5
        self.kin_mlp = nn.Sequential(
            nn.Linear(N_KIN, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # (c) max/LSE features:
        #   [max_hit_pileup, lse_hit_pileup,
        #    max_slice_multi, lse_slice_multi,
        #    lse_one_minus_trigger]
        N_MAX = 5
        self.max_mlp = nn.Sequential(
            nn.Linear(N_MAX, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, hidden // 2),
        )

        # dedicated per-hit pileup classifier (own gradient path)
        self.pileup_node_head = nn.Sequential(
            nn.Linear(jk_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + hidden // 2, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def compute_pileup_kinematics(self, f):
        """ATAR-only structural pileup features."""
        B = f['B']

        # number of valid ATAR slices per graph (clean events: ~3 — pi entry,
        # pi stop, e+; pileup events drift higher)
        n_slices = global_add_pool(
            torch.ones_like(f['B_slice_idx'], dtype=torch.float).unsqueeze(-1),
            f['B_slice_idx'], size=B,
        ) / 4.0

        # span of slice mean times (clean events stay tight)
        slice_t = f['slice_mean_t'].squeeze(-1)
        max_t = scatter_max_dense(slice_t, f['B_slice_idx'], B)
        min_t = -scatter_max_dense(-slice_t, f['B_slice_idx'], B)
        slice_span = ((max_t - min_t) / 100.0).unsqueeze(-1)

        # biggest non-trigger slice (low trigger prob)
        non_trigger = 1.0 - f['slice_trigger'].squeeze(-1)
        max_non_trigger = scatter_max_dense(
            non_trigger, f['B_slice_idx'], B).unsqueeze(-1)

        # number of slices with low trigger prob
        n_non_trigger = global_add_pool(
            (f['slice_trigger'].squeeze(-1) < 0.5).float().unsqueeze(-1),
            f['B_slice_idx'], size=B,
        ) / 4.0

        # span of all hit times in the event
        hit_t = f['hit_time']
        max_ht = scatter_max_dense(hit_t, f['batch_atar'], B)
        min_ht = -scatter_max_dense(-hit_t, f['batch_atar'], B)
        hit_span = ((max_ht - min_ht) / 100.0).unsqueeze(-1)

        return torch.cat([n_slices, slice_span, max_non_trigger,
                          n_non_trigger, hit_span], dim=-1)

    def compute_max_features(self, f):
        w_hit = f['pileup_prob']
        max_hit = scatter_max_dense(w_hit, f['batch_atar'], f['B'])
        lse_hit = scatter_logsumexp_dense(w_hit, f['batch_atar'], f['B'])

        slice_multi = torch.sigmoid(f['slice_multi'].squeeze(-1))
        max_slice = scatter_max_dense(slice_multi, f['B_slice_idx'], f['B'])
        lse_slice = scatter_logsumexp_dense(slice_multi, f['B_slice_idx'], f['B'])

        non_trigger = 1.0 - f['slice_trigger'].squeeze(-1)
        lse_non_trig = scatter_logsumexp_dense(
            non_trigger, f['B_slice_idx'], f['B'])

        return torch.stack(
            [max_hit, lse_hit, max_slice, lse_slice, lse_non_trig], dim=-1)

    def forward(self, features):
        f = dict(features)

        # own per-hit pileup classifier
        pileup_logit_node = self.pileup_node_head(f['h_atar']).squeeze(-1)
        f['pileup_prob'] = torch.sigmoid(pileup_logit_node)

        gated = f['h_atar'] * f['pileup_prob'].unsqueeze(-1)
        pool = self.pileup_gated_pool(gated, f['batch_atar'], dim_size=f['B'])
        pool_feat = F.gelu(self.pool_proj(pool))

        kin_feat = self.kin_mlp(self.compute_pileup_kinematics(f))
        max_feat = self.max_mlp(self.compute_max_features(f))

        z = torch.cat([pool_feat, kin_feat, max_feat], dim=-1)
        pileup_logit = self.fuse(z).squeeze(-1)

        return pileup_logit, pileup_logit_node


# ==========================================================================
# PieTaggerHead
# --------------------------------------------------------------------------
# Positive-tag classifier: "does this event look like a clean pi -> e nu
# chain?" Trained against is_pie (energy-blind), so its output is
# statistically independent of positron energy and does not warp the
# surviving energy spectrum across §1.2 bins.
#
# Transformer over anchor tokens (pion, positron, muon), physics-region
# pool tokens (along-start, along-end, late), and per-slice event tokens.
# Emits a single pie_logit per graph from the CLS read-out.
# ==========================================================================

class PieTaggerHead(nn.Module):
    def __init__(self, jk_dim, d_tail=D_TAIL, n_layers=2, n_heads=4):
        super().__init__()

        # anchor projections
        self.pion_anchor_proj = nn.Linear(jk_dim + 3, d_tail)
        self.positron_anchor_proj = nn.Linear(jk_dim + 3 + 3 + 1, d_tail)
        self.muon_anchor_proj = nn.Linear(jk_dim + 6, d_tail)  # muon_pool + kin(6)

        # per-graph physics-region pools
        def _make_gate():
            return AttentionalAggregation(nn.Sequential(
                nn.Linear(jk_dim, 64), nn.GELU(), nn.Linear(64, 1)))
        self.along_start_pool = _make_gate()
        self.along_end_pool = _make_gate()
        self.late_pool = _make_gate()
        self.along_start_proj = nn.Linear(jk_dim, d_tail)
        self.along_end_proj = nn.Linear(jk_dim, d_tail)
        self.late_proj = nn.Linear(jk_dim, d_tail)

        # per-slice token: event_token (256) + [slice_pdg(3), multi(1),
        #                                       trigger(1), energy(1), time(1)]
        self.slice_proj = nn.Linear(256 + 7, d_tail)

        # token-type embedding
        # 0:CLS 1:pion 2:pos 3:mu 4:start 5:end 6:late 7:slice
        self.token_type_emb = nn.Embedding(8, d_tail)
        self.cls = nn.Parameter(torch.randn(1, 1, d_tail) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_tail, nhead=n_heads, dim_feedforward=d_tail * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_tail, 1)

    def _along_track_weights(self, f):
        """Soft weights along the positron track.

        start_w: hits close to pion_stop (Bragg region)
        end_w:   hits far from pion_stop along positron_dir (MIP region)
        """
        # per-hit pion_stop and positron_dir (broadcast per graph to hit)
        pion_stop_h = f['pion_stop'][f['batch_atar']]          # [N_atar, 3]
        pos_dir_h = f['positron_dir'][f['batch_atar']]         # [N_atar, 3]
        rel = f['hit_pos'] - pion_stop_h                        # [N_atar, 3]
        proj = (rel * pos_dir_h).sum(dim=-1)                   # [N_atar] signed
        # in units of ~1 (positions are already normalized upstream); the exact
        # scale does not matter since we feed these as soft weights
        tau = 2.0
        start_w = torch.exp(-(proj.clamp(min=-10, max=10) ** 2) / (2 * tau ** 2))
        end_w = torch.sigmoid((proj - tau) * 0.5)
        return start_w, end_w

    def _late_weights(self, f):
        positron_t = f['positron_time'].squeeze(-1)[f['batch_atar']]
        return (f['hit_time'] - positron_t).clamp(min=0.0).sigmoid()

    def forward(self, f, muon_kinematics):
        B = f['B']
        dev = f['device']

        # --- anchor tokens ---
        pion_anchor = self.pion_anchor_proj(
            torch.cat([f['pion_pool'], f['pion_stop']], dim=-1))
        positron_anchor = self.positron_anchor_proj(
            torch.cat([f['mip_pool'], f['positron_dir'],
                       f['exit_dir'], f['positron_time']], dim=-1))
        muon_anchor = self.muon_anchor_proj(
            torch.cat([f['muon_pool'], muon_kinematics], dim=-1))

        # --- physics-region pool tokens ---
        start_w, end_w = self._along_track_weights(f)
        late_w = self._late_weights(f)
        start_vec = self.along_start_pool(
            f['h_atar'] * start_w.unsqueeze(-1), f['batch_atar'], dim_size=B)
        end_vec = self.along_end_pool(
            f['h_atar'] * end_w.unsqueeze(-1), f['batch_atar'], dim_size=B)
        late_vec = self.late_pool(
            f['h_atar'] * late_w.unsqueeze(-1), f['batch_atar'], dim_size=B)
        start_tok = self.along_start_proj(start_vec)
        end_tok = self.along_end_proj(end_vec)
        late_tok = self.late_proj(late_vec)

        # --- slice tokens ---
        slice_feats = torch.cat([
            f['event_tokens'], f['slice_pdg'], f['slice_multi'],
            f['slice_trigger'], f['slice_energy'], f['slice_mean_t'],
        ], dim=-1)
        slice_tokens = self.slice_proj(slice_feats)
        sort_idx = torch.argsort(f['B_slice_idx'])
        dense_slices, slice_pad_mask = to_dense_batch(
            slice_tokens[sort_idx], f['B_slice_idx'][sort_idx], batch_size=B)

        # --- type embeddings ---
        type_ids = {
            'cls': 0, 'pion': 1, 'pos': 2, 'mu': 3,
            'start': 4, 'end': 5, 'late': 6, 'slice': 7,
        }
        def typed(v, name):
            return v + self.token_type_emb(torch.full(
                v.shape[:-1], type_ids[name], dtype=torch.long, device=dev))

        cls_tok = self.cls.expand(B, -1, -1)
        fixed_tokens = torch.stack([
            typed(pion_anchor,     'pion'),
            typed(positron_anchor, 'pos'),
            typed(muon_anchor,     'mu'),
            typed(start_tok,       'start'),
            typed(end_tok,         'end'),
            typed(late_tok,        'late'),
        ], dim=1)
        dense_slices = typed(dense_slices, 'slice')
        seq = torch.cat([typed(cls_tok, 'cls'), fixed_tokens, dense_slices], dim=1)

        n_fixed = 1 + 6
        pad = torch.cat([
            torch.zeros(B, n_fixed, dtype=torch.bool, device=dev),
            ~slice_pad_mask,
        ], dim=1)

        out = self.transformer(seq, src_key_padding_mask=pad)
        return self.head(out[:, 0]).squeeze(-1)


# ==========================================================================
# Top-level model: trunk + two tail heads, 2D output.
# ==========================================================================

class PURITYTailModel(nn.Module):
    """ATAR-only PURITY trunk with three downstream heads:
    PieTaggerHead, MuonVetoHead, PileupVetoHead.

    Output dict includes:
        pie_logit         [B]       — clean pi -> e nu topology score (high = keep)
        muon_logit        [B]       — any-muon-evidence score (low = keep)
        pileup_logit      [B]       — any-ATAR-pileup score (low = keep)
        muon_node_logit   [N_atar]  — aux per-hit muon classifier
        pileup_node_logit [N_atar]  — aux per-hit pileup classifier
        ... plus all the upstream PURITY outputs (node PDG, slice PDG,
        endpoints, pion_stop, positron_dir, etc.)

    Downstream analysis cuts in the 3D (pie, muon, pileup) plane.

    freeze_trunk: if True (default), the backbone's parameters do not
    receive gradients. This is the Stage 1 setup — train heads against a
    converged PURITY trunk. Flip to False for Stage 2 end-to-end fine-tuning.
    """

    def __init__(self, hidden_dim=150, num_blocks=3, heads=5,
                 dropout=0.05, num_pdg_classes=3, freeze_trunk=True):
        super().__init__()
        self.backbone = PURITYTailBackbone(
            hidden_dim=hidden_dim, num_blocks=num_blocks, heads=heads,
            dropout=dropout, num_pdg_classes=num_pdg_classes,
        )
        jk_dim = self.backbone.jk_dim
        self.muon_veto   = MuonVetoHead(jk_dim)
        self.pileup_veto = PileupVetoHead(jk_dim)
        self.pie_tagger  = PieTaggerHead(jk_dim)

        if freeze_trunk:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x, batch):
        output = self.backbone(x, batch)
        if 'h_atar' not in output:
            return output

        f = assemble_tail_features(output, x, batch)
        muon_logit, muon_node_logit, muon_kin = self.muon_veto(f)
        pileup_logit, pileup_node_logit       = self.pileup_veto(f)
        pie_logit = self.pie_tagger(f, muon_kin)

        output['pie_logit']         = pie_logit
        output['muon_logit']        = muon_logit
        output['pileup_logit']      = pileup_logit
        output['muon_node_logit']   = muon_node_logit
        output['pileup_node_logit'] = pileup_node_logit
        return output


# ==========================================================================
# Loss
# --------------------------------------------------------------------------
# Three event-level BCEs + two per-hit aux BCEs. No clean-event masking —
# each head trains on the full event population so its output remains a
# valid axis of the 3D cut plane.
#
# pos_weights are tuned to the actual training-data class balances. The
# defaults below assume rates of:
#    is_pie:          ~20%   (minority — pos_weight > 1 to emphasize)
#    muon_present:    ~85%   (majority — pos_weight ~1, no extra emphasis)
#    pileup_present:  ~37%   (closer to balanced, mild pos_weight)
# Equal-loss-contribution heuristic: pos_weight ≈ rate_neg / rate_pos.
# ==========================================================================

def pie_tagger_loss(out, targets,
                    pos_weight_pie=4.0, pos_weight_muon=1.0, pos_weight_pileup=2.0,
                    pos_weight_aux_muon=10.0, pos_weight_aux_pileup=10.0,
                    w_pie=1.0, w_muon=1.0, w_pileup=1.0,
                    w_aux_muon=0.3, w_aux_pileup=0.3):
    """
    targets dict:
        'is_pie':           [B]       1 if event_type == 1
        'muon_present':     [B]       1 if any (atar_pdg & MUON) in event
        'pileup_present':   [B]       1 if any (atar_origin > 0)  ATAR-only
        'muon_hits':        [N_atar]  per-hit muon labels
        'pileup_hits':      [N_atar]  per-hit pileup labels (origin > 0)
        'muon_hit_mask':    [N_atar]  which hits participate in aux losses
    """
    device = out['pie_logit'].device

    pie_bce = F.binary_cross_entropy_with_logits(
        out['pie_logit'], targets['is_pie'].float(),
        pos_weight=torch.tensor(pos_weight_pie, device=device),
    )
    muon_bce = F.binary_cross_entropy_with_logits(
        out['muon_logit'], targets['muon_present'].float(),
        pos_weight=torch.tensor(pos_weight_muon, device=device),
    )
    pileup_bce = F.binary_cross_entropy_with_logits(
        out['pileup_logit'], targets['pileup_present'].float(),
        pos_weight=torch.tensor(pos_weight_pileup, device=device),
    )

    m = targets['muon_hit_mask'].bool()
    if m.any():
        aux_muon = F.binary_cross_entropy_with_logits(
            out['muon_node_logit'][m], targets['muon_hits'][m].float(),
            pos_weight=torch.tensor(pos_weight_aux_muon, device=device),
        )
        aux_pileup = F.binary_cross_entropy_with_logits(
            out['pileup_node_logit'][m], targets['pileup_hits'][m].float(),
            pos_weight=torch.tensor(pos_weight_aux_pileup, device=device),
        )
    else:
        aux_muon = torch.zeros((), device=device)
        aux_pileup = torch.zeros((), device=device)

    total = (w_pie * pie_bce + w_muon * muon_bce + w_pileup * pileup_bce
             + w_aux_muon * aux_muon + w_aux_pileup * aux_pileup)
    return total, dict(
        pie_bce=pie_bce, muon_bce=muon_bce, pileup_bce=pileup_bce,
        aux_muon=aux_muon, aux_pileup=aux_pileup,
    )
