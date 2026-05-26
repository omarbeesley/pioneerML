"""
Systematic diagnosis of positron direction S-shape bias.

Tests:
  1. Pre-normalization logit analysis
     - ||dir_logits|| vs θ_true  (does the norm vary with angle?)
     - Per-component (dx, dy, dz) residuals vs θ_true
     - θ_pred from raw logits vs F.normalize — isolates normalization effect

  2. Feedback propagation: acceptance with truth vs reco direction
     - If the direction bias creates a biased cos_sep_positron, the event
       builder accept/reject can inherit a θ-dependent acceptance shift.
     - Run inference twice: normal, and with positron_dir replaced by truth.

  3. Loss gradient geometry (analytical)
     - Cosine loss 1 - cos(Δ) gradient vs angular MSE gradient
     - Shows non-uniform learning pressure

  4. Hit selection bias
     - Number of positron-tagged hits per view vs θ_true
     - If forward/backward positrons have fewer qualifying hits,
       the direction head has less information at those angles.

  5. Training distribution
     - θ_true histogram: non-uniform → non-uniform loss pressure

Usage:
    python diagnose_direction_bias.py \\
        --checkpoint model_weights/PURITY_fast3_2x_v3_best.pth \\
        --eval_path /data/mixed_parquets/pie_benchmark_5_11/data.parquet \\
        --output_dir ./direction_bias_diagnostics
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_reco.models_v2 import PURITYHybridModelV2
from unified_reco.dataset import PURITYDataset
from unified_reco.constants import NORM_POS_ATAR

BATCH_SIZE = 50
MAX_HITS = 250
NUM_WORKERS = 2

TASK_WEIGHTS = {
    'w_atar_slice_multi':       0.05,
    'w_node_pdg':               1.0,
    'w_slice_pdg':              0.1,
    'w_endpoints':              0.025,
    'w_lyso_condensation':      0.25,
    'w_atar_trigger_slice':     0.5,
    'w_time_spread':            1.0,
    'time_spread_thresh_ns':    1.0,
    'time_spread_trig_floor':   0.25,
    'time_spread_mip_floor':    0.25,
    'w_pion_kinematics':        50.0,
    'w_positron_angle':         0.5,
    'w_event_builder':          0.1,
    'w_has_trigger_positron':   0.0,
    'w_dead_energy':            0.005,
}


# =========================================================================
# Inference with hooks
# =========================================================================

def run_hooked_inference(model, device, parquet_path, max_events=None):
    """Run inference capturing dir_logits (before F.normalize) and hit counts."""
    ds = PURITYDataset(parquet_path, max_hits=MAX_HITS)
    n_total = len(ds)
    n = n_total if max_events is None else min(n_total, max_events)
    if n < n_total:
        ds = Subset(ds, list(range(n)))
    df = PURITYDataset(parquet_path, max_hits=MAX_HITS).df.iloc[:n].reset_index(drop=True)

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))

    # truth
    theta_true = df['truth_theta'].to_numpy(dtype=np.float32)
    phi_true = df['truth_phi'].to_numpy(dtype=np.float32)
    truth_dir = np.column_stack([
        np.sin(theta_true) * np.cos(phi_true),
        np.sin(theta_true) * np.sin(phi_true),
        np.cos(theta_true),
    ])

    # hook storage
    captured_logits = []
    captured_gate_logits = []

    def hook_fn(module, inp, out):
        captured_logits.append(out.detach().cpu())

    def hook_gate_fn(module, inp, out):
        captured_gate_logits.append(out.detach().cpu())

    handle = model.positron_dir_head.register_forward_hook(hook_fn)
    handle_gate = model.dir_attn_pool.gate_nn.register_forward_hook(hook_gate_fn)

    # output storage
    all_logits = []
    all_pred_dir = []
    all_n_hits_x = []
    all_n_hits_y = []
    all_accepted = []

    # per-event attention statistics (computed from per-hit gate outputs)
    all_attn_entropy = []        # entropy of attention weights per graph
    all_attn_max = []            # max attention weight per graph
    all_attn_spatial_bias = []   # attention-weighted mean hit position per graph
    all_hit_positions = []       # mean hit z per graph (for reference)

    i0 = 0
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dl, desc='hooked inference'):
            batch = batch.to(device, non_blocking=True)
            B = batch.num_graphs
            anchor = getattr(batch, 'atar_triggering_pion_slice', None)

            captured_logits.clear()
            captured_gate_logits.clear()
            out = model(batch.x, batch.batch, task_weights=TASK_WEIGHTS,
                        triggering_pion_slice=anchor)

            # direction
            pred_dir = out['atar_positron_dir'].cpu().numpy()
            all_pred_dir.append(pred_dir)

            # pre-normalization logits
            if captured_logits:
                all_logits.append(captured_logits[0].numpy())
            else:
                all_logits.append(np.full((B, 3), np.nan))

            # hit counts per view
            is_atar = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
            is_x = batch.x[:, 5] > 0.5
            is_y = batch.x[:, 6] > 0.5
            batch_idx = batch.batch

            hit_trig = out.get('atar_hit_trigger_prob')
            hit_mip = out.get('atar_hit_mip_prob')

            if hit_trig is not None and hit_mip is not None:
                pos_mask = (hit_trig > 0.5) & (hit_mip > 0.5)
                is_x_atar = is_x[is_atar]
                is_y_atar = is_y[is_atar]
                batch_atar = batch_idx[is_atar]
                pos_x = pos_mask & is_x_atar
                pos_y = pos_mask & is_y_atar

                n_x = torch.zeros(B, device=device)
                n_y = torch.zeros(B, device=device)
                n_x.index_add_(0, batch_atar[pos_x], torch.ones(pos_x.sum(), device=device))
                n_y.index_add_(0, batch_atar[pos_y], torch.ones(pos_y.sum(), device=device))
                all_n_hits_x.append(n_x.cpu().numpy())
                all_n_hits_y.append(n_y.cpu().numpy())

                # --- Attention weight analysis ---
                # gate_nn is called twice: once for mask_x hits, once for mask_y
                # captured_gate_logits has 2 entries (x-view, y-view)
                # Compute per-graph attention statistics for each view
                for view_i, (view_mask, view_label) in enumerate(
                        [(pos_x, 'x'), (pos_y, 'y')]):
                    if view_i < len(captured_gate_logits) and view_mask.any():
                        gate_raw = captured_gate_logits[view_i].squeeze(-1)  # [N_view]
                        b_view = batch_atar[view_mask].cpu()
                        hit_pos_view = batch.x[is_atar][view_mask, :3].cpu()

                        # softmax per graph to get attention weights
                        from torch_geometric.utils import softmax as pyg_softmax
                        attn_weights = pyg_softmax(gate_raw, b_view, num_nodes=B)

                        # entropy per graph
                        log_w = torch.log(attn_weights.clamp(min=1e-10))
                        neg_wlogw = -attn_weights * log_w
                        entropy = torch.zeros(B)
                        entropy.index_add_(0, b_view, neg_wlogw)
                        if view_i == 0:
                            batch_entropy_x = entropy.numpy()
                        else:
                            batch_entropy_y = entropy.numpy()

                        # max weight per graph
                        max_w = torch.full((B,), -float('inf'))
                        max_w = max_w.scatter_reduce(
                            0, b_view, attn_weights, reduce='amax',
                            include_self=False)
                        max_w = torch.where(torch.isinf(max_w),
                                            torch.zeros_like(max_w), max_w)
                        if view_i == 0:
                            batch_max_x = max_w.numpy()
                        else:
                            batch_max_y = max_w.numpy()

                        # attention-weighted mean hit position (xyz)
                        w_pos = hit_pos_view * attn_weights.unsqueeze(-1)
                        wpos_sum = torch.zeros(B, 3)
                        wpos_sum.index_add_(0, b_view, w_pos)
                        if view_i == 0:
                            batch_wpos_x = wpos_sum.numpy()
                        else:
                            batch_wpos_y = wpos_sum.numpy()

                # store per-view stats as concatenated [entropy_x, entropy_y]
                if 'batch_entropy_x' in dir() and 'batch_entropy_y' in dir():
                    all_attn_entropy.append(np.stack(
                        [batch_entropy_x, batch_entropy_y], axis=-1))
                    all_attn_max.append(np.stack(
                        [batch_max_x, batch_max_y], axis=-1))
                    all_attn_spatial_bias.append(np.stack(
                        [batch_wpos_x, batch_wpos_y], axis=1))  # [B, 2, 3]
                else:
                    all_attn_entropy.append(np.full((B, 2), np.nan))
                    all_attn_max.append(np.full((B, 2), np.nan))
                    all_attn_spatial_bias.append(np.full((B, 2, 3), np.nan))
            else:
                all_n_hits_x.append(np.full(B, np.nan))
                all_n_hits_y.append(np.full(B, np.nan))
                all_attn_entropy.append(np.full((B, 2), np.nan))
                all_attn_max.append(np.full((B, 2), np.nan))
                all_attn_spatial_bias.append(np.full((B, 2, 3), np.nan))

            # acceptance
            es = out.get('event_summary', {})
            acc = es.get('accepted')
            if acc is not None:
                all_accepted.append(acc.float().cpu().numpy())
            else:
                all_accepted.append(np.full(B, np.nan))

            i0 += B

    handle.remove()
    handle_gate.remove()

    results = {
        'theta_true': theta_true[:i0],
        'phi_true': phi_true[:i0],
        'truth_dir': truth_dir[:i0],
        'pred_dir': np.concatenate(all_pred_dir),
        'dir_logits': np.concatenate(all_logits),
        'n_hits_x': np.concatenate(all_n_hits_x),
        'n_hits_y': np.concatenate(all_n_hits_y),
        'accepted': np.concatenate(all_accepted),
        'attn_entropy': np.concatenate(all_attn_entropy),     # [N, 2] x/y
        'attn_max': np.concatenate(all_attn_max),             # [N, 2]
        'attn_spatial': np.concatenate(all_attn_spatial_bias), # [N, 2, 3]
    }
    if 'truth_acceptance' in df.columns:
        results['truth_acceptance'] = df['truth_acceptance'].to_numpy(dtype=np.int32)[:i0]
    return results


# =========================================================================
# Test 1: Pre-normalization logit analysis
# =========================================================================

def plot_logit_analysis(r, output_dir):
    """Analyze dir_logits before F.normalize."""
    theta_deg = np.degrees(r['theta_true'])
    logits = r['dir_logits']
    pred = r['pred_dir']
    truth = r['truth_dir']

    valid = ~np.isnan(logits[:, 0])
    theta_deg = theta_deg[valid]
    logits = logits[valid]
    pred = pred[valid]
    truth = truth[valid]

    norm = np.linalg.norm(logits, axis=1)
    logit_dir = logits / np.maximum(norm[:, None], 1e-8)

    theta_from_logit = np.degrees(np.arccos(np.clip(logit_dir[:, 2], -1, 1)))
    theta_from_norm = np.degrees(np.arccos(np.clip(pred[:, 2], -1, 1)))
    theta_truth = np.degrees(np.arccos(np.clip(truth[:, 2], -1, 1)))

    # residuals
    res_logit = theta_truth - theta_from_logit
    res_norm = theta_truth - theta_from_norm
    res_diff = res_logit - res_norm  # bias introduced by normalization

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Test 1: Pre-normalization logit analysis', fontsize=14)

    # 1a: ||logits|| vs θ_true
    bins_theta = np.linspace(0, 180, 40)
    centers = 0.5 * (bins_theta[:-1] + bins_theta[1:])
    med_norm, q25_norm, q75_norm = [], [], []
    for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
        m = (theta_deg >= lo) & (theta_deg < hi)
        if m.sum() > 20:
            med_norm.append(np.median(norm[m]))
            q25_norm.append(np.percentile(norm[m], 25))
            q75_norm.append(np.percentile(norm[m], 75))
        else:
            med_norm.append(np.nan)
            q25_norm.append(np.nan)
            q75_norm.append(np.nan)

    ax = axes[0, 0]
    ax.plot(centers, med_norm, 'o-', markersize=3, label='median')
    ax.fill_between(centers, q25_norm, q75_norm, alpha=0.3, label='IQR')
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('||dir_logits||')
    ax.set_title('(a) Logit norm vs θ_true')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1b: Component residuals vs θ_true
    ax = axes[0, 1]
    for ci, (comp, color) in enumerate(zip(['dx', 'dy', 'dz'], ['C0', 'C1', 'C2'])):
        comp_res = truth[:, ci] - pred[:, ci]
        med_comp = []
        for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
            m = (theta_deg >= lo) & (theta_deg < hi)
            med_comp.append(np.median(comp_res[m]) if m.sum() > 20 else np.nan)
        ax.plot(centers, med_comp, 'o-', markersize=3, color=color, label=comp)
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('median(truth - pred)')
    ax.set_title('(b) Component residuals vs θ_true')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1c: θ residual from raw logits vs from F.normalize
    ax = axes[0, 2]
    med_logit, med_fnorm = [], []
    for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
        m = (theta_deg >= lo) & (theta_deg < hi)
        med_logit.append(np.median(res_logit[m]) if m.sum() > 20 else np.nan)
        med_fnorm.append(np.median(res_norm[m]) if m.sum() > 20 else np.nan)
    ax.plot(centers, med_logit, 'o-', markersize=3, label='from raw logits', color='C0')
    ax.plot(centers, med_fnorm, 's-', markersize=3, label='from F.normalize', color='C3')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('median(θ_true - θ_pred) [deg]')
    ax.set_title('(c) θ bias: raw logits vs normalized')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1d: Bias INTRODUCED by normalization
    ax = axes[1, 0]
    med_diff = []
    for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
        m = (theta_deg >= lo) & (theta_deg < hi)
        med_diff.append(np.median(res_diff[m]) if m.sum() > 20 else np.nan)
    ax.plot(centers, med_diff, 'o-', markersize=3, color='C4')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('Δ(median residual) [deg]')
    ax.set_title('(d) Bias from normalization (logit − normalized)')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.grid(True, alpha=0.3)

    # 1e: 2D histogram: ||logits|| vs θ_true
    ax = axes[1, 1]
    ok = norm < np.percentile(norm, 99.5)
    ax.hist2d(theta_deg[ok], norm[ok],
              bins=[np.linspace(0, 180, 60), np.linspace(0, np.percentile(norm[ok], 99), 60)],
              cmap='viridis', norm=LogNorm())
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('||dir_logits||')
    ax.set_title('(e) Logit norm distribution')
    ax.axvline(90, color='w', ls=':', lw=0.8)
    ax.axvline(120, color='w', ls='--', lw=0.8)

    # 1f: dz component of logits vs θ_true
    ax = axes[1, 2]
    ax.hist2d(theta_deg, logits[:, 2],
              bins=[np.linspace(0, 180, 60), np.linspace(-np.percentile(np.abs(logits[:, 2]), 99),
                                                          np.percentile(np.abs(logits[:, 2]), 99), 60)],
              cmap='viridis', norm=LogNorm())
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('dir_logits[z]')
    ax.set_title('(f) Raw z-logit vs θ_true')
    ax.axvline(90, color='w', ls=':', lw=0.8)
    ax.axvline(120, color='w', ls='--', lw=0.8)

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test1_logit_analysis.png'), dpi=150)
    plt.close()
    print("  wrote test1_logit_analysis.png")


# =========================================================================
# Test 2: Acceptance with truth vs reco direction
# =========================================================================

def plot_acceptance_feedback(r, output_dir):
    """Compare acceptance using reco direction vs truth direction at 120° cut."""
    if 'truth_acceptance' not in r:
        print("  [skip] test 2: no truth_acceptance in data")
        return

    theta_deg = np.degrees(r['theta_true'])
    pred = r['pred_dir']
    truth = r['truth_dir']
    truth_accept = r['truth_acceptance']

    theta_pred = np.degrees(np.arccos(np.clip(pred[:, 2], -1, 1)))
    theta_truth_from_vec = np.degrees(np.arccos(np.clip(truth[:, 2], -1, 1)))

    cut = 120.0
    reco_accept = (theta_pred < cut).astype(int)
    truth_dir_accept = (theta_truth_from_vec < cut).astype(int)

    bins = np.linspace(0, 180, 40)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Test 2: Acceptance bias from direction reconstruction', fontsize=14)

    # 2a: Acceptance efficiency vs θ_true (events that should be accepted)
    ax = axes[0]
    should_accept = truth_accept == 1
    eff_reco, eff_truth_dir = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (theta_deg >= lo) & (theta_deg < hi) & should_accept
        if m.sum() > 10:
            eff_reco.append(reco_accept[m].mean())
            eff_truth_dir.append(truth_dir_accept[m].mean())
        else:
            eff_reco.append(np.nan)
            eff_truth_dir.append(np.nan)
    ax.plot(centers, eff_reco, 'o-', markersize=3, label='reco direction', color='C3')
    ax.plot(centers, eff_truth_dir, 's-', markersize=3, label='truth direction', color='C0')
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('acceptance efficiency')
    ax.set_title('(a) Efficiency for true-accept events')
    ax.axvline(cut, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2b: Acceptance difference (reco - truth)
    ax = axes[1]
    diff_eff = [r - t if not (np.isnan(r) or np.isnan(t)) else np.nan
                for r, t in zip(eff_reco, eff_truth_dir)]
    ax.plot(centers, diff_eff, 'o-', markersize=4, color='C4')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('Δ(efficiency)')
    ax.set_title('(b) Efficiency difference (reco − truth dir)')
    ax.axvline(cut, color='gray', ls='--', lw=0.8)
    ax.grid(True, alpha=0.3)

    # 2c: Net migration across the cut
    ax = axes[2]
    near_cut = (theta_deg > cut - 15) & (theta_deg < cut + 15)
    reco_pass = reco_accept[near_cut]
    truth_pass = truth_dir_accept[near_cut]
    gained = ((reco_pass == 1) & (truth_pass == 0)).sum()
    lost = ((reco_pass == 0) & (truth_pass == 1)).sum()
    ax.bar(['gained\n(reco yes, truth no)', 'lost\n(reco no, truth yes)'],
           [gained, lost], color=['C2', 'C3'])
    ax.set_ylabel('events')
    ax.set_title(f'(c) Migration across {cut}° cut\n(θ_true ∈ [{cut-15}°, {cut+15}°])')
    ax.grid(True, alpha=0.3, axis='y')
    net = gained - lost
    total = near_cut.sum()
    ax.text(0.5, 0.95, f'net migration: {net:+d} ({net/max(total,1):.3%} of {total})',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test2_acceptance_feedback.png'), dpi=150)
    plt.close()
    print("  wrote test2_acceptance_feedback.png")


# =========================================================================
# Test 3: Loss gradient geometry (analytical)
# =========================================================================

def plot_loss_geometry(output_dir):
    """Show gradient magnitude of different loss functions vs angle error."""
    delta = np.linspace(0.01, 30, 500)  # angle error in degrees
    delta_rad = np.radians(delta)

    # Cosine loss: L = 1 - cos(Δ), dL/dΔ = sin(Δ)
    grad_cosine = np.sin(delta_rad)

    # Angular MSE: L = Δ², dL/dΔ = 2Δ
    grad_angular_mse = 2 * delta_rad

    # Angular MAE: L = Δ, dL/dΔ = 1
    grad_angular_mae = np.ones_like(delta_rad)

    # Huber-like on angle: L = Δ² for Δ < δ₀, else δ₀(2Δ - δ₀)
    delta0 = np.radians(5)
    grad_huber = np.where(delta_rad < delta0, 2 * delta_rad, 2 * delta0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Test 3: Loss function gradient geometry', fontsize=14)

    ax = axes[0]
    ax.plot(delta, grad_cosine, label='cosine: sin(Δ)', lw=2)
    ax.plot(delta, grad_angular_mse, label='angular MSE: 2Δ', lw=2)
    ax.plot(delta, grad_angular_mae, label='angular MAE: 1', lw=2)
    ax.plot(delta, grad_huber, label='angular Huber (5°)', lw=2, ls='--')
    ax.set_xlabel('angle error Δ [deg]')
    ax.set_ylabel('|dL/dΔ|')
    ax.set_title('(a) Gradient magnitude vs error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Gradient relative to cosine at small errors
    ax = axes[1]
    ratio_mse = grad_angular_mse / np.maximum(grad_cosine, 1e-10)
    ax.plot(delta, ratio_mse, label='angular MSE / cosine', lw=2)
    ax.set_xlabel('angle error Δ [deg]')
    ax.set_ylabel('gradient ratio')
    ax.set_title('(b) Angular MSE pushes harder at small errors')
    ax.set_ylim(0, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Effective learning rate: how much does a 1° error move the prediction?
    # For cosine loss on unit vectors: the gradient on the logits depends
    # on both the angle error AND the logit norm.
    # dL/d(logit_z) = -(truth_z - cos(Δ) * pred_z) / ||logits||
    # At small errors, this ≈ -Δ * sin(θ_true) / ||logits|| for z-component
    ax = axes[2]
    theta_true_arr = np.linspace(5, 175, 100)
    theta_rad = np.radians(theta_true_arr)
    effective_grad_z = np.abs(np.sin(theta_rad))
    ax.plot(theta_true_arr, effective_grad_z, lw=2, color='C2')
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('|∂L/∂(logit_z)| (relative)')
    ax.set_title('(c) z-logit gradient sensitivity vs θ_true\n'
                 '(at fixed small error, unit norm)')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test3_loss_geometry.png'), dpi=150)
    plt.close()
    print("  wrote test3_loss_geometry.png")


# =========================================================================
# Test 4: Hit selection bias
# =========================================================================

def plot_hit_selection(r, output_dir):
    """Positron-tagged hit counts per view vs θ_true."""
    theta_deg = np.degrees(r['theta_true'])
    n_x = r['n_hits_x']
    n_y = r['n_hits_y']

    valid = ~np.isnan(n_x)
    theta_deg = theta_deg[valid]
    n_x = n_x[valid]
    n_y = n_y[valid]

    bins = np.linspace(0, 180, 40)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Test 4: Positron hit selection vs θ_true', fontsize=14)

    for ai, (n_hits, label) in enumerate([(n_x, 'x-view'), (n_y, 'y-view'),
                                           (n_x + n_y, 'total')]):
        ax = axes[ai]
        med, q25, q75 = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (theta_deg >= lo) & (theta_deg < hi)
            if m.sum() > 20:
                med.append(np.median(n_hits[m]))
                q25.append(np.percentile(n_hits[m], 25))
                q75.append(np.percentile(n_hits[m], 75))
            else:
                med.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)
        ax.plot(centers, med, 'o-', markersize=3, label='median')
        ax.fill_between(centers, q25, q75, alpha=0.3, label='IQR')
        ax.set_xlabel('θ_true [deg]')
        ax.set_ylabel(f'# positron hits ({label})')
        ax.set_title(f'({chr(97+ai)}) {label} positron hits')
        ax.axvline(90, color='gray', ls=':', lw=0.8)
        ax.axvline(120, color='gray', ls='--', lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test4_hit_selection.png'), dpi=150)
    plt.close()
    print("  wrote test4_hit_selection.png")


# =========================================================================
# Test 5: Training distribution
# =========================================================================

def plot_training_distribution(r, output_dir):
    """θ_true distribution and implied loss weighting."""
    theta_deg = np.degrees(r['theta_true'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Test 5: Angular distribution and implied loss weighting', fontsize=14)

    # 5a: raw distribution
    ax = axes[0]
    ax.hist(theta_deg, bins=np.linspace(0, 180, 60), alpha=0.7, density=True)
    # overlay sin(θ) for isotropic
    th = np.linspace(0, 180, 200)
    ax.plot(th, np.sin(np.radians(th)) / 2 * (np.pi / 180), 'r-', lw=2,
            label='isotropic sin(θ)/2')
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('density')
    ax.set_title('(a) θ_true distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5b: ratio to isotropic
    ax = axes[1]
    bins = np.linspace(0, 180, 40)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(theta_deg, bins=bins)
    isotropic = np.sin(np.radians(centers)) / 2 * (np.pi / 180) * len(theta_deg) * (bins[1] - bins[0])
    ratio = counts / np.maximum(isotropic, 1)
    ax.plot(centers, ratio, 'o-', markersize=3)
    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('data / isotropic')
    ax.set_title('(b) Ratio to isotropic distribution')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.grid(True, alpha=0.3)

    # 5c: effective gradient pressure = (events in bin) × (gradient sensitivity)
    # cosine loss z-gradient ∝ sin(θ) × density(θ)
    ax = axes[2]
    density = counts / (len(theta_deg) * (bins[1] - bins[0]))
    z_sensitivity = np.abs(np.sin(np.radians(centers)))
    pressure = density * z_sensitivity
    pressure_flat = density * 1.0  # angular MSE would have flat sensitivity
    ax.plot(centers, pressure / pressure.max(), 'o-', markersize=3,
            label='cosine loss', color='C0')
    ax.plot(centers, pressure_flat / pressure_flat.max(), 's-', markersize=3,
            label='angular MSE (flat grad)', color='C3')
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('relative learning pressure')
    ax.set_title('(c) Effective learning pressure on z-component')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test5_distribution.png'), dpi=150)
    plt.close()
    print("  wrote test5_distribution.png")


# =========================================================================
# Test 6: Attention weight analysis
# =========================================================================

def plot_attention_analysis(r, output_dir):
    """Analyze how the direction pooling attention varies with θ_true."""
    theta_deg = np.degrees(r['theta_true'])
    pred = r['pred_dir']
    truth = r['truth_dir']
    attn_entropy = r['attn_entropy']     # [N, 2] for x/y view
    attn_max = r['attn_max']             # [N, 2]
    attn_spatial = r['attn_spatial']     # [N, 2, 3] attn-weighted mean pos

    valid = ~np.isnan(attn_entropy[:, 0])
    theta_deg = theta_deg[valid]
    pred = pred[valid]
    truth = truth[valid]
    attn_entropy = attn_entropy[valid]
    attn_max = attn_max[valid]
    attn_spatial = attn_spatial[valid]

    theta_pred = np.degrees(np.arccos(np.clip(pred[:, 2], -1, 1)))
    theta_truth = np.degrees(np.arccos(np.clip(truth[:, 2], -1, 1)))
    angle_residual = theta_truth - theta_pred

    bins = np.linspace(0, 180, 40)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def binned_median(x, y):
        med = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (x >= lo) & (x < hi)
            med.append(np.median(y[m]) if m.sum() > 20 else np.nan)
        return np.array(med)

    def binned_percentile(x, y, pct):
        vals = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (x >= lo) & (x < hi)
            vals.append(np.percentile(y[m], pct) if m.sum() > 20 else np.nan)
        return np.array(vals)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Test 6: Direction pooling attention weights vs θ_true', fontsize=14)

    view_labels = ['x-view', 'y-view']
    view_colors = ['C0', 'C1']

    # Row 1: Attention entropy vs θ_true (measures concentration)
    ax = axes[0, 0]
    for vi, (vl, vc) in enumerate(zip(view_labels, view_colors)):
        med = binned_median(theta_deg, attn_entropy[:, vi])
        ax.plot(centers, med, 'o-', markersize=3, color=vc, label=vl)
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('attention entropy [nats]')
    ax.set_title('(a) Entropy of attention weights\n(low = concentrated on few hits)')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1: Max attention weight vs θ_true
    ax = axes[0, 1]
    for vi, (vl, vc) in enumerate(zip(view_labels, view_colors)):
        med = binned_median(theta_deg, attn_max[:, vi])
        ax.plot(centers, med, 'o-', markersize=3, color=vc, label=vl)
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('max attention weight')
    ax.set_title('(b) Max attention weight\n(high = one hit dominates)')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1: Correlation between entropy and angle residual
    ax = axes[0, 2]
    mean_entropy = attn_entropy.mean(axis=1)
    ax.hist2d(mean_entropy, angle_residual,
              bins=[np.linspace(0, np.percentile(mean_entropy, 99), 50),
                    np.linspace(-5, 5, 50)],
              cmap='viridis', norm=LogNorm())
    ax.set_xlabel('mean attention entropy')
    ax.set_ylabel('angle residual [deg]')
    ax.set_title('(c) Does attention concentration predict bias?')
    ax.axhline(0, color='w', ls='--', lw=0.8)

    # Row 2: Attention-weighted centroid z vs θ_true
    # This shows WHERE the attention focuses along z
    for vi, (vl, vc) in enumerate(zip(view_labels, view_colors)):
        ax = axes[1, vi]
        z_centroid = attn_spatial[:, vi, 2]  # z-component
        med = binned_median(theta_deg, z_centroid)
        q25 = binned_percentile(theta_deg, z_centroid, 25)
        q75 = binned_percentile(theta_deg, z_centroid, 75)
        ax.plot(centers, med, 'o-', markersize=3, color=vc, label='median')
        ax.fill_between(centers, q25, q75, alpha=0.3, color=vc, label='IQR')
        ax.set_xlabel('θ_true [deg]')
        ax.set_ylabel('attn-weighted centroid z [norm]')
        ax.set_title(f'({"d" if vi == 0 else "e"}) Where attention focuses in z ({vl})')
        ax.axvline(90, color='gray', ls=':', lw=0.8)
        ax.axvline(120, color='gray', ls='--', lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: z-centroid vs angle residual (does spatial attention bias predict direction bias?)
    ax = axes[1, 2]
    mean_z = attn_spatial[:, :, 2].mean(axis=1)
    ax.hist2d(mean_z, angle_residual,
              bins=[np.linspace(np.percentile(mean_z, 1), np.percentile(mean_z, 99), 50),
                    np.linspace(-5, 5, 50)],
              cmap='viridis', norm=LogNorm())
    ax.set_xlabel('mean attn-weighted z centroid')
    ax.set_ylabel('angle residual [deg]')
    ax.set_title('(f) z-centroid vs direction bias')
    ax.axhline(0, color='w', ls='--', lw=0.8)

    # Row 3: Attention-weighted centroid in transverse plane
    for vi, (vl, vc) in enumerate(zip(view_labels, view_colors)):
        ax = axes[2, vi]
        # transverse component: x for x-view, y for y-view
        trans_centroid = attn_spatial[:, vi, vi]  # x for view 0, y for view 1
        med = binned_median(theta_deg, trans_centroid)
        q25 = binned_percentile(theta_deg, trans_centroid, 25)
        q75 = binned_percentile(theta_deg, trans_centroid, 75)
        ax.plot(centers, med, 'o-', markersize=3, color=vc, label='median')
        ax.fill_between(centers, q25, q75, alpha=0.3, color=vc, label='IQR')
        ax.set_xlabel('θ_true [deg]')
        ax.set_ylabel(f'attn-weighted centroid {"x" if vi == 0 else "y"} [norm]')
        ax.set_title(f'({"g" if vi == 0 else "h"}) Transverse centroid ({vl})')
        ax.axvline(90, color='gray', ls=':', lw=0.8)
        ax.axvline(120, color='gray', ls='--', lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 3: Asymmetry between x-view and y-view attention
    ax = axes[2, 2]
    asym = attn_entropy[:, 0] - attn_entropy[:, 1]  # x entropy - y entropy
    med = binned_median(theta_deg, asym)
    ax.plot(centers, med, 'o-', markersize=4, color='C4')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('entropy(x) − entropy(y)')
    ax.set_title('(i) View asymmetry in attention\n(>0 = x-view more diffuse)')
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test6_attention_analysis.png'), dpi=150)
    plt.close()
    print("  wrote test6_attention_analysis.png")


# =========================================================================
# Test 7: Causal test — control for θ_true, check residual predictors
# =========================================================================

def plot_causal_test(r, output_dir):
    """Within narrow θ_true bins, does z-centroid / entropy / hit count
    predict the direction residual? If yes → causal. If no → confounded."""
    theta_deg = np.degrees(r['theta_true'])
    pred = r['pred_dir']
    truth = r['truth_dir']
    attn_entropy = r['attn_entropy']
    attn_spatial = r['attn_spatial']
    n_hits_x = r['n_hits_x']
    n_hits_y = r['n_hits_y']

    valid = (~np.isnan(attn_entropy[:, 0])) & (~np.isnan(n_hits_x))
    theta_deg = theta_deg[valid]
    pred = pred[valid]
    truth = truth[valid]
    attn_entropy = attn_entropy[valid]
    attn_spatial = attn_spatial[valid]
    n_hits = n_hits_x[valid] + n_hits_y[valid]

    theta_pred = np.degrees(np.arccos(np.clip(pred[:, 2], -1, 1)))
    theta_truth = np.degrees(np.arccos(np.clip(truth[:, 2], -1, 1)))
    angle_residual = theta_truth - theta_pred

    mean_z_centroid = attn_spatial[:, :, 2].mean(axis=1)
    mean_entropy = attn_entropy.mean(axis=1)

    # Define θ bins for conditioning
    theta_slices = [(10, 30), (30, 50), (50, 70), (70, 90),
                    (90, 110), (110, 130), (130, 150), (150, 170)]

    features = [
        ('z-centroid', mean_z_centroid),
        ('attention entropy', mean_entropy),
        ('total positron hits', n_hits),
    ]

    fig, axes = plt.subplots(len(features), len(theta_slices),
                             figsize=(4 * len(theta_slices), 4 * len(features)))
    fig.suptitle('Test 7: Causal test — does feature predict residual WITHIN θ bins?\n'
                 '(correlation here = causal, not confounded by θ)', fontsize=14)

    for fi, (feat_name, feat_vals) in enumerate(features):
        for ti, (t_lo, t_hi) in enumerate(theta_slices):
            ax = axes[fi, ti]
            m = (theta_deg >= t_lo) & (theta_deg < t_hi)
            if m.sum() < 100:
                ax.text(0.5, 0.5, f'n={m.sum()}\ntoo few',
                        transform=ax.transAxes, ha='center', va='center')
                continue

            fv = feat_vals[m]
            ar = angle_residual[m]

            # bin the feature into quintiles and compute median residual
            try:
                pcts = np.percentile(fv, np.linspace(0, 100, 11))
                pcts = np.unique(pcts)
                if len(pcts) < 3:
                    ax.text(0.5, 0.5, 'no variation',
                            transform=ax.transAxes, ha='center', va='center')
                    continue
                bin_centers = 0.5 * (pcts[:-1] + pcts[1:])
                med_res = []
                for lo, hi in zip(pcts[:-1], pcts[1:]):
                    sel = (fv >= lo) & (fv < hi)
                    med_res.append(np.median(ar[sel]) if sel.sum() > 10 else np.nan)
                med_res = np.array(med_res)

                ax.plot(bin_centers, med_res, 'o-', markersize=4, color='C0')
                ax.axhline(0, color='k', ls='--', lw=0.8)

                # linear regression for slope
                ok = ~np.isnan(med_res)
                if ok.sum() >= 3:
                    slope = np.polyfit(bin_centers[ok], med_res[ok], 1)[0]
                    corr = np.corrcoef(fv, ar)[0, 1]
                    ax.set_title(f'θ∈[{t_lo}°,{t_hi}°] n={m.sum()}\n'
                                 f'r={corr:.3f} slope={slope:.3f}', fontsize=8)
                else:
                    ax.set_title(f'θ∈[{t_lo}°,{t_hi}°] n={m.sum()}', fontsize=8)
            except Exception:
                ax.set_title(f'θ∈[{t_lo}°,{t_hi}°] n={m.sum()}', fontsize=8)

            if ti == 0:
                ax.set_ylabel(f'median residual [deg]\n({feat_name})')
            if fi == len(features) - 1:
                ax.set_xlabel(feat_name)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test7_causal_test.png'), dpi=150)
    plt.close()
    print("  wrote test7_causal_test.png")


# =========================================================================
# Summary: all biases on one plot
# =========================================================================

def plot_summary(r, output_dir):
    """Single summary: normalized vs raw logit bias, component breakdown."""
    theta_deg = np.degrees(r['theta_true'])
    logits = r['dir_logits']
    pred = r['pred_dir']
    truth = r['truth_dir']

    valid = ~np.isnan(logits[:, 0])
    theta_deg = theta_deg[valid]
    logits = logits[valid]
    pred = pred[valid]
    truth = truth[valid]
    norm = np.linalg.norm(logits, axis=1)

    logit_dir = logits / np.maximum(norm[:, None], 1e-8)
    theta_from_logit = np.degrees(np.arccos(np.clip(logit_dir[:, 2], -1, 1)))
    theta_from_norm = np.degrees(np.arccos(np.clip(pred[:, 2], -1, 1)))
    theta_truth_deg = np.degrees(np.arccos(np.clip(truth[:, 2], -1, 1)))

    res_logit = theta_truth_deg - theta_from_logit
    res_norm = theta_truth_deg - theta_from_norm

    bins = np.linspace(0, 180, 40)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Summary: Where does the S-shape come from?', fontsize=14)

    for res, label, color, marker in [
        (res_logit, 'bias in raw logits (before normalize)', 'C0', 'o'),
        (res_norm, 'bias after F.normalize', 'C3', 's'),
    ]:
        med = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (theta_deg >= lo) & (theta_deg < hi)
            med.append(np.median(res[m]) if m.sum() > 20 else np.nan)
        ax.plot(centers, med, f'{marker}-', markersize=4, label=label, color=color)

    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.axvline(90, color='gray', ls=':', lw=0.8)
    ax.axvline(120, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('θ_true [deg]')
    ax.set_ylabel('median(θ_true − θ_pred) [deg]')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('If curves overlap → bias is in the logits, not from normalization.\n'
                 'If they differ → F.normalize introduces/changes the bias.')

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_bias_source.png'), dpi=150)
    plt.close()
    print("  wrote summary_bias_source.png")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--eval_path', required=True)
    parser.add_argument('--output_dir', default='./direction_bias_diagnostics')
    parser.add_argument('--max_events', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=50)
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # Load model
    model = PURITYHybridModelV2().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"loaded checkpoint: {args.checkpoint} "
          f"(epoch {ckpt.get('epoch', '?')})")

    # Run inference with hooks
    print("\n=== Running hooked inference ===")
    r = run_hooked_inference(model, device, args.eval_path,
                             max_events=args.max_events)
    print(f"  collected {len(r['theta_true'])} events")

    # Generate all diagnostic plots
    print("\n=== Test 1: Pre-normalization logit analysis ===")
    plot_logit_analysis(r, args.output_dir)

    print("\n=== Test 2: Acceptance feedback ===")
    plot_acceptance_feedback(r, args.output_dir)

    print("\n=== Test 3: Loss gradient geometry ===")
    plot_loss_geometry(args.output_dir)

    print("\n=== Test 4: Hit selection bias ===")
    plot_hit_selection(r, args.output_dir)

    print("\n=== Test 5: Training distribution ===")
    plot_training_distribution(r, args.output_dir)

    print("\n=== Test 6: Attention weight analysis ===")
    plot_attention_analysis(r, args.output_dir)

    print("\n=== Test 7: Causal test (control for θ) ===")
    plot_causal_test(r, args.output_dir)

    print("\n=== Summary ===")
    plot_summary(r, args.output_dir)

    print(f"\nAll plots written to {args.output_dir}/")


if __name__ == '__main__':
    main()
