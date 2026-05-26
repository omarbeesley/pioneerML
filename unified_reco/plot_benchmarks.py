"""
Benchmark Plotting Script for PURITY Model

Reads the parquet files produced by benchmark.py and generates diagnostic plots.
No PyTorch required — runs outside the container on the GPU node.

Usage:
    python plot_benchmarks.py --results_dir ../benchmark_results/ --output_dir ../benchmark_plots/

Expected input files in results_dir:
    pie_eval_events.parquet   pimu_eval_events.parquet
    pie_eval_slices.parquet   pimu_eval_slices.parquet
    pie_eval_lyso_hits.parquet  pimu_eval_lyso_hits.parquet  (optional)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

SENTINEL = -999.0


# ========================================================================== #
#  Utilities                                                                  #
# ========================================================================== #

def watermark(ax=None):
    """Add a small PURITY watermark."""
    if ax is None:
        ax = plt.gca()
    ax.text(0.98, 0.02, 'PURITY', transform=ax.transAxes,
            fontsize=7, color='gray', alpha=0.5, ha='right', va='bottom')


def load_dataset(results_dir, tag):
    """Load event, slice, and (optionally) LYSO parquets for a given tag."""
    events_path = os.path.join(results_dir, f'{tag}_events.parquet')
    slices_path = os.path.join(results_dir, f'{tag}_slices.parquet')
    lyso_path = os.path.join(results_dir, f'{tag}_lyso_hits.parquet')

    events = pd.read_parquet(events_path) if os.path.exists(events_path) else None
    slices = pd.read_parquet(slices_path) if os.path.exists(slices_path) else None
    lyso = pd.read_parquet(lyso_path) if os.path.exists(lyso_path) else None

    if events is not None:
        print(f"  {tag} events: {len(events)}")
    if slices is not None:
        print(f"  {tag} slices: {len(slices)}")
    if lyso is not None:
        print(f"  {tag} LYSO hits: {len(lyso)}")

    return events, slices, lyso


# ========================================================================== #
#  1. Acceptance Confusion Matrix                                             #
# ========================================================================== #

def plot_acceptance_confusion(datasets, output_dir):
    """2x1 acceptance confusion matrices (pie, pimu)."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, (tag, raw_tag, ev, color) in zip(axes, datasets):
        if ev is None:
            ax.set_visible(False)
            continue

        t = ev['truth_acceptance'].values == 1
        p = (ev['pred_accepted'].values != SENTINEL) & (ev['pred_accepted'].values >= 0.5)

        tp = int((t & p).sum())
        fn = int((t & ~p).sum())
        fp = int((~t & p).sum())
        tn = int((~t & ~p).sum())
        cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

        eff = tp / max(tp + fn, 1)
        fake = fp / max(fp + tn, 1)
        pur = tp / max(tp + fp, 1)

        im = ax.imshow(cm, cmap=color, aspect='equal')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['reject', 'accept'])
        ax.set_yticklabels(['reject', 'accept'])
        ax.set_xlabel('predicted'); ax.set_ylabel('truth')
        vmax = cm.max()
        for i in range(2):
            for j in range(2):
                txt_color = 'white' if cm[i, j] > 0.6 * vmax else 'black'
                ax.text(j, i, f'{cm[i, j]:,}',
                        ha='center', va='center', color=txt_color, fontsize=14)

        ax.set_title(f'{tag}\n'
                     rf'$\epsilon={eff:.3f}\ \ f={fake:.4f}\ \ P={pur:.3f}$',
                     fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        print(f"  {tag}: TN={tn}  FP={fp}  FN={fn}  TP={tp}  "
              f"eff={eff:.4f}  fake={fake:.5f}  purity={pur:.4f}")

    fig.tight_layout()
    path = os.path.join(output_dir, 'acceptance_confusion.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


# ========================================================================== #
#  1b. Acceptance False-Positive Breakdown                                     #
# ========================================================================== #

def plot_acceptance_fp_breakdown(datasets, output_dir):
    """For each dataset, break down false positives (truth=reject, pred=accept)
    by which truth criterion caused the rejection:
      - pion out of fiducial (z or xy)
      - angle > 120°
      - positron not in window (no trigger positron)
    Events can fail multiple criteria; shown as a stacked bar + Venn-style table.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, (tag, raw_tag, ev, color) in zip(axes, datasets):
        if ev is None:
            ax.set_visible(False)
            continue

        # Diagnostic: truth theta distribution and scatter rates around 120° cut
        theta_deg = np.degrees(ev['truth_theta'].values)
        theta_reco_deg = np.degrees(np.arccos(np.clip(ev['pred_positron_dir_z'].values, -1, 1)))
        print(f"    {tag} truth theta counts around 120°:")
        for lo, hi in [(105, 110), (110, 115), (115, 120),
                        (120, 125), (125, 130), (130, 135)]:
            n = int(((theta_deg > lo) & (theta_deg <= hi)).sum())
            print(f"      [{lo}°, {hi}°]: {n:,}")

        # Direct scatter-across-120° analysis (angle only, ignoring acceptance)
        valid_reco = ev['pred_positron_dir_z'].values != SENTINEL
        htp_ok = (ev['truth_htp'].values == 1) if 'truth_htp' in ev.columns else np.ones(len(ev), dtype=bool)
        reco_mask = valid_reco & htp_ok

        for width in [2, 5, 10]:
            below_band = reco_mask & (theta_deg > 120 - width) & (theta_deg <= 120)
            above_band = reco_mask & (theta_deg > 120) & (theta_deg <= 120 + width)
            scatter_out = int((below_band & (theta_reco_deg > 120)).sum())  # truth<120, reco>120
            scatter_in = int((above_band & (theta_reco_deg <= 120)).sum())  # truth>120, reco<120
            n_below = int(below_band.sum())
            n_above = int(above_band.sum())
            print(f"      ±{width}°: below={n_below:,} scatter_out={scatter_out:,} ({100*scatter_out/max(n_below,1):.1f}%) | "
                  f"above={n_above:,} scatter_in={scatter_in:,} ({100*scatter_in/max(n_above,1):.1f}%) | "
                  f"ratio(in/out)={scatter_in/max(scatter_out,1):.2f}")

        t_acc = ev['truth_acceptance'].values == 1
        p_acc = (ev['pred_accepted'].values != SENTINEL) & (ev['pred_accepted'].values >= 0.5)

        # False positives: truth=reject, pred=accept
        fp_mask = (~t_acc) & p_acc
        n_fp = int(fp_mask.sum())

        if n_fp == 0:
            ax.text(0.5, 0.5, 'No false positives', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_title(tag)
            continue

        # Decompose truth rejection reasons on the FP events
        pz = ev['truth_pion_stop_z'].values[fp_mask]
        px = ev['truth_pion_stop_x'].values[fp_mask]
        py = ev['truth_pion_stop_y'].values[fp_mask]
        theta = ev['truth_theta'].values[fp_mask]
        htp = ev['truth_htp'].values[fp_mask] if 'truth_htp' in ev.columns else np.ones(n_fp)

        fail_fid_z = ~((pz > 1.2) & (pz < 4.8))
        fail_fid_xy = ~((np.abs(px) < 8.0) & (np.abs(py) < 8.0))
        fail_fiducial = fail_fid_z | fail_fid_xy
        fail_angle = np.degrees(theta) >= 120.0
        fail_htp = htp < 0.5

        # Count each failure mode (events can fail multiple)
        categories = {
            'Fiducial (z)': fail_fid_z,
            'Fiducial (xy)': fail_fid_xy,
            'Angle > 120°': fail_angle,
            'No positron\nin window': fail_htp,
        }

        labels = list(categories.keys())
        counts = [int(v.sum()) for v in categories.values()]
        fracs = [c / n_fp * 100 for c in counts]

        bars = ax.barh(labels, fracs, color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'])
        ax.set_xlabel('% of false positives')
        ax.set_xlim(0, max(fracs) * 1.3 if max(fracs) > 0 else 100)

        for bar, count, frac in zip(bars, counts, fracs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{count:,} ({frac:.1f}%)', va='center', fontsize=10)

        # Also show how many fail ONLY one criterion
        only_fid = fail_fiducial & ~fail_angle & ~fail_htp
        only_angle = ~fail_fiducial & fail_angle & ~fail_htp
        only_htp = ~fail_fiducial & ~fail_angle & fail_htp
        multi = (fail_fiducial.astype(int) + fail_angle.astype(int) + fail_htp.astype(int)) > 1

        summary = (f'Total FP: {n_fp:,}\n'
                   f'Only fiducial: {int(only_fid.sum()):,}\n'
                   f'Only angle: {int(only_angle.sum()):,}\n'
                   f'Only no-positron: {int(only_htp.sum()):,}\n'
                   f'Multiple: {int(multi.sum()):,}')
        ax.text(0.95, 0.05, summary, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

        ax.set_title(f'{tag}\nFalse positive breakdown (n={n_fp:,})', fontsize=12)

    fig.tight_layout()
    path = os.path.join(output_dir, 'acceptance_fp_breakdown.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")

    # --- Detailed diagnostic for angle-only FP events ---
    for tag, raw_tag, ev, color in datasets:
        if ev is None:
            continue

        t_acc = ev['truth_acceptance'].values == 1
        p_acc = (ev['pred_accepted'].values != SENTINEL) & (ev['pred_accepted'].values >= 0.5)
        fp_mask = (~t_acc) & p_acc

        theta_truth = ev['truth_theta'].values
        theta_deg = np.degrees(theta_truth)
        pz = ev['truth_pion_stop_z'].values
        px = ev['truth_pion_stop_x'].values
        py = ev['truth_pion_stop_y'].values
        htp = ev['truth_htp'].values if 'truth_htp' in ev.columns else np.ones(len(ev))

        fail_fid = ~((pz > 1.2) & (pz < 4.8) & (np.abs(px) < 8.0) & (np.abs(py) < 8.0))
        fail_angle = theta_deg >= 120.0
        fail_htp = htp < 0.5
        only_angle_fp = fp_mask & (~fail_fid) & fail_angle & (~fail_htp)
        n_oa = int(only_angle_fp.sum())

        if n_oa < 10:
            continue

        theta_true_fp = theta_deg[only_angle_fp]
        theta_reco_fp = np.degrees(np.arccos(np.clip(
            ev['pred_positron_dir_z'].values[only_angle_fp], -1, 1)))
        residual_fp = theta_true_fp - theta_reco_fp

        # Also get pion stop errors for these events
        ps_err_x = ev['pred_pion_stop_x'].values[only_angle_fp] - ev['truth_pion_stop_x'].values[only_angle_fp]
        ps_err_y = ev['pred_pion_stop_y'].values[only_angle_fp] - ev['truth_pion_stop_y'].values[only_angle_fp]
        ps_err_z = ev['pred_pion_stop_z'].values[only_angle_fp] - ev['truth_pion_stop_z'].values[only_angle_fp]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # Row 1: angle diagnostics
        axes[0, 0].hist(theta_true_fp, bins=60, range=(120, 180), color='tab:blue', alpha=0.7)
        axes[0, 0].set_xlabel(r'$\theta_{\mathrm{True}}$ [deg]')
        axes[0, 0].set_title(f'Truth θ of angle-only FPs')
        axes[0, 0].axvline(np.median(theta_true_fp), color='red', ls='--',
                           label=f'median={np.median(theta_true_fp):.1f}°')
        axes[0, 0].legend(fontsize=9)

        axes[0, 1].hist(theta_reco_fp, bins=60, range=(0, 180), color='tab:orange', alpha=0.7)
        axes[0, 1].set_xlabel(r'$\theta_{\mathrm{Reco}}$ [deg]')
        axes[0, 1].set_title(f'Reco θ of angle-only FPs')
        axes[0, 1].axvline(120, color='red', ls=':', label='120° cut')
        axes[0, 1].axvline(np.median(theta_reco_fp), color='blue', ls='--',
                           label=f'median={np.median(theta_reco_fp):.1f}°')
        axes[0, 1].legend(fontsize=9)

        axes[0, 2].hist(residual_fp, bins=60, range=(-30, 60), color='tab:green', alpha=0.7)
        axes[0, 2].set_xlabel(r'$\theta_{\mathrm{True}} - \theta_{\mathrm{Reco}}$ [deg]')
        axes[0, 2].set_title(f'Residual (positive = pred too forward)')
        axes[0, 2].axvline(0, color='k', ls='-', lw=0.8)
        axes[0, 2].axvline(np.mean(residual_fp), color='red', ls='--',
                           label=f'mean={np.mean(residual_fp):.1f}°')
        axes[0, 2].legend(fontsize=9)

        # Row 2: 2D truth vs reco angle, pion stop error, IoU
        h = axes[1, 0].hist2d(theta_true_fp, theta_reco_fp,
                              bins=[40, 40], range=[[120, 180], [0, 180]], cmin=1)
        axes[1, 0].plot([120, 180], [120, 180], 'r--', lw=0.8)
        axes[1, 0].axhline(120, color='white', ls=':', lw=0.8)
        axes[1, 0].set_xlabel(r'$\theta_{\mathrm{True}}$ [deg]')
        axes[1, 0].set_ylabel(r'$\theta_{\mathrm{Reco}}$ [deg]')
        axes[1, 0].set_title('Truth vs Reco (angle-only FPs)')
        fig.colorbar(h[3], ax=axes[1, 0])

        ps_err_3d = np.sqrt(ps_err_x**2 + ps_err_y**2 + ps_err_z**2)
        axes[1, 1].hist(ps_err_3d, bins=50, range=(0, 5), color='tab:purple', alpha=0.7)
        axes[1, 1].set_xlabel('Pion stop 3D error [mm]')
        axes[1, 1].set_title(f'Pion stop error (median={np.median(ps_err_3d):.2f} mm)')

        iou = ev['pred_pos_iou'].values[only_angle_fp]
        axes[1, 2].hist(iou[np.isfinite(iou)], bins=50, range=(0, 1),
                        color='tab:red', alpha=0.7)
        axes[1, 2].set_xlabel('Positron IoU')
        axes[1, 2].set_title(f'Hit IoU (median={np.nanmedian(iou):.3f})')

        fig.suptitle(f'{tag}: Angle-only FP diagnostics (n={n_oa:,})', fontsize=14)
        fig.tight_layout()
        watermark()
        path = os.path.join(output_dir, f'acceptance_fp_angle_diag_{raw_tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")


# ========================================================================== #
#  1c. Acceptance False-Negative Breakdown                                     #
# ========================================================================== #

def plot_acceptance_fn_breakdown(datasets, output_dir):
    """For each dataset, break down false negatives (truth=accept, pred=reject)
    by which *predicted* criterion caused the rejection:
      - predicted pion stop out of fiducial (z or xy)
      - predicted angle > 120°
      - no trigger positron predicted (HTP < 0.5)
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, (tag, raw_tag, ev, color) in zip(axes, datasets):
        if ev is None:
            ax.set_visible(False)
            continue

        t_acc = ev['truth_acceptance'].values == 1
        p_acc = (ev['pred_accepted'].values != SENTINEL) & (ev['pred_accepted'].values >= 0.5)

        # False negatives: truth=accept, pred=reject
        fn_mask = t_acc & ~p_acc
        n_fn = int(fn_mask.sum())

        if n_fn == 0:
            ax.text(0.5, 0.5, 'No false negatives', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_title(tag)
            continue

        # Decompose by which PREDICTED criterion caused rejection
        pred_pz = ev['pred_pion_stop_z'].values[fn_mask]
        pred_px = ev['pred_pion_stop_x'].values[fn_mask]
        pred_py = ev['pred_pion_stop_y'].values[fn_mask]
        pred_dir_z = ev['pred_positron_dir_z'].values[fn_mask]
        pred_htp = ev['pred_htp'].values[fn_mask] if 'pred_htp' in ev.columns else np.ones(n_fn)

        _cos_120 = np.cos(np.radians(120.0))
        fail_fid_z = ~((pred_pz > 1.2) & (pred_pz < 4.8))
        fail_fid_xy = ~((np.abs(pred_px) < 8.0) & (np.abs(pred_py) < 8.0))
        fail_fiducial = fail_fid_z | fail_fid_xy
        fail_angle = pred_dir_z <= _cos_120
        fail_htp = (pred_htp < 0.5) | (pred_htp == SENTINEL)

        categories = {
            'Pred fiducial (z)': fail_fid_z,
            'Pred fiducial (xy)': fail_fid_xy,
            'Pred angle > 120°': fail_angle,
            'Pred no trigger\npositron': fail_htp,
        }

        labels = list(categories.keys())
        counts = [int(v.sum()) for v in categories.values()]
        fracs = [c / n_fn * 100 for c in counts]

        bars = ax.barh(labels, fracs, color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'])
        ax.set_xlabel('% of false negatives')
        ax.set_xlim(0, max(fracs) * 1.3 if max(fracs) > 0 else 100)

        for bar, count, frac in zip(bars, counts, fracs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{count:,} ({frac:.1f}%)', va='center', fontsize=10)

        only_fid = fail_fiducial & ~fail_angle & ~fail_htp
        only_angle = ~fail_fiducial & fail_angle & ~fail_htp
        only_htp = ~fail_fiducial & ~fail_angle & fail_htp
        multi = (fail_fiducial.astype(int) + fail_angle.astype(int) + fail_htp.astype(int)) > 1

        summary = (f'Total FN: {n_fn:,}\n'
                   f'Only fiducial: {int(only_fid.sum()):,}\n'
                   f'Only angle: {int(only_angle.sum()):,}\n'
                   f'Only no-HTP: {int(only_htp.sum()):,}\n'
                   f'Multiple: {int(multi.sum()):,}')
        ax.text(0.95, 0.05, summary, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

        ax.set_title(f'{tag}\nFalse negative breakdown (n={n_fn:,})', fontsize=12)

    fig.tight_layout()
    path = os.path.join(output_dir, 'acceptance_fn_breakdown.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


# ========================================================================== #
#  2. Role Confusion Matrix                                                   #
# ========================================================================== #

def plot_role_confusion(datasets_slices, output_dir, include_anchor=False):
    """Per-tag role confusion matrices (counts + row-normalized)."""
    names = ['Background', r'$\mu$', r'$e^+$']

    for tag, raw_tag, sl in datasets_slices:
        if sl is None:
            continue

        mask = np.ones(len(sl), dtype=bool)
        if not include_anchor and 'is_anchor' in sl.columns:
            mask &= ~sl['is_anchor'].values.astype(bool)

        rt = sl['role_truth'].values[mask]
        rp = sl['role_pred'].values[mask]

        cm = np.zeros((3, 3), dtype=np.int64)
        for t, p in zip(rt, rp):
            if 0 <= t < 3 and 0 <= p < 3:
                cm[int(t), int(p)] += 1
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for a, m, ttl, fmt in [(axes[0], cm, 'Counts', 'd'),
                               (axes[1], cm_norm, 'Row-normalized', '.4f')]:
            im = a.imshow(m, cmap='Blues')
            a.set_xticks(range(3)); a.set_yticks(range(3))
            a.set_xticklabels(names); a.set_yticklabels(names)
            a.set_xlabel('Predicted'); a.set_ylabel('Truth')
            a.set_title(f'{tag}: {ttl}')
            for i in range(3):
                for j in range(3):
                    a.text(j, i, format(m[i, j], fmt), ha='center', va='center',
                           color='white' if m[i, j] > m.max() * 0.5 else 'black')
            fig.colorbar(im, ax=a)
        fig.tight_layout()
        watermark(axes[0])
        path = os.path.join(output_dir, f'role_confusion_{raw_tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")


# ========================================================================== #
#  3. Positron Angle RMS vs cos(theta)                                        #
# ========================================================================== #

def angle_rms_curve_cos(ev, bins_cos, gate):
    """Compute RMS of angle residual in bins of cos(theta_true)."""
    theta_true = np.arccos(ev['pred_positron_dir_z'].values[gate].clip(-1, 1))
    theta_pred = ev['pred_polar_angle'].values[gate]

    # Truth angle from truth theta
    truth_theta = ev['truth_theta'].values[gate]

    residual_deg = np.degrees(truth_theta - theta_pred)
    cos_true = np.cos(truth_theta)

    centers = 0.5 * (bins_cos[:-1] + bins_cos[1:])
    rms_vals = []
    err_vals = []
    for lo, hi in zip(bins_cos[:-1], bins_cos[1:]):
        m = (cos_true >= lo) & (cos_true < hi)
        if m.sum() > 5:
            r = residual_deg[m]
            rms = np.sqrt(np.mean(r ** 2))
            err = rms / np.sqrt(2 * m.sum())
            rms_vals.append(rms)
            err_vals.append(err)
        else:
            rms_vals.append(np.nan)
            err_vals.append(np.nan)

    return centers, np.array(rms_vals), np.array(err_vals)


def plot_angle_rms(datasets_events, output_dir, iou_min=0.95):
    """Positron angle RMS vs cos(theta_true)."""
    bins_cos = np.linspace(-1.0, 1.0, 21)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue

        htp_gate = (ev['truth_htp'].values == 1) & \
                   (ev['pred_htp'].values != SENTINEL) & \
                   (ev['pred_htp'].values > 0.5)
        iou_ok = ev['pred_pos_iou'].values >= iou_min
        gate = htp_gate & iou_ok
        n_post = int(gate.sum())

        c, rms, err = angle_rms_curve_cos(ev, bins_cos, gate)
        ax.errorbar(c, rms, yerr=err, fmt='o-', color=color, capsize=3,
                    label=f'{tag}  (n={n_post})')

    ax.set_xlabel(r'$\cos(\theta_{\mathrm{True}})$')
    ax.set_ylabel(r'RMS($\theta_{\mathrm{True}} - \theta_{\mathrm{Reco}}$) [deg]')
    ax.set_title('PURITY Positron angle RMS')
    ax.set_ylim(bottom=0)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    watermark()
    path = os.path.join(output_dir, 'angle_rms_vs_costheta.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")

    # Same plot with extended y-axis to show full range
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue
        htp_gate = (ev['truth_htp'].values == 1) & \
                   (ev['pred_htp'].values != SENTINEL) & \
                   (ev['pred_htp'].values > 0.5)
        iou_ok = ev['pred_pos_iou'].values >= iou_min
        gate = htp_gate & iou_ok
        n_post = int(gate.sum())
        c, rms, err = angle_rms_curve_cos(ev, bins_cos, gate)
        ax.errorbar(c, rms, yerr=err, fmt='o-', color=color, capsize=3,
                    label=f'{tag}  (n={n_post})')
    ax.set_xlabel(r'$\cos(\theta_{\mathrm{True}})$')
    ax.set_ylabel(r'RMS($\theta_{\mathrm{True}} - \theta_{\mathrm{Reco}}$) [deg]')
    ax.set_title('PURITY Positron angle RMS')
    ax.set_ylim(2, 14)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    watermark()
    path = os.path.join(output_dir, 'angle_rms_vs_costheta_wide.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


def plot_angle_median_bias(datasets_events, output_dir, iou_min=0.95):
    """Median angle residual (truth - reco) vs theta_true in degrees.
    Positive = model predicts too forward (reco < truth)."""
    bins_theta = np.linspace(0, 180, 37)  # 5° bins

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue

        htp_gate = (ev['truth_htp'].values == 1) & \
                   (ev['pred_htp'].values != SENTINEL) & \
                   (ev['pred_htp'].values > 0.5)
        iou_ok = ev['pred_pos_iou'].values >= iou_min
        gate = htp_gate & iou_ok
        n_post = int(gate.sum())

        truth_theta = ev['truth_theta'].values[gate]
        pred_theta = ev['pred_polar_angle'].values[gate]
        residual_deg = np.degrees(truth_theta - pred_theta)
        truth_deg = np.degrees(truth_theta)

        centers = 0.5 * (bins_theta[:-1] + bins_theta[1:])
        median_vals, q25_vals, q75_vals = [], [], []
        for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
            m = (truth_deg >= lo) & (truth_deg < hi)
            if m.sum() > 10:
                r = residual_deg[m]
                median_vals.append(np.median(r))
                q25_vals.append(np.percentile(r, 25))
                q75_vals.append(np.percentile(r, 75))
            else:
                median_vals.append(np.nan)
                q25_vals.append(np.nan)
                q75_vals.append(np.nan)

        median_vals = np.array(median_vals)
        q25_vals = np.array(q25_vals)
        q75_vals = np.array(q75_vals)

        ax.plot(centers, median_vals, 'o-', color=color, markersize=4,
                label=f'{tag}  (n={n_post})')
        ax.fill_between(centers, q25_vals, q75_vals, alpha=0.15, color=color)

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.axvline(90, color='gray', linestyle=':', linewidth=0.8, label=r'$\theta = 90°$')
    ax.axvline(120, color='gray', linestyle='--', linewidth=0.8, label=r'$\theta = 120°$ cut')
    ax.set_xlabel(r'$\theta_{\mathrm{True}}$ [deg]')
    ax.set_ylabel(r'Median($\theta_{\mathrm{True}} - \theta_{\mathrm{Reco}}$) [deg]')
    ax.set_title('Positron angle median bias (positive = pred too forward)\nshaded = IQR')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    watermark()
    path = os.path.join(output_dir, 'angle_median_bias_vs_theta.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


def plot_kappa_vs_theta(datasets_events, output_dir, iou_min=0.95):
    """Median vMF concentration (kappa) vs theta_true."""
    bins_theta = np.linspace(0, 180, 37)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    has_data = False
    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue
        if 'pred_positron_log_kappa' not in ev.columns:
            continue

        htp_gate = (ev['truth_htp'].values == 1) & \
                   (ev['pred_htp'].values != SENTINEL) & \
                   (ev['pred_htp'].values > 0.5)
        iou_ok = ev['pred_pos_iou'].values >= iou_min
        lk_ok = ev['pred_positron_log_kappa'].values != SENTINEL
        gate = htp_gate & iou_ok & lk_ok

        truth_theta = np.degrees(ev['truth_theta'].values[gate])
        log_kappa = ev['pred_positron_log_kappa'].values[gate]
        kappa = np.exp(np.clip(log_kappa, -5, 20))

        centers = 0.5 * (bins_theta[:-1] + bins_theta[1:])
        med_kappa, q25_kappa, q75_kappa = [], [], []
        med_lk = []
        for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
            m = (truth_theta >= lo) & (truth_theta < hi)
            if m.sum() > 10:
                k = kappa[m]
                med_kappa.append(np.median(k))
                q25_kappa.append(np.percentile(k, 25))
                q75_kappa.append(np.percentile(k, 75))
                med_lk.append(np.median(log_kappa[m]))
            else:
                med_kappa.append(np.nan)
                q25_kappa.append(np.nan)
                q75_kappa.append(np.nan)
                med_lk.append(np.nan)

        med_kappa = np.array(med_kappa)
        q25_kappa = np.array(q25_kappa)
        q75_kappa = np.array(q75_kappa)
        med_lk = np.array(med_lk)

        axes[0].plot(centers, med_kappa, 'o-', color=color, markersize=4,
                     label=tag)
        axes[0].fill_between(centers, q25_kappa, q75_kappa, alpha=0.15, color=color)
        axes[1].plot(centers, med_lk, 'o-', color=color, markersize=4)
        has_data = True

    if not has_data:
        plt.close()
        return

    axes[0].set_ylabel(r'$\kappa$ (concentration)')
    axes[0].set_title(r'vMF concentration $\kappa$ vs $\theta_{\mathrm{True}}$'
                      '\nshaded = IQR')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(90, color='gray', linestyle=':', linewidth=0.8)
    axes[0].axvline(120, color='gray', linestyle='--', linewidth=0.8)

    axes[1].set_ylabel(r'$\log\kappa$')
    axes[1].set_xlabel(r'$\theta_{\mathrm{True}}$ [deg]')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(90, color='gray', linestyle=':', linewidth=0.8)
    axes[1].axvline(120, color='gray', linestyle='--', linewidth=0.8)

    fig.tight_layout()
    watermark(axes[1])
    path = os.path.join(output_dir, 'kappa_vs_theta.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


def plot_angle_mean_bias(datasets_events, output_dir, iou_min=0.95):
    """Mean angle residual (truth - reco) vs cos(theta_true).
    Positive = model predicts too forward."""
    bins_cos = np.linspace(-1.0, 1.0, 21)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue

        htp_gate = (ev['truth_htp'].values == 1) & \
                   (ev['pred_htp'].values != SENTINEL) & \
                   (ev['pred_htp'].values > 0.5)
        iou_ok = ev['pred_pos_iou'].values >= iou_min
        gate = htp_gate & iou_ok
        n_post = int(gate.sum())

        truth_theta = ev['truth_theta'].values[gate]
        pred_theta = ev['pred_polar_angle'].values[gate]
        residual_deg = np.degrees(truth_theta - pred_theta)
        cos_true = np.cos(truth_theta)

        centers = 0.5 * (bins_cos[:-1] + bins_cos[1:])
        mean_vals, err_vals = [], []
        for lo, hi in zip(bins_cos[:-1], bins_cos[1:]):
            m = (cos_true >= lo) & (cos_true < hi)
            if m.sum() > 5:
                r = residual_deg[m]
                mean_vals.append(np.mean(r))
                err_vals.append(np.std(r) / np.sqrt(m.sum()))
            else:
                mean_vals.append(np.nan)
                err_vals.append(np.nan)

        ax.errorbar(centers, mean_vals, yerr=err_vals, fmt='o-', color=color,
                    capsize=3, label=f'{tag}  (n={n_post})')

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel(r'$\cos(\theta_{\mathrm{True}})$')
    ax.set_ylabel(r'Mean($\theta_{\mathrm{True}} - \theta_{\mathrm{Reco}}$) [deg]')
    ax.set_title('PURITY Positron angle mean bias (positive = pred too forward)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    watermark()
    path = os.path.join(output_dir, 'angle_mean_bias_vs_costheta.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


def plot_angle_residual_at_cut(datasets_events, output_dir, iou_min=0.95):
    """Histogram of angle residual (truth - reco) for events near the 120°
    acceptance cut. Shows whether the error distribution is skewed."""

    fig, axes = plt.subplots(1, len(datasets_events), figsize=(6 * len(datasets_events), 5))
    if len(datasets_events) == 1:
        axes = [axes]

    for ax, (tag, raw_tag, ev, color) in zip(axes, datasets_events):
        if ev is None:
            ax.set_visible(False)
            continue

        htp_gate = (ev['truth_htp'].values == 1) & \
                   (ev['pred_htp'].values != SENTINEL) & \
                   (ev['pred_htp'].values > 0.5)
        iou_ok = ev['pred_pos_iou'].values >= iou_min
        gate = htp_gate & iou_ok

        truth_theta_deg = np.degrees(ev['truth_theta'].values[gate])
        pred_theta_deg = np.degrees(ev['pred_polar_angle'].values[gate])
        residual_deg = truth_theta_deg - pred_theta_deg

        # Select events with truth theta in [110, 130] — straddling the 120° cut
        near_cut = (truth_theta_deg > 110) & (truth_theta_deg < 130)
        r = residual_deg[near_cut]
        n = int(near_cut.sum())

        # Split into below-cut and above-cut
        below = (truth_theta_deg > 110) & (truth_theta_deg <= 120)
        above = (truth_theta_deg > 120) & (truth_theta_deg < 130)
        r_below = residual_deg[below]
        r_above = residual_deg[above]

        bins = np.linspace(-15, 15, 61)
        ax.hist(r_below, bins=bins, alpha=0.5, color='tab:blue', density=True,
                label=rf'truth $\theta \in [110°, 120°]$ (n={int(below.sum()):,})')
        ax.hist(r_above, bins=bins, alpha=0.5, color='tab:red', density=True,
                label=rf'truth $\theta \in [120°, 130°]$ (n={int(above.sum()):,})')

        # Stats
        for r_sub, c, side in [(r_below, 'tab:blue', 'below'),
                                (r_above, 'tab:red', 'above')]:
            if len(r_sub) > 0:
                mu = np.mean(r_sub)
                sig = np.std(r_sub)
                skew = np.mean(((r_sub - mu) / sig) ** 3) if sig > 0 else 0
                ax.axvline(mu, color=c, linestyle='--', linewidth=1.5)
                ax.text(0.98 if side == 'above' else 0.02,
                        0.95 if side == 'above' else 0.85,
                        f'{side} cut:\n'
                        f'  mean={mu:.2f}°\n'
                        f'  RMS={sig:.2f}°\n'
                        f'  skew={skew:.2f}',
                        transform=ax.transAxes, fontsize=9,
                        ha='right' if side == 'above' else 'left',
                        va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

        ax.axvline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r'$\theta_{\mathrm{True}} - \theta_{\mathrm{Reco}}$ [deg]')
        ax.set_ylabel('density')
        ax.set_title(f'{tag}: Residual near 120° cut (n={n:,})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    watermark()
    path = os.path.join(output_dir, 'angle_residual_at_cut.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


def plot_angle_2d(datasets_events, output_dir):
    """2D truth vs reco polar angle for each dataset (htp-gated)."""
    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue

        m = ((ev['truth_htp'].values == 1) &
             (ev['pred_positron_dir_x'].values != SENTINEL))

        theta_true = np.degrees(ev['truth_theta'].values[m])
        theta_reco = np.degrees(np.arccos(np.clip(ev['pred_positron_dir_z'].values[m], -1, 1)))

        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

        # Left: full range
        h = axes[0].hist2d(theta_true, theta_reco, bins=[90, 90],
                           range=[[0, 180], [0, 180]], cmin=1)
        axes[0].plot([0, 180], [0, 180], 'r--', lw=0.8)
        axes[0].set_xlabel(r'$\theta_{\mathrm{True}}$ [deg]')
        axes[0].set_ylabel(r'$\theta_{\mathrm{Reco}}$ [deg]')
        axes[0].set_title(f'{tag} | truth htp == 1 (n={int(m.sum()):,})')
        fig.colorbar(h[3], ax=axes[0])

        # Middle: zoomed around 60° (below 90°)
        h1 = axes[1].hist2d(theta_true, theta_reco, bins=[60, 60],
                            range=[[40, 80], [40, 80]], cmin=1)
        axes[1].plot([40, 80], [40, 80], 'r--', lw=0.8)
        axes[1].axvline(60, color='white', linestyle=':', lw=0.8)
        axes[1].axhline(60, color='white', linestyle=':', lw=0.8)
        axes[1].set_xlabel(r'$\theta_{\mathrm{True}}$ [deg]')
        axes[1].set_ylabel(r'$\theta_{\mathrm{Reco}}$ [deg]')
        axes[1].set_title(f'{tag} | zoom near 60°')
        fig.colorbar(h1[3], ax=axes[1])

        # Right: zoomed around 120° cut
        h2 = axes[2].hist2d(theta_true, theta_reco, bins=[60, 60],
                            range=[[100, 140], [100, 140]], cmin=1)
        axes[2].plot([100, 140], [100, 140], 'r--', lw=0.8)
        axes[2].axvline(120, color='white', linestyle=':', lw=0.8)
        axes[2].axhline(120, color='white', linestyle=':', lw=0.8)
        axes[2].set_xlabel(r'$\theta_{\mathrm{True}}$ [deg]')
        axes[2].set_ylabel(r'$\theta_{\mathrm{Reco}}$ [deg]')
        axes[2].set_title(f'{tag} | zoom near 120°')
        fig.colorbar(h2[3], ax=axes[2])

        fig.tight_layout()
        watermark()
        path = os.path.join(output_dir, f'angle_2d_{raw_tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")


# ========================================================================== #
#  4. Positron Energy Spectrum                                                #
# ========================================================================== #

def plot_energy_spectrum(datasets_events, output_dir, all_lyso=None, p_hit_threshold=0.5):
    """Reconstructed positron energy spectra.

    If all_lyso is provided (dict of tag -> lyso hits DataFrame), the reco
    energy is recomputed from per-hit p_hit with the given threshold instead
    of using the baked-in pred_positron_energy (which used p_hit > 0.5).
    """
    n_ds = len(datasets_events)
    bins = np.linspace(0, 90, 90)

    def _get_energies(ev, raw_tag, require_truth_htp=False, add_dead=False):
        v = ((ev['pred_positron_energy'].values != SENTINEL) &
             np.isfinite(ev['truth_live_E'].values) &
             (ev['pred_htp'].values != SENTINEL) &
             (ev['pred_htp'].values > 0.5) &
             (ev['pred_accepted'].values >= 0.5))
        if require_truth_htp and 'truth_htp' in ev.columns:
            v &= (ev['truth_htp'].values == 1)

        te = ev['truth_live_E'].values.copy()
        if add_dead and 'truth_dead_E' in ev.columns:
            te = te + np.where(np.isfinite(ev['truth_dead_E'].values),
                               ev['truth_dead_E'].values, 0.0)
        te = te[v]

        if all_lyso is not None and raw_tag in all_lyso and all_lyso[raw_tag] is not None:
            lyso = all_lyso[raw_tag]
            lyso_old = lyso[lyso['p_hit'] > 0.5].groupby('event_id')['E'].sum()
            lyso_new = lyso[lyso['p_hit'] > p_hit_threshold].groupby('event_id')['E'].sum()
            pred_E = ev['pred_positron_energy'].values.copy()
            lyso_old_arr = np.zeros(len(ev))
            for eid, esum in lyso_old.items():
                if eid < len(lyso_old_arr):
                    lyso_old_arr[eid] = esum
            atar_pos_E = np.maximum(pred_E - lyso_old_arr, 0.0)
            lyso_new_arr = np.zeros(len(ev))
            for eid, esum in lyso_new.items():
                if eid < len(lyso_new_arr):
                    lyso_new_arr[eid] = esum
            dead_E = ev['pred_dead_energy'].values.copy()
            dead_E = np.where(dead_E == SENTINEL, 0.0, dead_E)
            pe = (atar_pos_E + lyso_new_arr + dead_E)[v]
            thresh_label = f'reco (p>{p_hit_threshold})'
        else:
            pe = ev['pred_positron_energy'].values.copy()
            if add_dead and 'pred_dead_energy' in ev.columns:
                pred_dead = ev['pred_dead_energy'].values.copy()
                pred_dead = np.where(pred_dead == SENTINEL, 0.0, pred_dead)
                pe = pe + pred_dead
            pe = pe[v]
            thresh_label = 'reco'
        return te, pe, thresh_label

    # Log scale plot
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4.5), sharey=False)
    if n_ds == 1: axes = [axes]
    for ax, (tag, raw_tag, ev, color) in zip(axes, datasets_events):
        if ev is None:
            ax.set_visible(False)
            continue
        te, pe, thresh_label = _get_energies(ev, raw_tag)
        ax.hist(te, bins=bins, histtype='stepfilled', color=color, alpha=0.30,
                edgecolor=color, linewidth=1.0, label='truth (live E)')
        ax.hist(pe, bins=bins, histtype='step', color='black', linewidth=1.8,
                label=thresh_label)
        ax.set_xlabel(r'Energy [MeV]')
        ax.set_ylabel('counts')
        ax.set_yscale('log')
        ax.set_xlim(0, 90)
        ax.set_title(f'{tag}  (N={len(te):,})')
        ax.legend(loc='upper right')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        if 'pie' in raw_tag and 'pimu' not in raw_tag and len(te) > 0:
            tail_t = 100 * (te < 56).sum() / len(te)
            tail_r = 100 * (pe < 56).sum() / len(pe)
            ax.text(0.03, 0.97,
                    f'Tail (E<56 MeV):\n  truth={tail_t:.2f}%\n  reco={tail_r:.2f}%',
                    transform=ax.transAxes, fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))
    fig.tight_layout()
    path = os.path.join(output_dir, 'energy_spectrum.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")

    # Linear scale plot
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4.5), sharey=False)
    if n_ds == 1: axes = [axes]
    for ax, (tag, raw_tag, ev, color) in zip(axes, datasets_events):
        if ev is None:
            ax.set_visible(False)
            continue
        te, pe, thresh_label = _get_energies(ev, raw_tag)
        ax.hist(te, bins=bins, histtype='stepfilled', color=color, alpha=0.30,
                edgecolor=color, linewidth=1.0, label='truth (live E)')
        ax.hist(pe, bins=bins, histtype='step', color='black', linewidth=1.8,
                label=thresh_label)
        ax.set_xlabel(r'Energy [MeV]')
        ax.set_ylabel('counts')
        ax.set_xlim(0, 90)
        ax.set_title(f'{tag}  (N={len(te):,})')
        ax.legend(loc='upper right')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    fig.tight_layout()
    path = os.path.join(output_dir, 'energy_spectrum_linear.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")

    # Log scale — truth_htp == 1 only
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4.5), sharey=False)
    if n_ds == 1: axes = [axes]
    for ax, (tag, raw_tag, ev, color) in zip(axes, datasets_events):
        if ev is None:
            ax.set_visible(False)
            continue
        te, pe, thresh_label = _get_energies(ev, raw_tag, require_truth_htp=True)
        ax.hist(te, bins=bins, histtype='stepfilled', color=color, alpha=0.30,
                edgecolor=color, linewidth=1.0, label='truth (live E)')
        ax.hist(pe, bins=bins, histtype='step', color='black', linewidth=1.8,
                label=thresh_label)
        ax.set_xlabel(r'Energy [MeV]')
        ax.set_ylabel('counts')
        ax.set_yscale('log')
        ax.set_xlim(0, 90)
        ax.set_title(f'{tag}  truth htp=1  (N={len(te):,})')
        ax.legend(loc='upper right')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    fig.tight_layout()
    path = os.path.join(output_dir, 'energy_spectrum_htp.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")

    # Linear scale — truth_htp == 1 only
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4.5), sharey=False)
    if n_ds == 1: axes = [axes]
    for ax, (tag, raw_tag, ev, color) in zip(axes, datasets_events):
        if ev is None:
            ax.set_visible(False)
            continue
        te, pe, thresh_label = _get_energies(ev, raw_tag, require_truth_htp=True)
        ax.hist(te, bins=bins, histtype='stepfilled', color=color, alpha=0.30,
                edgecolor=color, linewidth=1.0, label='truth (live E)')
        ax.hist(pe, bins=bins, histtype='step', color='black', linewidth=1.8,
                label=thresh_label)
        ax.set_xlabel(r'Energy [MeV]')
        ax.set_ylabel('counts')
        ax.set_xlim(0, 90)
        ax.set_title(f'{tag}  truth htp=1  (N={len(te):,})')
        ax.legend(loc='upper right')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    fig.tight_layout()
    path = os.path.join(output_dir, 'energy_spectrum_htp_linear.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")

    # Zoomed pie-only peak: linear scale, 50-80 MeV, 100 bins + Gaussian fits
    from scipy.optimize import curve_fit
    bins_zoom = np.linspace(50, 80, 101)

    def _gauss(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def _fit_peak(data, bins, fit_lo=65, fit_hi=73):
        """Fit a Gaussian to histogram counts in [fit_lo, fit_hi]."""
        counts, _ = np.histogram(data, bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        mask = (centers >= fit_lo) & (centers <= fit_hi)
        x_fit, y_fit = centers[mask], counts[mask]
        if len(x_fit) < 3 or y_fit.max() == 0:
            return None, None, None, None, None
        try:
            p0 = [y_fit.max(), x_fit[np.argmax(y_fit)], 1.0]
            popt, _ = curve_fit(_gauss, x_fit, y_fit, p0=p0,
                                bounds=([0, fit_lo, 0.01], [np.inf, fit_hi, 20.0]))
            A, mu, sigma = popt
            return A, mu, abs(sigma), centers, counts
        except RuntimeError:
            return None, None, None, centers, counts

    for tag, raw_tag, ev, color in datasets_events:
        if ev is None or ('pimu' in raw_tag or 'michel' in raw_tag):
            continue
        te, pe, thresh_label = _get_energies(ev, raw_tag, require_truth_htp=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(te, bins=bins_zoom, histtype='stepfilled', color=color, alpha=0.30,
                edgecolor=color, linewidth=1.0, label='truth (live E)')
        ax.hist(pe, bins=bins_zoom, histtype='step', color='black', linewidth=1.8,
                label=thresh_label)

        # Fit Gaussians
        x_plot = np.linspace(50, 80, 300)
        stats_lines = []

        A_t, mu_t, sig_t, _, _ = _fit_peak(te, bins_zoom)
        if sig_t is not None:
            ax.plot(x_plot, _gauss(x_plot, A_t, mu_t, sig_t),
                    color=color, linewidth=1.5, linestyle='--',
                    label=rf'truth fit  $\mu$={mu_t:.2f} MeV')

        A_r, mu_r, sig_r, _, _ = _fit_peak(pe, bins_zoom)
        if sig_r is not None:
            ax.plot(x_plot, _gauss(x_plot, A_r, mu_r, sig_r),
                    color='black', linewidth=1.5, linestyle='--',
                    label=rf'reco fit  $\mu$={mu_r:.2f} MeV')

        if sig_t is not None and sig_r is not None and sig_r > sig_t:
            sig_ml = np.sqrt(sig_r**2 - sig_t**2)
            ax.text(0.03, 0.55,
                    rf'$\sigma_{{ML}}$ = {sig_ml:.2f} MeV',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

        ax.set_xlabel(r'Energy [MeV]')
        ax.set_ylabel('counts')
        ax.set_xlim(50, 80)
        ax.set_title(f'{tag}  truth htp=1  (N={len(te):,})')
        ax.legend(loc='upper right')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        fig.tight_layout()
        path = os.path.join(output_dir, f'energy_spectrum_peak_{raw_tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")

    # Same peak plot but with dead material added back
    for tag, raw_tag, ev, color in datasets_events:
        if ev is None or ('pimu' in raw_tag or 'michel' in raw_tag):
            continue
        te, pe, thresh_label = _get_energies(ev, raw_tag, require_truth_htp=True, add_dead=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(te, bins=bins_zoom, histtype='stepfilled', color=color, alpha=0.30,
                edgecolor=color, linewidth=1.0, label='truth (live + dead E)')
        ax.hist(pe, bins=bins_zoom, histtype='step', color='black', linewidth=1.8,
                label=thresh_label + ' + dead')

        x_plot = np.linspace(50, 80, 300)

        A_t, mu_t, sig_t, _, _ = _fit_peak(te, bins_zoom)
        if sig_t is not None:
            ax.plot(x_plot, _gauss(x_plot, A_t, mu_t, sig_t),
                    color=color, linewidth=1.5, linestyle='--',
                    label=rf'truth fit  $\mu$={mu_t:.2f} MeV')

        A_r, mu_r, sig_r, _, _ = _fit_peak(pe, bins_zoom)
        if sig_r is not None:
            ax.plot(x_plot, _gauss(x_plot, A_r, mu_r, sig_r),
                    color='black', linewidth=1.5, linestyle='--',
                    label=rf'reco fit  $\mu$={mu_r:.2f} MeV')

        if sig_t is not None and sig_r is not None and sig_r > sig_t:
            sig_ml = np.sqrt(sig_r**2 - sig_t**2)
            ax.text(0.03, 0.55,
                    rf'$\sigma_{{ML}}$ = {sig_ml:.2f} MeV',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

        ax.set_xlabel(r'Energy [MeV]')
        ax.set_ylabel('counts')
        ax.set_xlim(50, 80)
        ax.set_title(f'{tag}  truth htp=1  (live + dead E)  (N={len(te):,})')
        ax.legend(loc='upper left')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        fig.tight_layout()
        path = os.path.join(output_dir, f'energy_spectrum_peak_dead_{raw_tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")


def plot_energy_overlay(datasets_events, output_dir, all_lyso=None,
                        p_hit_threshold=0.5, pimu_scale=10000,
                        time_window=None):
    """Overlay pie and pimu energy densities with pimu scaled by pimu_scale.

    Produces two panels: truth (left) and reco (right), each showing pie
    density and pimu density × pimu_scale on the same axes.
    """
    # Collect pie and pimu data
    pie_truth, pie_reco = None, None
    pimu_truth, pimu_reco = None, None
    pie_n_before_tcut, pimu_n_before_tcut = 0, 0

    bins = np.linspace(0, 140, 140)

    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue

        v = ((ev['pred_positron_energy'].values != SENTINEL) &
             np.isfinite(ev['truth_live_E'].values) &
             (ev['pred_htp'].values != SENTINEL) &
             (ev['pred_htp'].values > 0.5) &
             (ev['pred_accepted'].values >= 0.5))

        n_before_tcut = int(v.sum())

        # Positron time window cut (optional)
        if time_window is not None and 'pred_positron_time_ns' in ev.columns:
            pt = ev['pred_positron_time_ns'].values
            v &= (pt > time_window[0]) & (pt < time_window[1]) & (pt != SENTINEL)

        te = ev['truth_live_E'].values[v]

        # Recompute reco energy if LYSO data available
        if all_lyso is not None and raw_tag in all_lyso and all_lyso[raw_tag] is not None:
            lyso = all_lyso[raw_tag]
            lyso_old = lyso[lyso['p_hit'] > 0.5].groupby('event_id')['E'].sum()
            lyso_new = lyso[lyso['p_hit'] > p_hit_threshold].groupby('event_id')['E'].sum()

            pred_E = ev['pred_positron_energy'].values.copy()
            lyso_old_arr = np.zeros(len(ev))
            for eid, esum in lyso_old.items():
                if eid < len(lyso_old_arr):
                    lyso_old_arr[eid] = esum
            atar_pos_E = np.maximum(pred_E - lyso_old_arr, 0.0)

            lyso_new_arr = np.zeros(len(ev))
            for eid, esum in lyso_new.items():
                if eid < len(lyso_new_arr):
                    lyso_new_arr[eid] = esum

            dead_E = ev['pred_dead_energy'].values.copy()
            dead_E = np.where(dead_E == SENTINEL, 0.0, dead_E)
            pe = (atar_pos_E + lyso_new_arr + dead_E)[v]
        else:
            pe = ev['pred_positron_energy'].values[v]

        if 'pie' in raw_tag and 'pimu' not in raw_tag:
            pie_truth, pie_reco = te, pe
            pie_n_before_tcut = n_before_tcut
        elif 'pimu' in raw_tag or 'michel' in raw_tag:
            pimu_truth, pimu_reco = te, pe
            pimu_n_before_tcut = n_before_tcut

    if pie_truth is None or pimu_truth is None:
        print("  (skipped — need both pie and pimu datasets)")
        return

    # Track total events before time cut to adjust branching ratio scale.
    # pimu_scale represents the physical BR (e.g. 10000 = 1 pimu per 10000 pie).
    # If a time window removes different fractions of pie vs pimu, the
    # effective scale must be adjusted:
    #   effective = pimu_scale × (frac_pimu_surviving / frac_pie_surviving)
    if pie_n_before_tcut > 0 and pimu_n_before_tcut > 0:
        frac_pie = len(pie_truth) / pie_n_before_tcut
        frac_pimu = len(pimu_truth) / pimu_n_before_tcut
        effective_scale = pimu_scale * (frac_pimu / max(frac_pie, 1e-9))
    else:
        effective_scale = pimu_scale

    print(f"    pie gated: {len(pie_truth)} (of {pie_n_before_tcut}), "
          f"pimu gated: {len(pimu_truth)} (of {pimu_n_before_tcut}), "
          f"effective scale: {effective_scale:.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _smooth(y, window=5):
        """Simple uniform moving-average smoother."""
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode='same')

    for ax, pie_data, pimu_data, title in [
        (axes[0], pie_truth, pimu_truth, 'Truth (live E)'),
        (axes[1], pie_reco, pimu_reco, 'Reco'),
    ]:
        bin_width = bins[1] - bins[0]
        centers = 0.5 * (bins[:-1] + bins[1:])

        pie_counts, _ = np.histogram(pie_data, bins=bins)
        pimu_counts, _ = np.histogram(pimu_data, bins=bins)

        # Normalize to density, then scale pimu by effective BR
        pie_density = pie_counts / (pie_counts.sum() * bin_width)
        pimu_density = pimu_counts / (pimu_counts.sum() * bin_width) * effective_scale

        # Two regimes: raw histogram below the Michel edge, smoothed above.
        # Then log-interpolate across the gap of zero bins to connect them.
        crossover = 56  # MeV
        idx_cross = np.searchsorted(centers, crossover)

        # Smooth the high-energy tail in isolation
        tail = pimu_density[idx_cross:]
        tail_smooth = _smooth(tail, window=21)

        # Assemble: raw below crossover, smoothed above
        pimu_plot = pimu_density.copy()
        pimu_plot[idx_cross:] = tail_smooth

        # Find the zero-bin gap between the two regimes and log-interpolate
        gap_region = (centers > 45) & (centers < 75)
        gap_indices = np.where(gap_region & (pimu_plot <= 0))[0]
        if len(gap_indices) > 0:
            left = gap_indices[0] - 1
            right = gap_indices[-1] + 1
            if left >= 0 and right < len(pimu_plot) and pimu_plot[left] > 0 and pimu_plot[right] > 0:
                log_left = np.log10(pimu_plot[left])
                log_right = np.log10(pimu_plot[right])
                for j in gap_indices:
                    alpha = (j - left) / (right - left)
                    pimu_plot[j] = 10 ** (log_left + alpha * (log_right - log_left))

        ax.step(centers, pie_density, where='mid', color='tab:red',
                linewidth=1.8, label=r'$\pi \to e$')
        ax.plot(centers, pimu_plot, color='tab:blue',
                linewidth=1.8, linestyle='--',
                label=rf'$\pi \to \mu \to e\ (\times{effective_scale:,.0f})$')

        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel('density (a.u.)')
        ax.set_yscale('log')
        ax.set_xlim(0, 90)
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(rf'Energy density overlay (BR $\times${pimu_scale:,.0f}, '
                 rf'effective $\times${effective_scale:,.0f})', fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, 'energy_overlay.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


# ========================================================================== #
#  5. Pion Stop Resolution                                                    #
# ========================================================================== #

def plot_pion_stop_resolution(datasets_events, output_dir):
    """Pion stop residual distributions per axis."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axis_names = ['x', 'y', 'z']
    bins = np.linspace(-1.0, 1.0, 80)

    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue

        gate = (ev['pred_accepted'].values >= 0.5) & (ev['pred_accepted'].values != SENTINEL)

        for i, axis in enumerate(axis_names):
            truth = ev[f'truth_pion_stop_{axis}'].values[gate]
            pred = ev[f'pred_pion_stop_{axis}'].values[gate]
            valid = np.isfinite(truth) & np.isfinite(pred) & (pred != SENTINEL)
            residual = pred[valid] - truth[valid]

            axes[i].hist(residual, bins=bins, histtype='step', color=color,
                         linewidth=1.5,
                         label=f'{tag} (rms={np.std(residual):.3f} mm)')
            axes[i].set_xlabel(f'{axis} residual [mm]')
            axes[i].set_ylabel('counts')
            axes[i].set_yscale('log')
            axes[i].set_title(f'Pion stop {axis}')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'pion_stop_resolution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")

    # 3D distance error overlay (pie vs pimu on same axes, normalized)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins_3d = np.linspace(0, 2, 200)
    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue
        gate = (ev['pred_accepted'].values >= 0.5) & (ev['pred_accepted'].values != SENTINEL)
        dx = ev['pred_pion_stop_x'].values[gate] - ev['truth_pion_stop_x'].values[gate]
        dy = ev['pred_pion_stop_y'].values[gate] - ev['truth_pion_stop_y'].values[gate]
        dz = ev['pred_pion_stop_z'].values[gate] - ev['truth_pion_stop_z'].values[gate]
        valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz) & \
                (ev['pred_pion_stop_x'].values[gate] != SENTINEL)
        dist = np.sqrt(dx[valid]**2 + dy[valid]**2 + dz[valid]**2)
        ax.hist(dist, bins=bins_3d, histtype='step', color=color,
                linewidth=1.8, density=True,
                label=f'{tag}  (median={np.median(dist):.3f} mm)')

    ax.set_xlabel('Pion stop 3D error [mm]')
    ax.set_ylabel('Counts [arb]')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Pion stop 3D reconstruction error')
    fig.tight_layout()
    path = os.path.join(output_dir, 'pion_stop_3d_error.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


# ========================================================================== #
#  6. Endpoint Resolution                                                     #
# ========================================================================== #

def plot_endpoint_resolution(datasets_slices, output_dir):
    """Start/stop endpoint residuals per axis."""
    axis_names = ['x', 'y', 'z']
    bins = np.linspace(-2.0, 2.0, 80)

    for point, point_label in [('start', 'Start'), ('stop', 'Stop')]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for tag, raw_tag, sl in datasets_slices:
            if sl is None:
                continue
            for i, axis in enumerate(axis_names):
                truth_col = f'truth_{point}_{axis}'
                pred_col = f'pred_{point}_{axis}'
                if truth_col not in sl.columns or pred_col not in sl.columns:
                    continue
                truth = sl[truth_col].values
                pred = sl[pred_col].values
                valid = np.isfinite(truth) & np.isfinite(pred)
                residual = pred[valid] - truth[valid]

                axes[i].hist(residual, bins=bins, histtype='step', linewidth=1.5,
                             label=f'{tag} (rms={np.std(residual):.3f} mm)')
                axes[i].set_xlabel(f'{axis} residual [mm]')
                axes[i].set_ylabel('counts')
                axes[i].set_title(f'{point_label} endpoint {axis}')
                axes[i].legend(fontsize=8)
                axes[i].grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(output_dir, f'endpoint_{point}_resolution.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")


# ========================================================================== #
#  7. Positron Hit IoU Distribution                                           #
# ========================================================================== #

def plot_positron_iou(datasets_events, output_dir):
    """Distribution of positron hit-level IoU."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 50)

    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue
        iou = ev['pred_pos_iou'].values
        valid = np.isfinite(iou) & (iou >= 0)
        iou_v = iou[valid]
        ax.hist(iou_v, bins=bins, histtype='step', color=color, linewidth=1.5,
                label=f'{tag} (median={np.median(iou_v):.3f}, n={len(iou_v)})')

    ax.set_xlabel('Positron hit IoU')
    ax.set_ylabel('counts')
    ax.set_title('Positron hit-level IoU')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='IoU=0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    watermark()
    path = os.path.join(output_dir, 'positron_iou.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


# ========================================================================== #
#  8. Positron Precision / Recall                                             #
# ========================================================================== #

def plot_precision_recall(datasets_events, output_dir):
    """Positron hit-level precision and recall distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(0, 1, 50)

    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue
        for ax, col, label in [(axes[0], 'pred_pos_precision', 'Precision'),
                                (axes[1], 'pred_pos_recall', 'Recall')]:
            vals = ev[col].values
            valid = np.isfinite(vals)
            v = vals[valid]
            ax.hist(v, bins=bins, histtype='step', color=color, linewidth=1.5,
                    label=f'{tag} (median={np.median(v):.3f})')
            ax.set_xlabel(label)
            ax.set_ylabel('counts')
            ax.set_title(f'Positron {label}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    watermark(axes[0])
    path = os.path.join(output_dir, 'positron_precision_recall.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


# ========================================================================== #
#  9. Energy Resolution (2D)                                                  #
# ========================================================================== #

def plot_energy_resolution_2d(datasets_events, output_dir):
    """2D truth vs reco energy, and residual vs truth."""
    for tag, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue

        v = ((ev['pred_positron_energy'].values != SENTINEL) &
             np.isfinite(ev['truth_live_E'].values) &
             (ev['pred_htp'].values != SENTINEL) &
             (ev['pred_htp'].values > 0.5) &
             (ev['pred_accepted'].values >= 0.5))

        te = ev['truth_live_E'].values[v]
        pe = ev['pred_positron_energy'].values[v]
        residual = pe - te

        from matplotlib.colors import LogNorm

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Truth vs Reco
        _, _, _, im0 = axes[0].hist2d(te, pe, bins=80, cmap='viridis',
                                       range=[[0, 140], [0, 140]],
                                       norm=LogNorm(vmin=1))
        axes[0].plot([0, 140], [0, 140], 'r--', linewidth=1, alpha=0.7)
        axes[0].set_xlabel('Truth Live Energy [MeV]')
        axes[0].set_ylabel('Reco Energy [MeV]')
        axes[0].set_title(f'{tag}: Truth vs Reco')
        fig.colorbar(im0, ax=axes[0])

        # Residual vs Truth
        _, _, _, im1 = axes[1].hist2d(te, residual, bins=80, cmap='viridis',
                                       range=[[0, 140], [-40, 40]],
                                       norm=LogNorm(vmin=1))
        axes[1].axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.7)
        axes[1].set_xlabel('Truth Live Energy [MeV]')
        axes[1].set_ylabel('Reco - Truth [MeV]')
        axes[1].set_title(f'{tag}: Energy Residual')
        fig.colorbar(im1, ax=axes[1])

        fig.tight_layout()
        path = os.path.join(output_dir, f'energy_resolution_2d_{raw_tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")


# ========================================================================== #
#  10. Role Probability Distributions                                         #
# ========================================================================== #

def plot_role_probabilities(datasets_slices, output_dir):
    """Distribution of predicted role probabilities by true class."""
    role_names = ['none', 'muon', 'positron']
    prob_cols = ['role_prob_none', 'role_prob_muon', 'role_prob_positron']

    for tag, raw_tag, sl in datasets_slices:
        if sl is None:
            continue

        mask = ~sl['is_anchor'].values.astype(bool) if 'is_anchor' in sl.columns else \
               np.ones(len(sl), dtype=bool)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        bins = np.linspace(0, 1, 50)

        for ax, prob_col, rname in zip(axes, prob_cols, role_names):
            for true_class in range(3):
                class_mask = mask & (sl['role_truth'].values == true_class)
                if class_mask.sum() == 0:
                    continue
                probs = sl[prob_col].values[class_mask]
                ax.hist(probs, bins=bins, histtype='step', linewidth=1.5,
                        label=f'truth={role_names[true_class]}', alpha=0.8)
            ax.set_xlabel(f'P({rname})')
            ax.set_ylabel('counts')
            ax.set_title(f'{tag}: P({rname})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(output_dir, f'role_probabilities_{raw_tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")


# ========================================================================== #
#  11. Positron Time Spectra (accepted, split by energy threshold)             #
# ========================================================================== #

def plot_positron_time_spectra(datasets_events, output_dir, e_threshold=56.0,
                               acceptance_cut=120.0):
    """Time spectra of accepted positrons, split above/below energy threshold.

    Reconstructed energy = pred_positron_energy + pred_dead_energy.
    Acceptance = pred_polar_angle < acceptance_cut.
    Also requires pred_htp > 0.5 (identified trigger positron).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    time_bins = np.linspace(-300, 500, 801)

    for label, raw_tag, ev, color in datasets_events:
        if ev is None:
            continue

        t = ev['pred_positron_time_ns'].values
        angle = ev['pred_polar_angle'].values
        e_pos = ev['pred_positron_energy'].values
        e_dead = ev['pred_dead_energy'].values
        htp = ev['pred_htp'].values if 'pred_htp' in ev.columns else np.ones(len(ev))
        accepted_flag = ev['pred_accepted'].values if 'pred_accepted' in ev.columns else np.ones(len(ev))

        valid = ((t > SENTINEL + 1) & (angle > SENTINEL + 1) &
                 (e_pos > SENTINEL + 1) & (e_dead > SENTINEL + 1) &
                 (htp > 0.5) & (accepted_flag >= 0.5))

        dead = np.where(e_dead > SENTINEL + 1, e_dead, 0.0)
        e_total = e_pos + dead
        above = valid & (e_total >= e_threshold)
        below = valid & (e_total < e_threshold)

        n_above, n_below = above.sum(), below.sum()
        tail_frac = 100.0 * n_below / max(n_above + n_below, 1)

        ax = axes[0]
        ax.hist(t[above], bins=time_bins, histtype='step', linewidth=1.5,
                color=color,
                label=f'{label} E≥{e_threshold:.0f} (n={n_above})')
        ax.hist(t[below], bins=time_bins, histtype='step', linewidth=1.5,
                color=color, linestyle='--', alpha=0.6,
                label=f'{label} E<{e_threshold:.0f} (n={n_below}, {tail_frac:.1f}%)')

        ax = axes[1]
        if n_above > 0 and n_below > 0:
            h_above, _ = np.histogram(t[above], bins=time_bins)
            h_below, _ = np.histogram(t[below], bins=time_bins)
            bin_centers = 0.5 * (time_bins[:-1] + time_bins[1:])
            h_total = h_above + h_below
            frac_above = np.where(h_total > 0, h_above / h_total, np.nan)
            ax.plot(bin_centers, frac_above, color=color, linewidth=1.5,
                    label=f'{label}')

    axes[0].set_xlabel('Positron time [ns]')
    axes[0].set_ylabel('Events / 1 ns')
    axes[0].set_title(f'Accepted positron time spectra')
    axes[0].legend(fontsize=8)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    watermark(axes[0])

    axes[1].set_xlabel('Positron time [ns]')
    axes[1].set_ylabel(f'Fraction with E ≥ {e_threshold:.0f} MeV')
    axes[1].set_title(f'High-energy fraction vs time')
    axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    watermark(axes[1])

    fig.tight_layout()
    path = os.path.join(output_dir, 'positron_time_spectra.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


# ========================================================================== #
#  Main                                                                       #
# ========================================================================== #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', type=str, default='benchmark_results',
                        help='Directory containing benchmark parquet files')
    parser.add_argument('--output_dir', type=str, default='benchmark_plots',
                        help='Directory to save plots')
    parser.add_argument('--tags', nargs='+', default=None,
                        help='Tags to plot (default: auto-detect from filenames)')
    parser.add_argument('--p_hit_threshold', type=float, default=0.5,
                        help='p_hit threshold for LYSO energy sum (default 0.5). '
                             'Recomputes reco energy from per-hit parquet.')
    parser.add_argument('--pimu_scale', type=float, default=8330,
                        help='Scale factor for pimu density in overlay plot (default 8330 = 1/BR).')
    parser.add_argument('--time_window', type=float, nargs=2, default=None,
                        metavar=('T_MIN', 'T_MAX'),
                        help='Positron time window [ns] for energy overlay (e.g. --time_window 2 35).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect tags from filenames
    if args.tags is None:
        files = os.listdir(args.results_dir)
        tags = sorted(set(
            f.replace('_events.parquet', '').replace('_slices.parquet', '')
             .replace('_lyso_hits.parquet', '').replace('_lyso_clusters.parquet', '')
            for f in files if f.endswith('.parquet')
        ))
        print(f"Auto-detected tags: {tags}")
    else:
        tags = args.tags

    # Load all datasets
    print("Loading data...")
    all_events = {}
    all_slices = {}
    all_lyso = {}
    for tag in tags:
        ev, sl, lyso = load_dataset(args.results_dir, tag)
        all_events[tag] = ev
        all_slices[tag] = sl
        all_lyso[tag] = lyso

    # Define color scheme
    colors = {
        'pie_eval': 'tab:red', 'pimu_eval': 'tab:blue',
        'pie': 'tab:red', 'pimu': 'tab:blue',
    }
    labels = {
        'pie_eval': r'$\pi \to e$', 'pimu_eval': r'$\pi \to \mu \to e$',
        'pie': r'$\pi \to e$', 'pimu': r'$\pi \to \mu \to e$',
    }

    def get_color(tag):
        if tag in colors:
            return colors[tag]
        if 'pimu' in tag or 'michel' in tag:
            return 'tab:blue'
        if 'pie' in tag:
            return 'tab:red'
        return 'tab:green'

    def get_label(tag):
        if tag in labels:
            return labels[tag]
        if 'pimu' in tag or 'michel' in tag:
            return r'$\pi \to \mu \to e$'
        if 'pie' in tag:
            return r'$\pi \to e$'
        return tag

    # Build dataset lists for plotting functions
    # Each tuple carries (label, raw_tag, data, ...) so labels go in titles and raw_tag in filenames
    datasets_events = [(get_label(t), t, all_events[t], get_color(t)) for t in tags if all_events[t] is not None]
    datasets_slices = [(get_label(t), t, all_slices[t]) for t in tags if all_slices[t] is not None]
    datasets_acceptance = [(get_label(t), t, all_events[t],
                           'Reds' if ('pie' in t and 'pimu' not in t) else 'Blues')
                          for t in tags if all_events[t] is not None]

    # Generate all plots
    print("\nGenerating plots...")

    print("  1. Acceptance confusion matrices")
    plot_acceptance_confusion(datasets_acceptance, args.output_dir)

    print("  1b. Acceptance false-positive breakdown")
    plot_acceptance_fp_breakdown(datasets_acceptance, args.output_dir)

    print("  1c. Acceptance false-negative breakdown")
    plot_acceptance_fn_breakdown(datasets_acceptance, args.output_dir)

    print("  2. Role confusion matrices")
    plot_role_confusion(datasets_slices, args.output_dir)

    print("  3. Positron angle RMS")
    plot_angle_rms(datasets_events, args.output_dir)

    print("  3b. Positron angle median bias vs theta")
    plot_angle_median_bias(datasets_events, args.output_dir)

    print("  3b2. Positron angle mean bias vs cos(theta)")
    plot_angle_mean_bias(datasets_events, args.output_dir)

    print("  3b3. vMF kappa vs theta")
    plot_kappa_vs_theta(datasets_events, args.output_dir)

    print("  3c. Angle residual at 120° cut")
    plot_angle_residual_at_cut(datasets_events, args.output_dir)

    print("  3d. Angle 2D (truth vs reco)")
    plot_angle_2d(datasets_events, args.output_dir)

    print(f"  4. Positron energy spectrum (p_hit > {args.p_hit_threshold})")
    plot_energy_spectrum(datasets_events, args.output_dir,
                         all_lyso=all_lyso if args.p_hit_threshold != 0.5 else None,
                         p_hit_threshold=args.p_hit_threshold)

    tw_str = f", t={args.time_window[0]}-{args.time_window[1]}ns" if args.time_window else ""
    print(f"  4b. Energy density overlay (pie vs pimu x{args.pimu_scale}{tw_str})")
    plot_energy_overlay(datasets_events, args.output_dir,
                        all_lyso=all_lyso if args.p_hit_threshold != 0.5 else None,
                        p_hit_threshold=args.p_hit_threshold,
                        pimu_scale=args.pimu_scale,
                        time_window=args.time_window)

    print("  5. Pion stop resolution")
    plot_pion_stop_resolution(datasets_events, args.output_dir)

    print("  6. Endpoint resolution")
    plot_endpoint_resolution(datasets_slices, args.output_dir)

    print("  7. Positron hit IoU")
    plot_positron_iou(datasets_events, args.output_dir)

    print("  8. Positron precision/recall")
    plot_precision_recall(datasets_events, args.output_dir)

    print("  9. Energy resolution 2D")
    plot_energy_resolution_2d(datasets_events, args.output_dir)

    print("  10. Role probability distributions")
    plot_role_probabilities(datasets_slices, args.output_dir)

    print("  11. Positron time spectra (above/below 56 MeV)")
    plot_positron_time_spectra(datasets_events, args.output_dir)

    print(f"\nDone! All plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()