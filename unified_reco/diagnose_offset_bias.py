"""
Test whether the angular bias is consistent with the model predicting
positron direction from the wrong reference point (e.g. calorimeter
centre / origin) instead of from the true pion stop.

For each event we compute the "parallax direction" — the direction a
calorimeter-centred observer would infer — and check whether the
model's error correlates with that geometric parallax.

Usage:
    python diagnose_offset_bias.py --results_dir /pioneerML/benchmark_results_v2c \
                                   --output_dir /pioneerML/offset_diagnostics
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SENTINEL = -999.0


def load_pie_events(results_dir):
    for f in sorted(os.listdir(results_dir)):
        if f.endswith('_events.parquet') and 'pie' in f:
            df = pd.read_parquet(os.path.join(results_dir, f))
            return f.replace('_events.parquet', ''), df
    return None, None


def gate_events(ev):
    htp_gate = (ev['truth_htp'].values == 1) & \
               (ev['pred_htp'].values != SENTINEL) & \
               (ev['pred_htp'].values > 0.5)
    iou_ok = ev['pred_pos_iou'].values >= 0.95
    return htp_gate & iou_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output_dir', default='./offset_diagnostics')
    parser.add_argument('--calo_radius', type=float, default=260.0,
                        help='Approximate LYSO inner radius [mm]')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tag, ev = load_pie_events(args.results_dir)
    if ev is None:
        print("No pie events parquet found!")
        return

    gate = gate_events(ev)
    print(f"Loaded {tag}: {gate.sum()} gated events out of {len(ev)}")

    # --- Extract quantities ---
    theta_true = ev['truth_theta'].values[gate]
    phi_true = ev['truth_phi'].values[gate]
    theta_pred_rad = ev['pred_polar_angle'].values[gate]

    d_true = np.column_stack([
        np.sin(theta_true) * np.cos(phi_true),
        np.sin(theta_true) * np.sin(phi_true),
        np.cos(theta_true),
    ])

    d_pred = np.column_stack([
        ev['pred_positron_dir_x'].values[gate],
        ev['pred_positron_dir_y'].values[gate],
        ev['pred_positron_dir_z'].values[gate],
    ])
    d_pred_norm = np.linalg.norm(d_pred, axis=1, keepdims=True)
    d_pred_norm = np.where(d_pred_norm > 0, d_pred_norm, 1.0)
    d_pred = d_pred / d_pred_norm

    ps_true = np.column_stack([
        ev['truth_pion_stop_x'].values[gate],
        ev['truth_pion_stop_y'].values[gate],
        ev['truth_pion_stop_z'].values[gate],
    ])
    ps_pred = np.column_stack([
        ev['pred_pion_stop_x'].values[gate],
        ev['pred_pion_stop_y'].values[gate],
        ev['pred_pion_stop_z'].values[gate],
    ])

    theta_true_deg = np.degrees(theta_true)
    theta_pred_deg = np.degrees(theta_pred_rad)
    angle_residual = theta_true_deg - theta_pred_deg  # >0 means model under-predicts theta

    # --- Figure 1: Angle residual vs pion stop position ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{tag}: angle residual vs pion stop position (n={gate.sum()})', fontsize=13)

    labels = ['x', 'y', 'z']
    ranges_mm = [(-9, 9), (-9, 9), (3, 4)]

    for i, (lab, rng) in enumerate(zip(labels, ranges_mm)):
        # vs truth pion stop
        axes[0, i].hist2d(ps_true[:, i], angle_residual,
                          bins=[np.linspace(rng[0], rng[1], 60),
                                np.linspace(-10, 10, 60)],
                          cmap='viridis', norm=LogNorm())
        axes[0, i].axhline(0, color='w', ls='--', lw=0.8)
        axes[0, i].set_xlabel(f'truth pion stop {lab} [mm]')
        axes[0, i].set_ylabel('angle residual [deg]')
        axes[0, i].set_title(f'vs truth pion stop {lab}')

        # vs predicted pion stop
        axes[1, i].hist2d(ps_pred[:, i], angle_residual,
                          bins=[np.linspace(rng[0], rng[1], 60),
                                np.linspace(-10, 10, 60)],
                          cmap='viridis', norm=LogNorm())
        axes[1, i].axhline(0, color='w', ls='--', lw=0.8)
        axes[1, i].set_xlabel(f'pred pion stop {lab} [mm]')
        axes[1, i].set_ylabel('angle residual [deg]')
        axes[1, i].set_title(f'vs pred pion stop {lab}')

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'angle_res_vs_pion_stop_2d.png'), dpi=150)
    plt.close()

    # --- Figure 2: Binned profiles (median residual vs pion stop) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{tag}: median angle residual vs pion stop position', fontsize=13)

    for i, (lab, rng) in enumerate(zip(labels, ranges_mm)):
        bins = np.linspace(rng[0], rng[1], 25)
        centers = 0.5 * (bins[:-1] + bins[1:])
        med_truth, med_pred = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mt = (ps_true[:, i] >= lo) & (ps_true[:, i] < hi)
            mp = (ps_pred[:, i] >= lo) & (ps_pred[:, i] < hi)
            med_truth.append(np.median(angle_residual[mt]) if mt.sum() > 20 else np.nan)
            med_pred.append(np.median(angle_residual[mp]) if mp.sum() > 20 else np.nan)

        axes[i].plot(centers, med_truth, 'o-', color='C0', markersize=3, label='vs truth')
        axes[i].plot(centers, med_pred, 's-', color='C3', markersize=3, label='vs pred')
        axes[i].axhline(0, color='k', ls='--', lw=0.8)
        axes[i].set_xlabel(f'pion stop {lab} [mm]')
        axes[i].set_ylabel('median angle residual [deg]')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'angle_res_vs_pion_stop_profiles.png'), dpi=150)
    plt.close()

    # --- Figure 2b: 1D pion stop z distributions (truth vs pred) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{tag}: pion stop z distribution — truth vs predicted', fontsize=13)
    fine_bins = np.linspace(2.5, 5.0, 200)
    axes[0].hist(ps_true[:, 2], bins=fine_bins, alpha=0.6, label='truth', density=True)
    axes[0].hist(ps_pred[:, 2], bins=fine_bins, alpha=0.6, label='predicted', density=True)
    axes[0].set_xlabel('pion stop z [mm]')
    axes[0].set_ylabel('density')
    axes[0].set_title('Fine-binned z distribution (peak region)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    coarse_bins = np.linspace(0, 6, 120)
    axes[1].hist(ps_true[:, 2], bins=coarse_bins, alpha=0.6, label='truth', density=True)
    axes[1].hist(ps_pred[:, 2], bins=coarse_bins, alpha=0.6, label='predicted', density=True)
    axes[1].set_xlabel('pion stop z [mm]')
    axes[1].set_ylabel('density')
    axes[1].set_title('Full z range')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pion_stop_z_1d.png'), dpi=150)
    plt.close()

    # --- Figure 3: Parallax model ---
    # If the model predicts direction from some offset point P instead of
    # the true pion stop, we can compute the expected angular error.
    # Calorimeter hit ≈ pion_stop + L * d_true, where L = calo_radius / sin(theta)
    # Direction from offset point: d_model = (calo_hit - P_offset) / |...|
    R = args.calo_radius

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Parallax model: expected bias if direction measured from offset point', fontsize=13)

    # Compute calorimeter hit position (propagate along true direction to radius R)
    # Spherical shell: |pion_stop + t * d_true| = R
    # t^2 |d|^2 + 2t (ps · d) + |ps|^2 - R^2 = 0
    ps_dot_d = np.sum(ps_true * d_true, axis=1)
    ps_sq = np.sum(ps_true**2, axis=1)
    discriminant = ps_dot_d**2 - (ps_sq - R**2)
    valid = discriminant > 0
    t_hit = np.zeros(len(d_true))
    t_hit[valid] = -ps_dot_d[valid] + np.sqrt(discriminant[valid])
    calo_hit = ps_true + t_hit[:, None] * d_true

    # Test several offset hypotheses
    offsets = [
        ("origin (0,0,0)", np.array([0.0, 0.0, 0.0])),
        ("z+0.3mm", np.array([0.0, 0.0, 0.3])),
        ("pred pion stop", None),  # special: use predicted pion stop
    ]

    for col, (offset_label, offset_vec) in enumerate(offsets):
        if offset_vec is not None:
            ref_point = offset_vec[None, :]
        else:
            ref_point = ps_pred

        d_from_ref = calo_hit - ref_point
        d_from_ref_norm = d_from_ref / np.linalg.norm(d_from_ref, axis=1, keepdims=True)

        # Theta from this reference direction
        theta_from_ref = np.degrees(np.arccos(np.clip(d_from_ref_norm[:, 2], -1, 1)))
        expected_bias = theta_true_deg - theta_from_ref  # parallax-induced error

        # Top: expected bias vs observed bias
        ok = valid & (np.abs(expected_bias) < 15) & (np.abs(angle_residual) < 15)
        axes[0, col].hist2d(expected_bias[ok], angle_residual[ok],
                            bins=[np.linspace(-5, 5, 80), np.linspace(-5, 5, 80)],
                            cmap='viridis', norm=LogNorm())
        axes[0, col].plot([-5, 5], [-5, 5], 'r--', lw=1, label='y=x')
        axes[0, col].set_xlabel(f'expected bias from {offset_label} [deg]')
        axes[0, col].set_ylabel('observed angle residual [deg]')
        axes[0, col].set_title(f'ref = {offset_label}')
        axes[0, col].legend(fontsize=8)

        # Bottom: expected bias vs theta
        axes[1, col].hist2d(theta_true_deg[ok], expected_bias[ok],
                            bins=[np.linspace(0, 180, 60), np.linspace(-5, 5, 60)],
                            cmap='viridis', norm=LogNorm())
        axes[1, col].axhline(0, color='w', ls='--', lw=0.8)
        axes[1, col].set_xlabel('theta_true [deg]')
        axes[1, col].set_ylabel(f'expected bias from {offset_label} [deg]')

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'parallax_model.png'), dpi=150)
    plt.close()

    # --- Figure 4: Scan over z-offset values ---
    # For each offset, compute correlation between expected and observed bias
    z_offsets = np.linspace(-5, 5, 51)
    correlations = []
    slopes = []

    for dz in z_offsets:
        ref = np.array([[0.0, 0.0, dz]])
        d_ref = calo_hit - ref
        d_ref /= np.linalg.norm(d_ref, axis=1, keepdims=True)
        theta_ref = np.degrees(np.arccos(np.clip(d_ref[:, 2], -1, 1)))
        exp_bias = theta_true_deg - theta_ref
        ok = valid & np.isfinite(exp_bias) & np.isfinite(angle_residual)
        if ok.sum() > 100:
            corr = np.corrcoef(exp_bias[ok], angle_residual[ok])[0, 1]
            correlations.append(corr)
            A = np.column_stack([exp_bias[ok], np.ones(ok.sum())])
            slope, _ = np.linalg.lstsq(A, angle_residual[ok], rcond=None)[0]
            slopes.append(slope)
        else:
            correlations.append(np.nan)
            slopes.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(z_offsets, correlations, 'b-', lw=2)
    axes[0].axvline(0.3, color='r', ls='--', label='ATAR offset (0.3mm)')
    axes[0].set_xlabel('assumed z offset [mm]')
    axes[0].set_ylabel('correlation(expected bias, observed bias)')
    axes[0].set_title('Parallax model: z-offset scan')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z_offsets, slopes, 'b-', lw=2)
    axes[1].axvline(0.3, color='r', ls='--', label='ATAR offset (0.3mm)')
    axes[1].axhline(1.0, color='gray', ls=':', lw=0.8, label='slope=1 (perfect match)')
    axes[1].set_xlabel('assumed z offset [mm]')
    axes[1].set_ylabel('slope (observed vs expected)')
    axes[1].set_title('Best-fit slope of observed vs expected bias')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    best_idx = np.nanargmax(np.abs(correlations))
    best_z = z_offsets[best_idx]
    best_corr = correlations[best_idx]
    print(f"\nBest z-offset: {best_z:.2f} mm (correlation = {best_corr:.4f})")
    print(f"ATAR offset (0.3mm): correlation = {correlations[np.argmin(np.abs(z_offsets - 0.3))]:.4f}")

    fig.suptitle(f'Z-offset scan (best = {best_z:.1f}mm, r = {best_corr:.3f})', fontsize=13)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'z_offset_scan.png'), dpi=150)
    plt.close()

    # --- Figure 5: 3D offset scan (x, y, z independently) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Offset scan: correlation with observed bias per axis', fontsize=13)

    for ax_idx, axis_label in enumerate(['x', 'y', 'z']):
        offsets_1d = np.linspace(-5, 5, 51)
        corrs = []
        for d in offsets_1d:
            ref = np.zeros((1, 3))
            ref[0, ax_idx] = d
            d_ref = calo_hit - ref
            d_ref /= np.linalg.norm(d_ref, axis=1, keepdims=True)
            theta_ref = np.degrees(np.arccos(np.clip(d_ref[:, 2], -1, 1)))
            exp_bias = theta_true_deg - theta_ref
            ok = valid & np.isfinite(exp_bias) & np.isfinite(angle_residual)
            if ok.sum() > 100:
                corrs.append(np.corrcoef(exp_bias[ok], angle_residual[ok])[0, 1])
            else:
                corrs.append(np.nan)

        axes[ax_idx].plot(offsets_1d, corrs, 'b-', lw=2)
        if axis_label == 'z':
            axes[ax_idx].axvline(0.3, color='r', ls='--', label='ATAR offset')
        axes[ax_idx].set_xlabel(f'{axis_label} offset [mm]')
        axes[ax_idx].set_ylabel('correlation')
        axes[ax_idx].set_title(f'{axis_label}-offset scan')
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].legend()

    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'offset_scan_3d.png'), dpi=150)
    plt.close()

    # --- Figure 6: Pion-stop correction ---
    # Take the model's predicted direction, propagate from the origin to the
    # calorimeter, then recompute the direction from the true pion stop to
    # that calorimeter hit.  Compare uncorrected vs corrected.
    R = args.calo_radius

    # Propagate predicted direction from origin to calorimeter shell
    # |origin + t * d_pred| = R  =>  t = R (since origin = 0 and |d_pred| = 1)
    # More precisely: t^2 - R^2 = 0 => t = R
    # But let's also try from predicted pion stop
    ref_points = [
        ("from origin", np.zeros((1, 3))),
        ("from pred pion stop", ps_pred),
        ("from truth pion stop", ps_true),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Pion-stop correction: recompute direction from pion stop to calo hit', fontsize=13)

    for col, (ref_label, ref_pt) in enumerate(ref_points):
        # Propagate d_pred from ref_pt to calorimeter shell |ref + t*d| = R
        rd = np.sum(ref_pt * d_pred, axis=1)
        r2 = np.sum(ref_pt**2, axis=1)
        disc = rd**2 - (r2 - R**2)
        ok = disc > 0
        t_calo = np.zeros(len(d_pred))
        t_calo[ok] = -rd[ok] + np.sqrt(disc[ok])
        calo_hit_pred = ref_pt + t_calo[:, None] * d_pred

        # Recompute direction from TRUE pion stop to this calo hit
        d_corrected = calo_hit_pred - ps_true
        d_corrected /= np.linalg.norm(d_corrected, axis=1, keepdims=True)
        theta_corrected = np.degrees(np.arccos(np.clip(d_corrected[:, 2], -1, 1)))
        residual_corrected = theta_true_deg - theta_corrected

        # Row 0: corrected residual vs theta
        m = ok & np.isfinite(residual_corrected)
        axes[0, col].hist2d(theta_true_deg[m], residual_corrected[m],
                            bins=[np.linspace(0, 180, 60), np.linspace(-10, 10, 60)],
                            cmap='viridis', norm=LogNorm())
        axes[0, col].axhline(0, color='w', ls='--', lw=0.8)
        axes[0, col].set_xlabel('theta_true [deg]')
        axes[0, col].set_ylabel('corrected residual [deg]')
        axes[0, col].set_title(f'corrected ({ref_label})')

        # Row 1: corrected residual vs pion stop z
        axes[1, col].hist2d(ps_true[m, 2], residual_corrected[m],
                            bins=[np.linspace(-15, 15, 60), np.linspace(-10, 10, 60)],
                            cmap='viridis', norm=LogNorm())
        axes[1, col].axhline(0, color='w', ls='--', lw=0.8)
        axes[1, col].set_xlabel('truth pion stop z [mm]')
        axes[1, col].set_ylabel('corrected residual [deg]')

        # Row 2: profile — median corrected residual vs pion stop z
        bins_z = np.linspace(-15, 15, 25)
        centers_z = 0.5 * (bins_z[:-1] + bins_z[1:])
        med_orig, med_corr = [], []
        for lo, hi in zip(bins_z[:-1], bins_z[1:]):
            mz = m & (ps_true[:, 2] >= lo) & (ps_true[:, 2] < hi)
            med_corr.append(np.median(residual_corrected[mz]) if mz.sum() > 20 else np.nan)
            med_orig.append(np.median(angle_residual[mz]) if mz.sum() > 20 else np.nan)

        axes[2, col].plot(centers_z, med_orig, 'o-', color='C3', markersize=3, label='uncorrected')
        axes[2, col].plot(centers_z, med_corr, 's-', color='C0', markersize=3, label='corrected')
        axes[2, col].axhline(0, color='k', ls='--', lw=0.8)
        axes[2, col].set_xlabel('truth pion stop z [mm]')
        axes[2, col].set_ylabel('median residual [deg]')
        axes[2, col].legend(fontsize=9)
        axes[2, col].set_title(f'profile ({ref_label})')

        # Print summary
        rms_orig = np.sqrt(np.mean(angle_residual[m]**2))
        rms_corr = np.sqrt(np.mean(residual_corrected[m]**2))
        bias_orig = np.median(angle_residual[m])
        bias_corr = np.median(residual_corrected[m])
        print(f"\n{ref_label}:")
        print(f"  RMS:    {rms_orig:.3f} -> {rms_corr:.3f} deg")
        print(f"  median: {bias_orig:.3f} -> {bias_corr:.3f} deg")

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pion_stop_correction.png'), dpi=150)
    plt.close()

    print("\nDone. Plots saved to", args.output_dir)


if __name__ == '__main__':
    main()
