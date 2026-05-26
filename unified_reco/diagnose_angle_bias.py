"""
Diagnose the source of angle bias and FP/FN asymmetry at the 120° cut.

Examines:
  1. Error distribution shape near 120° — symmetric or skewed?
  2. Bias as a function of positron hit count — does more info reduce bias?
  3. Bias as a function of pion stop quality — correlated?
  4. Predicted direction components (dx, dy, dz) vs truth — which component is biased?
  5. Dir head input features: are they systematically different above/below 120°?

Usage (inside container):
    python diagnose_angle_bias.py --results_dir /pioneerML/benchmark_results_v2 \
                                  --output_dir /pioneerML/angle_bias_diagnostics
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


def load_events(results_dir):
    datasets = []
    for f in sorted(os.listdir(results_dir)):
        if f.endswith('_events.parquet'):
            tag = f.replace('_events.parquet', '')
            df = pd.read_parquet(os.path.join(results_dir, f))
            datasets.append((tag, df))
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output_dir', default='./angle_bias_diagnostics')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = load_events(args.results_dir)

    for tag, ev in datasets:
        if 'pie' not in tag:
            continue

        print(f"\n{'='*60}")
        print(f"  {tag}")
        print(f"{'='*60}")

        htp_ok = (ev['truth_htp'].values == 1) & \
                 (ev['pred_htp'].values != SENTINEL) & \
                 (ev['pred_htp'].values > 0.5)
        iou_ok = ev['pred_pos_iou'].values >= 0.95
        gate = htp_ok & iou_ok

        theta_true_rad = ev['truth_theta'].values[gate]
        phi_true_rad = ev['truth_phi'].values[gate]
        theta_true = np.degrees(theta_true_rad)
        theta_reco = np.degrees(ev['pred_polar_angle'].values[gate])
        residual = theta_true - theta_reco

        truth_dz = np.cos(theta_true_rad)
        truth_dx = np.sin(theta_true_rad) * np.cos(phi_true_rad)
        truth_dy = np.sin(theta_true_rad) * np.sin(phi_true_rad)
        pred_dz = ev['pred_positron_dir_z'].values[gate]
        pred_dx = ev['pred_positron_dir_x'].values[gate]
        pred_dy = ev['pred_positron_dir_y'].values[gate]

        iou = ev['pred_pos_iou'].values[gate]

        ps_err = np.sqrt(
            (ev['truth_pion_stop_x'].values[gate] - ev['pred_pion_stop_x'].values[gate])**2 +
            (ev['truth_pion_stop_y'].values[gate] - ev['pred_pion_stop_y'].values[gate])**2 +
            (ev['truth_pion_stop_z'].values[gate] - ev['pred_pion_stop_z'].values[gate])**2
        )

        # =============================================================
        # 1. Error distribution near the 120° cut — symmetric or skewed?
        # =============================================================
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (lo, hi, label) in zip(axes, [
            (115, 125, '115°-125° (cut region)'),
            (105, 115, '105°-115° (below cut)'),
            (125, 135, '125°-135° (above cut)'),
        ]):
            m = (theta_true >= lo) & (theta_true < hi)
            r = residual[m]
            n_events = m.sum()
            ax.hist(r, bins=np.linspace(-20, 20, 81), color='steelblue', alpha=0.7)
            ax.axvline(0, color='k', ls='--', lw=1)
            ax.axvline(np.median(r), color='red', ls='-', lw=1.5,
                       label=f'median={np.median(r):+.2f}°')
            ax.axvline(np.mean(r), color='orange', ls='-', lw=1.5,
                       label=f'mean={np.mean(r):+.2f}°')

            # Skewness
            from scipy.stats import skew
            sk = skew(r)
            ax.set_title(f'{label}\nn={n_events:,}  skew={sk:+.3f}')
            ax.set_xlabel('θ_true - θ_reco [deg]')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{tag}: Error distribution near 120° cut', fontsize=13)
        fig.tight_layout()
        path = os.path.join(args.output_dir, f'error_dist_near_cut_{tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")

        # =============================================================
        # 2. Component-level bias: dz, dx, dy residuals vs theta
        # =============================================================
        bins_theta = np.linspace(0, 180, 37)
        centers = 0.5 * (bins_theta[:-1] + bins_theta[1:])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        components = [
            ('dz', truth_dz, pred_dz),
            ('dx', truth_dx, pred_dx),
            ('dy', truth_dy, pred_dy),
        ]
        for ax, (name, truth_c, pred_c) in zip(axes, components):
            comp_res = truth_c - pred_c
            medians = []
            for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
                m = (theta_true >= lo) & (theta_true < hi)
                medians.append(np.median(comp_res[m]) if m.sum() > 10 else np.nan)
            ax.plot(centers, medians, 'o-', markersize=3, color='steelblue')
            ax.axhline(0, color='k', ls='--', lw=0.8)
            ax.axvline(90, color='gray', ls=':', lw=0.8)
            ax.axvline(120, color='gray', ls='--', lw=0.8)
            ax.set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')
            ax.set_ylabel(f'median(truth_{name} - pred_{name})')
            ax.set_title(f'{name} component bias')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{tag}: Direction component biases vs θ', fontsize=13)
        fig.tight_layout()
        path = os.path.join(args.output_dir, f'component_bias_{tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")

        # =============================================================
        # 3. Bias vs IoU quality — does better positron ID reduce bias?
        # =============================================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        iou_cuts = [(0.95, 0.99, 'IoU 0.95-0.99'), (0.99, 1.01, 'IoU > 0.99')]
        for label_iou, (ilo, ihi, lbl) in enumerate(iou_cuts):
            iou_m = (iou >= ilo) & (iou < ihi)
            medians = []
            for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
                m = (theta_true >= lo) & (theta_true < hi) & iou_m
                medians.append(np.median(residual[m]) if m.sum() > 10 else np.nan)
            axes[0].plot(centers, medians, 'o-', markersize=3,
                         label=f'{lbl} (n={iou_m.sum():,})')

        axes[0].axhline(0, color='k', ls='--', lw=0.8)
        axes[0].axvline(90, color='gray', ls=':', lw=0.8)
        axes[0].axvline(120, color='gray', ls='--', lw=0.8)
        axes[0].set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')
        axes[0].set_ylabel('median angle residual [deg]')
        axes[0].set_title('Angle bias by IoU quality')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Bias vs pion stop quality
        ps_cuts = [(0, 0.1, 'ps err < 0.1mm'), (0.1, 0.3, 'ps err 0.1-0.3mm'),
                   (0.3, 5.0, 'ps err > 0.3mm')]
        for ps_lo, ps_hi, lbl in ps_cuts:
            ps_m = (ps_err >= ps_lo) & (ps_err < ps_hi)
            medians = []
            for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
                m = (theta_true >= lo) & (theta_true < hi) & ps_m
                medians.append(np.median(residual[m]) if m.sum() > 10 else np.nan)
            axes[1].plot(centers, medians, 'o-', markersize=3,
                         label=f'{lbl} (n={ps_m.sum():,})')

        axes[1].axhline(0, color='k', ls='--', lw=0.8)
        axes[1].axvline(90, color='gray', ls=':', lw=0.8)
        axes[1].axvline(120, color='gray', ls='--', lw=0.8)
        axes[1].set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')
        axes[1].set_ylabel('median angle residual [deg]')
        axes[1].set_title('Angle bias by pion stop quality')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f'{tag}: Does event quality affect the bias?', fontsize=13)
        fig.tight_layout()
        path = os.path.join(args.output_dir, f'bias_vs_quality_{tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")

        # =============================================================
        # 4. Predicted dz vs truth dz — is there a slope != 1?
        # =============================================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist2d(truth_dz, pred_dz,
                       bins=[np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)],
                       cmap='viridis', norm=LogNorm())
        axes[0].plot([-1, 1], [-1, 1], 'r--', lw=1)
        axes[0].set_xlabel('truth dz')
        axes[0].set_ylabel('pred dz')
        axes[0].set_title('Predicted vs truth dz (= cos θ)')
        axes[0].grid(True, alpha=0.2)

        # Binned median pred_dz vs truth_dz
        bins_dz = np.linspace(-1, 1, 41)
        centers_dz = 0.5 * (bins_dz[:-1] + bins_dz[1:])
        med_pred_dz = []
        for lo, hi in zip(bins_dz[:-1], bins_dz[1:]):
            m = (truth_dz >= lo) & (truth_dz < hi)
            med_pred_dz.append(np.median(pred_dz[m]) if m.sum() > 10 else np.nan)
        med_pred_dz = np.array(med_pred_dz)

        axes[1].plot(centers_dz, med_pred_dz, 'o-', markersize=3, color='steelblue',
                     label='median pred dz')
        axes[1].plot([-1, 1], [-1, 1], 'r--', lw=1, label='perfect')
        axes[1].axvline(np.cos(np.radians(120)), color='gray', ls='--', lw=0.8,
                        label='cos(120°)')

        # Fit slope
        valid = ~np.isnan(med_pred_dz)
        if valid.sum() > 2:
            slope, intercept = np.polyfit(centers_dz[valid], med_pred_dz[valid], 1)
            axes[1].plot(centers_dz, slope * centers_dz + intercept, 'g-', lw=1.5,
                         label=f'fit: slope={slope:.4f}, int={intercept:.4f}')

        axes[1].set_xlabel('truth dz')
        axes[1].set_ylabel('median pred dz')
        axes[1].set_title('Median predicted dz vs truth dz\n(slope < 1 = regression to mean)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f'{tag}: cos(θ) prediction analysis', fontsize=13)
        fig.tight_layout()
        path = os.path.join(args.output_dir, f'dz_regression_{tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")

        # =============================================================
        # 5. FP vs FN: what distinguishes them?
        # =============================================================
        truth_pass = theta_true < 120
        reco_pass = theta_reco < 120
        fp_mask = ~truth_pass & reco_pass
        fn_mask = truth_pass & ~reco_pass

        print(f"\n  Angle-gated FP: {fp_mask.sum():,}   FN: {fn_mask.sum():,}"
              f"   ratio: {fp_mask.sum()/max(fn_mask.sum(),1):.2f}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # FP: truth theta distribution
        axes[0, 0].hist(theta_true[fp_mask], bins=np.linspace(100, 160, 61),
                        color='red', alpha=0.6, label=f'FP (n={fp_mask.sum():,})')
        axes[0, 0].hist(theta_true[fn_mask], bins=np.linspace(100, 140, 41),
                        color='blue', alpha=0.6, label=f'FN (n={fn_mask.sum():,})')
        axes[0, 0].axvline(120, color='k', ls='--')
        axes[0, 0].set_xlabel('truth θ [deg]')
        axes[0, 0].set_title('FP and FN: truth θ distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # FP vs FN: IoU distribution
        axes[0, 1].hist(iou[fp_mask], bins=np.linspace(0.95, 1.0, 51),
                        color='red', alpha=0.6, density=True, label='FP')
        axes[0, 1].hist(iou[fn_mask], bins=np.linspace(0.95, 1.0, 51),
                        color='blue', alpha=0.6, density=True, label='FN')
        axes[0, 1].set_xlabel('positron IoU')
        axes[0, 1].set_title('FP vs FN: positron hit quality')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # FP vs FN: pion stop error
        axes[1, 0].hist(ps_err[fp_mask], bins=np.linspace(0, 0.5, 51),
                        color='red', alpha=0.6, density=True, label='FP')
        axes[1, 0].hist(ps_err[fn_mask], bins=np.linspace(0, 0.5, 51),
                        color='blue', alpha=0.6, density=True, label='FN')
        axes[1, 0].set_xlabel('pion stop 3D error [mm]')
        axes[1, 0].set_title('FP vs FN: pion stop quality')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # FP vs FN: angle error magnitude
        axes[1, 1].hist(np.abs(residual[fp_mask]), bins=np.linspace(0, 20, 81),
                        color='red', alpha=0.6, density=True, label='FP')
        axes[1, 1].hist(np.abs(residual[fn_mask]), bins=np.linspace(0, 20, 81),
                        color='blue', alpha=0.6, density=True, label='FN')
        axes[1, 1].set_xlabel('|θ_true - θ_reco| [deg]')
        axes[1, 1].set_title('FP vs FN: angle error magnitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(f'{tag}: FP vs FN characterization', fontsize=13)
        fig.tight_layout()
        path = os.path.join(args.output_dir, f'fp_vs_fn_{tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")

        # =============================================================
        # 6. dz bias vs other event properties
        # =============================================================
        dz_residual = truth_dz - pred_dz

        truth_ps_x = ev['truth_pion_stop_x'].values[gate]
        truth_ps_y = ev['truth_pion_stop_y'].values[gate]
        truth_ps_z = ev['truth_pion_stop_z'].values[gate]
        pred_ps_z = ev['pred_pion_stop_z'].values[gate]

        truth_energy = ev['truth_positron_energy'].values[gate] \
            if 'truth_positron_energy' in ev.columns else None
        pred_energy = ev['pred_positron_energy'].values[gate] \
            if 'pred_positron_energy' in ev.columns else None

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 6a: dz bias vs truth pion stop z
        bins_psz = np.linspace(1.2, 4.8, 25)
        centers_psz = 0.5 * (bins_psz[:-1] + bins_psz[1:])
        med_dz_vs_psz = []
        for lo, hi in zip(bins_psz[:-1], bins_psz[1:]):
            m = (truth_ps_z >= lo) & (truth_ps_z < hi)
            med_dz_vs_psz.append(np.median(dz_residual[m]) if m.sum() > 10 else np.nan)
        axes[0, 0].plot(centers_psz, med_dz_vs_psz, 'o-', markersize=3, color='steelblue')
        axes[0, 0].axhline(0, color='k', ls='--', lw=0.8)
        axes[0, 0].set_xlabel('truth pion stop z [mm]')
        axes[0, 0].set_ylabel('median(truth_dz - pred_dz)')
        axes[0, 0].set_title('dz bias vs pion stop z position')
        axes[0, 0].grid(True, alpha=0.3)

        # 6b: dz bias vs truth pion stop x
        bins_psx = np.linspace(-8, 8, 33)
        centers_psx = 0.5 * (bins_psx[:-1] + bins_psx[1:])
        med_dz_vs_psx = []
        for lo, hi in zip(bins_psx[:-1], bins_psx[1:]):
            m = (truth_ps_x >= lo) & (truth_ps_x < hi)
            med_dz_vs_psx.append(np.median(dz_residual[m]) if m.sum() > 10 else np.nan)
        axes[0, 1].plot(centers_psx, med_dz_vs_psx, 'o-', markersize=3, color='steelblue')
        axes[0, 1].axhline(0, color='k', ls='--', lw=0.8)
        axes[0, 1].set_xlabel('truth pion stop x [mm]')
        axes[0, 1].set_ylabel('median(truth_dz - pred_dz)')
        axes[0, 1].set_title('dz bias vs pion stop x position')
        axes[0, 1].grid(True, alpha=0.3)

        # 6c: dz bias vs truth phi
        bins_phi = np.linspace(-180, 180, 37)
        centers_phi = 0.5 * (bins_phi[:-1] + bins_phi[1:])
        phi_deg = np.degrees(phi_true_rad)
        med_dz_vs_phi = []
        for lo, hi in zip(bins_phi[:-1], bins_phi[1:]):
            m = (phi_deg >= lo) & (phi_deg < hi)
            med_dz_vs_phi.append(np.median(dz_residual[m]) if m.sum() > 10 else np.nan)
        axes[0, 2].plot(centers_phi, med_dz_vs_phi, 'o-', markersize=3, color='steelblue')
        axes[0, 2].axhline(0, color='k', ls='--', lw=0.8)
        axes[0, 2].set_xlabel('truth φ [deg]')
        axes[0, 2].set_ylabel('median(truth_dz - pred_dz)')
        axes[0, 2].set_title('dz bias vs azimuthal angle φ')
        axes[0, 2].grid(True, alpha=0.3)

        # 6d: theta bias vs truth phi (is the bias phi-dependent?)
        med_theta_vs_phi = []
        for lo, hi in zip(bins_phi[:-1], bins_phi[1:]):
            m = (phi_deg >= lo) & (phi_deg < hi)
            med_theta_vs_phi.append(np.median(residual[m]) if m.sum() > 10 else np.nan)
        axes[1, 0].plot(centers_phi, med_theta_vs_phi, 'o-', markersize=3, color='steelblue')
        axes[1, 0].axhline(0, color='k', ls='--', lw=0.8)
        axes[1, 0].set_xlabel('truth φ [deg]')
        axes[1, 0].set_ylabel('median(θ_true - θ_reco) [deg]')
        axes[1, 0].set_title('θ bias vs azimuthal angle φ')
        axes[1, 0].grid(True, alpha=0.3)

        # 6e: dz bias vs positron energy
        if truth_energy is not None:
            bins_e = np.linspace(0, 90, 31)
            centers_e = 0.5 * (bins_e[:-1] + bins_e[1:])
            med_dz_vs_e = []
            for lo, hi in zip(bins_e[:-1], bins_e[1:]):
                m = (truth_energy >= lo) & (truth_energy < hi)
                med_dz_vs_e.append(np.median(dz_residual[m]) if m.sum() > 10 else np.nan)
            axes[1, 1].plot(centers_e, med_dz_vs_e, 'o-', markersize=3, color='steelblue')
            axes[1, 1].axhline(0, color='k', ls='--', lw=0.8)
            axes[1, 1].set_xlabel('truth positron energy [MeV]')
            axes[1, 1].set_ylabel('median(truth_dz - pred_dz)')
            axes[1, 1].set_title('dz bias vs positron energy')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'no energy column', transform=axes[1, 1].transAxes,
                           ha='center')

        # 6f: 2D — theta bias vs (theta, phi) to see if there's a preferred direction
        theta_2d = theta_true
        phi_2d = phi_deg
        nbins = 30
        bias_2d = np.full((nbins, nbins), np.nan)
        theta_edges = np.linspace(0, 180, nbins + 1)
        phi_edges = np.linspace(-180, 180, nbins + 1)
        for i in range(nbins):
            for j in range(nbins):
                m = ((theta_2d >= theta_edges[i]) & (theta_2d < theta_edges[i+1]) &
                     (phi_2d >= phi_edges[j]) & (phi_2d < phi_edges[j+1]))
                if m.sum() > 20:
                    bias_2d[i, j] = np.median(residual[m])
        im = axes[1, 2].imshow(bias_2d, origin='lower', aspect='auto',
                               extent=[-180, 180, 0, 180], cmap='RdBu_r',
                               vmin=-2, vmax=2)
        axes[1, 2].set_xlabel('truth φ [deg]')
        axes[1, 2].set_ylabel('truth θ [deg]')
        axes[1, 2].set_title('Median θ bias map (θ, φ)')
        axes[1, 2].axhline(90, color='gray', ls=':', lw=0.8)
        axes[1, 2].axhline(120, color='gray', ls='--', lw=0.8)
        plt.colorbar(im, ax=axes[1, 2], label='median bias [deg]')

        fig.suptitle(f'{tag}: dz bias correlations with event properties', fontsize=13)
        fig.tight_layout()
        path = os.path.join(args.output_dir, f'dz_correlations_{tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")

        # =============================================================
        # 7. Print numerical summary
        # =============================================================
        print(f"\n  --- Median dz bias at key angles ---")
        for angle in [30, 60, 90, 120, 150]:
            m = (theta_true >= angle - 5) & (theta_true < angle + 5)
            if m.sum() > 10:
                print(f"    θ={angle}°:  dz bias = {np.median(dz_residual[m]):+.5f}"
                      f"  (θ bias = {np.median(residual[m]):+.3f}°)")

        print(f"\n  --- Scatter rates across 120° cut ---")
        for band_lo, band_hi in [(118, 120), (116, 120), (114, 120),
                                  (120, 122), (120, 124), (120, 126)]:
            m = (theta_true >= band_lo) & (theta_true < band_hi)
            n_total = m.sum()
            if n_total == 0:
                continue
            crossed = (theta_reco >= 120) if band_hi <= 120 else (theta_reco < 120)
            n_crossed = (m & crossed).sum()
            pct = 100 * n_crossed / n_total
            direction = "below→above" if band_hi <= 120 else "above→below"
            print(f"    truth [{band_lo:3d}°,{band_hi:3d}°): "
                  f"{n_crossed:5,}/{n_total:6,} crossed ({pct:5.1f}%) [{direction}]")


if __name__ == '__main__':
    main()
