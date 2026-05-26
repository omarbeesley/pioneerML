"""
Diagnose vMF angle head: correlations between kappa, angle error,
pion stop residuals, and true angle from benchmark parquets.

Usage (inside container):
    python diagnose_vmf.py --results_dir /pioneerML/benchmark_results_v2c \
                           --output_dir /pioneerML/vmf_diagnostics
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
    """Load all *_events.parquet files, return list of (tag, df)."""
    datasets = []
    for f in sorted(os.listdir(results_dir)):
        if f.endswith('_events.parquet'):
            tag = f.replace('_events.parquet', '')
            df = pd.read_parquet(os.path.join(results_dir, f))
            datasets.append((tag, df))
    return datasets


def gate_events(ev):
    """Apply HTP + IoU gate, return mask."""
    htp_gate = (ev['truth_htp'].values == 1) & \
               (ev['pred_htp'].values != SENTINEL) & \
               (ev['pred_htp'].values > 0.5)
    iou_ok = ev['pred_pos_iou'].values >= 0.95
    lk_ok = ev['pred_positron_log_kappa'].values != SENTINEL
    return htp_gate & iou_ok & lk_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output_dir', default='./vmf_diagnostics')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = load_events(args.results_dir)

    if not datasets:
        print("No event parquets found!")
        return

    tag_colors = {}
    for tag, _ in datasets:
        if 'pie' in tag:
            tag_colors[tag] = 'C3'
        elif 'pimu' in tag:
            tag_colors[tag] = 'C0'
        else:
            tag_colors[tag] = 'C2'

    # ------------------------------------------------------------------ #
    #  Figure 1: kappa distributions and correlations (per channel)       #
    # ------------------------------------------------------------------ #
    for tag, ev in datasets:
        gate = gate_events(ev)
        color = tag_colors[tag]

        log_kappa = ev['pred_positron_log_kappa'].values[gate]
        kappa = np.exp(np.clip(log_kappa, -5, 20))
        theta_true = np.degrees(ev['truth_theta'].values[gate])
        theta_reco = np.degrees(ev['pred_polar_angle'].values[gate])
        angle_residual = theta_true - theta_reco

        pion_stop_res_x = ev['truth_pion_stop_x'].values[gate] - ev['pred_pion_stop_x'].values[gate]
        pion_stop_res_y = ev['truth_pion_stop_y'].values[gate] - ev['pred_pion_stop_y'].values[gate]
        pion_stop_res_z = ev['truth_pion_stop_z'].values[gate] - ev['pred_pion_stop_z'].values[gate]
        pion_stop_err_3d = np.sqrt(pion_stop_res_x**2 + pion_stop_res_y**2 + pion_stop_res_z**2)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'{tag}: vMF diagnostics (n={gate.sum()})', fontsize=13)

        # 1a: log_kappa distribution
        axes[0, 0].hist(log_kappa, bins=100, color=color, alpha=0.7)
        axes[0, 0].axvline(np.median(log_kappa), color='k', ls='--',
                           label=f'median={np.median(log_kappa):.2f}')
        axes[0, 0].set_xlabel(r'$\log\kappa$')
        axes[0, 0].set_ylabel('counts')
        axes[0, 0].set_title(r'$\log\kappa$ distribution')
        axes[0, 0].legend()

        # 1b: 2D histogram of log_kappa vs angle error
        axes[0, 1].hist2d(np.abs(angle_residual), log_kappa,
                          bins=[np.linspace(0, 20, 80), np.linspace(-3, 12, 80)],
                          cmap='viridis', norm=LogNorm())
        axes[0, 1].set_xlabel('|angle residual| [deg]')
        axes[0, 1].set_ylabel(r'$\log\kappa$')
        axes[0, 1].set_title(r'$\log\kappa$ vs angle error')

        # 1c: 2D histogram of log_kappa vs theta_true
        axes[0, 2].hist2d(theta_true, log_kappa,
                          bins=[np.linspace(0, 180, 60), np.linspace(-3, 12, 60)],
                          cmap='viridis', norm=LogNorm())
        axes[0, 2].set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')
        axes[0, 2].set_ylabel(r'$\log\kappa$')
        axes[0, 2].set_title(r'$\log\kappa$ vs $\theta_{\mathrm{true}}$')
        axes[0, 2].axvline(90, color='w', ls=':', lw=0.8)
        axes[0, 2].axvline(120, color='w', ls='--', lw=0.8)

        # 2a: pion stop 3D error vs log_kappa
        axes[1, 0].hist2d(log_kappa, pion_stop_err_3d,
                          bins=[np.linspace(-3, 12, 60), np.linspace(0, 1.0, 60)],
                          cmap='viridis', norm=LogNorm())
        axes[1, 0].set_xlabel(r'$\log\kappa$')
        axes[1, 0].set_ylabel('pion stop 3D error [mm]')
        axes[1, 0].set_title('pion stop error vs kappa')

        # 2b: pion stop z residual vs theta_true
        axes[1, 1].hist2d(theta_true, pion_stop_res_z,
                          bins=[np.linspace(0, 180, 60), np.linspace(-1, 1, 60)],
                          cmap='viridis', norm=LogNorm())
        axes[1, 1].axhline(0, color='w', ls='--', lw=0.8)
        axes[1, 1].axvline(90, color='w', ls=':', lw=0.8)
        axes[1, 1].set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')
        axes[1, 1].set_ylabel('pion stop z residual [mm]')
        axes[1, 1].set_title('pion stop z residual vs theta')

        # 2c: angle residual vs pion stop z residual
        axes[1, 2].hist2d(pion_stop_res_z, angle_residual,
                          bins=[np.linspace(-1, 1, 60), np.linspace(-15, 15, 60)],
                          cmap='viridis', norm=LogNorm())
        axes[1, 2].axhline(0, color='w', ls='--', lw=0.8)
        axes[1, 2].axvline(0, color='w', ls='--', lw=0.8)
        axes[1, 2].set_xlabel('pion stop z residual [mm]')
        axes[1, 2].set_ylabel('angle residual [deg]')
        axes[1, 2].set_title('angle error vs pion stop z error')

        for ax in axes.flat:
            ax.grid(True, alpha=0.2)

        fig.tight_layout()
        path = os.path.join(args.output_dir, f'vmf_diag_{tag}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> {path}")

    # ------------------------------------------------------------------ #
    #  Figure 2: binned profiles — median kappa and pion stop error       #
    #            vs theta, both channels overlaid                         #
    # ------------------------------------------------------------------ #
    bins_theta = np.linspace(0, 180, 37)
    centers = 0.5 * (bins_theta[:-1] + bins_theta[1:])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for tag, ev in datasets:
        gate = gate_events(ev)
        color = tag_colors[tag]

        log_kappa = ev['pred_positron_log_kappa'].values[gate]
        kappa = np.exp(np.clip(log_kappa, -5, 20))
        theta_true = np.degrees(ev['truth_theta'].values[gate])
        theta_reco = np.degrees(ev['pred_polar_angle'].values[gate])
        angle_residual = theta_true - theta_reco

        ps_res_z = ev['truth_pion_stop_z'].values[gate] - ev['pred_pion_stop_z'].values[gate]
        ps_err_3d = np.sqrt(
            (ev['truth_pion_stop_x'].values[gate] - ev['pred_pion_stop_x'].values[gate])**2 +
            (ev['truth_pion_stop_y'].values[gate] - ev['pred_pion_stop_y'].values[gate])**2 +
            ps_res_z**2
        )

        med_kappa, med_angle_res, med_ps_z, rms_ps_3d = [], [], [], []
        for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
            m = (theta_true >= lo) & (theta_true < hi)
            if m.sum() > 10:
                med_kappa.append(np.median(kappa[m]))
                med_angle_res.append(np.median(angle_residual[m]))
                med_ps_z.append(np.median(ps_res_z[m]))
                rms_ps_3d.append(np.sqrt((ps_err_3d[m]**2).mean()))
            else:
                med_kappa.append(np.nan)
                med_angle_res.append(np.nan)
                med_ps_z.append(np.nan)
                rms_ps_3d.append(np.nan)

        short = r'$\pi{\to}e$' if 'pie' in tag else r'$\pi{\to}\mu{\to}e$'
        axes[0, 0].plot(centers, med_kappa, 'o-', color=color, markersize=3, label=short)
        axes[0, 1].plot(centers, med_angle_res, 'o-', color=color, markersize=3, label=short)
        axes[1, 0].plot(centers, med_ps_z, 'o-', color=color, markersize=3, label=short)
        axes[1, 1].plot(centers, rms_ps_3d, 'o-', color=color, markersize=3, label=short)

    for ax in axes.flat:
        ax.axvline(90, color='gray', ls=':', lw=0.8)
        ax.axvline(120, color='gray', ls='--', lw=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')

    axes[0, 0].set_ylabel(r'median $\kappa$')
    axes[0, 0].set_title(r'vMF concentration vs $\theta$')
    axes[0, 1].set_ylabel('median angle residual [deg]')
    axes[0, 1].set_title('angle bias vs theta')
    axes[0, 1].axhline(0, color='k', ls='--', lw=0.8)
    axes[1, 0].set_ylabel('median pion stop z residual [mm]')
    axes[1, 0].set_title('pion stop z bias vs theta')
    axes[1, 0].axhline(0, color='k', ls='--', lw=0.8)
    axes[1, 1].set_ylabel('pion stop 3D RMS [mm]')
    axes[1, 1].set_title('pion stop 3D RMS vs theta')

    fig.suptitle('vMF diagnostics: binned profiles vs theta', fontsize=13)
    fig.tight_layout()
    path = os.path.join(args.output_dir, 'vmf_profiles_vs_theta.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


if __name__ == '__main__':
    main()
