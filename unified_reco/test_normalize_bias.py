"""
Verify that F.normalize on noisy logits creates an S-shaped bias in theta.

No neural network — just sample true directions, add Gaussian noise,
normalize, and measure the median bias in theta vs theta_true.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sample_sphere(n):
    z = np.random.uniform(-1, 1, n)
    phi = np.random.uniform(0, 2 * np.pi, n)
    r = np.sqrt(1 - z**2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def measure_bias(n=1_000_000, noise_std=0.1):
    true_dir = sample_sphere(n)
    noise = np.random.randn(n, 3) * noise_std
    noisy = true_dir + noise
    pred_dir = normalize(noisy)

    theta_true = np.degrees(np.arccos(np.clip(true_dir[:, 2], -1, 1)))
    theta_reco = np.degrees(np.arccos(np.clip(pred_dir[:, 2], -1, 1)))
    residual = theta_true - theta_reco

    return theta_true, residual


def main():
    np.random.seed(42)

    noise_levels = [0.05, 0.1, 0.2, 0.4]
    bins = np.linspace(0, 180, 37)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for sigma in noise_levels:
        theta_true, residual = measure_bias(noise_std=sigma)

        medians = []
        q25s, q75s = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (theta_true >= lo) & (theta_true < hi)
            r = residual[m]
            medians.append(np.median(r))
            q25s.append(np.percentile(r, 25))
            q75s.append(np.percentile(r, 75))

        medians = np.array(medians)
        axes[0].plot(centers, medians, 'o-', markersize=3,
                     label=rf'$\sigma$ = {sigma}')

        rms_vals = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (theta_true >= lo) & (theta_true < hi)
            rms_vals.append(np.degrees(np.sqrt((np.radians(residual[m])**2).mean())))
        axes[1].plot(centers, rms_vals, 'o-', markersize=3,
                     label=rf'$\sigma$ = {sigma}')

    axes[0].axhline(0, color='k', ls='--', lw=0.8)
    axes[0].axvline(90, color='gray', ls=':', lw=0.8)
    axes[0].axvline(120, color='gray', ls='--', lw=0.8)
    axes[0].set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')
    axes[0].set_ylabel(r'Median($\theta_{\mathrm{true}} - \theta_{\mathrm{reco}}$) [deg]')
    axes[0].set_title('Median bias from normalize + arccos\n(no neural network, just noise + normalize)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axvline(90, color='gray', ls=':', lw=0.8)
    axes[1].axvline(120, color='gray', ls='--', lw=0.8)
    axes[1].set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')
    axes[1].set_ylabel('RMS angle error [deg]')
    axes[1].set_title('RMS vs theta\n(should be flat if normalize is unbiased)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig('normalize_bias_test.png', dpi=150)
    plt.close()
    print("Saved: normalize_bias_test.png")

    # Print the bias at key angles for the noise level closest to PURITY
    sigma = 0.1
    theta_true, residual = measure_bias(noise_std=sigma)
    for angle in [10, 45, 90, 120, 170]:
        m = (theta_true >= angle - 5) & (theta_true < angle + 5)
        print(f"  theta={angle:3d}°:  median bias = {np.median(residual[m]):+.3f}°"
              f"   IQR = [{np.percentile(residual[m], 25):+.2f}, "
              f"{np.percentile(residual[m], 75):+.2f}]°")

    # --- FP / FN analysis at the 120° acceptance cut ---
    print("\n" + "=" * 60)
    print("  FP / FN at 120° acceptance cut (normalize bias only)")
    print("=" * 60)

    cut_deg = 120.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for sigma in noise_levels:
        n = 5_000_000
        true_dir = sample_sphere(n)
        noise = np.random.randn(n, 3) * sigma
        pred_dir = normalize(true_dir + noise)

        theta_true = np.degrees(np.arccos(np.clip(true_dir[:, 2], -1, 1)))
        theta_reco = np.degrees(np.arccos(np.clip(pred_dir[:, 2], -1, 1)))

        truth_pass = theta_true < cut_deg
        reco_pass = theta_reco < cut_deg

        tp = (truth_pass & reco_pass).sum()
        fp = (~truth_pass & reco_pass).sum()
        fn = (truth_pass & ~reco_pass).sum()
        tn = (~truth_pass & ~reco_pass).sum()

        ratio = fp / fn if fn > 0 else float('inf')
        print(f"\n  σ = {sigma}:")
        print(f"    TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
        print(f"    FP/FN = {ratio:.3f}")
        print(f"    FP rate (of true fail): {fp/(fp+tn)*100:.2f}%")
        print(f"    FN rate (of true pass): {fn/(fn+tp)*100:.2f}%")

        # FP and FN theta distributions
        theta_fp = theta_true[~truth_pass & reco_pass]
        theta_fn = theta_true[truth_pass & ~reco_pass]

        fp_bins = np.linspace(100, 160, 61)
        fn_bins = np.linspace(100, 140, 41)

        axes[0].hist(theta_fp, bins=fp_bins, alpha=0.4, density=True,
                     label=rf'$\sigma$={sigma} (n={len(theta_fp):,})')
        axes[1].hist(theta_fn, bins=fn_bins, alpha=0.4, density=True,
                     label=rf'$\sigma$={sigma} (n={len(theta_fn):,})')

    axes[0].axvline(cut_deg, color='k', ls='--', lw=1)
    axes[0].set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')
    axes[0].set_ylabel('density')
    axes[0].set_title('False positives: true θ distribution\n(events above 120° predicted below)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axvline(cut_deg, color='k', ls='--', lw=1)
    axes[1].set_xlabel(r'$\theta_{\mathrm{true}}$ [deg]')
    axes[1].set_ylabel('density')
    axes[1].set_title('False negatives: true θ distribution\n(events below 120° predicted above)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Acceptance cut FP/FN from normalize bias alone', fontsize=13)
    fig.tight_layout()
    path = 'normalize_bias_fpfn.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")

    # --- Summary table ---
    print("\n" + "-" * 60)
    print(f"  {'σ':>6}  {'FP/FN':>8}  {'bias@120°':>10}  {'RMS@120°':>10}")
    print("-" * 60)
    for sigma in noise_levels:
        n = 5_000_000
        true_dir = sample_sphere(n)
        noise_v = np.random.randn(n, 3) * sigma
        pred_dir = normalize(true_dir + noise_v)
        theta_true = np.degrees(np.arccos(np.clip(true_dir[:, 2], -1, 1)))
        theta_reco = np.degrees(np.arccos(np.clip(pred_dir[:, 2], -1, 1)))
        residual = theta_true - theta_reco

        near_cut = (theta_true >= 115) & (theta_true < 125)
        bias_at_cut = np.median(residual[near_cut])
        rms_at_cut = np.degrees(np.sqrt((np.radians(residual[near_cut])**2).mean()))

        truth_pass = theta_true < cut_deg
        reco_pass = theta_reco < cut_deg
        fp = (~truth_pass & reco_pass).sum()
        fn = (truth_pass & ~reco_pass).sum()
        ratio = fp / fn if fn > 0 else float('inf')

        print(f"  {sigma:6.2f}  {ratio:8.3f}  {bias_at_cut:+10.3f}°  {rms_at_cut:10.2f}°")


if __name__ == '__main__':
    main()
