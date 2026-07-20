"""
Offline cut analysis on the per-event predictions.parquet from eval_tail.py.
Pure pandas/numpy/matplotlib -- no torch, runs on the host.

Per-head INDEPENDENT box cuts only (no logit-combining). Produces:
  roc_sweep.png        each head's pi->e nu efficiency vs Michel leak
  eff_heatmaps.png     per head, a 2-D map of pi->e nu efficiency over
                       (deposited energy x cut threshold)  [coarse, --n_bins]
  box_eff_vs_energy.png  efficiency vs deposited energy at the optimized box cut
  + console: the jointly-optimized 4 thresholds, with eff / Michel-leak / energy bias.

Energy-ratio bias = eff(E>cut)/eff(E<cut) - 1 on the pi->e nu spectrum
(0 = the cut preserves the high/low-energy ratio, i.e. no R_e/mu bias).

Usage:
    python sweep_cuts.py --pred tail_reveal_eval/predictions.parquet \
        --output_dir tail_reveal_eval/sweep \
        --target_leak 0 --energy_cut 56 --max_ratio_bias 0.01 --grid_points 10
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# head -> (column, keep direction). +1: keep if score > thr; -1 (veto): keep if score < thr.
HEADS = [("pie_score", +1), ("topo_score", +1), ("muon_score", -1), ("pileup_score", -1)]


def roc_sweep(s, direction, is_pie, n=2000):
    sig = np.sort(s[is_pie == 1]); bkg = np.sort(s[is_pie == 0])
    thr = np.linspace(np.quantile(s, 1e-4), np.quantile(s, 1 - 1e-4), n)
    if direction > 0:
        pie_eff = 1.0 - np.searchsorted(sig, thr, "right") / len(sig)
        leak    = 1.0 - np.searchsorted(bkg, thr, "right") / len(bkg)
    else:
        pie_eff = np.searchsorted(sig, thr, "left") / len(sig)
        leak    = np.searchsorted(bkg, thr, "left") / len(bkg)
    return thr, pie_eff, leak


def ratio_bias(energy_pie, keep_pie, ecut):
    """eff(E>ecut)/eff(E<ecut) - 1 on the signal (0 = unbiased high/low ratio)."""
    hi, lo = energy_pie > ecut, energy_pie < ecut
    if hi.sum() == 0 or lo.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    eff_hi, eff_lo = float(keep_pie[hi].mean()), float(keep_pie[lo].mean())
    bias = (eff_hi / eff_lo - 1.0) if eff_lo > 0 else float("inf")
    return bias, eff_hi, eff_lo


# --------------------------------------------------------------------------
def per_head_roc(df, is_pie, output_dir, target_leak):
    targets = sorted(set([target_leak, 0.0, 1e-4, 1e-3]))
    hdr = f"{'head':<13} " + "  ".join(f"eff@leak<={t:g}" for t in targets)
    print(hdr); print("-" * len(hdr))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for col, d in HEADS:
        s = df[col].values
        thr, pe, lk = roc_sweep(s, d, is_pie)
        ax.plot(np.clip(lk, 1e-7, 1.0), pe, label=col, lw=1.6)
        cells = []
        for t in targets:
            ok = lk <= t
            cells.append(f"{pe[ok].max():9.4f}" if ok.any() else "     -   ")
        print(f"{col:<13} " + "  ".join(cells))
    ax.set_xscale("log"); ax.set_xlabel("Michel leak-through (FPR)")
    ax.set_ylabel("pi->e nu efficiency"); ax.set_title("per-head cut sweep")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{output_dir}/roc_sweep.png", dpi=130); plt.close(fig)


def eff_heatmaps(df, is_pie, energy, output_dir, n_bins, energy_cut):
    """Per head: pi->e nu efficiency over (deposited energy x cut threshold)."""
    pie = is_pie == 1
    e = energy[pie]
    ebins = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
    if len(ebins) < 3:
        print("[warn] deposited_energy has no spread — skipping heatmaps")
        return
    eidx = np.clip(np.digitize(e, ebins) - 1, 0, len(ebins) - 2)
    counts = np.bincount(eidx, minlength=len(ebins) - 1)
    ecenters = 0.5 * (ebins[:-1] + ebins[1:])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (col, d) in zip(axes.flat, HEADS):
        sc = df[col].values[pie]
        # evenly-spaced thresholds across the score range (not quantiles, so the
        # y-axis is uniform and the labels aren't bunched where the score piles up)
        thr_vals = np.linspace(sc.min(), sc.max(), n_bins)
        H = np.full((n_bins, len(ebins) - 1), np.nan)
        for yi, t in enumerate(thr_vals):
            keep = (sc > t) if d > 0 else (sc < t)
            H[yi] = np.bincount(eidx, weights=keep.astype(float),
                                minlength=len(ebins) - 1) / np.maximum(counts, 1)
        im = ax.imshow(H, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis",
                       extent=[ebins[0], ebins[-1], -0.5, n_bins - 0.5])
        ax.axvline(energy_cut, color="w", ls=":", lw=1.0)
        ax.set_yticks(range(n_bins))
        ax.set_yticklabels([f"{t:.3g}" for t in thr_vals], fontsize=7)
        keepdir = ">" if d > 0 else "<"
        ax.set_title(f"{col}   (keep if score {keepdir} thr)", fontsize=10)
        ax.set_xlabel("deposited energy (MeV)"); ax.set_ylabel("cut threshold")
        fig.colorbar(im, ax=ax, label="pi->e nu efficiency", fraction=0.046, pad=0.04)
    fig.suptitle("per-head pi->e nu efficiency vs (deposited energy, cut threshold)")
    fig.tight_layout(); fig.savefig(f"{output_dir}/eff_heatmaps.png", dpi=130); plt.close(fig)


def box_optimize(df, is_pie, energy, args):
    """Grid-search ALL FOUR per-head thresholds simultaneously: maximize pi->e nu
    efficiency subject to michel_leak <= target_leak AND |ratio_bias| <= max_ratio_bias.
    Searches on a (subsampled) grid for speed, then reports the winner on FULL data."""
    cols = [c for c, _ in HEADS]
    signs = np.array([d for _, d in HEADS], dtype=float)        # keep if X > t (signed)
    ops = {c: (">" if d > 0 else "<") for c, d in HEADS}
    X = df[cols].values * signs
    sig, bkg = is_pie == 1, is_pie == 0
    energy_pie = energy[sig]
    Xs_full, Xb_full = X[sig], X[bkg]

    rng = np.random.default_rng(0)
    si = np.where(sig)[0]; bi = np.where(bkg)[0]
    if len(si) > args.scan_cap:
        si = si[rng.choice(len(si), args.scan_cap, replace=False)]
    if len(bi) > args.scan_cap * 3:
        bi = bi[rng.choice(len(bi), args.scan_cap * 3, replace=False)]
    Xs, Xb, es = X[si], X[bi], energy[si]
    hi, lo = es > args.energy_cut, es < args.energy_cut

    n = args.grid_points
    grids = [np.quantile(Xs_full[:, d], np.linspace(0.0, 0.95, n)) for d in range(4)]
    passS = [Xs[:, d][:, None] > grids[d][None, :] for d in range(4)]   # [ns, n]
    passB = [Xb[:, d][:, None] > grids[d][None, :] for d in range(4)]   # [nb, n]

    print(f"\n=== joint box-cut optimizer: {n}^4={n**4} grid combos "
          f"(scan on {len(si)} pie / {len(bi)} michel) ===")
    print(f"objective: max pi->e nu eff  s.t.  michel_leak <= {args.target_leak:g}"
          + (f"  AND  |ratio_bias| <= {args.max_ratio_bias:g}"
             if np.isfinite(args.max_ratio_bias) else ""))
    best = None
    for i, j, k, l in itertools.product(range(n), repeat=4):
        kb = passB[0][:, i] & passB[1][:, j] & passB[2][:, k] & passB[3][:, l]
        if kb.mean() > args.target_leak:
            continue
        ks = passS[0][:, i] & passS[1][:, j] & passS[2][:, k] & passS[3][:, l]
        eh = ks[hi].mean() if hi.any() else np.nan
        el = ks[lo].mean() if lo.any() else np.nan
        bias = (eh / el - 1.0) if el > 0 else np.inf
        if abs(bias) > args.max_ratio_bias:
            continue
        eff = ks.mean()
        if best is None or eff > best["eff"]:
            best = dict(t=np.array([grids[d][[i, j, k, l][d]] for d in range(4)]), eff=eff)
    if best is None:
        print("  no grid combo met the constraints "
              "(raise --grid_points, or loosen --target_leak / --max_ratio_bias)")
        return
    # verify the winner on FULL statistics
    t = best["t"]
    keepS = (Xs_full > t).all(1); keepB = (Xb_full > t).all(1)
    eff = float(keepS.mean()); leak = float(keepB.mean())
    bias, eff_hi, eff_lo = ratio_bias(energy_pie, keepS, args.energy_cut)
    print("optimal box cut (verified on full statistics):")
    for d, c in enumerate(cols):
        print(f"   {c:<13} {ops[c]} {t[d] * signs[d]:.4f}")
    print(f"   -> pi->e nu eff = {eff:.4f}   michel_leak = {leak:.3e} "
          f"({int(keepB.sum())}/{len(Xb_full)})")
    print(f"      energy ratio_bias = {bias:+.4f}   "
          f"[eff(E>{args.energy_cut:g})={eff_hi:.4f}, eff(E<{args.energy_cut:g})={eff_lo:.4f}]")

    # efficiency vs deposited energy at the optimum
    ebins = np.unique(np.quantile(energy_pie, np.linspace(0, 1, args.n_bins + 1)))
    if len(ebins) >= 3:
        eidx = np.clip(np.digitize(energy_pie, ebins) - 1, 0, len(ebins) - 2)
        cnt = np.bincount(eidx, minlength=len(ebins) - 1)
        num = np.bincount(eidx, weights=keepS.astype(float), minlength=len(ebins) - 1)
        eff_e = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
        ctr = 0.5 * (ebins[:-1] + ebins[1:])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ctr[cnt > 0], eff_e[cnt > 0], "o-")
        ax.axvline(args.energy_cut, color="k", ls=":", lw=0.8)
        ax.set_xlabel("truth deposited energy (MeV)")
        ax.set_ylabel("pi->e nu efficiency (survivors/total)")
        ax.set_title(f"optimized box cut: ratio_bias={bias:+.3f}, eff={eff:.3f}, leak={leak:.1e}")
        ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(f"{args.output_dir}/box_eff_vs_energy.png", dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="predictions.parquet from eval_tail.py")
    p.add_argument("--output_dir", default="sweep")
    p.add_argument("--target_leak", type=float, default=0.0,
                   help="Rejection goal: max Michel leak (0 = reject ALL Michel).")
    p.add_argument("--energy_cut", type=float, default=56.0,
                   help="Deposited-energy boundary (MeV) for the bias metric.")
    p.add_argument("--max_ratio_bias", type=float, default=float("inf"),
                   help="Max allowed |eff(E>cut)/eff(E<cut) - 1| at the optimum. Default: off.")
    p.add_argument("--n_bins", type=int, default=10, help="Heatmap / energy-bin coarseness.")
    p.add_argument("--grid_points", type=int, default=10,
                   help="Threshold values per head in the joint optimizer (cost ~ this^4).")
    p.add_argument("--scan_cap", type=int, default=200000,
                   help="Max pi->e nu events sampled for the grid scan (speed).")
    p.add_argument("--emax", type=float, default=float("inf"),
                   help="Drop events with deposited_energy above this (MeV). A single "
                        "positron can't exceed ~71 MeV, so ~75-80 removes the known "
                        "LYSO-energy-doubling artifact (~1e-4 of events near 2x70).")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.pred)
    if np.isfinite(args.emax):
        n0 = len(df)
        df = df[df["deposited_energy"] <= args.emax].reset_index(drop=True)
        print(f"dropped {n0 - len(df)} events with deposited_energy > {args.emax} MeV "
              f"(unphysical LYSO-doubling tail)")
    is_pie = df["is_pie"].values.astype(int)
    energy = df["deposited_energy"].values
    n_sig, n_bkg = int((is_pie == 1).sum()), int((is_pie == 0).sum())
    print(f"events: pie={n_sig}, michel={n_bkg}  "
          f"(leak=0 is stats-limited; floor ~ {1/n_bkg:.2e})\n")

    per_head_roc(df, is_pie, args.output_dir, args.target_leak)
    eff_heatmaps(df, is_pie, energy, args.output_dir, args.n_bins, args.energy_cut)
    box_optimize(df, is_pie, energy, args)

    print(f"\nwrote roc_sweep.png, eff_heatmaps.png, box_eff_vs_energy.png to {args.output_dir}/")
    print("\nExplore arbitrary cuts yourself:")
    print("  df = pd.read_parquet('predictions.parquet')")
    print("  keep = (df.pie_score>0.6)&(df.topo_score>0.5)&(df.muon_score<0.3)&(df.pileup_score<0.5)")
    print("  eff = keep[df.is_pie==1].mean();  leak = keep[df.is_pie==0].mean()")


if __name__ == "__main__":
    main()
