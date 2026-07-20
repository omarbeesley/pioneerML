"""
Plot the pi->e nu tail-fraction bias (eff_tail/eff_peak) in COARSE bins of true
positron polar angle, for the muDIF and piDIF veto heads, off an eval_tail
predictions.parquet. Within each theta bin the veto cut is set to keep pie_eff of
that bin's pie (per-bin threshold), so the plotted ratio isolates whether the cut
sculpts the energy spectrum DIFFERENTLY at different angles (theta-dependent bias)
or uniformly (a flat offset = intrinsic, not angle).

Energy = deposited_energy + dead_E (= lyso + atar_posE + dead, the complete deposit;
deposited_energy already contains atar_posE). peak = e_split<=E<=e_max, tail = E<e_split.
muDIF uses full acceptance; piDIF uses angle-only (<120, no pion-stop fiducial).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADS = [("muon_dif_score", "is_mudif", "muDIF", "full",  "tab:green"),
         ("pion_dif_score", "is_pidif", "piDIF", "angle", "tab:purple")]


def bias_in_bins(df, score_col, name, acc_mode, edges, pie_eff, e_split, e_max, accept_min,
                 angle_max):
    E = (df.deposited_energy + df.dead_E).values.astype(float)
    th = np.degrees(df.positron_theta.values)
    if acc_mode == "angle":
        acc = th < angle_max
    else:
        acc = df.acceptance.values >= accept_min
    pie = (df.is_pie.values == 1) & acc & np.isfinite(E) & (E <= e_max)
    s = df[score_col].values
    cx, cy, ce, nt = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = pie & (th >= lo) & (th < hi)
        peak = m & (E >= e_split)
        tail = m & (E < e_split)
        np_, nt_ = int(peak.sum()), int(tail.sum())
        if nt_ < 10 or np_ < 50:
            continue
        thr = np.quantile(s[m], pie_eff)
        ep = (s[peak] < thr).mean()
        et = (s[tail] < thr).mean()
        if ep <= 0:
            continue
        b = et / ep
        be = b * np.sqrt(et * (1 - et) / (nt_ * et ** 2 + 1e-12)
                         + ep * (1 - ep) / (np_ * ep ** 2 + 1e-12))
        cx.append(0.5 * (lo + hi)); cy.append(b); ce.append(be); nt.append(nt_)
    return np.array(cx), np.array(cy), np.array(ce), nt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", default="tail_reveal_eval_pidif/predictions.parquet")
    p.add_argument("--out", default="tail_reveal_eval_pidif/bias_vs_theta.png")
    p.add_argument("--pie_eff", type=float, default=0.5)
    p.add_argument("--e_split", type=float, default=56.0)
    p.add_argument("--e_max", type=float, default=75.0)
    p.add_argument("--accept_min", type=float, default=0.5)
    p.add_argument("--angle_max", type=float, default=120.0)
    p.add_argument("--theta_edges", type=float, nargs="*",
                   default=[0, 50, 70, 85, 100, 120], help="coarse theta bin edges (deg)")
    args = p.parse_args()

    df = pd.read_parquet(args.pred)
    edges = np.array(args.theta_edges, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for score_col, _pos, name, acc_mode, col in HEADS:
        if score_col not in df.columns:
            print(f"[skip] {name}: no {score_col}")
            continue
        cx, cy, ce, nt = bias_in_bins(df, score_col, name, acc_mode, edges, args.pie_eff,
                                      args.e_split, args.e_max, args.accept_min, args.angle_max)
        if len(cx) == 0:
            print(f"[skip] {name}: no populated bins"); continue
        ax.errorbar(cx, cy, yerr=ce, fmt="o-", color=col, lw=1.8, ms=5, capsize=3,
                    label=f"{name} ({acc_mode} acc)")
        print(f"{name}:")
        for x, y, e, n in zip(cx, cy, ce, nt):
            print(f"   theta~{x:5.1f} deg   bias={y:.3f} +/- {e:.3f}   (tail N={n})")
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label="unbiased")
    for x in edges:
        ax.axvline(x, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("true positron polar angle theta (deg)")
    ax.set_ylabel(f"tail-fraction bias  eff(tail)/eff(peak)  @ pie eff={args.pie_eff:g}")
    ax.set_title("pi->e nu tail bias vs true theta  (per-bin cut; E=live+dead; "
                 f"tail<{args.e_split:g} MeV)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(args.out, dpi=130); plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
