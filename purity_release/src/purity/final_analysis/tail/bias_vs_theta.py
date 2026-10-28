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
from pathlib import Path
import json
from ._runtime import AnalysisRun
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, required=True, help="Explicit saved eval_tail prediction table; no checkpoint version is selected automatically")
    p.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory")
    p.add_argument("--hash-inputs", action="store_true", help="Also SHA256 hash the full input parquet")
    p.add_argument("--pie-eff", type=float, default=0.5)
    p.add_argument("--e-split", type=float, default=56.0)
    p.add_argument("--e-max", type=float, default=75.0)
    p.add_argument("--accept-min", type=float, default=0.5)
    p.add_argument("--angle-max", type=float, default=120.0)
    p.add_argument("--theta-edges", type=float, nargs="*",
                   default=[0, 50, 70, 85, 100, 120], help="coarse theta bin edges (deg)")
    args = p.parse_args(argv)
    if not args.pred.is_file(): p.error('Prediction input does not exist')
    if not 0 < args.pie_eff < 1: p.error('--pie-eff must lie strictly between zero and one')
    if len(args.theta_edges) < 2 or not np.all(np.diff(args.theta_edges) > 0):
        p.error('--theta-edges must contain at least two strictly increasing values')
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f'Output directory must be new or empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = AnalysisRun('bias_vs_theta', args.output_dir, vars(args) | {
        'pred': str(args.pred.resolve()), 'output_dir': str(args.output_dir.resolve())}, args.hash_inputs)
    try:
        result = run(args, record)
    except BaseException as exc:
        record.finish(error=exc)
        raise
    else:
        record.finish()
    return result


def run(args, record):
    df = record.read_parquet(args.pred)
    required = {'deposited_energy', 'dead_E', 'positron_theta', 'acceptance', 'is_pie'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing required prediction columns: {sorted(missing)}')
    if df.empty: raise ValueError('Prediction input is empty')
    result = {'rows': len(df), 'heads': {}, 'skipped': {}}
    edges = np.array(args.theta_edges, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for score_col, _pos, name, acc_mode, col in HEADS:
        if score_col not in df.columns:
            result["skipped"][name] = f"Missing {score_col}"
            print(f"[skip] {name}: no {score_col}")
            continue
        cx, cy, ce, nt = bias_in_bins(df, score_col, name, acc_mode, edges, args.pie_eff,
                                      args.e_split, args.e_max, args.accept_min, args.angle_max)
        if len(cx) == 0:
            result["skipped"][name] = "No bin has at least 10 tail and 50 peak events with positive peak efficiency"
            print(f"[skip] {name}: no populated bins"); continue
        result['heads'][name] = {'score_column': score_col, 'acceptance_mode': acc_mode,
                                  'theta_deg': cx.tolist(), 'eff_tail_over_eff_peak': cy.tolist(),
                                  'uncertainty': ce.tolist(), 'tail_events': nt}
        ax.errorbar(cx, cy, yerr=ce, fmt="o-", color=col, lw=1.8, ms=5, capsize=3,
                    label=f"{name} ({acc_mode} acc)")
        print(f"{name}:")
        for x, y, e, n in zip(cx, cy, ce, nt):
            print(f"   theta~{x:5.1f} deg   bias={y:.3f} +/- {e:.3f}   (tail N={n})")
    if not result['heads']:
        plt.close(fig)
        raise ValueError('No head has populated angular bins; refusing to write a blank bias figure')
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label="unbiased")
    for x in edges:
        ax.axvline(x, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("true positron polar angle theta (deg)")
    ax.set_ylabel(f"tail-fraction bias  eff(tail)/eff(peak)  @ pie eff={args.pie_eff:g}")
    ax.set_title("pi->e nu tail bias vs true theta  (per-bin cut; E=live+dead; "
                 f"tail<{args.e_split:g} MeV)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(args.output_dir / f'bias_vs_theta.{ext}', dpi=130)
    plt.close(fig)
    (args.output_dir / 'bias_vs_theta.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"\nwrote {args.output_dir}/bias_vs_theta.png")
    return result


if __name__ == "__main__":
    main()
