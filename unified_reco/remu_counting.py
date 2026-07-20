"""Counting R_e/mu (no time fit).

At the 10^-4 level the per-bin TIME fit is stats-limited and is really a check on time-fit
DISTORTIONS, not the primary measurement.  Assuming accidentals are removed (accepted &
in-window & clean), R_e/mu follows directly from four gen_weight-weighted accepted counts:

              recoE >= 56 MeV      recoE < 56 MeV
  pi->e            N_pie_hi             N_pie_lo         (pi->e is ~70 MeV -> mostly high; low = radiative/reco tail)
  pi->mu->e        N_mue_hi             N_mue_lo         (michel endpoint 52.8 -> mostly low; high = reco leakage)

  R_e/mu = (N_pie_acc / eff_pie) / (N_mue_acc / eff_mue),   N_acc = N_hi + N_lo,
           eff = accepted-in-window-clean / all   (per-channel acceptance x reco efficiency)
"""
import argparse, numpy as np
from purity_analysis import io


def counting(pie_path, michel_path, ecut=56.0, emax=float("inf"),
             time_col="pred_positron_time_consensus_ns"):
    cfg = io.AnalysisConfig(mode="reco", energy="deposited", time_col=time_col)
    df = io.load_all(pie_path, michel_path, cfg)
    df = df[np.isfinite(df.E) & (df.E <= emax)]
    R = {}
    for cls in ("pie", "mue"):
        d = df[df.cls == cls]
        tot = float(d.w.sum())
        sel = d[d.accepted & d.in_window & d.clean]        # accidentals removed
        hi = float(sel[sel.E >= ecut].w.sum()); lo = float(sel[sel.E < ecut].w.sum())
        R[cls] = dict(tot=tot, hi=hi, lo=lo, acc=hi + lo, eff=(hi + lo) / tot if tot else 0.0,
                      n_hi=int((sel.E >= ecut).sum()), n_lo=int((sel.E < ecut).sum()))
    npie = R["pie"]["acc"] / R["pie"]["eff"]; nmue = R["mue"]["acc"] / R["mue"]["eff"]
    return R, npie / nmue, R["pie"]["acc"] / R["mue"]["acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie", required=True); ap.add_argument("--michel", required=True)
    ap.add_argument("--ecut", type=float, default=56.0); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    R, remu, remu_raw = counting(args.pie, args.michel, ecut=args.ecut)
    p, m = R["pie"], R["mue"]
    print(f"\n=== Counting R_e/mu {args.tag} (E_cut={args.ecut:g} MeV) ===")
    print(f"{'':10}{'recoE>=cut':>16}{'recoE<cut':>16}{'  accepted':>14}{'  eff(acc)':>12}")
    print(f"{'pi->e':10}{p['hi']:16.4g}{p['lo']:16.4g}{p['acc']:14.4g}{p['eff']:12.4f}   (n={p['n_hi']}+{p['n_lo']})")
    print(f"{'pi->mu->e':10}{m['hi']:16.4g}{m['lo']:16.4g}{m['acc']:14.4g}{m['eff']:12.4f}   (n={m['n_hi']}+{m['n_lo']})")
    ftail = p['lo'] / p['acc'] if p['acc'] else 0.0
    fleak = m['hi'] / m['acc'] if m['acc'] else 0.0
    print(f"\n  pi->e tail below cut : {ftail*100:.2f}%   (radiative + reco leakage below {args.ecut:g} MeV)")
    print(f"  michel leak above cut: {fleak*100:.3f}%   (michel reco above the 52.8 MeV endpoint)")
    print(f"\n  R_e/mu (eff-corrected) = {remu:.4e}")
    print(f"  R_e/mu (raw acc ratio) = {remu_raw:.4e}")
    print(f"  R_e/mu (SM)            = 1.2352e-04")
    print(f"  closure  meas/SM       = {remu/1.2352e-4:.4f}\n")


if __name__ == "__main__":
    main()
