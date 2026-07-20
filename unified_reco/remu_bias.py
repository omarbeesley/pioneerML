"""Counting R_e/mu (standard analysis): acceptance reco-vs-truth bias decomposition.

The measurement selects events with the MODEL's acceptance (pred_accepted) and corrects by the
acceptance efficiency.  If the model's acceptance differs from the TRUE acceptance
(truth_acceptance) by a DIFFERENT amount for pi->e than for michel, R_e/mu is biased:

    R_e/mu(reco) / R_e/mu(input) = (eff_reco/eff_true)_pie / (eff_reco/eff_true)_michel

This isolates that ratio and shows the false-neg / false-pos acceptance flips driving it.
Energy migration across the 56 MeV split does NOT bias the counting total (hi+lo summed).
"""
import argparse, numpy as np, pandas as pd, pyarrow.parquet as pq

COLS = ['truth_acceptance', 'pred_accepted', 'truth_live_E', 'truth_dead_E',
        'pred_positron_energy', 'pred_dead_energy', 'truth_positron_t',
        'pred_positron_time_consensus_ns', 'truth_gen_weight', 'truth_event_type', 'truth_positron_energy']


def load(path, is_pie):
    import glob
    files = sorted(glob.glob(path)) if "*" in path else [path]
    d = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in files], ignore_index=True)
    if not is_pie:  # michel pool: drop kPienu(0x1), kMurad(0x200), above-endpoint (io.py logic)
        et = d.truth_event_type.astype('int64')
        d = d[((et & 1) == 0) & ((et & 0x200) == 0) & (d.truth_positron_energy <= 55)].reset_index(drop=True)
    return d


def analyze(pie_path, michel_path, ecut=56.0, emax=float("inf")):
    P, M = load(pie_path, True), load(michel_path, False)
    n_pie, n_mic = len(P), len(M)
    R = {}
    for name, d, is_pie in [("pie", P, True), ("mue", M, False)]:
        w = d.truth_gen_weight.values.astype(float)
        if is_pie:
            w = w * (n_mic / n_pie)                                  # luminosity match to michel pool
        tot = float(w.sum())
        acc_t = d.truth_acceptance.values > 0.5                      # TRUE acceptance
        tr = d.pred_positron_time_consensus_ns.values
        Er = np.clip(d.pred_positron_energy.values, 0, None) + np.clip(d.pred_dead_energy.values, 0, None)
        clean = np.abs(tr - d.truth_positron_t.values) < 5.0
        acc_r = (d.pred_accepted.values >= 0.5) & (tr >= -300) & (tr <= 500) & clean & (Er <= emax)  # RECO
        R[name] = dict(tot=tot,
                       eff_t=float(w[acc_t].sum()) / tot, eff_r=float(w[acc_r].sum()) / tot,
                       hi=float(w[acc_r & (Er >= ecut)].sum()), lo=float(w[acc_r & (Er < ecut)].sum()),
                       fn=float(w[acc_t & ~acc_r].sum()) / tot,   # true-acc, reco-rejected
                       fp=float(w[~acc_t & acc_r].sum()) / tot)   # true-rej, reco-accepted
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie", required=True); ap.add_argument("--michel", required=True)
    ap.add_argument("--tag", default=""); ap.add_argument("--ecut", type=float, default=56.0)
    ap.add_argument("--emax", type=float, default=float("inf"),
                    help="upper reco-E cap (default none; 75 was the biased legacy value, cross-check only)")
    args = ap.parse_args()
    R = analyze(args.pie, args.michel, ecut=args.ecut, emax=args.emax)
    p, m = R["pie"], R["mue"]
    remu_in = p["tot"] / m["tot"]
    remu_reco = (p["hi"] + p["lo"]) / p["eff_t"] / ((m["hi"] + m["lo"]) / m["eff_t"])
    rp, rm = p["eff_r"] / p["eff_t"], m["eff_r"] / m["eff_t"]

    print(f"\n=== R_e/mu counting -- acceptance reco-vs-truth bias {args.tag} ===\n")
    print(f"{'':11}{'eff(TRUE acc)':>14}{'eff(RECO acc)':>15}{'reco/true':>11}{'  FN%':>8}{'  FP%':>8}")
    print(f"{'pi->e':11}{p['eff_t']:14.4f}{p['eff_r']:15.4f}{rp:11.4f}{p['fn']*100:8.2f}{p['fp']*100:8.2f}")
    print(f"{'pi->mu->e':11}{m['eff_t']:14.4f}{m['eff_r']:15.4f}{rm:11.4f}{m['fn']*100:8.2f}{m['fp']*100:8.2f}")
    print(f"\n  N_pie:  hi(>= {args.ecut:g})={p['hi']:.4g}  lo(< {args.ecut:g})={p['lo']:.4g}")
    print(f"  N_mue:  hi(>= {args.ecut:g})={m['hi']:.4g}  lo(< {args.ecut:g})={m['lo']:.4g}")
    print(f"\n  R_e/mu (input, MC truth)      = {remu_in:.4e}")
    print(f"  R_e/mu (reco sel, truth-eff)  = {remu_reco:.4e}")
    print(f"  acceptance bias = (reco/true)_pie / (reco/true)_mue = {rp/rm:.4f}")
    print(f"                  -> R_e/mu bias  = {(rp/rm - 1)*100:+.2f}%   [reco/input = {remu_reco/remu_in:.4f}]")
    print(f"\n  dominant source: pi->e FN={p['fn']*100:.2f}% vs michel FN={m['fn']*100:.2f}%  "
          f"(differential acceptance loss)\n")


if __name__ == "__main__":
    main()
