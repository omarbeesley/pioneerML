"""
muDIF/piDIF-veto suppression / bias table, computed OFFLINE from an eval_tail
predictions.parquet (no re-inference). Reproduces the survival-vs-cut numbers
behind {mudif,pidif}_eff_vs_cut_accept.png with an added pi->e nu peak-vs-tail
BIAS column. Pick the head with --head {mudif,pidif}.

For each veto cut (the veto KEEPS events with muDIF-score < cut), on the
acceptance-passing population (truth_acceptance >= accept_min):

  pie_surv     fraction of pi->e nu kept            (signal efficiency)
  muDIF_leak   fraction of muDIF kept               (leakage); suppression = 1/leak
  suppression  1 / muDIF_leak
  eff_peak     pi->e nu kept with e_split<=E<=e_max  (deposited)  [peak]
  eff_tail     pi->e nu kept with E<e_split          (deposited)  [tail]
  bias         eff_tail / eff_peak                  (1.0 = no energy-dependent sculpting)

pi->e nu is two-body (monoenergetic ~69.8 MeV), so DEPOSITED energy carries the
peak/tail structure; deposits > e_max (default 75 MeV) are dropped as parquet
volume / double-counting artifacts. Rows at standard pie efficiencies
(--pie_effs) are added and annotated.

Usage:
  python mudif_suppression_table.py --pred path/to/predictions.parquet
  python mudif_suppression_table.py --pred ... --cuts 0.02 0.05 0.1 0.2
  python mudif_suppression_table.py --pred ... --pie_effs 0.25 0.5 0.9 0.95 0.99
"""
import argparse

import numpy as np
import pandas as pd


def load(pred, accept_min, e_max, e_split, score_col, pos_col, include_dead=True):
    df = pd.read_parquet(pred)
    if score_col not in df.columns or pos_col not in df.columns:
        raise KeyError(f"{pred} lacks {score_col}/{pos_col} — re-run eval_tail to "
                       f"regenerate predictions with the piDIF columns.")
    score = df[score_col].to_numpy()
    is_pie = df["is_pie"].to_numpy().astype(int)
    is_mudif = df[pos_col].to_numpy().astype(int)
    # Reco positron energy = live deposit + dead-material loss (dead loss grows with
    # angle, so a live-only tail is partly an angle artifact). Default adds it back.
    dep = df["deposited_energy"].to_numpy().astype(float)
    if include_dead and "dead_E" in df.columns:
        dep = dep + df["dead_E"].to_numpy().astype(float)
    acc = (df["acceptance"].to_numpy() >= accept_min) if "acceptance" in df.columns \
        else np.ones(len(df), bool)
    fin = np.isfinite(dep)
    pie = (is_pie == 1) & acc & fin & (dep <= e_max)
    mud = (is_mudif == 1) & acc
    return dict(
        score=score,
        pie=pie,
        mud=mud,
        peak=pie & (dep >= e_split),
        tail=pie & (dep < e_split),
        n_art=int(((is_pie == 1) & acc & fin & (dep > e_max)).sum()),
    )


def frac_below(score, mask, cut):
    """(efficiency, n) for 'score < cut' within mask."""
    s = score[mask]
    n = len(s)
    if n == 0:
        return float("nan"), 0
    return float((s < cut).mean()), n


def metrics(D, cut):
    sc = D["score"]
    ps, _ = frac_below(sc, D["pie"], cut)
    ml, _ = frac_below(sc, D["mud"], cut)
    ep, n_ep = frac_below(sc, D["peak"], cut)
    et, n_et = frac_below(sc, D["tail"], cut)
    supp = (1.0 / ml) if ml and ml > 0 else float("inf")
    if ep and ep > 0 and et == et:        # ep>0 and et not NaN
        bias = et / ep
        # independent-binomial error on the ratio
        v_ep = ep * (1.0 - ep) / max(n_ep, 1)
        v_et = et * (1.0 - et) / max(n_et, 1)
        bias_err = (bias * np.sqrt(v_et / et ** 2 + v_ep / ep ** 2)) if et > 0 else 0.0
    else:
        bias, bias_err = float("nan"), float("nan")
    return ps, ml, supp, ep, et, bias, bias_err


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred",
                    default="tail_reveal_eval/predictions.parquet",
                    help="eval_tail predictions parquet")
    ap.add_argument("--e_split", type=float, default=56.0,
                    help="peak/tail boundary (MeV, deposited energy)")
    ap.add_argument("--e_max", type=float, default=75.0,
                    help="drop pi->e nu with deposited E above this (volume artifacts)")
    ap.add_argument("--accept_min", type=float, default=0.5,
                    help="truth_acceptance threshold")
    ap.add_argument("--cuts", type=float, nargs="*",
                    default=[0.02, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50],
                    help="explicit veto-score cuts to tabulate")
    ap.add_argument("--pie_effs", type=float, nargs="*",
                    default=[0.25, 0.5, 0.9, 0.95, 0.99],
                    help="also add (and annotate) the cut achieving each pie efficiency")
    ap.add_argument("--head", choices=["mudif", "pidif"], default="mudif",
                    help="which DIF veto head to tabulate (selects score/label columns)")
    ap.add_argument("--no_dead_E", action="store_true",
                    help="split on the LIVE calorimeter deposit only (default adds dead_E back)")
    args = ap.parse_args()

    score_col = {"mudif": "muon_dif_score", "pidif": "pion_dif_score"}[args.head]
    pos_col   = {"mudif": "is_mudif",       "pidif": "is_pidif"}[args.head]
    name = {"mudif": "muDIF", "pidif": "piDIF"}[args.head]

    D = load(args.pred, args.accept_min, args.e_max, args.e_split, score_col, pos_col,
             include_dead=not args.no_dead_E)
    n_pie, n_mud = int(D["pie"].sum()), int(D["mud"].sum())
    n_peak, n_tail = int(D["peak"].sum()), int(D["tail"].sum())
    print(f"pred: {args.pred}   head: {name}")
    print(f"accepted: pie={n_pie} (peak {n_peak} / tail {n_tail})  {name}={n_mud}  "
          f"| dropped {D['n_art']} pie with E>{args.e_max:g} MeV (artifacts)")
    if n_pie == 0 or n_mud == 0:
        print(f"[error] need both accepted pie and accepted {name} events")
        return

    # cut where pie efficiency == pie_eff, annotated
    annot = {}
    cuts = list(args.cuts)
    for pe in args.pie_effs:
        c = float(np.quantile(D["score"][D["pie"]], pe))
        cuts.append(c)
        annot[round(c, 6)] = f"<- pie eff={pe:g}"
    cuts = sorted(set(round(c, 6) for c in cuts))

    print(f"\npeak = {args.e_split:g}<=E<={args.e_max:g} MeV   tail = E<{args.e_split:g} MeV   "
          f"(deposited);  veto keeps score < cut")
    print(f"\n{'cut':>7}  {'pie_surv':>8}  {name+'_leak':>10}  {'suppr':>7}  "
          f"{'eff_peak':>8}  {'eff_tail':>8}  {'bias(t/p)':>13}")
    for c in cuts:
        ps, ml, supp, ep, et, bias, be = metrics(D, c)
        supp_s = f"{supp:6.1f}x" if np.isfinite(supp) else "   inf"
        bias_s = f"{bias:.3f}+/-{be:.3f}" if bias == bias else "      nan"
        print(f"{c:7.4f}  {ps:8.4f}  {ml:10.5f}  {supp_s:>7}  "
              f"{ep:8.4f}  {et:8.4f}  {bias_s:>13}   {annot.get(round(c, 6), '')}")

    if n_tail < 30:
        print(f"\n[warn] only {n_tail} accepted-pie tail events (E<{args.e_split:g} MeV) "
              f"-> eff_tail / bias are statistics-limited")


if __name__ == "__main__":
    main()
