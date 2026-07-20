"""
Per-head cut tables for the tail-reveal model, computed OFFLINE from an eval_tail
predictions.parquet (no re-inference). One table per model output head, in the same
format as mudif_suppression_table.py but for every head and its relevant background.

For each head, at a grid of pi->e nu (signal) efficiencies, reports:
  cut          the score threshold achieving that pie efficiency
  pie_surv     accepted pi->e nu kept (== the target efficiency, as a check)
  <bkg>_leak   fraction of that head's BACKGROUND kept (leakage); suppression = 1/leak
  suppr        1 / <bkg>_leak
  eff_peak     pi->e nu kept with e_split<=E<=e_max (deposited)   [peak]
  eff_tail     pi->e nu kept with E<e_split        (deposited)    [tail]
  bias(t/p)    eff_tail / eff_peak  (1.0 = no energy-dependent sculpting of the signal)

Cut DIRECTION and BACKGROUND per head:
  pie  / topo  TAGGER -> keep score > cut   vs Michel  (is_pie==0 & is_mudif==0 & is_pidif==0)
  muon         VETO   -> keep score < cut   vs stopped muon (muon_present==1)
  pileup       VETO   -> keep score < cut   vs pileup       (pileup_present==1)
  muon_dif     VETO   -> keep score < cut   vs muDIF        (is_mudif==1)
  pion_dif     VETO   -> keep score < cut   vs piDIF        (is_pidif==1)

ACCEPTANCE per head:
  All heads except pion_dif use the FULL truth_acceptance flag (positron survives window
  AND pion-stop fiducial 1.2<z<4.8,|x|<8,|y|<8 AND positron angle<120). For pion_dif the
  pion decays IN FLIGHT and never makes the fiducial stop, so the fiducial part would
  reject real piDIF -> pion_dif instead requires ONLY positron polar angle < 120 deg
  (reconstructed from the positron_theta column; no pion-stop fiducial).

The 75 MeV cap is applied to the PIE population only (deposited energy is used solely to
bin the signal into peak/tail; pi->e nu is monoenergetic ~69.8 MeV so >75 is a volume /
double-counting artifact). Backgrounds are NOT energy-capped (their energy isn't used).

Heads / backgrounds / columns missing from the parquet are skipped with a message.

Usage:
  python cut_tables.py --pred tail_reveal_eval_pidif/predictions.parquet
  python cut_tables.py --pred ... --heads pion_dif muon_dif --pie_effs 0.5 0.9 0.95 0.99
"""
import argparse

import numpy as np
import pandas as pd

# name, score col, direction ("high"=tagger keeps >cut / "low"=veto keeps <cut),
# background spec, background label, acceptance mode ("full" / "angle")
HEADS = [
    ("pie",      "pie_score",      "high", "michel",         "Michel", "full"),
    ("topo",     "topo_score",     "high", "michel",         "Michel", "full"),
    ("muon",     "muon_score",     "low",  "muon_present",   "muon",   "full"),
    ("pileup",   "pileup_score",   "low",  "pileup_present", "pileup", "full"),
    ("muon_dif", "muon_dif_score", "low",  "is_mudif",       "muDIF",  "full"),
    ("pion_dif", "pion_dif_score", "low",  "is_pidif",       "piDIF",  "angle"),
]


def accept_mask(df, mode, accept_min, angle_max_deg):
    """Per-head acceptance. 'full' = the stored truth_acceptance flag; 'angle' =
    positron polar angle < angle_max_deg ONLY (no pion-stop fiducial — for piDIF,
    whose pion decays in flight). Returns None if the needed column is absent."""
    if mode == "angle":
        if "positron_theta" not in df.columns:
            return None
        return np.degrees(df["positron_theta"].to_numpy()) < angle_max_deg
    if "acceptance" in df.columns:
        return df["acceptance"].to_numpy() >= accept_min
    return np.ones(len(df), bool)


def build_pops(df, dep, acc, e_max, e_split):
    """Pie (signal) population + peak/tail sub-bins for a given acceptance mask.
    The 75 MeV cap applies HERE (signal only), since dep is used to bin the signal."""
    fin = np.isfinite(dep)
    is_pie = df["is_pie"].to_numpy() == 1
    pie = is_pie & acc & fin & (dep <= e_max)
    n_art = int((is_pie & acc & fin & (dep > e_max)).sum())
    return pie, pie & (dep >= e_split), pie & (dep < e_split), n_art


def bkg_mask(df, spec, acc):
    """Accepted background population (NOT energy-capped — energy isn't used here)."""
    if spec == "michel":
        base = (df["is_pie"].to_numpy() == 0)
        for c in ("is_mudif", "is_pidif"):
            if c in df.columns:
                base &= (df[c].to_numpy() == 0)
        return base & acc
    if spec not in df.columns:
        return None
    return (df[spec].to_numpy() > 0.5) & acc


def kept(score, mask, thr, keep_high):
    """(kept fraction, binomial err, n) within mask for the given cut/direction."""
    s = score[mask]
    n = len(s)
    if n == 0:
        return float("nan"), float("nan"), 0
    k = (s > thr) if keep_high else (s < thr)
    e = float(k.mean())
    return e, float(np.sqrt(max(e * (1.0 - e), 0.0) / n)), n


def thr_for_eff(pie_score, eff, keep_high):
    """Threshold giving signal efficiency `eff`. keep_high: frac(>thr)=eff ->
    quantile(1-eff); keep_low: frac(<thr)=eff -> quantile(eff)."""
    return float(np.quantile(pie_score, (1.0 - eff) if keep_high else eff))


def head_table(df, head, dep, args):
    name, score_col, direction, bkg_spec, bkg_label, acc_mode = head
    keep_high = (direction == "high")

    if score_col not in df.columns:
        print(f"\n### {name:8s} [skip] no '{score_col}' column in parquet")
        return
    acc = accept_mask(df, acc_mode, args.accept_min, args.angle_max)
    if acc is None:
        print(f"\n### {name:8s} [skip] acceptance mode '{acc_mode}' needs the "
              f"'positron_theta' column — re-run eval_tail to regenerate predictions")
        return

    score = df[score_col].to_numpy()
    pie, peak, tail, n_art = build_pops(df, dep, acc, args.e_max, args.e_split)
    bkg = bkg_mask(df, bkg_spec, acc)
    n_pie, n_bkg = int(pie.sum()), (0 if bkg is None else int(bkg.sum()))
    if n_pie == 0:
        print(f"\n### {name:8s} [skip] no accepted pi->e nu under this acceptance")
        return
    if bkg is None or n_bkg == 0:
        print(f"\n### {name:8s} [skip] background '{bkg_label}' ({bkg_spec}) absent/empty")
        return

    acc_desc = (f"angle<{args.angle_max:g} deg only (no pion-stop fiducial)"
                if acc_mode == "angle" else "full truth_acceptance")
    arrow = ">" if keep_high else "<"
    print(f"\n### {name}  ({'TAGGER' if keep_high else 'VETO'}: keep score {arrow} cut)  "
          f"vs {bkg_label}")
    print(f"    acc: {acc_desc}  |  accepted pie={n_pie} "
          f"(peak {int(peak.sum())} / tail {int(tail.sum())}, dropped {n_art} >"
          f"{args.e_max:g}MeV)  N_bkg={n_bkg}")
    print(f"{'cut':>7}  {'pie_surv':>8}  {bkg_label + '_leak':>12}  {'suppr':>8}  "
          f"{'eff_peak':>8}  {'eff_tail':>8}  {'bias(t/p)':>14}")

    for eff in args.pie_effs:
        thr = thr_for_eff(score[pie], eff, keep_high)
        ps, _, _ = kept(score, pie, thr, keep_high)
        ml, _, _ = kept(score, bkg, thr, keep_high)
        ep, e_ep, _ = kept(score, peak, thr, keep_high)
        et, e_et, _ = kept(score, tail, thr, keep_high)
        supp = (1.0 / ml) if (ml == ml and ml > 0) else float("inf")
        if ep > 0 and et == et:
            bias = et / ep
            be = (bias * np.sqrt((e_et / et) ** 2 + (e_ep / ep) ** 2)) if et > 0 else 0.0
        else:
            bias, be = float("nan"), float("nan")
        supp_s = f"{supp:6.1f}x" if np.isfinite(supp) else "    inf"
        bias_s = f"{bias:.3f}+/-{be:.3f}" if bias == bias else "         nan"
        print(f"{thr:7.4f}  {ps:8.4f}  {ml:12.5f}  {supp_s:>8}  "
              f"{ep:8.4f}  {et:8.4f}  {bias_s:>14}   <- pie eff={eff:g}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="tail_reveal_eval_pidif/predictions.parquet",
                    help="eval_tail predictions parquet")
    ap.add_argument("--e_split", type=float, default=56.0,
                    help="peak/tail boundary (MeV, deposited energy)")
    ap.add_argument("--e_max", type=float, default=75.0,
                    help="drop pi->e nu (signal) with deposited E above this (volume artifacts)")
    ap.add_argument("--accept_min", type=float, default=0.5,
                    help="truth_acceptance threshold (full-acceptance heads)")
    ap.add_argument("--angle_max", type=float, default=120.0,
                    help="positron polar-angle acceptance ceiling (deg) for the angle-only mode")
    ap.add_argument("--pie_effs", type=float, nargs="*",
                    default=[0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99],
                    help="signal-efficiency operating points (one row each)")
    ap.add_argument("--heads", nargs="*", default=[h[0] for h in HEADS],
                    help="which heads to tabulate (default: all present)")
    ap.add_argument("--no_dead_E", action="store_true",
                    help="split on the LIVE calorimeter deposit only. Default adds the "
                         "dead-material loss back (deposited_energy + dead_E) so the tail "
                         "is real lost energy, not an angle-correlated dead-material artifact.")
    args = ap.parse_args()

    df = pd.read_parquet(args.pred)
    dep = df["deposited_energy"].to_numpy().astype(float)
    e_label = "deposited"
    if not args.no_dead_E:
        if "dead_E" in df.columns:
            dep = dep + df["dead_E"].to_numpy().astype(float)
            e_label = "deposited+dead"
        else:
            print("[warn] no dead_E column; falling back to live deposited energy only")
    print(f"pred: {args.pred}   ({len(df)} rows)")
    print(f"peak = {args.e_split:g}<=E<={args.e_max:g} MeV   tail = E<{args.e_split:g} MeV  "
          f"({e_label} energy; {args.e_max:g} MeV cap on signal only)")

    by_name = {h[0]: h for h in HEADS}
    for name in args.heads:
        if name not in by_name:
            print(f"\n### {name} [skip] unknown head (known: {', '.join(by_name)})")
            continue
        head_table(df, by_name[name], dep, args)


if __name__ == "__main__":
    main()
