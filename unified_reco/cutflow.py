"""
Sequential cut-flow on the per-event predictions.parquet from eval_tail.py.
Cuts are applied CUMULATIVELY in this order:
   0. (no cut)     -- the raw overlaid spectra
   1. acceptance   -- truth_acceptance==1: pion stop in fiducial box
                      (1.2<z<4.8, |x|,|y|<8) AND positron polar angle <120 deg
                      (AND triggering positron survives the readout window)
   2. muon veto    (keep if muon_score   < --muon_cut)
   3. pileup veto  (keep if pileup_score < --pileup_cut)
   4. pie topo     (keep if topo_score   > --topo_cut)
   5. pie cut      (keep if pie_score    > --pie_cut)

After each stage it writes an overlaid deposited-energy spectrum (Michel vs
pi->e nu) and reports, in the legend:
   - cumulative pi->e nu efficiency   (+/- binomial error)
   - cumulative Michel efficiency
   - the acceptance bias to the (E>cut)/(E<cut) ratio (+/- error), measured
     RELATIVE TO THE POST-ACCEPTANCE sample (the acceptance stage is the
     reference, bias=0; pre-acceptance stages show n/a):
       bias = eff(E>cut)/eff(E<cut) - 1   over post-acceptance pi->e nu.

Pure pandas/numpy/matplotlib -- run on the host.

Usage:
  python cutflow.py --pred tail_reveal_eval/predictions.parquet \
    --output_dir tail_reveal_eval/cutflow \
    --muon_cut 0.5 --pileup_cut 0.5 --topo_cut 0.5 --pie_cut 0.5 \
    --energy_cut 56 --emax 75
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def eff_err(k, n):
    """Binomial efficiency k/n and its error."""
    if n == 0:
        return float("nan"), float("nan")
    e = k / n
    return e, float(np.sqrt(max(e * (1.0 - e), 0.0) / n))


def ratio_bias_err(energy_pie, keep_pie, ref_pie, ecut):
    """Bias relative to the REFERENCE (post-acceptance) pi->e nu sample: among
    ref_pie events, eff(E>ecut)/eff(E<ecut) - 1 of the cumulative cut, with
    propagated binomial error."""
    hi = (energy_pie > ecut) & ref_pie
    lo = (energy_pie < ecut) & ref_pie
    e_hi, s_hi = eff_err(int(keep_pie[hi].sum()), int(hi.sum()))
    e_lo, s_lo = eff_err(int(keep_pie[lo].sum()), int(lo.sum()))
    if not (e_lo > 0) or not (e_hi > 0):
        return (e_hi / e_lo - 1.0 if e_lo > 0 else float("nan")), float("nan")
    r = e_hi / e_lo
    rel = np.sqrt((s_hi / e_hi) ** 2 + (s_lo / e_lo) ** 2)
    return r - 1.0, float(r * rel)


def bias_text(mode, bias, berr):
    if mode == "count":
        return f"{bias:+.4f} +/- {berr:.4f}"
    if mode == "reference":
        return "0 (post-acceptance reference)"
    return "n/a (pre-acceptance)"


def plot_spectrum(E, is_pie, keep, ref_pie, n_pie0, n_mic0, bins, ecut,
                  title, path, bias_mode):
    kp, km = keep & is_pie, keep & ~is_pie
    pie_eff, pie_err = eff_err(int(kp.sum()), n_pie0)
    mic_eff, _ = eff_err(int(km.sum()), n_mic0)
    if bias_mode == "count":
        bias, berr = ratio_bias_err(E[is_pie], keep[is_pie], ref_pie, ecut)
    else:
        bias, berr = (0.0, 0.0) if bias_mode == "reference" else (float("nan"), float("nan"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(E[kp], bins=bins, histtype="step", lw=1.9, color="tab:blue",
            label=f"pi->e nu  (eff = {pie_eff:.4f} +/- {pie_err:.4f})")
    ax.hist(E[km], bins=bins, histtype="step", lw=1.9, color="tab:red",
            label=f"Michel    (eff = {mic_eff:.3e})")
    ax.axvline(ecut, color="k", ls=":", lw=0.9, label=f"E = {ecut:g} MeV")
    ax.plot([], [], " ", label=f"accept. bias = {bias_text(bias_mode, bias, berr)}")
    ax.set_yscale("log")
    ax.set_xlabel("deposited energy (MeV)"); ax.set_ylabel("events / bin")
    ax.set_title(title); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return pie_eff, pie_err, mic_eff, bias, berr


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="predictions.parquet from eval_tail.py")
    p.add_argument("--output_dir", default="cutflow")
    p.add_argument("--muon_cut", type=float, default=0.5)
    p.add_argument("--pileup_cut", type=float, default=0.5)
    p.add_argument("--topo_cut", type=float, default=0.5)
    p.add_argument("--pie_cut", type=float, default=0.5)
    p.add_argument("--energy_cut", type=float, default=56.0)
    p.add_argument("--emax", type=float, default=float("inf"),
                   help="Drop events with deposited_energy above this (MeV); ~75 removes "
                        "the unphysical LYSO-doubling tail.")
    p.add_argument("--n_ebins", type=int, default=40, help="Energy-spectrum histogram bins.")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.pred)
    if np.isfinite(args.emax):
        n0 = len(df)
        df = df[df["deposited_energy"] <= args.emax].reset_index(drop=True)
        print(f"dropped {n0 - len(df)} events with deposited_energy > {args.emax} MeV")

    is_pie = df["is_pie"].values.astype(bool)
    E = df["deposited_energy"].values
    n_pie0, n_mic0 = int(is_pie.sum()), int((~is_pie).sum())
    bins = np.linspace(0.0, float(E.max()), args.n_ebins + 1)
    print(f"start: pie={n_pie0}, michel={n_mic0}\n")

    acc_present = "acceptance" in df.columns
    acc_full = (df["acceptance"].values == 1) if acc_present else np.ones(len(df), bool)
    ref_pie = acc_full[is_pie]          # bias reference = post-acceptance pi->e nu
    if not acc_present:
        print("[warn] no 'acceptance' column -- re-run eval_tail.py to get the acceptance "
              "stage; bias will be counted relative to ALL pie instead.\n")

    stages = []
    if acc_present:
        stages.append(("1_acceptance", "acceptance", acc_full,
                       "pion-stop fiducial & theta<120 (truth_acceptance==1)"))
    stages += [
        ("2_muonveto",   "muon veto",   df["muon_score"].values   < args.muon_cut,   f"muon_score < {args.muon_cut:g}"),
        ("3_pileupveto", "pileup veto", df["pileup_score"].values < args.pileup_cut, f"pileup_score < {args.pileup_cut:g}"),
        ("4_topo",       "pie topo",    df["topo_score"].values   > args.topo_cut,   f"topo_score > {args.topo_cut:g}"),
        ("5_pie",        "pie cut",     df["pie_score"].values    > args.pie_cut,    f"pie_score > {args.pie_cut:g}"),
    ]

    print(f"{'stage':<13} {'pie_eff':>20} {'michel_eff':>13}   "
          f"accept_bias(>{args.energy_cut:g}/<{args.energy_cut:g})")

    def row(name, pe, pee, me, mode, bias, berr):
        print(f"{name:<13} {pe:>12.4f}+/-{pee:.4f} {me:>13.3e}   {bias_text(mode, bias, berr)}")

    # stage 0: before any cuts (bias not counted yet)
    keep = np.ones(len(df), bool)
    pe, pee, me, bi, be = plot_spectrum(E, is_pie, keep, ref_pie, n_pie0, n_mic0, bins,
                                        args.energy_cut, "before any cuts",
                                        f"{args.output_dir}/cutflow_0_nocut.png", "none")
    row("no cut", pe, pee, me, "none", bi, be)

    # cumulative cuts
    for tag, label, passmask, descr in stages:
        keep = keep & passmask
        if acc_present:
            mode = "reference" if tag == "1_acceptance" else "count"
        else:
            mode = "count"
        title = f"after {tag[0]}. {label}  ({descr})  [cumulative]"
        pe, pee, me, bi, be = plot_spectrum(E, is_pie, keep, ref_pie, n_pie0, n_mic0, bins,
                                            args.energy_cut, title,
                                            f"{args.output_dir}/cutflow_{tag}.png", mode)
        row(label, pe, pee, me, mode, bi, be)

    print(f"\nwrote cutflow_0_nocut.png + cutflow_*.png to {args.output_dir}/")


if __name__ == "__main__":
    main()
