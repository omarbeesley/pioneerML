#!/usr/bin/env python
"""FIGURE 10 -- why an aggressive muDIF veto biases the tail.

Three stacked panels against the true positron polar angle, all from the train-mode
central cache (58 MB, safe outside SLURM):

  (a) muDIF suppression of a tight in-flight-muon veto;
  (b) the fraction of accepted pie events that veto removes;
  (c) the pie tail fraction, the quantity the selection must not sculpt.

The point of the figure is the alignment of (b) and (c): the veto removes pie
preferentially at the angles where the tail fraction is largest, which is how an
energy-blind topological cut acquires an energy bias.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tail_figstyle

tail_figstyle.apply(scale=0.86)

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI_final"
BLUE, VERM = "#0072B2", "#D55E00"
ES = 56.0
PIE_EFF = 0.50          # the aggressive working point: the veto keeps 50% of pie

pr = pd.read_parquet(f"{P}/tail_reveal_eval10/predictions_drop.parquet",
                     columns=["is_pie", "is_mudif", "acceptance", "muon_dif_score",
                              "deposited_energy", "positron_theta"])
pr["positron_theta"] = np.degrees(pr.positron_theta)   # stored in radians
acc = pr[pr.acceptance >= 1]
pie = acc[acc.is_pie == 1]
mud = acc[acc.is_mudif == 1]

# threshold at the requested pie efficiency (veto keeps score < thr)
thr = float(np.quantile(pie.muon_dif_score, PIE_EFF))
print(f"threshold for {100*PIE_EFF:.0f}% pie efficiency: muon_dif_score < {thr:.4f}")

edges = np.linspace(0, 180, 19)
ctr = 0.5 * (edges[1:] + edges[:-1])


def binned(theta, mask):
    n, _ = np.histogram(theta, bins=edges)
    k, _ = np.histogram(theta[mask], bins=edges)
    return n, k


fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(5.6, 6.4), sharex=True)

# (a) muDIF suppression
n, k = binned(mud.positron_theta.values, (mud.muon_dif_score.values < thr).astype(bool))
ok = (n > 40) & (k > 0)
supp = n[ok] / k[ok]
a1.errorbar(ctr[ok], supp, yerr=supp * np.sqrt(1 / k[ok]), fmt="o", ms=3.2,
            color="#CC79A7", lw=1.0)
a1.set_ylabel(r"$\mu$DIF suppression")
a1.text(0.02, 0.88, "(a)", transform=a1.transAxes, fontsize=9)

# (b) pie removed by the veto
n, k = binned(pie.positron_theta.values, (pie.muon_dif_score.values >= thr).astype(bool))
ok = n > 400
rem = k[ok] / n[ok]
a2.errorbar(ctr[ok], 100 * rem, yerr=100 * np.sqrt(rem * (1 - rem) / n[ok]), fmt="s",
            ms=3.0, color=BLUE, lw=1.0)
a2.set_ylabel(r"$\pi\to e$ removed [%]")
a2.text(0.02, 0.88, "(b)", transform=a2.transAxes, fontsize=9)

# (c) pie tail fraction
n, k = binned(pie.positron_theta.values, (pie.deposited_energy.values < ES).astype(bool))
ok = n > 400
tf = k[ok] / n[ok]
a3.errorbar(ctr[ok], 100 * tf, yerr=100 * np.sqrt(tf * (1 - tf) / n[ok]), fmt="^",
            ms=3.2, color=VERM, lw=1.0)
a3.set_ylabel("tail fraction [%]")
a3.set_xlabel(r"true polar angle $\theta$ [deg]")
for a in (a1, a2, a3):
    a.axvline(120, color="0.55", ls=":", lw=0.9)
a3.set_xlim(0, 130)
a3.text(0.02, 0.88, "(c)", transform=a3.transAxes, fontsize=9)

fig.align_ylabels()
fig.tight_layout(h_pad=0.4)
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig10_mudif_angular.{ext}")
print(f"wrote {OUT}/fig10_mudif_angular.pdf")

# numbers for the caption
print(f"pie sample {len(pie):,}, muDIF sample {len(mud):,} (accepted)")
print(f"overall suppression at this threshold: "
      f"{len(mud) / max((mud.muon_dif_score < thr).sum(), 1):.0f}")
