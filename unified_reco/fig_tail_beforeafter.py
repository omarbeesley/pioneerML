#!/usr/bin/env python
"""FIGURE -- the whole tail measurement in two panels: every channel before and after.

Left: \pie and the three backgrounds at acceptance, at their physical rates per stopped
pion.  Right: the same four after the complete selection chain.  The Michel and piDIF
spectra vanish, the muDIF residue survives at the loose working point, and the \pie
low-energy tail stands revealed below the 56 MeV line.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tail_figstyle

tail_figstyle.apply(scale=0.9)

OUT = "/home/obeesley/thesis_ref/figures/AI_final"
ES = 56.0
COLS = dict(pie="#0072B2", mic="#de2d26", pid="#009E73", mud="#CC79A7")
LABS = dict(pie=r"$\pi\to e$", mic=r"$\pi\to\mu\to e$", pid=r"$\pi$DIF", mud=r"$\mu$DIF")

d = np.load(f"{OUT}/tail_reveal_cache.npz")
bins = d["bins"]

fig, (aL, aR) = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)

for tag in ("mic", "pid", "mud", "pie"):
    aL.stairs(np.maximum(d[f"{tag}_acc_h"], 0), bins, color=COLS[tag], lw=1.6,
              label=LABS[tag])

after = {"pie": "pie_full_pidif_h", "mic": "mic_full_pidif_h",
         "pid": "pid_full_pidif_h", "mud": "mud_muon_h"}
for tag in ("mic", "pid", "mud", "pie"):
    h = d[after[tag]]
    if h.max() > 0:
        aR.stairs(np.maximum(h, 0), bins, color=COLS[tag], lw=1.6)

pie_fin = d["pie_full_pidif_h"]
aR.stairs(np.maximum(pie_fin, 0), bins, color=COLS["pie"], fill=True, alpha=0.14)

ymin = np.min(pie_fin[pie_fin > 0]) / 4
ymax = d["mic_acc_h"].max() * 3
for ax, title in ((aL, "before the selection (at acceptance)"),
                  (aR, "after the full chain")):
    ax.axvline(ES, color="k", ls=":", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("reconstructed positron energy [MeV]")
    ax.set_title(title, fontsize=10, pad=6)
aL.set_ylabel("rate per stopped pion / bin")
aL.legend(fontsize=8.5, loc="upper right", frameon=False)
lo = bins[:-1] < ES
bias = 100 * ((d["pie_full_pidif_h"][lo].sum() / d["pie_full_pidif_h"][~lo].sum())
              / (d["pie_time_h"][lo].sum() / d["pie_time_h"][~lo].sum()) - 1)
import json
scan = json.load(open(f"{OUT}/tail_stats.json"))["mudif_scan_retuned"]
contam = next(r["contam_pct"] for r in scan if r["mudif_cut"] == 0.46)
# event-level at exactly 56 MeV on the same full chain; the histogram bins have an edge
# at 55.5, so a bin-summed version spills past the cut and understates the contamination
aR.text(0.035, 0.935, rf"$\pi\to e$ tail bias: ${bias:+.1f} \pm 1.1$%"
        "\n" rf"$\mu$DIF in the low bin: {contam:.0f}%",
        transform=aR.transAxes, fontsize=8.5, color="0.2", va="top", linespacing=1.6)

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/tail_beforeafter.{ext}")
print(f"wrote {OUT}/tail_beforeafter.pdf")
