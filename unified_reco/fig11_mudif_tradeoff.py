#!/usr/bin/env python
"""FIGURE 11 -- the muDIF veto trade-off: contamination against induced tail bias.

One curve on the two axes the requirement is written in. Both quantities are plotted so that
"worse" runs the same way, the bias as a magnitude on a logarithmic axis, so the desirable
corner is the empty bottom left and the failure to reach it is geometry rather than assertion.
The veto threshold rides on the marker colour, which keeps the scan knob visible without a
single label on the plane. No fit is drawn: the points bend away from a power law at the loose
end, and a line through them would misstate the data exactly where the reader looks.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

import tail_figstyle

tail_figstyle.apply(scale=0.9)

OUT = "/home/obeesley/thesis_ref/figures/AI_final"

# The scan saturates at the loose end: past ~0.7 the veto is effectively off and the bias
# floor belongs to the rest of the chain, so those thresholds pile onto one point. Plot a
# spread subset, keeping the effectively-unvetoed endpoint and the nominal working point.
SHOWN = (0.95, 0.46, 0.20, 0.05, 0.02, 0.005)
scan = sorted([r for r in json.load(open(f"{OUT}/tail_stats.json"))["mudif_scan_retuned"]
               if r["mudif_cut"] in SHOWN], key=lambda r: -r["contam_pct"])
x = np.array([r["contam_pct"] for r in scan])
b = np.abs(np.array([r["bias_pct"] for r in scan]))
be = np.array([r["bias_unc_pct"] for r in scan])
cut = np.array([r["mudif_cut"] for r in scan])

YLO = 0.12
lo = np.maximum(b - be, YLO * 1.02)          # log axis: clip bars that would cross zero
yerr = np.vstack([b - lo, be])

fig, ax = plt.subplots(figsize=(5.4, 4.0))

ax.axhline(1.0, color="0.75", ls=":", lw=0.9, zorder=0)
ax.axvline(1.0, color="0.75", ls=":", lw=0.9, zorder=0)

ax.errorbar(x, b, yerr=yerr, fmt="none", ecolor="0.72", elinewidth=0.9, capsize=0, zorder=2)
ax.plot(x, b, "-", color="0.55", lw=1.0, zorder=3)
norm = LogNorm(vmin=cut.min(), vmax=cut.max())
CMAP = LinearSegmentedColormap.from_list(
    "blues_hi", plt.get_cmap("Blues_r")(np.linspace(0.0, 0.72, 256)))
ax.scatter(x, b, c=cut, cmap=CMAP, norm=norm, s=46, ec="0.2", lw=0.5, zorder=4)

cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=ax, pad=0.015, aspect=26)
cb.set_label(r"$\mu$DIF veto threshold")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.8, 26)
ax.set_ylim(YLO, 60)
ax.set_xlabel(r"$\mu$DIF contamination of the low-energy bin [%]")
ax.set_ylabel(r"induced tail bias [%]")
for a in (ax.xaxis, ax.yaxis):
    a.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v:g}"))

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig11_mudif_tradeoff.{ext}")
print(f"wrote {OUT}/fig11_mudif_tradeoff.pdf")
print(f"contamination {x.min():.2f}-{x.max():.2f}%, |bias| {b.min():.2f}-{b.max():.2f}%")
