#!/usr/bin/env python
"""FIGURE 12 -- the irreducible muDIF population, as two densities on one scale.

The plane is (energy the muon deposited in the ATAR) against (its kinetic energy at the
moment of decay).  Left: all accepted muDIF events.  Right: the survivors of the complete
selection chain with the veto at its tightest.  Each panel is normalized to its own sample,
so the two densities are directly comparable: the accepted population runs down the
energy-conservation band, while the survivors abandon it for the top-left corner, full
birth energy and less than a single MIP deposit.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import tail_figstyle

tail_figstyle.apply(scale=0.9)

OUT = "/home/obeesley/thesis_ref/figures/AI_final"
BLUE = "#0072B2"

d = np.load(f"{OUT}/tail_reveal_cache.npz")
aE, aK = d["acc_muE"], d["acc_dke"]
sE, sK = d["surv_muE"], d["surv_dke"]
mip_mpv = float(d["mip_mpv"])

XLO, XHI = 3e-3, 5.0
bx = np.geomspace(XLO, XHI, 49)
by = np.linspace(0, 4.25, 35)


def dens(E, K):
    h, _, _ = np.histogram2d(np.clip(E, XLO * 1.001, XHI * 0.999), K, bins=(bx, by))
    return h / max(len(E), 1)


fig, (aL, aR) = plt.subplots(1, 2, figsize=(8.2, 3.7), sharey=True,
                             constrained_layout=True)
norm = LogNorm(vmin=3e-5, vmax=0.3)

for ax, (E, K), title in ((aL, (aE, aK), f"all accepted $\\mu$DIF   (n = {len(aE):,})"),
                          (aR, (sE, sK),
                           f"survivors of the full chain, tight veto   (n = {len(sE):,})")):
    pc = ax.pcolormesh(bx, by, dens(E, K).T, cmap="Purples", norm=norm, rasterized=True)
    ax.axvline(mip_mpv, color=BLUE, ls="--", lw=1.1)
    ax.axhline(4.12, color="0.4", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(0, 4.45)
    ax.set_xlabel("muon deposit in the ATAR [MeV]")
    ax.set_title(title, fontsize=9.5, pad=5)

aL.set_ylabel(r"muon kinetic energy at decay $K_\mu$ [MeV]")
aL.text(mip_mpv * 1.13, 0.5, "one MIP hit", color=BLUE, fontsize=8, rotation=90,
        va="bottom")
aL.text(XHI * 0.85, 4.16, r"$K_\mu$ at production", color="0.3", fontsize=8, ha="right",
        va="bottom")

cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="Purples"), ax=(aL, aR),
                  pad=0.015, aspect=26)
cb.set_label("fraction of the sample / bin")

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig12_mudif_irreducible.{ext}")
print(f"wrote {OUT}/fig12_mudif_irreducible.pdf")
print(f"survivors: n={len(sE):,}, K>4.0: {(sK > 4.0).mean() * 100:.1f}%, "
      f"below one MIP: {(sE < mip_mpv).mean() * 100:.1f}%")
