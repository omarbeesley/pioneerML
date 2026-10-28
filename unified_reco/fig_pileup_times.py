#!/usr/bin/env python
"""Generation times of the accidental components across the readout window.

Reads the recorder output written by scratchpad/pileup_fig/run_pileup_times.py,
which mixes events with the physical pileup configuration and stores the time at
which every accidental component was placed.  Nothing here is analytic: each
histogram is the realized output of the mixer.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tail_figstyle

tail_figstyle.apply(scale=0.82)

NPZ = os.environ.get("PILEUP_NPZ", "/home/obeesley/pioneerML/scratchpad/pileup_fig/pileup_times.npz")
OUT = os.environ.get("FIGDIR", "/home/obeesley/thesis_ref/figures/AI")

BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
INK = "0.15"

d = np.load(NPZ)
N = int(d["n_events"])
lo, hi = float(d["window_lo"]), float(d["window_hi"])
BW = 20.0
edges = np.arange(lo, hi + BW, BW)
ctr = 0.5 * (edges[1:] + edges[:-1])


def dens(key):
    """Placements per event per bin."""
    h, _ = np.histogram(d[key], bins=edges)
    return h / N, np.sqrt(h) / N


old, e_old = dens("old_muon")
new, e_new = dens("new_pion_positron")
tot, e_tot = old + new, np.hypot(e_old, e_new)
calo, e_calo = dens("calo_only")
radio, e_radio = dens("radioactivity")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.1))

# ---- panel (a): the accidental positrons -------------------------------------------------
ax.errorbar(ctr, tot, yerr=e_tot, fmt="o", ms=3.0, color=INK, lw=1.0,
            label="Total", zorder=5)
ax.errorbar(ctr, old, yerr=e_old, fmt="s", ms=2.6, color=BLUE, lw=1.0,
            label="Muon stopped before the window")
ax.errorbar(ctr, new, yerr=e_new, fmt="^", ms=2.8, color=VERM, lw=1.0,
            label="Pion entering the window")
ax.set_xlabel("Time in the readout window [ns]")
ax.set_ylabel(f"Accidental positrons per event per {BW:.0f} ns")
ax.set_xlim(lo, hi)
ax.set_ylim(0, max(tot.max() * 1.75, 1e-9))
ax.legend(loc="upper center", ncol=1, fontsize=6.6, handletextpad=0.5, borderpad=0.35)
ax.text(0.03, 0.955, "(a)", transform=ax.transAxes, va="top", fontsize=9)

# ---- panel (b): the calorimeter-only activity --------------------------------------------
bx.errorbar(ctr, radio, yerr=e_radio, fmt="o", ms=3.0, color=PURPLE, lw=1.0,
            label="LYSO radioactivity")
bx.errorbar(ctr, calo, yerr=e_calo, fmt="s", ms=2.8, color=GREEN, lw=1.0,
            label="Michel positron missing the ATAR")
bx.set_yscale("log")
bx.set_xlabel("Time in the readout window [ns]")
bx.set_ylabel(f"Calorimeter deposits per event per {BW:.0f} ns")
bx.set_xlim(lo, hi)
bx.legend(loc="center right", fontsize=6.6, handletextpad=0.5, borderpad=0.35)
bx.text(0.03, 0.955, "(b)", transform=bx.transAxes, va="top", fontsize=9)

fig.tight_layout()

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"pileup_time_components.{ext}")
    fig.savefig(p)
    print("wrote", p)

# ---- numbers the caption needs -----------------------------------------------------------
def slope_pct_per_window(y, e):
    """Fractional change across the window from a straight-line fit, in percent."""
    w = 1.0 / np.maximum(e, 1e-12) ** 2
    p, cov = np.polyfit(ctr, y, 1, w=np.sqrt(w), cov=True)
    span = (hi - lo)
    mean = y.mean()
    return 100 * p[0] * span / mean, 100 * np.sqrt(cov[0, 0]) * span / mean


print(f"events                    {N}")
print(f"old-muon   per event      {d['old_muon'].size / N:.4f}")
print(f"new-pion positrons/event  {d['new_pion_positron'].size / N:.4f}"
      f"  (stubs {d['new_pion_stub'].size / N:.4f}, "
      f"in-window fraction {d['new_pion_positron'].size / max(d['new_pion_stub'].size,1):.4f})")
print(f"calo-only  per event      {d['calo_only'].size / N:.4f}")
print(f"radioactivity per event   {d['radioactivity'].size / N:.3f}")
print(f"total accidental positrons per event {(d['old_muon'].size + d['new_pion_positron'].size)/N:.4f}")
if "acc_per_event" in d and d["acc_per_event"].size:
    print(f"mixer-booked accidentals per event   {d['acc_per_event'].mean():.4f}")
s, es = slope_pct_per_window(tot, e_tot)
print(f"total-rate slope across the window   {s:+.2f} +- {es:.2f} % per {hi-lo:.0f} ns")
s, es = slope_pct_per_window(old, e_old)
print(f"old-muon slope                       {s:+.2f} +- {es:.2f} %")
s, es = slope_pct_per_window(new, e_new)
print(f"new-pion slope                       {s:+.2f} +- {es:.2f} %")
