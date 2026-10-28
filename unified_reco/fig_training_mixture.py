#!/usr/bin/env python
"""Composition of the two PURITY training mixtures, for the AI chapter.

Three panels: what triggers a main-model event, what pileup augmentation is laid
on top of it, and what triggers a tail-reveal event.  The main-model fractions are
the production mixer settings (generate_benchmarks.TRAIN_VAL_OPTS); the
tail-reveal fractions were measured directly on the production training parquet,
/data/nvme0/prod_ml_data/mixed_parquets/tail_reveal_piDIF/tail_reveal_training.parquet,
which holds 1e6 events at 30.05 / 30.00 / 25.00 / 14.95 percent.

Colors are the Okabe-Ito qualitative palette, which is designed to stay separable
under the common forms of color vision deficiency.  Every wedge is labeled, so the
identity of a class never rests on its color alone.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tail_figstyle

tail_figstyle.apply(scale=0.82)

OUT = os.environ.get("FIGDIR", "/home/obeesley/thesis_ref/figures/AI")

# Okabe-Ito
BLUE = "#0072B2"     # pie
RED = "#de2d26"      # michel, matching the chapter convention
GREEN = "#009E73"    # muDIF
PURPLE = "#CC79A7"   # piDIF
ORANGE = "#E69F00"
GREY = "#BFBFBF"     # the "none" category

PANELS = [
    dict(
        title="Main model:\ntriggering decay",
        labels=[r"$\pi\to\mu\to e$", r"$\pi\to e$"],
        values=[80.0, 20.0],
        colors=[RED, BLUE],
    ),
    dict(
        title="Main model:\npileup augmentation",
        labels=["Additional calorimeter-only\npileup", "Additional accidental\npositron in ATAR", "No overlay added"],
        values=[50.0, 10.0, 40.0],
        colors=[GREEN, ORANGE, GREY],
    ),
    dict(
        title="Tail-reveal model:\ntriggering decay",
        labels=[r"$\pi\to\mu\to e$", r"$\pi\to e$", r"$\mu$DIF", r"$\pi$DIF"],
        values=[30.0, 30.0, 25.0, 15.0],
        colors=[RED, BLUE, GREEN, PURPLE],
    ),
]

fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.3))

for ax, p in zip(axes, PANELS):
    wedges, _, autotexts = ax.pie(
        p["values"],
        colors=p["colors"],
        startangle=90,
        counterclock=False,
        autopct=lambda v: f"{v:.0f}%",
        pctdistance=0.62,
        wedgeprops=dict(width=0.62, edgecolor="white", linewidth=1.1),
        textprops=dict(color="white", fontsize=8),
    )
    for t in autotexts:
        t.set_fontweight("bold")
    # the grey "none" wedge needs dark text to stay legible
    for w, t in zip(wedges, autotexts):
        if w.get_facecolor()[:3] == matplotlib.colors.to_rgb(GREY):
            t.set_color("0.25")
    ax.set_title(p["title"], pad=8, fontsize=8.5)
    ax.legend(wedges, p["labels"], loc="upper center", bbox_to_anchor=(0.5, -0.02),
              frameon=False, fontsize=6.4, ncol=2 if len(p["labels"]) > 2 else 1,
              handlelength=1.1, handletextpad=0.6, columnspacing=1.2)
    ax.set_aspect("equal")

fig.subplots_adjust(wspace=0.30, bottom=0.20, top=0.84)

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    path = os.path.join(OUT, f"training_mixture.{ext}")
    fig.savefig(path)
    print("wrote", path)
