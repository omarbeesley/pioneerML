"""Shared style for the ML-background pedagogical figures.

No system LaTeX is available, so equations are rendered with matplotlib
mathtext using the Computer-Modern font set paired with a serif text font,
which reads consistently with thesis body type. Palette reuses the house
analysis colours (red/blue/green) so the pedagogy figures match the rest of
the document.
"""
import os
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.axisbelow": True,
    "axes.linewidth": 0.9,
    "lines.antialiased": True,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8",
})

# House palette + semantic roles for these figures.
PAL = dict(
    red="#d62728",
    blue="#1f77b4",
    green="#2ca02c",
    orange="#ff7f0e",
    grey="#bbbbbb",
    signal="#1f77b4",      # signal cluster (inside the circle)
    background="#d62728",   # background (outside the circle)
    truth="#111111",        # reference / analytic answer
)


def save(fig, stem):
    """Save a figure as both vector PDF (for LaTeX) and PNG (for review)."""
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(OUTDIR, f"{stem}.{ext}")
        fig.savefig(p)
        paths.append(p)
    return paths
