"""Thesis figure style for the tail figures.

Follows the convention of the analysis plots: serif text with Computer Modern mathtext,
ticks pointing inward on all four sides, minor ticks visible, and no grid.
"""
import matplotlib.pyplot as plt


def apply(scale=1.0):
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 220, "savefig.bbox": "tight",
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 11 * scale,
        "axes.labelsize": 12 * scale, "axes.titlesize": 12 * scale,
        "legend.fontsize": 10 * scale,
        "xtick.labelsize": 10.5 * scale, "ytick.labelsize": 10.5 * scale,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 6, "xtick.minor.size": 3,
        "ytick.major.size": 6, "ytick.minor.size": 3,
        "axes.grid": False, "axes.linewidth": 0.9, "axes.axisbelow": True,
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "0.8",
    })
