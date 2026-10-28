"""Render the three reconstructed background cutflows from cached spectra.

Ported from fig_tail_reveal_cutflows.py; see docs/tail_analysis.md for the historical population.
"""
from ._runtime import command


def run(*, pipeline_dir=None, cache_dir=None, output_dir, record,
        skip_cutflows=False, michel_nmax=107):
    import os

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import style as tail_figstyle

    tail_figstyle.apply(scale=0.9)

    OUT = str(output_dir)
    BLUE = "#0072B2"
    ES = 56.0

    d = record.load_npz(cache_dir / "tail_reveal_cache.npz")
    bins = d["bins"]


    def step(ax, h, **kw):
        ax.stairs(np.maximum(h, 0), bins, **kw)


    def pct(n, n0):
        """Surviving fraction of the accepted sample, formatted for a legend."""
        if n == 0:
            return "none"
        f = 100.0 * n / n0
        if f >= 1:
            return f"{f:.0f}%" if f > 99 else f"{f:.1f}%"
        if f >= 0.1:
            return f"{f:.1g}%"
        e = int(np.floor(np.log10(f)))
        return f"${f / 10**e:.0f}\\times10^{{{e}}}$%"


    PANELS = [
        dict(tag="mic", name="tail_reveal_michel", blab=r"$\pi\to\mu\to e$",
             stages=[("acc", "acceptance"), ("time", "+ prompt window"),
                     ("topo", r"+ $\pi\to e$ topology"), ("muon", "+ muon vetoes")],
             shades=["#99000d", "#de2d26", "#fc9272", "0.35"]),
        dict(tag="pid", name="tail_reveal_pidif", blab=r"$\pi$DIF",
             stages=[("acc", "acceptance"), ("time", "+ prompt window"),
                     ("pdveto", "+ pion-DIF veto"), ("full_pidif", r"+ all $\mu$DAR cuts")],
             shades=["#00543c", "#009E73", "#7fcdae", "0.35"]),
        dict(tag="mud", name="tail_reveal_mudif", blab=r"$\mu$DIF",
             stages=[("acc", "acceptance"), ("time", "+ prompt window"),
                     ("topo", r"+ $\pi\to e$ topology"), ("muon", "+ loose muon veto")],
             shades=["#6d3f66", "#CC79A7", "#e7b8d4", "0.35"]),
    ]

    for p in PANELS:
        fig, ax = plt.subplots(figsize=(5.9, 4.4))

        pie_fin = d["pie_full_pidif_h"]
        step(ax, d["pie_acc_h"], color=BLUE, lw=1.0, ls=":",
             label=r"$\pi\to e$, acceptance (100%)")
        step(ax, pie_fin, color=BLUE, lw=1.7, fill=True, alpha=0.14)
        step(ax, pie_fin, color=BLUE, lw=1.7,
             label=rf"$\pi\to e$, full selection ({pct(d['pie_full_pidif_n'], d['pie_acc_n'])})")

        n0 = int(d[f"{p['tag']}_acc_n"])
        for (key, lab), col in zip(p["stages"], p["shades"]):
            n = int(d[f"{p['tag']}_{key}_n"])
            label = f"{p['blab']}, {lab} ({pct(n, n0)})"
            if n == 0:
                ax.plot([], [], " ", label=label)
            else:
                step(ax, d[f"{p['tag']}_{key}_h"], color=col, lw=1.5, label=label)

        ax.axvline(ES, color="k", ls=":", lw=0.9)
        ax.set_yscale("log")
        ymin = np.min(pie_fin[pie_fin > 0]) / 4
        ymax = max(d[f"{p['tag']}_acc_h"].max(), d["pie_acc_h"].max()) * 3.5
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("reconstructed positron energy [MeV]")
        ax.set_ylabel("rate per stopped pion / bin")
        ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  frameon=False, handlelength=1.5, columnspacing=1.3, labelspacing=0.4)

        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"{OUT}/{p['name']}.{ext}")
        print(f"wrote {OUT}/{p['name']}.pdf")


def main(argv=None):
    return command('cutflows', 'cache', run, __doc__, argv=argv,
                   diagnostics=False)


if __name__ == "__main__":
    main()
