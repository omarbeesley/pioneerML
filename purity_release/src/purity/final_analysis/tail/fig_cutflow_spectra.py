"""The tail-reveal cutflow as one energy spectrum per cut, for talks.

The thesis figure stacks every stage of the background into a single panel in successively
lighter shades.  On a page that is the denser and better figure: the reader can trace one curve
down through the stages at their own pace.  In a talk it asks the audience to hold four curves
and a two-column legend at once.  Here the stage is the panel instead of the shade, so each
panel carries exactly two curves -- signal and background, always the same two colours -- and
the eye reads left to right watching the background empty out from under the tail.

The four panels are the settled muDAR chain.  Both spectra keep the same axes and the same
vertical range throughout, so a curve that shrinks between panels really did shrink; nothing is
renormalised panel to panel.  The dotted line is the 56 MeV kinematic edge of the michel
spectrum: to its right is where the pi->e signal lives.

Each panel quotes the surviving fraction of each channel's OWN accepted sample, which is what
makes the two numbers comparable when the raw rates differ by orders of magnitude.

Portable copy of figures/talk/fig_cutflow_spectra.py; all helpers are local.
"""
from ._runtime import command


def run(*, pipeline_dir=None, cache_dir=None, output_dir, record,
        skip_cutflows=False, michel_nmax=107):

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgb

    from . import style as tail_figstyle

    tail_figstyle.apply(scale=0.95)

    CACHE = cache_dir / "tail_reveal_cache.npz"
    OUT = str(output_dir)
    BLUE = "#0072B2"
    ES = 56.0
    # One legend style shared by every talk figure.  A light frame rather than none: floated over
    # white the keys read as stray marks, and these panels have live curves close to where the
    # legend sits.
    LEGEND = dict(frameon=True, framealpha=0.95, edgecolor="0.75", facecolor="white",
                  fancybox=False, borderpad=0.55, labelspacing=0.45, handlelength=1.5,
                  handletextpad=0.7, borderaxespad=0.55, fontsize=9.5)

    PANELS = [
        dict(tag="mic", name="cutflow_spectra_michel", blab=r"$\pi\to\mu\to e$", bcol="#99000d",
             stages=[("acc", "acceptance"), ("time", "+ prompt window"),
                     ("topo", r"+ $\pi\to e$ topology"), ("muon", "+ muon vetoes")]),
        dict(tag="pid", name="cutflow_spectra_pidif", blab=r"$\pi$DIF", bcol="#00543c",
             stages=[("acc", "acceptance"), ("time", "+ prompt window"),
                     ("pdveto", "+ pion-DIF veto"), ("full_pidif", r"+ all $\mu$DAR cuts")]),
        dict(tag="mud", name="cutflow_spectra_mudif", blab=r"$\mu$DIF", bcol="#6d3f66",
             stages=[("acc", "acceptance"), ("time", "+ prompt window"),
                     ("topo", r"+ $\pi\to e$ topology"), ("muon", "+ loose muon veto")]),
    ]

    d = record.load_npz(CACHE)
    bins = d["bins"]


    def pct(n, n0):
        if n == 0:
            return "none"
        f = 100.0 * n / n0
        if f >= 99.95:
            return "100%"
        if f >= 1:
            return f"{f:.1f}%"
        # Plain decimals rather than scientific: at these magnitudes "0.0061%" is read straight
        # off the slide, where "6.1e-3%" makes the audience do arithmetic mid-sentence.  Two
        # significant figures throughout.
        dp = max(0, 1 - int(np.floor(np.log10(f))))
        return f"{f:.{dp}f}%"


    for p in PANELS:
        st = p["stages"]
        n0_b, n0_s = float(d[f"{p['tag']}_acc_n"]), float(d["pie_acc_n"])
        have_pie = [f"pie_{k}_n" in d for k, _ in st]

        # One vertical range for the whole row.  Rescaling per panel would make the background look
        # unchanged from one cut to the next, which is the opposite of the point.
        hs = [d[f"{p['tag']}_{k}_h"] for k, _ in st] + \
             [d[f"pie_{k}_h"] for (k, _), ok in zip(st, have_pie) if ok]
        pos = np.concatenate([h[h > 0] for h in hs if (h > 0).any()])
        # Headroom for the surviving fractions in the top-left of every panel and the key in
        # the top-right of the first: at the acceptance stage the background curve runs
        # right through where both would otherwise sit.
        ylo, yhi = pos.min() / 3.0, max(h.max() for h in hs) * 700.0

        fig, axes = plt.subplots(1, len(st), figsize=(3.05 * len(st) + 0.5, 3.5),
                                 sharex=True, sharey=True)
        for ax, (k, lab), ok in zip(np.atleast_1d(axes), st, have_pie):
            nb = int(d[f"{p['tag']}_{k}_n"])
            if ok:
                hp = d[f"pie_{k}_h"]
                ax.stairs(np.maximum(hp, 0), bins, color=BLUE, lw=1.5, fill=True, alpha=0.14)
                ax.stairs(np.maximum(hp, 0), bins, color=BLUE, lw=1.5)
            if nb > 0:
                ax.stairs(np.maximum(d[f"{p['tag']}_{k}_h"], 0), bins, color=p["bcol"], lw=1.5)

            ax.axvline(ES, color="k", ls=":", lw=0.9)
            ax.set_title(lab, pad=6)
            # The surviving fraction belongs to a channel, so it is quoted in that channel's own
            # legend entry rather than set loose in a corner of the panel.  Each panel then reads on
            # its own, and no number has to be matched back to a colour.
            h = []
            if ok:
                h.append(Patch(facecolor=(*to_rgb(BLUE), 0.14), edgecolor=BLUE, lw=1.5,
                               label=rf"$\pi\to e$  ({pct(int(d[f'pie_{k}_n']), n0_s)})"))
            if nb > 0:
                h.append(Line2D([], [], color=p["bcol"], lw=1.5,
                                label=f"{p['blab']}  ({pct(nb, n0_b)})"))
            else:
                # No line key where there is no curve: a key with nothing behind it would say the
                # channel is drawn somewhere in the panel.
                h.append(Line2D([], [], ls="none", marker="", label=f"{p['blab']}  (none)"))
            ax.legend(handles=h, loc="upper right", **LEGEND)

        ax0 = np.atleast_1d(axes)[0]
        ax0.set_yscale("log")
        ax0.set_ylim(ylo, yhi)
        ax0.set_xlim(bins[0], bins[-1])
        ax0.set_ylabel("rate per stopped pion / bin")
        fig.supxlabel("reconstructed positron energy [MeV]", y=0.02)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"{OUT}/{p['name']}.{ext}")
        print(f"wrote {p['name']}.pdf", flush=True)
    print("SPECTRA_DONE", flush=True)


def main(argv=None):
    return command('fig_cutflow_spectra', 'cache', run, __doc__, argv=argv,
                   diagnostics=False)


if __name__ == '__main__':
    main()
