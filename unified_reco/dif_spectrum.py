"""muon-DIF and pion-DIF (two SEPARATE figures) reconstructed deposited-energy spectra, before
vs after the DIF veto at pi->e efficiencies 99/90/50/30/10%, with a right panel tracking how the
pi->e TAIL FRACTION (E<56, the low bin) evolves as the veto tightens -- i.e. whether the veto
distorts the pi->e tail while suppressing the DIF background.  No annotations; working point
encoded by color (darker = tighter veto).
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm, colors

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline/tail_reveal_eval10/predictions.parquet"
OUTDIR = "/home/obeesley/pioneerML/unified_reco/updated_plots"
ECUT = 56.0
EFFS = [99, 90, 50, 30, 10]
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

d = pd.read_parquet(P, columns=["is_pie", "is_mudif", "is_pidif", "muon_dif_score",
                                "pion_dif_score", "deposited_energy", "acceptance"])
d = d[(d.acceptance == 1) & (d.deposited_energy <= 75)].reset_index(drop=True)
pie = d[d.is_pie == 1]
bins = np.linspace(0, 75, 51)


def make(dcol, scol, name, out, cmap_name):
    dif = d[d[dcol] == 1]
    ps = pie[scol].to_numpy(); pE = pie.deposited_energy.to_numpy(); n_pie = len(pie)
    cmap = cm.get_cmap(cmap_name)
    cval = lambda e: cmap(0.25 + 0.6 * (1 - e / 100.0))     # darker = tighter (lower eff)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[2, 1]))
    # LEFT: DIF spectrum, before (grey) + after at each working point
    axL.hist(dif.deposited_energy, bins=bins, histtype="stepfilled", color="0.86", ec="0.6", zorder=0)
    for e in EFFS:
        cut = np.percentile(ps, e)
        surv = dif[dif[scol] < cut]
        axL.hist(surv.deposited_energy, bins=bins, histtype="step", lw=1.9, color=cval(e), zorder=3)
    axL.axvline(ECUT, color="k", ls=":", lw=0.9, zorder=1)
    axL.set_yscale("log"); axL.set_xlabel("reconstructed deposited energy [MeV]")
    axL.set_ylabel("accepted events / bin"); axL.set_title(name, fontweight="bold")
    sm = cm.ScalarMappable(norm=colors.Normalize(10, 99), cmap=cmap)
    cb = fig.colorbar(sm, ax=axL, pad=0.01); cb.set_label(r"$\pi\to e$ efficiency [%]")

    # RIGHT: pi->e tail fraction (E<56) vs pi->e efficiency, swept over the cut
    thr = np.percentile(ps, np.linspace(0.5, 100, 200))
    eff_sweep, tf_sweep = [], []
    for t in thr:
        k = ps < t
        if k.sum() == 0:
            continue
        eff_sweep.append(100.0 * k.sum() / n_pie)
        tf_sweep.append(100.0 * (pE[k] < ECUT).sum() / k.sum())
    axR.plot(eff_sweep, tf_sweep, "-", color="0.4", lw=1.5, zorder=2)
    for e in EFFS:
        k = ps < np.percentile(ps, e)
        tf = 100.0 * (pE[k] < ECUT).sum() / k.sum()
        axR.plot(e, tf, "o", color=cval(e), ms=9, zorder=4)
    axR.set_xlabel(r"$\pi\to e$ efficiency [%]"); axR.set_ylabel(r"$\pi\to e$ tail fraction  $E<56$ MeV [%]")
    axR.invert_xaxis()                                       # pi->e decreasing to the right (veto tightening)
    axR.set_title("tail fraction vs veto", fontsize=10)

    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, out)); plt.close(fig)
    print(f"wrote {out}  ({name}: {len(dif):,} accepted DIF)")


make("is_mudif", "muon_dif_score", "muon-DIF", "08_mudif_spectrum.png", "Reds")
make("is_pidif", "pion_dif_score", "pion-DIF", "09_pidif_spectrum.png", "Purples")
