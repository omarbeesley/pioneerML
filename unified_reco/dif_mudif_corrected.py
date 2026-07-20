"""muon-DIF: suppression spectrum (left) + the delta-ray-corrected pi->e TAIL-BIAS answer (right).

The delta-ray/Bhabha fix (Ecorr = live + dead + Epure, Epure = ATAR electron-only hit energy that
atar_posE drops) matters for the PI->E TAIL, not the muDIF background.  So:
  LEFT  : muDIF suppression spectrum, before + after veto at 99/90/50/30/10% pi->e eff
          (energy uncorrected -- the fix barely shifts the background).
  RIGHT : pi->e tail fraction (E<56) vs pi->e efficiency, CURRENT (live+dead) vs DELTA-RAY
          CORRECTED (Ecorr) -- shows how much of the tail erosion is the bookkeeping artifact.
Cut calibrated on the norad pie (with Epure); PURITY_TAIL10 scores are comparable across samples.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm, colors

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/pioneerML/unified_reco/updated_plots/08_mudif_spectrum.png"
ESPLIT, EMAX = 56.0, 75.0
EFFS = [99, 90, 50, 30, 10]
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

# pie with Epure (norad) -> defines the veto cuts AND the tail-fraction comparison
pie = pd.concat([pd.read_parquet(f"{P}/norad_joined_v2.parquet"),
                 pd.read_parquet(f"{P}/norad_joined2_v2.parquet")], ignore_index=True)
p_Edep = (pie.deposited_energy + pie.dead_E).to_numpy()
p_Ecorr = (p_Edep + pie.Epure).to_numpy()
p_sc = pie.muon_dif_score.to_numpy()
p_acc = ((pie.acceptance >= 1.0) & (pie.is_pie == 1)).to_numpy() & np.isfinite(p_Edep) & (p_Edep <= EMAX)
cut = lambda pe: np.quantile(p_sc[p_acc], pe / 100.0)

# muDIF spectrum (uncorrected energy) for the suppression view
md = pd.read_parquet(f"{P}/tail_reveal_eval10/predictions.parquet",
                     columns=["is_mudif", "muon_dif_score", "deposited_energy", "dead_E", "acceptance"])
md = md[md.is_mudif == 1]
md_E = (md.deposited_energy + md.dead_E).to_numpy(); md_sc = md.muon_dif_score.to_numpy()
md_acc = (md.acceptance >= 1.0).to_numpy() & np.isfinite(md_E) & (md_E <= EMAX)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[2, 1]))
bins = np.linspace(0, 75, 51)
axL.hist(md_E[md_acc], bins=bins, histtype="stepfilled", color="0.86", ec="0.6", zorder=0)
cmap = cm.get_cmap("Reds"); cval = lambda e: cmap(0.25 + 0.6 * (1 - e / 100.0))
for e in EFFS:
    surv = md_acc & (md_sc < cut(e))
    axL.hist(md_E[surv], bins=bins, histtype="step", lw=1.9, color=cval(e), zorder=3)
axL.axvline(ESPLIT, color="k", ls=":", lw=0.9); axL.set_yscale("log")
axL.set_xlabel("deposited energy  live+dead [MeV]"); axL.set_ylabel("accepted events / bin")
axL.set_title("muon-DIF suppression", fontweight="bold")
# colorbar built from the ACTUAL line colors so 10% eff (tightest veto, most suppression) reads dark
cb_cmap = colors.LinearSegmentedColormap.from_list("cb", [cval(e) for e in np.linspace(10, 99, 256)])
sm = cm.ScalarMappable(norm=colors.Normalize(10, 99), cmap=cb_cmap)
fig.colorbar(sm, ax=axL, pad=0.01).set_label(r"$\pi\to e$ efficiency [%]")

# RIGHT: pi->e tail fraction, NORMALIZED to its no-veto value (=1) so the RELATIVE veto bias is
# shown -- the absolute fraction is energy-definition-dependent (live vs live+dead) and misleading.
thr = np.quantile(p_sc[p_acc], np.linspace(0.005, 1.0, 200))
def sweep(E):
    ef, tf = [], []
    for t in thr:
        k = p_acc & (p_sc < t)
        if k.sum() == 0: continue
        ef.append(100.0 * k.sum() / p_acc.sum()); tf.append((E[k] < ESPLIT).sum() / k.sum())
    ef, tf = np.array(ef), np.array(tf)
    tf = tf / tf[np.argmax(ef)]
    keep = ef >= 5.0                                        # drop the extreme-cut low-stats tail
    return ef[keep], tf[keep]
e1, t1 = sweep(p_Ecorr)
axR.axhline(1.0, color="0.6", ls="--", lw=1.0, zorder=1)
axR.plot(e1, t1, "-", color="#2ca02c", lw=2.0)
axR.set_xlabel(r"$\pi\to e$ efficiency [%]")
axR.set_ylabel("Tail Fraction Bias")
axR.set_ylim(0.90, 1.13); axR.invert_xaxis()

fig.tight_layout(); fig.savefig(OUT); plt.close(fig)
print(f"wrote {OUT}")
print(f"pie accepted={p_acc.sum():,}  (delta-ray-corrected tail fraction, normalized to no-veto)")
