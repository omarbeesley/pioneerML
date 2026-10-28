"""Where the muDIF residual lives after the retuned chain (prompt window -> topo>0.99 ->
muon veto -> pion-DIF veto), on the reconstructed quantities.

Outputs figures/AI/tail_mudif_retuned.png: left, the reconstructed-energy spectrum of
accepted muDIF, of the retuned-chain survivors, and of the full-chain pie, at the 20:1
acceptance normalization; right, the muon decay kinetic energy of the survivors against all
accepted muDIF (the instant-decay population). Stats appended to tail_stats.json.
"""
import json
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
import tail_figstyle
tail_figstyle.apply()
C_PIE, C_MUD = "#d62728", "#9467bd"
ES, EM = 56.0, 75.0
import tail_rates as TR
R_PIE, RATIO = TR.BR_PIE, 20.0
import glob, os

TCOLS = ["muon_score", "pileup_score", "topo_score", "pie_score", "muon_dif_score",
         "pion_dif_score", "deposited_energy"]
MCOLS = ["truth_live_E", "pred_accepted", "pred_positron_energy", "pred_dead_energy",
         "pred_positron_time_consensus_ns"]


def f32(x):
    return np.asarray(x, dtype=np.float32)


def paired(tail, main):
    tE, mE = f32(tail.deposited_energy.values), f32(main.truth_live_E.values)
    ti, pt, pm = 0, [], []
    for mi in range(len(main)):
        while ti < len(tail) and tE[ti] != mE[mi]:
            ti += 1
        if ti >= len(tail):
            break
        pt.append(ti); pm.append(mi); ti += 1
    assert len(pt) >= 0.999 * len(main)
    j = tail.iloc[pt].reset_index(drop=True)
    mm = main.iloc[pm].reset_index(drop=True)
    for c in mm.columns:
        j["m_" + c] = mm[c].values
    return j


# pie (big pool) for the window anchor, efficiency, and overlay
rows = []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet")):
    g = ev.split("shard_")[-1].split("/")[0]
    tp = f"{P}/norad_eval2_rs/ev_{g}/predictions.parquet"
    if not os.path.exists(tp):
        continue
    m = pd.read_parquet(ev, columns=MCOLS + ["truth_is_pie"])
    t = pd.read_parquet(tp, columns=TCOLS + ["is_pie"])
    if len(m) != len(t):
        continue
    j = paired(t, m)
    rows.append(j[(j.is_pie == 1) & (j.m_truth_is_pie == 1)])
pie = pd.concat(rows, ignore_index=True)

mt = pd.read_parquet(f"{P}/tail_reveal_eval10/predictions_evalmode_bak.parquet",
                     columns=TCOLS + ["is_mudif", "muon_decay_ke"])
mt = mt[mt.is_mudif == 1].reset_index(drop=True)
mmain = pd.read_parquet(f"{P}/purity_eval/tail_osl/mudif_osl_events.parquet", columns=MCOLS)
mud = paired(mt, mmain)
print(f"pie {len(pie):,}  mudif {len(mud):,}", flush=True)


def reco(j):
    E = j.m_pred_positron_energy.values + j.m_pred_dead_energy.values
    acc = (j.m_pred_accepted.values >= 1) & np.isfinite(E) & (E >= 0) & (E <= EM)
    return E, acc, j.m_pred_positron_time_consensus_ns.values


Ep, pacc, tp_ = reco(pie)
onset = np.quantile(tp_[pacc & np.isfinite(tp_)], 0.005)
TLO, THI = onset + 2.0, onset + 32.0
Em, macc, tm = reco(mud)


def chain(j, acc, t):
    return acc & (t >= TLO) & (t <= THI) & (j.topo_score.values > 0.99) \
           & (j.muon_score.values < 0.02) & (j.muon_dif_score.values < 0.46) \
           & (j.pion_dif_score.values < 0.01)


kp = chain(pie, pacc, tp_)
km = chain(mud, macc, tm)
wpie = R_PIE / pacc.sum()
wdif = (R_PIE / RATIO) / macc.sum()
plo = wpie * (kp & (Ep < ES)).sum()
mlo = wdif * (km & (Em < ES)).sum()
TS = TR.tail_rescale(Ep[pacc])
contam = TR.contamination(mlo, plo, TS)
supp = macc.sum() / max(km.sum(), 1)
dke = mud.muon_decay_ke.values
stats = dict(mudif_accepted=int(macc.sum()), mudif_survivors=int(km.sum()),
             suppression=float(supp), low_bin_contam_pct=float(contam),
             pie_eff_pct=float(100 * kp.sum() / pacc.sum()),
             survivor_frac_full_ke=float((dke[km] > 3.9).mean()),
             survivor_median_Ereco=float(np.median(Em[km])),
             survivor_frac_below_56=float((Em[km] < ES).mean()))
print(json.dumps(stats, indent=1), flush=True)

fig, (aL, aR) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
bins = np.linspace(0, EM, 51)
aL.hist(Em[macc], bins=bins, weights=np.full(macc.sum(), wdif), histtype="stepfilled",
        color="0.88", ec="0.6", label=r"$\mu$DIF at acceptance")
aL.hist(Em[km], bins=bins, weights=np.full(km.sum(), wdif), histtype="step", lw=1.9,
        color=C_MUD, label=r"$\mu$DIF after the retuned chain")
aL.hist(Ep[kp], bins=bins, weights=np.full(kp.sum(), wpie), histtype="step", lw=1.6,
        color=C_PIE, label=r"$\pi\to e$ after the retuned chain")
aL.axvline(ES, color="k", ls=":", lw=0.9); aL.set_yscale("log")
aL.set_xlabel("reconstructed energy [MeV]"); aL.set_ylabel("branching-weighted rate [a.u.]")
aL.text(0.03, 0.03, f"suppression: {supp:,.0f}$\\times$\nlow-bin contam: {contam:.1f}%",
        transform=aL.transAxes, va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.92))
aL.legend(fontsize=8.5, loc="upper left")
b2 = np.linspace(0, 4.2, 29)
aR.hist(dke[macc], bins=b2, weights=np.full(macc.sum(), 1 / macc.sum()), histtype="step",
        lw=1.8, color="0.45", label=f"all accepted (n={int(macc.sum()):,})")
aR.hist(dke[km], bins=b2, weights=np.full(km.sum(), 1 / max(km.sum(), 1)), histtype="stepfilled",
        color=C_MUD, alpha=0.45, ec=C_MUD, lw=1.8, label=f"survivors (n={int(km.sum())})")
aR.set_xlabel("muon kinetic energy at decay [MeV]")
aR.set_ylabel("fraction of events / bin"); aR.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/tail_mudif_retuned.png"); plt.close(fig)

allstats = json.load(open(f"{OUT}/tail_stats.json"))
allstats["mudif_after_retune"] = stats
json.dump(allstats, open(f"{OUT}/tail_stats.json", "w"), indent=1)
print("ALL DONE", flush=True)
