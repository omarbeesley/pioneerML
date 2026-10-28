"""High-statistics realistic-reco muDIF trade-off: the 10M norad pie pool.

pie side: per-shard join of the eval-mode tail predictions (norad_eval2_rs/ev_G) with the
nodrop-ep12 main-model summaries (purity_eval/nodrop_ep12_pie/shard_G) -- same input file,
same dataset filter, so rows correspond one to one (verified per shard by float32 equality
of tail.deposited_energy vs main.truth_live_E).
muDIF side: the corrected-flags osl evaluation joined as in tail_reco_join.py.

Outputs: overwrites figures/AI/tail_reco_bias.png with the high-statistics curve and writes
the 'reco_bigpie' block into figures/AI/tail_stats.json.
"""
import glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
import tail_figstyle
tail_figstyle.apply()
C_MUD = "#9467bd"
ES, EM = 56.0, 75.0
R_PIE, RATIO = 1e-4, 20.0


def f32(x):
    return np.asarray(x, dtype=np.float32)


# ---------------- pie: shard-wise join of tail scores and main reco ----------------------
rows = []
nsh = 0
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet")):
    g = ev.split("shard_")[-1].split("/")[0]
    tp = f"{P}/norad_eval2_rs/ev_{g}/predictions.parquet"
    if not os.path.exists(tp):
        continue
    m = pd.read_parquet(ev, columns=["truth_live_E", "truth_dead_E", "truth_is_pie", "pred_accepted",
                                     "pred_positron_energy", "pred_dead_energy",
                                     "pred_positron_time_consensus_ns"])
    t = pd.read_parquet(tp, columns=["muon_dif_score", "deposited_energy", "is_pie"])
    if len(m) != len(t):
        print(f"shard {g}: row mismatch main={len(m)} tail={len(t)}, SKIP", flush=True)
        continue
    eq = (f32(t.deposited_energy.values) == f32(m.truth_live_E.values)).mean()
    if eq < 0.999:
        print(f"shard {g}: alignment check failed (eq={eq:.4f}), SKIP", flush=True)
        continue
    d = pd.DataFrame({
        "mudif_score": t.muon_dif_score.values,
        "is_pie": (t.is_pie.values == 1) & (m.truth_is_pie.values == 1),
        "acc": m.pred_accepted.values >= 1,
        "Ereco": m.pred_positron_energy.values + m.pred_dead_energy.values,
        "Edep": m.truth_live_E.values + m.truth_dead_E.values,
    })
    rows.append(d[d.is_pie])
    nsh += 1
pie = pd.concat(rows, ignore_index=True)
print(f"big pie: {nsh} shards, {len(pie):,} pie rows", flush=True)

Ep = pie.Ereco.to_numpy()
pa = pie.acc.to_numpy() & np.isfinite(Ep) & (Ep >= 0) & (Ep <= EM)
psc = pie.mudif_score.to_numpy()
n_tail = int((pa & (Ep < ES)).sum())
tf = n_tail / pa.sum()
print(f"accepted {pa.sum():,}  reco tail fraction {100*tf:.3f}%  tail events {n_tail:,}", flush=True)

# ---------------- muDIF: corrected osl eval, joined as in tail_reco_join -----------------
mt = pd.read_parquet(f"{P}/tail_reveal_eval10/predictions_evalmode_bak.parquet",
                     columns=["muon_dif_score", "deposited_energy", "is_mudif"])
mt = mt[mt.is_mudif == 1].reset_index(drop=True)
mm = pd.read_parquet(f"{P}/purity_eval/tail_osl/mudif_osl_events.parquet",
                     columns=["truth_live_E", "pred_accepted", "pred_positron_energy",
                              "pred_dead_energy"])
tE, mE = f32(mt.deposited_energy.values), f32(mm.truth_live_E.values)
ti, pt, pm_ = 0, [], []
for mi in range(len(mm)):
    while ti < len(mt) and tE[ti] != mE[mi]:
        ti += 1
    if ti >= len(mt):
        break
    pt.append(ti); pm_.append(mi); ti += 1
print(f"mudif: paired {len(pt):,}/{len(mm):,}", flush=True)
assert len(pt) >= 0.999 * len(mm)
msc = mt.muon_dif_score.values[pt]
Em = (mm.pred_positron_energy.values + mm.pred_dead_energy.values)[pm_]
macc = (mm.pred_accepted.values >= 1)[pm_] & np.isfinite(Em) & (Em >= 0) & (Em <= EM)

# ---------------- trade-off on reconstructed energy, high statistics ---------------------
rlo = (pa & (Ep < ES)).sum(); rhi = (pa & (Ep >= ES)).sum()
wpie = R_PIE / pa.sum(); wdif = (R_PIE / RATIO) / macc.sum()
effs = [99, 95, 90, 80, 70, 50, 30]
X, Y, YE, tab = [], [], [], []
for e in effs:
    c = np.quantile(psc[pa], e / 100.0)
    pk = pa & (psc < c); mk = macc & (msc < c)
    plo = wpie * (pk & (Ep < ES)).sum(); mlo = wdif * (mk & (Em < ES)).sum()
    contam = 100 * mlo / (mlo + plo) if (mlo + plo) > 0 else 0.0
    slo = (pk & (Ep < ES)).sum(); shi = (pk & (Ep >= ES)).sum()
    el, eh = slo / max(rlo, 1), shi / max(rhi, 1); r = el / max(eh, 1e-12)
    err = 100 * r * np.sqrt((1 - el) / max(slo, 1) + (1 - eh) / max(shi, 1))
    X.append(contam); Y.append(100 * (r - 1)); YE.append(err)
    tab.append(dict(pie_eff=e, contam_pct=float(contam), bias_pct=float(100 * (r - 1)),
                    bias_unc_pct=float(err), suppression=float(macc.sum() / max(mk.sum(), 1))))
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.add_patch(patches.Rectangle((0, -1), 1, 2, fc="#2ca02c", alpha=0.13, ec="none", zorder=0))
ax.errorbar(X, Y, yerr=YE, fmt="o-", color=C_MUD, lw=1.8, ms=5.5, capsize=3)
for x, y, e in zip(X, Y, effs):
    ax.annotate(f"{e}%", (x, y), textcoords="offset points", xytext=(8, 7), fontsize=9.5)
ax.axhline(0, color="0.6", ls="--", lw=1.0)
ax.set_xlabel(r"$\mu$DIF contamination in the low bin [%] (reconstructed energy)")
ax.set_ylabel(r"$\pi\to e$ tail bias [%] (reconstructed energy)")
ax.set_xlim(left=0)
fig.tight_layout(); fig.savefig(f"{OUT}/tail_reco_bias.png"); plt.close(fig)

# decompose the reco tail: genuine low deposit vs reco-migrated (true deposit >= 56)
Edep = pie.Edep.to_numpy()
tailm = pa & (Ep < ES)
true_tail = tailm & (Edep < ES); migr = tailm & (Edep >= ES)
decomp = dict(n_tail=int(tailm.sum()), n_true_tail=int(true_tail.sum()), n_migrated=int(migr.sum()))
for e in (90, 70, 50):
    c = np.quantile(psc[pa], e / 100.0)
    decomp[f"veto_keep_eff_at_{e}"] = dict(
        peak=float((psc[pa & (Ep >= ES)] < c).mean()),
        true_tail=float((psc[true_tail] < c).mean()),
        migrated=float((psc[migr] < c).mean()))
print("TAIL DECOMPOSITION:", json.dumps(decomp, indent=1), flush=True)

allstats = json.load(open(f"{OUT}/tail_stats.json"))
allstats["reco_bigpie"] = dict(shards=nsh, pie_rows=int(len(pie)), accepted=int(pa.sum()), tail_decomposition=decomp,
                               tail_events=n_tail, tail_fraction_pct=float(100 * tf),
                               tradeoff=tab)
json.dump(allstats, open(f"{OUT}/tail_stats.json", "w"), indent=1)
print(json.dumps(tab, indent=1))
print("ALL DONE", flush=True)
