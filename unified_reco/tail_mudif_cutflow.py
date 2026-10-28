"""muDIF cutflow on the retuned reconstructed chain + aggressiveness sweep of the muDIF veto.

Figure 1 (tail_cutflow_mudif_reco.png): staged cutflow of muDIF against pie at the 20:1
acceptance normalization, on reconstructed energy/acceptance/time, stages
  (0) acceptance -> (1) prompt window -> (2) topology>0.99 -> (3) muon veto -> (4) piDIF veto.
Figure 2 (tail_mudif_retuned_scan.png): tail bias vs residual contamination of the FULL
retuned chain as the muon-DIF threshold tightens from 0.46, each point labeled by the
threshold, with the pie efficiency alongside.
Stats appended to tail_stats.json (mudif_cutflow_reco, mudif_scan_retuned).
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
C_PIE, C_MUD = "#d62728", "#9467bd"
ES, EM = 56.0, 75.0
import tail_rates as TR
R_PIE, RATIO = TR.BR_PIE, 20.0
bins = np.linspace(0, EM, 51)

TCOLS = ["muon_score", "pileup_score", "topo_score", "muon_dif_score", "pion_dif_score",
         "deposited_energy"]
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
                     columns=TCOLS + ["is_mudif"])
mt = mt[mt.is_mudif == 1].reset_index(drop=True)
mud = paired(mt, pd.read_parquet(f"{P}/purity_eval/tail_osl/mudif_osl_events.parquet", columns=MCOLS))
print(f"pie {len(pie):,}  mudif {len(mud):,}", flush=True)


def reco(j):
    E = j.m_pred_positron_energy.values + j.m_pred_dead_energy.values
    acc = (j.m_pred_accepted.values >= 1) & np.isfinite(E) & (E >= 0) & (E <= EM)
    return E, acc, j.m_pred_positron_time_consensus_ns.values


Ep, pacc, tp_ = reco(pie)
onset = np.quantile(tp_[pacc & np.isfinite(tp_)], 0.005)
TLO, THI = onset + 2.0, onset + 32.0
Em, macc, tm = reco(mud)


def stages(j, acc, t):
    tw = (t >= TLO) & (t <= THI)
    s0 = acc
    s1 = s0 & tw
    s2 = s1 & (j.topo_score.values > 0.99)
    s3 = s2 & (j.muon_score.values < 0.02) & (j.muon_dif_score.values < 0.46)
    s4 = s3 & (j.pion_dif_score.values < 0.01)
    return [("(0) acceptance", s0), ("(1) + prompt time window", s1),
            ("(2) + $\\pi\\to e$ topology $>0.99$", s2), ("(3) + muon veto", s3),
            ("(4) + pion-DIF veto", s4)]


sp = stages(pie, pacc, tp_)
sm = stages(mud, macc, tm)
wpie = R_PIE / pacc.sum()
wdif = (R_PIE / RATIO) / macc.sum()
TS = TR.tail_rescale(Ep[pacc])
print(f"[rates] tail rescale ts = {TS:.3f}", flush=True)

fig, axs2 = plt.subplots(2, 3, figsize=(13.0, 10.6), sharex=True, sharey=True)
axes = axs2.ravel()[:5]
axs2.ravel()[5].axis("off")
TF0 = None
stab = []
for i, (ax, (name, pk), (_, mk)) in enumerate(zip(axes, sp, sm)):
    ax.hist(Ep[pk], bins=bins, weights=np.full(pk.sum(), wpie), histtype="step", lw=1.6,
            color=C_PIE, label=r"$\pi\to e$" if i == 0 else None)
    ax.hist(Em[mk], bins=bins, weights=np.full(mk.sum(), wdif), histtype="step", lw=1.8,
            color=C_MUD, label=r"$\mu$DIF" if i == 0 else None)
    ax.axvline(ES, color="k", ls=":", lw=0.9); ax.set_yscale("log")
    ax.set_ylim(3e-10, 3e-1)
    ax.set_title(name, fontsize=11)
    mlo = wdif * (mk & (Em < ES)).sum(); plo = wpie * (pk & (Ep < ES)).sum()
    mhi = wdif * (mk & (Em >= ES)).sum(); phi = wpie * (pk & (Ep >= ES)).sum()
    clo = TR.contamination(mlo, plo, TS)
    chi = 100 * mhi / (mhi + phi) if (mhi + phi) > 0 else 0.0
    peff = 100 * pk.sum() / pacc.sum()
    tf = (Ep[pk] < ES).sum() / max((Ep[pk] >= ES).sum(), 1)
    if TF0 is None and i == 1:
        TF0 = tf
    bias = 100 * (tf / TF0 - 1) if TF0 else 0.0
    ax.text(0.96, 0.96, f"contam ($E\\geq$56): {chi:.2f}%\ncontam ($E<$56): {clo:.1f}%\n"
                        f"$\\pi\\to e$ eff: {peff:.1f}%\ntail bias: {bias:+.1f}%",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.92))
    if i == 0:
        ax.legend(fontsize=9, loc="upper left")
    stab.append(dict(stage=name, contam_lo=float(clo), contam_hi=float(chi),
                     pie_eff=float(peff), tail_bias=float(bias), mudif_events=int(mk.sum())))
for a in (axes[2], axes[3], axes[4]):
    a.set_xlabel("reconstructed energy [MeV]")
axes[2].tick_params(labelbottom=True)
axes[0].set_ylabel("branching-weighted rate [a.u.]")
axes[3].set_ylabel("branching-weighted rate [a.u.]")
fig.tight_layout(); fig.savefig(f"{OUT}/tail_cutflow_mudif_reco.png"); plt.close(fig)
for s in stab:
    print(f"  {s['stage'][:36]:36} contam_lo {s['contam_lo']:5.1f}%  evts {s['mudif_events']:>7,}  "
          f"pie eff {s['pie_eff']:5.1f}%  bias {s['tail_bias']:+.2f}%", flush=True)

# ---------------- aggressiveness sweep of the muDIF threshold on the retuned chain --------
tw_p = pacc & (tp_ >= TLO) & (tp_ <= THI)
tw_m = macc & (tm >= TLO) & (tm <= THI)
base_p = tw_p & (pie.topo_score.values > 0.99) & (pie.muon_score.values < 0.02) \
         & (pie.pion_dif_score.values < 0.01)
base_m = tw_m & (mud.topo_score.values > 0.99) & (mud.muon_score.values < 0.02) \
         & (mud.pion_dif_score.values < 0.01)
rlo_ref = (tw_p & (Ep < ES)).sum(); rhi_ref = (tw_p & (Ep >= ES)).sum()
tf_ref = rlo_ref / max(rhi_ref, 1)
thresholds = [0.46, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
scan = []
for thr in thresholds:
    kp = base_p & (pie.muon_dif_score.values < thr)
    km = base_m & (mud.muon_dif_score.values < thr)
    plo = wpie * (kp & (Ep < ES)).sum(); mlo = wdif * (km & (Em < ES)).sum()
    contam = TR.contamination(mlo, plo, TS)
    slo = (kp & (Ep < ES)).sum(); shi = (kp & (Ep >= ES)).sum()
    tf = slo / max(shi, 1); r = tf / tf_ref
    el, eh = slo / max(rlo_ref, 1), shi / max(rhi_ref, 1)      # nested-binomial, as in mudif_tradeoff.py
    err = 100 * r * np.sqrt((1 - el) / max(slo, 1) + (1 - eh) / max(shi, 1))
    scan.append(dict(mudif_cut=thr, mudif_survivors=int(km.sum()),
                     suppression=float(macc.sum() / max(km.sum(), 1)),
                     contam_pct=float(contam), pie_eff=float(100 * kp.sum() / pacc.sum()),
                     bias_pct=float(100 * (r - 1)), bias_unc_pct=float(err)))
print(f"\n{'cut':>6} {'surv':>6} {'supp':>7} {'contam':>8} {'pie eff':>8} {'bias':>15}")
for s in scan:
    print(f"{s['mudif_cut']:6.3f} {s['mudif_survivors']:6d} {s['suppression']:6.0f}x "
          f"{s['contam_pct']:7.1f}% {s['pie_eff']:7.2f}% {s['bias_pct']:+7.2f}+/-{s['bias_unc_pct']:4.2f}%", flush=True)

fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.add_patch(patches.Rectangle((0, -1), 1, 2, fc="#2ca02c", alpha=0.13, ec="none", zorder=0))
X = [s["contam_pct"] for s in scan]; Y = [s["bias_pct"] for s in scan]
YE = [s["bias_unc_pct"] for s in scan]
ax.errorbar(X, Y, yerr=YE, fmt="o-", color=C_MUD, lw=1.8, ms=5.5, capsize=3)
for s in scan:
    ax.annotate(f"{s['mudif_cut']:g}", (s["contam_pct"], s["bias_pct"]),
                textcoords="offset points", xytext=(8, 7), fontsize=9.5)
ax.axhline(0, color="0.6", ls="--", lw=1.0)
ax.set_xlabel(r"$\mu$DIF contamination in the low bin [%] (reconstructed energy)")
ax.set_ylabel(r"$\pi\to e$ tail bias [%] (full retuned chain)")
ax.set_xlim(left=0)
fig.tight_layout(); fig.savefig(f"{OUT}/tail_mudif_retuned_scan.png"); plt.close(fig)

allstats = json.load(open(f"{OUT}/tail_stats.json"))
allstats["mudif_cutflow_reco"] = stab
allstats["mudif_scan_retuned"] = scan
json.dump(allstats, open(f"{OUT}/tail_stats.json", "w"), indent=1)
print("ALL DONE", flush=True)
