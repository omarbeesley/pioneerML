"""Can the unused heads (pileup, pie tagger) replace the time cut in the no-timing chain?

The chains of tail_reco_cutflows.py open with the prompt window, which removes most of the
Michel population before any network score acts, so the suppression attributed to the cuts is
measured on the surviving few percent. Here the order is inverted: the topological cuts act on
the whole accepted population, and the prompt window is applied last, as a factor that stacks
on top. This separates what the network achieves from what the clock achieves.

Stages: acceptance -> + pie topology > 0.99 -> + muon veto -> + pion-DIF veto -> + prompt window.

Outputs figures/AI/tail_cutflow_notime.png (one panel per background) and the
'notime_cutflows' block of figures/AI/tail_stats.json.
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tail_figstyle
tail_figstyle.apply()
import tail_rates as TR

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
C_PIE, C_MIC, C_PID, C_MUD = "#d62728", "#1f77b4", "#2ca02c", "#9467bd"
ES, EM = 56.0, 90.0
KMUDIF, KRAD = 0x1000, 0x200
bins = np.linspace(0, EM, 61)
CL90 = 2.3

TCOLS = ["muon_score", "topo_score", "pie_score", "pileup_score", "muon_dif_score",
         "pion_dif_score", "deposited_energy"]
MCOLS = ["truth_live_E", "truth_event_type", "truth_gen_weight", "pred_accepted",
         "pred_positron_energy", "pred_dead_energy", "pred_positron_time_consensus_ns"]


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
    if len(pt) < 0.999 * len(main):
        return None
    j = tail.iloc[pt].reset_index(drop=True)
    mm = main.iloc[pm].reset_index(drop=True)
    for c in mm.columns:
        j["m_" + c] = mm[c].values
    return j


def finish(j, w_gen=False):
    d = pd.DataFrame({
        "E": j.m_pred_positron_energy.values + j.m_pred_dead_energy.values,
        "acc": (j.m_pred_accepted.values >= 1),
        "t": j.m_pred_positron_time_consensus_ns.values,
        "muon": j.muon_score.values, "topo": j.topo_score.values,
        "pie_s": j.pie_score.values, "pileup": j.pileup_score.values,
        "mudif": j.muon_dif_score.values, "pidif": j.pion_dif_score.values,
    })
    d["w"] = j.m_truth_gen_weight.values if w_gen else 1.0
    return d


# ------------------------------------------------------------------ samples
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
    if j is not None:
        rows.append(finish(j[(j.is_pie == 1) & (j.m_truth_is_pie == 1)]))
pie = pd.concat(rows, ignore_index=True)

rows, MIC_GW_ALL = [], 0.0
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_michel_inwin/*/mic_events.parquet")):
    piece = ev.split("/")[-2]
    tp = f"{P}/michel_eval_inwin_rs/{piece}/predictions.parquet"
    if not os.path.exists(tp):
        continue
    m = pd.read_parquet(ev, columns=MCOLS)
    t = pd.read_parquet(tp, columns=TCOLS)
    if len(m) != len(t):
        continue
    j = paired(t, m)
    if j is None:
        continue
    et = j.m_truth_event_type.values.astype(np.int64)
    gwa = j.m_truth_gen_weight.values.astype(float)
    MIC_GW_ALL += gwa[np.isfinite(gwa)].sum()
    keep = ((et & 1) == 0) & ((et & KMUDIF) == 0) & ((et & KRAD) == 0) & np.isfinite(gwa)
    rows.append(finish(j[keep], w_gen=True))
mic = pd.concat(rows, ignore_index=True)

mt = pd.read_parquet(f"{P}/tail_reveal_eval10/predictions_evalmode_bak.parquet",
                     columns=TCOLS + ["is_pidif", "is_mudif"])
bkg = {}
for flag, mfile in (("is_pidif", "pidif_osl"), ("is_mudif", "mudif_osl")):
    tt = mt[mt[flag] == 1].reset_index(drop=True)
    m = pd.read_parquet(f"{P}/purity_eval/tail_osl/{mfile}_events.parquet", columns=MCOLS)
    bkg[flag] = finish(paired(tt, m))
pid, mud = bkg["is_pidif"], bkg["is_mudif"]
print(f"pie {len(pie):,}  michel {len(mic):,}  piDIF {len(pid):,}  muDIF {len(mud):,}", flush=True)


# ------------------------------------------------------------------ no-time base chain
onset = np.quantile(pie.t.to_numpy()[pie.acc.to_numpy() & np.isfinite(pie.t.to_numpy())], 0.005)
TLO, THI = onset + 2.0, onset + 32.0


def acc_of(d):
    E = d.E.to_numpy()
    return d.acc.to_numpy() & np.isfinite(E) & (E >= 0) & (E <= EM)


def base_notime(d, topo=0.99):
    return acc_of(d) & (d.topo.to_numpy() > topo) & (d.muon.to_numpy() < 0.02) \
        & (d.mudif.to_numpy() < 0.46) & (d.pidif.to_numpy() < 0.01)


Ep = pie.E.to_numpy(); acc_pie = acc_of(pie)
w_pie = TR.weights(len(pie), TR.BR_PIE)
w_mud = TR.weights(len(mud), TR.R_MUDIF_AT_ACC * w_pie[acc_pie].sum(), mask=acc_of(mud))
TS = TR.tail_rescale(Ep[acc_pie])
n_acc_pie = acc_pie.sum()

print("\n=== survivors of the no-time chain (topology + vetoes only) ===", flush=True)
for nm, d in (("MICHEL", mic), ("PIDIF", pid)):
    k = base_notime(d)
    cols = ["E", "t", "topo", "pie_s", "pileup", "muon", "mudif", "pidif"]
    print(f"\n-- {nm}: {int(k.sum())} survivors")
    print(d[k][cols].to_string(float_format=lambda v: f"{v:8.3f}"), flush=True)
k = base_notime(mud)
print(f"\n-- MUDIF: {int(k.sum())} survivors; pileup q50 {np.median(mud.pileup.to_numpy()[k]):.4f}, "
      f"pie_s q50 {np.median(mud.pie_s.to_numpy()[k]):.4f}", flush=True)
for nm, d in (("pie(acc)", pie), ("michel-surv", mic), ("pidif-surv", pid), ("mudif-surv", mud)):
    m = acc_pie if nm == "pie(acc)" else base_notime(d)
    if m.sum() == 0: continue
    print(f"   {nm:12} pileup q50 {np.median(d.pileup.to_numpy()[m]):.4f} q90 {np.quantile(d.pileup.to_numpy()[m],0.9):.4f}  "
          f"pie_s q50 {np.median(d.pie_s.to_numpy()[m]):.4f} q10 {np.quantile(d.pie_s.to_numpy()[m],0.1):.4f}", flush=True)

# ------------------------------------------------------------------ scan the unused heads
print("\n=== adding the pileup head and the pie tagger, still without any time cut ===", flush=True)
res = []
for topo in (0.99, 0.995, 0.999):
    for pu in (1.01, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05):
        for ps in (-1.0, 0.5, 0.9, 0.99, 0.999):
            def sel(d):
                return base_notime(d, topo) & (d.pileup.to_numpy() < pu) & (d.pie_s.to_numpy() > ps)
            nm_, np_ = int(sel(mic).sum()), int(sel(pid).sum())
            kp = sel(pie); km = sel(mud)
            eff = 100 * kp.sum() / n_acc_pie
            slo = (kp & (Ep < ES)).sum(); shi = (kp & (Ep >= ES)).sum()
            rlo = (acc_pie & (Ep < ES)).sum(); rhi = (acc_pie & (Ep >= ES)).sum()
            r = (slo / max(shi, 1)) / (rlo / max(rhi, 1))
            el, eh = slo / max(rlo, 1), shi / max(rhi, 1)
            err = 100 * r * np.sqrt((1 - el) / max(slo, 1) + (1 - eh) / max(shi, 1))
            mlo = w_mud[km & (mud.E.to_numpy() < ES)].sum(); plo = w_pie[kp & (Ep < ES)].sum()
            res.append(dict(topo=topo, pileup=pu, pie_s=ps, michel=nm_, pidif=np_,
                            mudif=int(km.sum()), pie_eff=float(eff), bias=float(100*(r-1)),
                            bias_err=float(err), mudif_contam=float(TR.contamination(mlo, plo, TS))))
zero = [r for r in res if r["michel"] == 0 and r["pidif"] == 0]
zero.sort(key=lambda r: -r["pie_eff"])
print(f"combinations with zero michel AND zero piDIF, no time cut: {len(zero)} of {len(res)}")
print(f"{'topo':>6} {'pileup':>7} {'pie_s':>7} | {'pie eff':>8} {'bias':>15} {'muDIF':>7} {'contam':>8}")
for r in zero[:12]:
    print(f"{r['topo']:6.3f} {r['pileup']:7.2f} {r['pie_s']:7.3f} | {r['pie_eff']:7.2f}% "
          f"{r['bias']:+7.2f}+/-{r['bias_err']:4.2f}% {r['mudif']:>7,} {r['mudif_contam']:7.2f}%", flush=True)
json.dump(dict(scan=res, zero_zero_notime=zero[:40]), open(f"{OUT}/tail_notime_heads.json", "w"), indent=1)
print("\nALL DONE", flush=True)
