"""Topological suppression measured without the prompt time window.

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


def acc_of(d):
    E = d.E.to_numpy()
    return d.acc.to_numpy() & np.isfinite(E) & (E >= 0) & (E <= EM)


def base_notime(d):
    return acc_of(d) & (d.topo.to_numpy() > 0.99) & (d.muon.to_numpy() < 0.02) \
        & (d.mudif.to_numpy() < 0.46) & (d.pidif.to_numpy() < 0.01)


HEADS = [("topo", "pie topology", True), ("pie_s", "pie tagger", True),
         ("muon", "stopped-muon veto", False), ("mudif", "muon-DIF veto", False),
         ("pidif", "pion-DIF veto", False), ("pileup", "pileup veto", False)]
ap = acc_of(pie)
print("=== score distribution of ACCEPTED pie (the reference) ===", flush=True)
ref = {}
for k, lab, hi in HEADS:
    v = pie[k].to_numpy()[ap]
    ref[k] = v
    print(f"  {lab:20} q50 {np.median(v):7.4f}  q90 {np.quantile(v,0.90):7.4f}  "
          f"q99 {np.quantile(v,0.99):7.4f}  q99.9 {np.quantile(v,0.999):7.4f}  max {v.max():7.4f}", flush=True)

surv = base_notime(mic)
s = mic[surv].copy()
s["key"] = np.round(s.t.to_numpy(), 2)
print(f"\n=== the {int(surv.sum())} surviving rows ({s.key.nunique()} distinct decays): "
      f"percentile of each score within accepted pie ===", flush=True)
for key, g in s.groupby("key"):
    r = g.iloc[0]
    print(f"\n  decay at t={key:.1f} ns  (x{len(g)} copies)  reco E {r.E:.1f} MeV", flush=True)
    for k, lab, hi in HEADS:
        val = float(r[k])
        pct = 100.0 * (ref[k] <= val).mean()
        flag = ""
        if not hi and pct > 99.0: flag = "   <-- ELEVATED"
        if hi and pct < 1.0: flag = "   <-- LOW"
        print(f"     {lab:20} {val:7.4f}   at the {pct:6.2f} percentile of pie{flag}", flush=True)
print("\n=== would a tighter veto catch them? ===", flush=True)
for k, lab, hi in HEADS:
    if hi: continue
    vals = sorted(float(g.iloc[0][k]) for _, g in s.groupby("key"))
    print(f"  {lab:20} survivor values: " + ", ".join(f"{v:.3f}" for v in vals), flush=True)
print("ALL DONE", flush=True)
