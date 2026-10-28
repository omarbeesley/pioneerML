"""What ARE the survivors of the timing-free chain, and should they be counted?

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
         "pred_positron_energy", "pred_dead_energy", "pred_positron_time_consensus_ns",
         "truth_positron_t", "truth_positron_energy", "truth_has_accidental",
         "truth_accidental_positron_t", "truth_has_atar_pileup",
         "truth_has_muon", "truth_theta", "truth_htp", "pred_htp", "pred_pos_iou",
         "pred_polar_angle", "event_id"]


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
        "t_true": j.m_truth_positron_t.values, "E_true": j.m_truth_positron_energy.values,
        "accid": j.m_truth_has_accidental.values,
        "t_acc": j.m_truth_accidental_positron_t.values, "atar_pu": j.m_truth_has_atar_pileup.values,
        "has_mu": j.m_truth_has_muon.values, "iou": j.m_pred_pos_iou.values,
        "truth_htp": j.m_truth_htp.values, "theta": np.degrees(j.m_truth_theta.values),
        "et": j.m_truth_event_type.values.astype(np.int64), "eid": j.m_event_id.values,
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




# ---------------------------------------------------------------- invisible-muon hypothesis
KMUTAR, KMUDAR = 0x4000, 0x2000


def acc_of(d):
    E = d.E.to_numpy()
    return d.acc.to_numpy() & np.isfinite(E) & (E >= 0) & (E <= EM)


def base_notime(d):
    return acc_of(d) & (d.topo.to_numpy() > 0.99) & (d.muon.to_numpy() < 0.02) \
        & (d.mudif.to_numpy() < 0.46) & (d.pidif.to_numpy() < 0.01)


a = acc_of(mic); et = mic.et.to_numpy(); hm = mic.has_mu.to_numpy()
surv = base_notime(mic)
no_mutar = (et & KMUTAR) == 0
no_muhit = hm == 0
print("=== michel: how rare is an invisible muon, and how well does it survive? ===", flush=True)
print(f"accepted michel rows            {a.sum():>12,}")
print(f"  muon did NOT decay in target  {int((a & no_mutar).sum()):>12,}  ({100*(a & no_mutar).sum()/a.sum():.4f}%)")
print(f"  no muon hit in the ATAR       {int((a & no_muhit).sum()):>12,}  ({100*(a & no_muhit).sum()/a.sum():.4f}%)")
print(f"  both                          {int((a & no_mutar & no_muhit).sum()):>12,}")
print(f"\ntopological survivors           {int(surv.sum()):>12,}  (all with no muon hit: {bool((surv & ~no_muhit).sum() == 0)})")
for lab, m in (("muon hit present", a & ~no_muhit), ("NO muon hit", a & no_muhit),
               ("muon decayed in target", a & ~no_mutar), ("muon decayed OUTSIDE target", a & no_mutar)):
    n = int(m.sum()); s = int((m & surv).sum())
    rate = s / n if n else 0
    print(f"  {lab:30} n {n:>12,}  survive {s:>4}  rate {rate:.3e}")
enh = ((a & no_muhit & surv).sum() / max((a & no_muhit).sum(), 1)) / \
      max((a & ~no_muhit & surv).sum() / max((a & ~no_muhit).sum(), 1), 1e-30)
print(f"\nan invisible-muon michel is {enh:,.0f}x more likely to survive the topological cuts")
print("\ntime distribution of the survivors (ns):", np.round(np.sort(mic.t.to_numpy()[surv]), 1), flush=True)
print("ALL DONE", flush=True)
