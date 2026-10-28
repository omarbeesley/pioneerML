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



# ------------------------------------------------------------------ survivor forensics
def acc_of(d):
    E = d.E.to_numpy()
    return d.acc.to_numpy() & np.isfinite(E) & (E >= 0) & (E <= EM)


def base_notime(d):
    return acc_of(d) & (d.topo.to_numpy() > 0.99) & (d.muon.to_numpy() < 0.02) \
        & (d.mudif.to_numpy() < 0.46) & (d.pidif.to_numpy() < 0.01)


BITS = {0x1: "kPienu", 0x10: "kPidif", 0x40: "kPitar", 0x200: "kMurad", 0x1000: "kMudif"}
for nm, d in (("MICHEL", mic), ("PIDIF", pid)):
    k = base_notime(d)
    s = d[k].copy()
    s["key"] = list(zip(np.round(s.t_true.values, 3), np.round(s.E_true.values, 3)))
    print(f"\n================ {nm}: {int(k.sum())} surviving ROWS, "
          f"{s.key.nunique()} distinct physical events ================", flush=True)
    for key, g in s.groupby("key"):
        r = g.iloc[0]
        dt = r.t_reco = r.t - r.t_true if np.isfinite(r.t_true) else float("nan")
        bits = " ".join(v for b, v in BITS.items() if int(r.et) & b)
        bits += f"   [raw 0x{int(r.et):x}]"
        print(f"\n  event x{len(g)} copies | reco E {r.E:6.2f} MeV, t {r.t:8.2f} ns")
        print(f"     truth: positron E {r.E_true:6.2f} MeV, t {r.t_true:8.2f} ns, theta {np.degrees(r.truth_theta) if 'truth_theta' in s else float('nan'):6.1f} deg"
              if False else
              f"     truth: positron E {r.E_true:6.2f} MeV, t {r.t_true:8.2f} ns   (reco - truth t = {r.t - r.t_true:+8.2f} ns)")
        print(f"     event_type bits: {bits or 'none'} | has_muon {int(r.has_mu)} | "
              f"accidental {int(r.accid)} (t={r.t_acc:.1f}) | atar pileup {int(r.atar_pu)} | truth htp {int(r.truth_htp) if np.isfinite(r.truth_htp) else -1}")
        print(f"     theta {r.theta:6.1f} deg | positron hit IoU {r.iou:.3f} | scores: topo {r.topo:.3f} pie {r.pie_s:.3f} "
              f"pileup {r.pileup:.3f} muon {r.muon:.3f} muDIF {r.mudif:.3f}", flush=True)

# how much of the sample is duplicated at all?
print("\n=== event_type bit census (accepted rows) ===", flush=True)
for nm, d in (("michel", mic), ("piDIF", pid), ("muDIF", mud)):
    a = acc_of(d); et = d.et.to_numpy()[a]
    allbits = sorted({b for v in np.unique(et) for b in (0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80,0x100,0x200,0x400,0x800,0x1000,0x2000,0x4000) if v & b})
    print(f"  {nm:7}: " + "  ".join(f"0x{b:x}={100*((et & b) > 0).mean():5.1f}%" for b in allbits), flush=True)

for nm, d in (("michel", mic), ("piDIF", pid), ("muDIF", mud)):
    a = acc_of(d)
    key = list(zip(np.round(d.t_true.values[a], 3), np.round(d.E_true.values[a], 3)))
    uniq = len(set(key))
    print(f"\n[dup] {nm:7} accepted rows {a.sum():>10,}  distinct (t_true,E_true) {uniq:>10,}  "
          f"mean copies {a.sum()/max(uniq,1):5.2f}", flush=True)
print("\nALL DONE", flush=True)
