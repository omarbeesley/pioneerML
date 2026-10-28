"""Scan every reconstructed variable in the eval parquets for one that separates the two
remaining muDAR survivors from the pi -> e population.

The reference population is pi -> e passing the SAME timing-free chain the survivors pass
(acceptance, topology > 0.99, muon < 0.02, muDIF < 0.40, piDIF < 0.01, pileup < 0.99), because
a variable is only useful if it kills the survivors while keeping that population.

Truth columns are excluded, and so are pred_pos_iou/precision/recall: benchmark.py builds those
by comparing the predicted positron mask against the truth mask, so they are diagnostics, not
analysis variables.

For each variable the survivor's percentile within the pi -> e reference is reported, together
with the pi -> e efficiency of the cut that would just remove it. Variables are ranked by how
little signal that cut costs.
"""
import glob, json, os
import numpy as np, pandas as pd

P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT = "/home/obeesley/thesis_ref/figures/AI"
EM = 90.0
KMUDIF, KRAD = 0x1000, 0x200

# reconstructed only.  pred_pos_* are truth-referenced diagnostics -> excluded from the ranking
# but still printed, because they say something about what the model saw.
DIAG = ["pred_pos_precision", "pred_pos_recall", "pred_pos_iou"]
MCOLS = ["event_id", "truth_live_E", "truth_event_type", "truth_is_pie",
         "pred_accepted", "pred_pion_stop_x", "pred_pion_stop_y", "pred_pion_stop_z",
         "pred_positron_dir_x", "pred_positron_dir_y", "pred_positron_dir_z",
         "pred_polar_angle", "pred_positron_energy", "pred_htp", "pred_dead_energy",
         "pred_positron_time_ns", "pred_positron_time_consensus_ns",
         "pred_positron_time_consensus_meanconf_ns", "pred_positron_time_argmax_role_ns",
         "pred_positron_time_spread_ns", "pred_positron_log_kappa"] + DIAG
TCOLS = ["pie_score", "topo_score", "muon_score", "pileup_score", "pie_logit", "topo_logit",
         "muon_logit", "pileup_logit", "muon_dif_score", "muon_dif_logit", "pion_dif_score",
         "pion_dif_logit", "keep_logit", "deposited_energy"]
SCOLS = ["event_id", "role_pred", "role_prob_none", "role_prob_muon", "role_prob_positron",
         "is_anchor", "pred_start_x", "pred_stop_x", "pred_start_y", "pred_stop_y",
         "pred_start_z", "pred_stop_z", "start_sigma_x", "stop_sigma_x", "start_sigma_y",
         "stop_sigma_y", "start_sigma_z", "stop_sigma_z"]


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


def slice_feats(sl, evids):
    """Per-event summaries of the slice table: how many slices, how muon-like the most
    muon-like one is, and where the positron slice starts relative to the pion stop."""
    g = sl.groupby("event_id")
    out = {}
    for eid, d in g:
        if eid not in evids:
            continue
        rp = d.role_pred.to_numpy()
        pm_ = d.role_prob_muon.to_numpy(); pp = d.role_prob_positron.to_numpy()
        rec = dict(n_slices=float(len(d)), n_role_muon=float((rp == 1).sum()),
                   n_role_pos=float((rp == 2).sum()), max_prob_muon=float(np.nanmax(pm_)),
                   sum_prob_muon=float(np.nansum(pm_)), max_prob_pos=float(np.nanmax(pp)),
                   n_anchor=float(d.is_anchor.to_numpy().sum()))
        k = int(np.nanargmax(pp))
        rec["pos_start_x"] = float(d.pred_start_x.values[k])
        rec["pos_start_y"] = float(d.pred_start_y.values[k])
        rec["pos_start_z"] = float(d.pred_start_z.values[k])
        rec["pos_sig_start"] = float(np.sqrt(d.start_sigma_x.values[k] ** 2
                                             + d.start_sigma_y.values[k] ** 2
                                             + d.start_sigma_z.values[k] ** 2))
        rec["pos_track_len"] = float(np.sqrt((d.pred_stop_x.values[k] - d.pred_start_x.values[k]) ** 2
                                             + (d.pred_stop_y.values[k] - d.pred_start_y.values[k]) ** 2
                                             + (d.pred_stop_z.values[k] - d.pred_start_z.values[k]) ** 2))
        out[eid] = rec
    return out


def derived(d, sf):
    """Reconstructed combinations that are not columns in their own right."""
    e = {}
    tc = d.m_pred_positron_time_consensus_ns.to_numpy()
    e["dt_consensus_argmax"] = np.abs(tc - d.m_pred_positron_time_argmax_role_ns.to_numpy())
    e["dt_consensus_meanconf"] = np.abs(tc - d.m_pred_positron_time_consensus_meanconf_ns.to_numpy())
    e["dt_consensus_plain"] = np.abs(tc - d.m_pred_positron_time_ns.to_numpy())
    dx = d.m_pred_positron_dir_x.to_numpy(); dy = d.m_pred_positron_dir_y.to_numpy()
    dz = d.m_pred_positron_dir_z.to_numpy()
    e["dir_norm"] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    e["dir_transverse"] = np.sqrt(dx ** 2 + dy ** 2)
    e["E_over_dep"] = (d.m_pred_positron_energy.to_numpy() + d.m_pred_dead_energy.to_numpy()) \
        / np.maximum(d.deposited_energy.to_numpy(), 1e-6)
    e["dead_frac"] = d.m_pred_dead_energy.to_numpy() / np.maximum(
        d.m_pred_positron_energy.to_numpy() + d.m_pred_dead_energy.to_numpy(), 1e-6)
    e["stop_r"] = np.sqrt(d.m_pred_pion_stop_x.to_numpy() ** 2 + d.m_pred_pion_stop_y.to_numpy() ** 2)
    # slice-derived, aligned by event_id
    eid = d.m_event_id.to_numpy()
    keys = ["n_slices", "n_role_muon", "n_role_pos", "max_prob_muon", "sum_prob_muon",
            "max_prob_pos", "n_anchor", "pos_sig_start", "pos_track_len"]
    for k in keys:
        e[k] = np.array([sf.get(i, {}).get(k, np.nan) for i in eid])
    px = np.array([sf.get(i, {}).get("pos_start_x", np.nan) for i in eid])
    py = np.array([sf.get(i, {}).get("pos_start_y", np.nan) for i in eid])
    pz = np.array([sf.get(i, {}).get("pos_start_z", np.nan) for i in eid])
    e["vertex_gap"] = np.sqrt((px - d.m_pred_pion_stop_x.to_numpy()) ** 2
                              + (py - d.m_pred_pion_stop_y.to_numpy()) ** 2
                              + (pz - d.m_pred_pion_stop_z.to_numpy()) ** 2)
    e["vertex_gap_z"] = np.abs(pz - d.m_pred_pion_stop_z.to_numpy())
    e["vertex_pull"] = e["vertex_gap"] / np.maximum(e["pos_sig_start"], 1e-3)
    return e


def chain(d, with_time=False, TLO=None, THI=None):
    E = d.m_pred_positron_energy.to_numpy() + d.m_pred_dead_energy.to_numpy()
    m = (d.m_pred_accepted.to_numpy() >= 1) & np.isfinite(E) & (E >= 0) & (E <= EM) \
        & (d.topo_score.to_numpy() > 0.99) & (d.muon_score.to_numpy() < 0.02) \
        & (d.muon_dif_score.to_numpy() < 0.40) & (d.pion_dif_score.to_numpy() < 0.01) \
        & (d.pileup_score.to_numpy() < 0.99)
    if with_time:
        t = d.m_pred_positron_time_consensus_ns.to_numpy()
        m &= (t >= TLO) & (t <= THI)
    return m


# ------------------------------------------------------------------ pie reference
pie_parts, pie_der = [], []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_pie/shard_*/pie_eval_events.parquet")):
    g = ev.split("shard_")[-1].split("/")[0]
    tp = f"{P}/norad_eval2_rs/ev_{g}/predictions.parquet"
    sp = ev.replace("_events.parquet", "_slices.parquet")
    if not (os.path.exists(tp) and os.path.exists(sp)):
        continue
    m = pd.read_parquet(ev, columns=MCOLS)
    t = pd.read_parquet(tp, columns=TCOLS + ["is_pie"])
    if len(m) != len(t):
        continue
    j = paired(t, m)
    if j is None:
        continue
    j = j[(j.is_pie == 1) & (j.m_truth_is_pie == 1)].reset_index(drop=True)
    k = chain(j)
    if k.sum() == 0:
        continue
    j = j[k].reset_index(drop=True)
    sf = slice_feats(pd.read_parquet(sp, columns=SCOLS), set(j.m_event_id.to_numpy().tolist()))
    pie_der.append(pd.DataFrame(derived(j, sf)))
    pie_parts.append(j)
    if sum(len(x) for x in pie_parts) > 400000:
        break
pie = pd.concat(pie_parts, ignore_index=True)
pieD = pd.concat(pie_der, ignore_index=True)
print(f"pie reference passing the timing-free chain: {len(pie):,}", flush=True)

# ------------------------------------------------------------------ michel survivors
srows, sder = [], []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_michel_inwin/*/mic_events.parquet")):
    piece = ev.split("/")[-2]
    tp = f"{P}/michel_eval_inwin_rs/{piece}/predictions.parquet"
    sp = ev.replace("_events.parquet", "_slices.parquet")
    if not (os.path.exists(tp) and os.path.exists(sp)):
        continue
    m = pd.read_parquet(ev, columns=MCOLS)
    t = pd.read_parquet(tp, columns=TCOLS)
    if len(m) != len(t):
        continue
    j = paired(t, m)
    if j is None:
        continue
    et = j.m_truth_event_type.to_numpy().astype(np.int64)
    ok = ((et & 1) == 0) & ((et & KMUDIF) == 0) & ((et & KRAD) == 0)
    k = chain(j) & ok
    if k.sum() == 0:
        continue
    jj = j[k].reset_index(drop=True)
    sf = slice_feats(pd.read_parquet(sp, columns=SCOLS), set(jj.m_event_id.to_numpy().tolist()))
    sder.append(pd.DataFrame(derived(jj, sf)))
    jj["piece"] = piece
    srows.append(jj)
surv = pd.concat(srows, ignore_index=True)
survD = pd.concat(sder, ignore_index=True)
print(f"michel survivors of the timing-free chain: {len(surv)} rows", flush=True)

# collapse duplicated triggers: one entry per distinct reconstructed time
surv["_k"] = surv.m_pred_positron_time_consensus_ns.round(1)
uniq = surv.drop_duplicates("_k").index.to_numpy()
print(f"  distinct events: {len(uniq)}  at t = "
      + ", ".join(f"{v:.1f}" for v in surv._k.values[uniq]) + " ns", flush=True)

# ------------------------------------------------------------------ the scan
ALL_P = {c: pie[c].to_numpy(float) for c in pie.columns
         if c not in ("piece", "_k") and pie[c].dtype.kind in "fiu"}
ALL_P.update({c: pieD[c].to_numpy(float) for c in pieD.columns})
ALL_S = {c: surv[c].to_numpy(float) for c in ALL_P if c in surv.columns}
ALL_S.update({c: survD[c].to_numpy(float) for c in survD.columns if c in ALL_P})

skip = {"m_truth_live_E", "m_truth_event_type", "m_truth_is_pie", "m_event_id",
        "deposited_energy", "is_pie"}
res = []
for c in sorted(ALL_P):
    if c in skip or c not in ALL_S:
        continue
    p = ALL_P[c]; p = p[np.isfinite(p)]
    if len(p) < 1000 or np.nanstd(p) == 0:
        continue
    for u in uniq:
        v = ALL_S[c][u]
        if not np.isfinite(v):
            continue
        below = float((p < v).mean())
        # the cut that just removes this survivor, whichever side it is extreme on
        keep_hi = float((p < v).mean())        # cut "variable < v" -> keeps this fraction
        keep_lo = float((p > v).mean())        # cut "variable > v"
        if keep_hi >= keep_lo:
            side, eff = "<", keep_hi
        else:
            side, eff = ">", keep_lo
        res.append(dict(name=c, tns=float(surv._k.values[u]), value=v, pctile=100 * below,
                        side=side, pie_eff=100 * eff))
r = pd.DataFrame(res)

print("\n=== every reconstructed variable, per survivor, ranked by pi->e kept ===", flush=True)
print(f"{'variable':<34}{'t [ns]':>8}{'value':>12}{'pctile':>9}{'cut':>5}{'pie kept':>10}", flush=True)
for _, row in r.sort_values("pie_eff", ascending=False).iterrows():
    nm = row["name"]
    flag = "   (truth-referenced, not usable)" if nm in DIAG else ""
    print(f"{nm:<34}{row['tns']:8.1f}{row['value']:12.4f}{row['pctile']:8.3f}%"
          f"{row['side']:>5}{row['pie_eff']:9.3f}%{flag}", flush=True)

print("\n=== variables where BOTH survivors are cut by one threshold ===", flush=True)
best = []
for c in r["name"].unique():
    d = r[r["name"] == c]
    if len(d) < len(uniq) or c in DIAG:
        continue
    if (d["side"] == "<").all():
        thr = d["value"].min(); eff = 100 * float((ALL_P[c][np.isfinite(ALL_P[c])] < thr).mean())
        best.append((c, "<", thr, eff))
    elif (d["side"] == ">").all():
        thr = d["value"].max(); eff = 100 * float((ALL_P[c][np.isfinite(ALL_P[c])] > thr).mean())
        best.append((c, ">", thr, eff))
best.sort(key=lambda x: -x[3])
if best:
    print(f"{'variable':<34}{'cut':>5}{'threshold':>12}{'pie kept':>11}", flush=True)
    for c, s, thr, eff in best:
        print(f"{c:<34}{s:>5}{thr:12.4f}{eff:10.3f}%", flush=True)
else:
    print("  none: no single variable puts both survivors on the same side.", flush=True)

json.dump(dict(n_pie=int(len(pie)), n_surv_rows=int(len(surv)), n_distinct=int(len(uniq)),
               scan=r.to_dict("records"), both=[dict(var=c, side=s, thr=float(t), pie_eff=float(e))
                                                for c, s, t, e in best]),
          open(f"{OUT}/tail_survivor_featscan.json", "w"), indent=1)
print("\nALL DONE", flush=True)
