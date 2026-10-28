import glob, os, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
P = "/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
MC = ["truth_live_E","truth_event_type","pred_accepted","pred_positron_energy",
      "pred_dead_energy","pred_positron_time_consensus_ns"]
TC = ["topo_score","muon_score","pileup_score","muon_dif_score","pion_dif_score","deposited_energy"]
f32 = lambda x: np.asarray(x, dtype=np.float32)
def paired(t, m):
    tE, mE = f32(t.deposited_energy.values), f32(m.truth_live_E.values)
    ti, pt, pm = 0, [], []
    for mi in range(len(m)):
        while ti < len(t) and tE[ti] != mE[mi]: ti += 1
        if ti >= len(t): break
        pt.append(ti); pm.append(mi); ti += 1
    if len(pt) < 0.999*len(m): return None
    j = t.iloc[pt].reset_index(drop=True); mm = m.iloc[pm].reset_index(drop=True)
    for c in mm.columns: j["m_"+c] = mm[c].values
    return j
hits = []
for ev in sorted(glob.glob(f"{P}/purity_eval/nodrop_ep12_michel_inwin/*/mic_events.parquet")):
    piece = ev.split("/")[-2]
    tp = f"{P}/michel_eval_inwin_rs/{piece}/predictions.parquet"
    if not os.path.exists(tp): continue
    m = pd.read_parquet(ev, columns=MC); t = pd.read_parquet(tp, columns=TC)
    if len(m) != len(t): continue
    j = paired(t, m)
    if j is None: continue
    E = j.m_pred_positron_energy.values + j.m_pred_dead_energy.values
    et = j.m_truth_event_type.values.astype(np.int64)
    k = ((et&1)==0)&((et&0x1000)==0)&((et&0x200)==0)&(j.m_pred_accepted.values>=1) \
        & np.isfinite(E)&(E>=0)&(E<=90.0)&(j.topo_score.values>0.99)&(j.muon_score.values<0.02) \
        & (j.muon_dif_score.values<0.40)&(j.pion_dif_score.values<0.01)&(j.pileup_score.values<0.99)
    if k.sum():
        for i in np.where(k)[0]:
            hits.append(dict(piece=piece, row=int(i),
                             t=round(float(j.m_pred_positron_time_consensus_ns.values[i]),1),
                             E=round(float(E[i]),2)))
print("=== survivor locations ===")
for h in hits: print("  ", h)
pieces = sorted({h["piece"] for h in hits})
print("PIECES:", " ".join(pieces))
json.dump(dict(hits=hits, pieces=pieces), open(f"{P}/purity_eval/survivor_pieces.json","w"), indent=1)
print("\n=== pie eval input sizes ===")
tot = 0
for i, f in enumerate(sorted(glob.glob(f"{P}/norad_mixed2_rs/mix_*/data.parquet"))[:14]):
    n = pq.ParquetFile(f).metadata.num_rows; tot += n
    print(f"   {f.split('/')[-2]}: {n:,}   cumulative {tot:,}")
