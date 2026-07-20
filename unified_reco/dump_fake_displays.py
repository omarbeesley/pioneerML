"""Dump per-hit model info for the PROMPT accidental-tagged fake events (for event displays).

Scans main_excl_ar shard evals for prompt fakes (accepted, high-bin, kPienu-cut michel,
reco pulled to 0-120 ns, mis-timed), reruns the epoch-10 exclusivity model on those events
in the faithful batch-200 context, and saves per-event list-columns of the ATAR graph hits
(positions, time, E, view, slice, truth origin/pdg[pion,muon,MIP]) + per-hit trigger/MIP
probabilities.  Loops shards until ~NEED events are collected.
"""
import sys, os, numpy as np, pandas as pd, torch
sys.path.insert(0, '/pioneerML'); sys.path.insert(0, '/pioneerML/unified_reco')
from unified_reco.models_v2 import PURITYHybridModelV2
from unified_reco.dataset import PURITYDataset
from unified_reco.constants import NORM_POS_ATAR, NORM_T_ATAR
from unified_reco.benchmark import TASK_WEIGHTS
from torch_geometric.loader import DataLoader

P = "/pipeline"
CKPT = f"{P}/model_weights/main_hybridv2_excl_epoch10.pth"
OUT = "/pioneerML/unified_reco/accidental_tag_displays/_fakes_sample.parquet"
NEED = 16

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(8)
model = PURITYHybridModelV2(dropout=0.1).to(dev)
model.load_state_dict(torch.load(CKPT, map_location=dev)['model'])
model.train()

TRUTHC = ["truth_positron_t", "truth_accidental_positron_t", "truth_positron_energy", "event_type",
          "truth_pion_stop_x", "truth_pion_stop_y", "truth_pion_stop_z",
          "truth_positron_start_x", "truth_positron_start_y", "truth_positron_start_z",
          "truth_positron_stop_x", "truth_positron_stop_y", "truth_positron_stop_z",
          "truth_muon_start_x", "truth_muon_start_y", "truth_muon_start_z",
          "truth_muon_stop_x", "truth_muon_stop_y", "truth_muon_stop_z"]

rows = []
for shn in range(30):
    SH = f"{shn:03d}"
    evf = f"{P}/purity_eval/main_excl_ar/shard_{SH}/pimu_eval_events.parquet"
    if not os.path.exists(evf):
        continue
    d = pd.read_parquet(evf)
    recoE = (np.clip(d.pred_positron_energy, 0, None) + np.clip(d.pred_dead_energy, 0, None)).to_numpy()
    tt = d.truth_positron_t.to_numpy(); pt = d.pred_positron_time_ns.to_numpy()
    et = d.truth_event_type.to_numpy().astype(np.int64)
    fake = ((d.pred_accepted.to_numpy() >= 0.5) & (tt > -999) & (recoE >= 56) & ((et & 1) == 0)
            & (d.truth_positron_energy.to_numpy() <= 55) & (pt > 0) & (pt < 120) & (np.abs(pt - tt) >= 5))
    idx = np.where(fake)[0]
    if len(idx) == 0:
        continue
    print(f"shard {SH}: {len(idx)} prompt fakes", flush=True)
    ds = PURITYDataset(f"{P}/mixed_10M/pimu_bench_{SH}.parquet", max_hits=250)
    dl = DataLoader(ds, batch_size=200, shuffle=False)
    want = set(idx.tolist()); i0 = 0
    with torch.inference_mode():
        for batch in dl:
            B = batch.num_graphs
            if not any((i0 + k) in want for k in range(B)):
                i0 += B; continue
            batch = batch.to(dev)
            out = model(batch.x, batch.batch, task_weights=TASK_WEIGHTS,
                        triggering_pion_slice=getattr(batch, 'atar_triggering_pion_slice', None))
            x = batch.x.cpu().numpy(); is_atar = (x[:, 5] > 0.5) | (x[:, 6] > 0.5)
            b_atar = batch.batch.cpu().numpy()[is_atar]
            trig = out['atar_hit_trigger_prob'].cpu().numpy(); mip = out['atar_hit_mip_prob'].cpu().numpy()
            origin = batch.atar_true_event_id.cpu().numpy(); pdg3 = batch.atar_node_pdg_target.cpu().numpy()
            xa = x[is_atar]
            for k in range(B):
                gi = i0 + k
                if gi not in want:
                    continue
                m = b_atar == k; r = ds.df.iloc[gi]
                row = dict(event_idx=int(gi), shard=SH,
                    hx=(xa[m, 0]*NORM_POS_ATAR).tolist(), hy=(xa[m, 1]*NORM_POS_ATAR).tolist(),
                    hz=(xa[m, 2]*NORM_POS_ATAR).tolist(), hE=xa[m, 3].tolist(),
                    ht=(xa[m, 4]*NORM_T_ATAR).tolist(), hview=(xa[m, 6] > 0.5).astype(int).tolist(),
                    hslice=xa[m, 8].tolist(), hslice_t=xa[m, 9].tolist(), horigin=origin[m].tolist(),
                    hpion=pdg3[m, 0].tolist(), hmuon=pdg3[m, 1].tolist(), hpos=pdg3[m, 2].tolist(),
                    htrig=trig[m].tolist(), hmip=mip[m].tolist(),
                    plain_t=float(out['positron_time_per_graph'][k])*NORM_T_ATAR,
                    meanconf_t=float(out['positron_time_consensus_meanconf'][k])*NORM_T_ATAR,
                    n_tagged=int(out['n_tagged_pos_slices'][k]))
                for c in TRUTHC:
                    row[c] = float(r[c]) if c in r else np.nan
                row["recoE"] = float(recoE[gi])
                rows.append(row)
            i0 += B
    print(f"  collected {len(rows)} total", flush=True)
    if len(rows) >= NEED:
        break
pd.DataFrame(rows).to_parquet(OUT)
print(f"wrote {len(rows)} events -> {OUT}", flush=True)
