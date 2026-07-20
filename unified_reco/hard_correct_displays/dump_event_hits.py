"""Extract per-hit graph data + model predictions for chosen hard events, ready for display.

Reads the event list (hard_events_100.parquet: columns orig_idx, category, ...), reruns the
frozen main model (main_hybridv2_excl_epoch10) on each event INSIDE its faithful batch-50 eval
window (MC-dropout averaged), and writes one row per event with the ATAR hit arrays, truth PDG
labels, and the model's per-hit positron tag + reconstructed energy/time/direction.

Output feeds make_event_displays.py.  Must be run inside pytorch.sif (see README).
"""
import sys, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0,'/pioneerML'); sys.path.insert(0,'/pioneerML/unified_reco')
from unified_reco.models_v2 import PURITYHybridModelV2
from unified_reco.dataset import PURITYDataset
from unified_reco.constants import NORM_POS_ATAR, NORM_T_ATAR
from unified_reco.benchmark import TASK_WEIGHTS
from torch_geometric.data import Batch
from collections import defaultdict

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--events', default='/scratch/hard_events_100.parquet', help='event list (orig_idx, category)')
ap.add_argument('--out',    default='/scratch/hard_events_hits.parquet', help='per-hit output parquet')
ap.add_argument('--input',  default='/pipeline/origin_resliced/pimu_bench_000.parquet',
                help='the RESLICED benchmark shard the eval ran on (shard 000)')
ap.add_argument('--ckpt',   default='/pipeline/model_weights/main_hybridv2_excl_epoch10.pth')
ap.add_argument('--category', default=None, help='only extract this category')
ap.add_argument('--limit', type=int, default=None, help='only the first N events')
ap.add_argument('--mc', type=int, default=1, help='MC-dropout passes to average; 1 matches benchmark.py exactly, >1 gives steadier hit tags')
ap.add_argument('--batch-size', type=int, default=50, help='eval batch size (do NOT change: predictions are batch-context dependent)')
args=ap.parse_args()
BS=args.batch_size; MC=args.mc
torch.manual_seed(0)

ev=pd.read_parquet(args.events)
if args.category: ev=ev[ev.category==args.category]
if args.limit: ev=ev.head(args.limit)
CAT={int(r.orig_idx):r.category for _,r in ev.iterrows()}
print(f"extracting {len(CAT)} events (mc={MC}, batch={BS})",flush=True)

ds=PURITYDataset(args.input, max_hits=250)   # same drop-filter + order as the eval
dev=torch.device('cpu'); torch.set_num_threads(8)
model=PURITYHybridModelV2(dropout=0.1).to(dev)
model.load_state_dict(torch.load(args.ckpt, map_location=dev)['model'])
model.train()   # MC-dropout eval, exactly as benchmark.py does

byb=defaultdict(list)
for gi in CAT: byb[gi//BS].append(gi)

rows=[]; done=0
for b,gis in sorted(byb.items()):
    lo=b*BS; hi=min((b+1)*BS,len(ds))
    batch=Batch.from_data_list([ds[j] for j in range(lo,hi)]).to(dev)
    x=batch.x.cpu().numpy(); bb=batch.batch.cpu().numpy()
    is_atar=(x[:,5]>0.5)|(x[:,6]>0.5)
    origin=batch.atar_true_event_id.cpu().numpy(); pdg3=batch.atar_node_pdg_target.cpu().numpy()
    is_trig=batch.is_trigger_target.cpu().bool().numpy()[is_atar]
    ba=bb[is_atar]; xa=x[is_atar]; na=int(is_atar.sum()); ng=batch.num_graphs
    aht=np.zeros(na); ahm=np.zeros(na); ahp=np.zeros(na)
    aacc=np.zeros(ng); ade=np.zeros(ng); ape=np.zeros(ng); apt=np.zeros(ng); adir=np.zeros((ng,3))
    with torch.inference_mode():
        for _ in range(MC):
            out=model(batch.x,batch.batch,task_weights=TASK_WEIGHTS,
                      triggering_pion_slice=getattr(batch,'atar_triggering_pion_slice',None))
            aht+=out['atar_hit_trigger_prob'].cpu().numpy().reshape(-1)
            ahm+=out['atar_hit_mip_prob'].cpu().numpy().reshape(-1)
            hp=out.get('atar_hit_pion_prob')
            if hp is not None: ahp+=hp.cpu().numpy().reshape(-1)
            es=out['event_summary']
            aacc+=es['accepted'].float().cpu().numpy().reshape(-1)
            ade+=es['dead_energy'].float().cpu().numpy().reshape(-1)
            ape+=es['positron_energy'].float().cpu().numpy().reshape(-1)
            apt+=out['positron_time_per_graph'].cpu().numpy().reshape(-1)*NORM_T_ATAR
            pdv=es.get('positron_dir')
            if pdv is not None: adir+=pdv.float().cpu().numpy().reshape(ng,3)
    htrig=aht/MC; hmip=ahm/MC; hpip=ahp/MC
    accv=aacc/MC; dev_e=ade/MC; pev=ape/MC; ptv=apt/MC; dirv=adir/MC
    for gi in gis:
        k=gi-lo; m=ba==k; r=ds.df.iloc[gi]
        pp=((htrig[m]>0.5)&(hmip[m]>0.5)); tp=((pdg3[m,2]>0.5)&is_trig[m])
        u=(pp|tp).sum(); iou=float((pp&tp).sum()/u) if u>0 else float('nan')
        rows.append(dict(orig_idx=int(gi), category=CAT[gi],
            hx=(xa[m,0]*NORM_POS_ATAR).tolist(), hy=(xa[m,1]*NORM_POS_ATAR).tolist(),
            hz=(xa[m,2]*NORM_POS_ATAR).tolist(), hE=xa[m,3].tolist(),
            ht=(xa[m,4]*NORM_T_ATAR).tolist(), hview=(xa[m,6]>0.5).astype(int).tolist(),
            hslice=xa[m,8].astype(int).tolist(), horigin=origin[m].tolist(),
            hpion=pdg3[m,0].astype(int).tolist(), hmuon=pdg3[m,1].astype(int).tolist(),
            hmip_truth=pdg3[m,2].astype(int).tolist(),
            htrig=htrig[m].tolist(), hmip=hmip[m].tolist(), hpion_prob=hpip[m].tolist(),
            lyso_z=list(np.asarray(r['lyso_z'],float)), lyso_E=list(np.asarray(r['lyso_E'],float)),
            lyso_t=list(np.asarray(r['lyso_t'],float)), lyso_pdg=list(np.asarray(r['lyso_pdg'],int)),
            htrig_pos=pp.astype(int).tolist(), htrue_pos=tp.astype(int).tolist(),
            pred_dir_x=float(dirv[k,0]), pred_dir_y=float(dirv[k,1]), pred_dir_z=float(dirv[k,2]),
            pred_accepted=float(accv[k]), recoE=float(max(pev[k],0)+max(dev_e[k],0)), pred_time=float(ptv[k]), iou=iou,
            truth_positron_energy=float(r['truth_positron_energy']), truth_positron_t=float(r['truth_positron_t']),
            truth_has_muon=int(r['truth_has_muon']), truth_has_atar_pileup=int(r['truth_has_atar_pileup']),
            truth_pion_stop_x=float(r['truth_pion_stop_x']), truth_pion_stop_y=float(r['truth_pion_stop_y']),
            truth_pion_stop_z=float(r['truth_pion_stop_z'])))
        done+=1
    print(f"  {done}/{len(CAT)} done (batch {b})",flush=True)
pd.DataFrame(rows).to_parquet(args.out)
print("wrote",args.out,len(rows),"events",flush=True)
