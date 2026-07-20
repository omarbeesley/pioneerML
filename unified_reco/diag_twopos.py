"""Diagnostic: for the fake-pie events, verify TWO positron slices are tagged, and test
the per-slice-MEAN-confidence time readout (argmax of average trig*mip) vs plain-mean & sum-consensus."""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, '/pioneerML')
from unified_reco.models_v2 import PURITYHybridModelV2
from unified_reco.dataset import PURITYDataset
from unified_reco.constants import NORM_T_ATAR
from unified_reco.benchmark import TASK_WEIGHTS
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

CKPT="/pipeline/model_weights/main_hybridv2_timefix_full_best.pth"
DATA="/pipeline/mixed_standard_acc/pimu_benchmark_5_11/data.parquet"
N=12000
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(16)
model=PURITYHybridModelV2(dropout=0.1).to(dev)
model.load_state_dict(torch.load(CKPT, map_location=dev)['model'])
model.train()

ds=PURITYDataset(DATA, max_hits=250); N=min(N,len(ds))
df=ds.df.iloc[:N].reset_index(drop=True)
dl=DataLoader(Subset(ds,list(range(N))), batch_size=200, shuffle=False, num_workers=4)

rows={k:np.full(N,np.nan,np.float32) for k in
      ['plain_t','sum_t','meanconf_t','n_slices','recoE','deadE','accepted']}
i0=0
with torch.inference_mode():
    for batch in dl:
        batch=batch.to(dev); B=batch.num_graphs; sl=slice(i0,i0+B)
        anchor=getattr(batch,'atar_triggering_pion_slice',None)
        out=model(batch.x,batch.batch,task_weights=TASK_WEIGHTS,triggering_pion_slice=anchor)
        es=out.get('event_summary',{})
        def top(k):
            t=out.get(k); return t.float().cpu().numpy()[:B] if isinstance(t,torch.Tensor) else None
        def sm(k):
            t=es.get(k); return t.float().cpu().numpy()[:B] if isinstance(t,torch.Tensor) else None
        for key,val in [('plain_t',top('positron_time_per_graph')),
                        ('sum_t',top('positron_time_consensus')),
                        ('meanconf_t',top('positron_time_consensus_meanconf')),
                        ('n_slices',top('n_tagged_pos_slices')),
                        ('recoE',sm('positron_energy')),('deadE',sm('dead_energy')),
                        ('accepted',sm('accepted'))]:
            if val is not None: rows[key][sl]=val
        i0+=B
o=pd.DataFrame(rows)
for c in ['plain_t','sum_t','meanconf_t']: o[c]=o[c]*NORM_T_ATAR
o['truth_t']=df['truth_positron_t'].to_numpy()[:N]
o['acc_t']=df['truth_accidental_positron_t'].to_numpy()[:N] if 'truth_accidental_positron_t' in df else -1000
o['truthKE']=df['truth_positron_energy'].to_numpy()[:N]
o['et']=df['event_type'].to_numpy()[:N] if 'event_type' in df else 0
o.to_parquet('/pipeline/diag_twopos.parquet')
print("wrote", len(o), "events")
