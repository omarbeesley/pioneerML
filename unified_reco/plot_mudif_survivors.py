"""Event displays of muDIF SURVIVORS — true muon-DIF (pion-at-rest, accepted) that LEAK through
the veto at 50% pie eff (muon_dif_score < cut). Same ATAR style as plot_mudif_events.py, with
the veto score in the title (these are the events the veto MISSED -> the invisible-muon leak).
Aligns predictions[is_mudif] with eval_mudif_osl survivors (exact PURITYDataset filter; verified corr=1.0)."""
import os, numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P="/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"; MAXH=250
PION,MUON,POSITRON,ELECTRON,GAMMA,OTHER=1,2,4,8,16,32
def hit_color(pdg):
    if pdg&MUON:return "tab:red"
    if pdg&POSITRON:return "tab:blue"
    if pdg&PION:return "dimgray"
    if pdg&ELECTRON:return "tab:cyan"
    if pdg&GAMMA:return "tab:orange"
    return "tab:green"
def _f(*v): return all(np.isfinite(x) for x in v)
# --- survivor selection from predictions ---
pred=pd.read_parquet(f"{P}/tail_reveal_eval10/predictions.parquet")
AT=pred.pion_decay_ke.to_numpy()==0.0; AC=pred.acceptance.to_numpy()==1
pie=pred[(pred.is_pie==1)&AT&AC]; cut=np.quantile(pie.muon_dif_score.to_numpy(),0.50)
mud=pred[pred.is_mudif==1].reset_index(drop=True)
surv=(mud.muon_dif_present.to_numpy()==1)&(mud.pion_decay_ke.to_numpy()==0.0)&(mud.acceptance.to_numpy()==1)&(mud.muon_dif_score.to_numpy()<cut)
pos=np.where(surv)[0]; sc=mud.muon_dif_score.to_numpy()[pos]; th=mud.positron_theta.to_numpy()[pos]
order=np.argsort(sc); pick=pos[order[np.linspace(0,len(pos)-1,18).astype(int)]]  # 18 spanning the survivor range
pickset={int(p):(float(mud.muon_dif_score.iloc[p]),float(mud.positron_theta.iloc[p])) for p in pick}
print(f"veto cut(50% pie)={cut:.4f}  survivors={len(pos)}  picking {len(pickset)}",flush=True)
# --- stream eval_mudif with EXACT filter; grab the picked rows (aligned by position) ---
COLS=["atar_x","atar_y","atar_z","atar_E","atar_view","atar_pdg","lyso_x",
      "truth_pion_stop_x","truth_pion_stop_y","truth_pion_stop_z","truth_muon_start_x","truth_muon_start_y","truth_muon_start_z",
      "truth_muon_stop_x","truth_muon_stop_y","truth_muon_stop_z","truth_positron_start_x","truth_positron_start_y","truth_positron_start_z",
      "truth_positron_stop_x","truth_positron_stop_y","truth_positron_stop_z","truth_theta","truth_positron_energy","muon_decay_ke"]
rows={}; j=0
for b in pq.ParquetFile(f"{P}/tail10/eval_mudif_osl.parquet").iter_batches(batch_size=8000,columns=COLS):
    d=b.to_pandas()
    ax=d.atar_x.apply(lambda x:len(x) if x is not None else 0).to_numpy()
    lx=d.lyso_x.apply(lambda x:len(x) if x is not None else 0).to_numpy()
    keep=((ax+lx)>0)&((ax+lx)<=MAXH); d=d[keep].reset_index(drop=True)
    for i in range(len(d)):
        if j in pickset: rows[j]=(d.iloc[i], pickset[j]); 
        j+=1
    if len(rows)>=len(pickset): break
print(f"matched {len(rows)} rows",flush=True)
OUT=f"{P}/mudif_survivor_displays"; os.makedirs(OUT,exist_ok=True)
def plot_event(row,score,thpred,path,idx):
    az=np.array(row["atar_z"],float)
    if len(az)==0: return False
    ax_=np.array(row["atar_x"],float); ay_=np.array(row["atar_y"],float); aE=np.array(row["atar_E"],float)
    aview=np.array(row["atar_view"],float); apdg=np.array(row["atar_pdg"],int)
    # sanity: predicted theta (predictions) vs truth theta (eval) should match
    dth=abs(np.degrees(float(row["truth_theta"]))-np.degrees(thpred))
    sizes=15.0+280.0*(aE/max(aE.max(),1e-6)); colors=np.array([hit_color(int(p)) for p in apdg])
    is_yz=aview>0.5; is_xz=~is_yz
    fig,axes=plt.subplots(2,1,figsize=(6,12),sharex=True)
    T=dict(pion=(row["truth_pion_stop_x"],row["truth_pion_stop_y"],row["truth_pion_stop_z"]),
           mu0=(row["truth_muon_start_x"],row["truth_muon_start_y"],row["truth_muon_start_z"]),
           mu1=(row["truth_muon_stop_x"],row["truth_muon_stop_y"],row["truth_muon_stop_z"]),
           e0=(row["truth_positron_start_x"],row["truth_positron_start_y"],row["truth_positron_start_z"]),
           e1=(row["truth_positron_stop_x"],row["truth_positron_stop_y"],row["truth_positron_stop_z"]))
    transv={0:ax_,1:ay_}
    for ax,sel,tc,tl in [(axes[0],is_xz,0,"x"),(axes[1],is_yz,1,"y")]:
        if sel.any(): ax.scatter(az[sel],transv[tc][sel],s=sizes[sel],c=list(colors[sel]),edgecolors="k",linewidths=0.3,alpha=0.85,zorder=3)
        px,py,pz=T["pion"]
        if _f(pz,px,py): ax.scatter([pz],[px if tc==0 else py],marker="*",s=240,c="k",zorder=4)
        def seg(a,b,c):
            if _f(a[2],b[2],a[tc],b[tc]): ax.plot([a[2],b[2]],[a[tc],b[tc]],color=c,ls="--",lw=1.6,alpha=0.75,zorder=2)
        seg(T["mu0"],T["mu1"],"tab:red"); seg(T["e0"],T["e1"],"tab:blue")
        ax.set_ylabel(f"{tl} (mm)"); ax.grid(alpha=0.3); ax.set_ylim(-11.5,11.5)
    axes[1].set_xlabel("z (mm)"); axes[1].set_xlim(-0.2,7.0)
    n_mu=int(((apdg&MUON)>0).sum()); n_pos=int(((apdg&POSITRON)>0).sum())
    # muon travel distance (truth)
    mdist=np.sqrt(sum((T["mu1"][k]-T["mu0"][k])**2 for k in range(3))) if _f(*T["mu0"],*T["mu1"]) else float('nan')
    fig.suptitle(f"muDIF SURVIVOR  veto_score={score:.3f} (< cut, LEAKED)   muon_travel={mdist:.2f} mm\n"
                 f"KE_mu@decay={float(row.get('muon_decay_ke',0)):.2f} MeV   E(e+)={float(row['truth_positron_energy']):.1f} MeV   "
                 f"theta={np.degrees(float(row['truth_theta'])):.0f} deg   ATAR {len(az)} hits (muon {n_mu}, e+ {n_pos})",fontsize=8.5)
    handles=[Line2D([0],[0],marker='o',ls='',mfc=c,mec='k',label=l) for c,l in
             [("tab:red","muon hit"),("tab:blue","e+ hit"),("dimgray","pion"),("tab:green","other")]]
    handles+=[Line2D([0],[0],marker='*',ls='',mfc='k',mec='k',label='pion stop (truth)'),
              Line2D([0],[0],color='tab:red',ls='--',label='muon track (truth)'),
              Line2D([0],[0],color='tab:blue',ls='--',label='e+ track (truth)')]
    axes[0].legend(handles=handles,fontsize=7,loc="best",ncol=2)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(path,dpi=130); plt.close(fig); return True
made=0
for j in sorted(rows):
    row,(score,thpred)=rows[j]
    if plot_event(row,score,thpred,f"{OUT}/survivor_{made:02d}.png",j):
        made+=1
print(f"wrote {made} survivor displays -> {OUT}/\nDONE",flush=True)
