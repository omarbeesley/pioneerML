"""Event displays of piDIF background events that the veto does NOT reject (survivors) —
i.e. accepted, low-bin (E<56) piDIF with a LOW pion_dif_score. Same ATAR display style as
plot_mudif_events.py (x-z top, y-z bottom; hits colored by PDG, sized by deposited energy;
pion-stop star + truth muon/e+ tracks). Annotated with the pion_dif_score so you can see how
'pie-like' the veto rated each one.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P="/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
PION,MUON,POSITRON,ELECTRON,GAMMA=1,2,4,8,16
NOTVETO_CUT=0.0349   # 90% pie-eff working point for pion_dif_score -> "not vetoed" below this
OUT=os.path.join(os.path.dirname(__file__),"pidif_notvetoed_displays"); os.makedirs(OUT,exist_ok=True)

def hit_color(pdg):
    if pdg & MUON: return "tab:red"
    if pdg & POSITRON: return "tab:blue"
    if pdg & PION: return "dimgray"
    if pdg & ELECTRON: return "tab:cyan"
    if pdg & GAMMA: return "tab:orange"
    return "tab:green"
def _fin(*v): return all(np.isfinite(x) for x in v)

def plot_event(row, path, idx):
    az=np.array(row["atar_z"],float)
    if len(az)==0: return False
    ax_=np.array(row["atar_x"],float); ay_=np.array(row["atar_y"],float)
    aE=np.array(row["atar_E"],float); aview=np.array(row["atar_view"],float); apdg=np.array(row["atar_pdg"],int)
    sizes=15.0+280.0*(aE/max(float(aE.max()),1e-6)); colors=np.array([hit_color(int(p)) for p in apdg])
    is_yz=aview>0.5; is_xz=~is_yz
    fig,axes=plt.subplots(2,1,figsize=(6,12),sharex=True)
    T=dict(pion=(row["truth_pion_stop_x"],row["truth_pion_stop_y"],row["truth_pion_stop_z"]),
           mu0=(row["truth_muon_start_x"],row["truth_muon_start_y"],row["truth_muon_start_z"]),
           mu1=(row["truth_muon_stop_x"],row["truth_muon_stop_y"],row["truth_muon_stop_z"]),
           e0=(row["truth_positron_start_x"],row["truth_positron_start_y"],row["truth_positron_start_z"]),
           e1=(row["truth_positron_stop_x"],row["truth_positron_stop_y"],row["truth_positron_stop_z"]))
    transv={0:ax_,1:ay_}
    for ax,sel,tc,tlab in [(axes[0],is_xz,0,"x"),(axes[1],is_yz,1,"y")]:
        if sel.any(): ax.scatter(az[sel],transv[tc][sel],s=sizes[sel],c=list(colors[sel]),edgecolors="k",linewidths=0.3,alpha=0.85,zorder=3)
        px,py,pz=T["pion"]
        if _fin(pz,px,py): ax.scatter([pz],[px if tc==0 else py],marker="*",s=240,c="k",zorder=4)
        def seg(a,b,c):
            if _fin(a[2],b[2],a[tc],b[tc]): ax.plot([a[2],b[2]],[a[tc],b[tc]],color=c,ls="--",lw=1.6,alpha=0.75,zorder=2)
        seg(T["mu0"],T["mu1"],"tab:red"); seg(T["e0"],T["e1"],"tab:blue")
        ax.set_ylabel(f"{tlab} (mm)"); ax.grid(alpha=0.3)
    axes[1].set_xlabel("z (mm)"); axes[1].set_xlim(-0.2,7.0)
    for ax in axes: ax.set_ylim(-11.5,11.5)
    n_mu=int(((apdg&MUON)>0).sum()); n_pos=int(((apdg&POSITRON)>0).sum())
    dep=float(row["live_E"])+float(row["dead_E"])
    fig.suptitle(f"piDIF NOT-VETOED #{idx}   pion_dif_score={float(row['pion_dif_score']):.4f} (< {NOTVETO_CUT:g} = kept)\n"
                 f"E_dep={dep:.1f} MeV (low bin)   E(e+)={float(row['truth_positron_energy']):.1f} MeV   "
                 f"theta={np.degrees(float(row['truth_theta'])):.0f} deg   KE_mu@decay={float(row.get('muon_decay_ke',0)):.2f} MeV\n"
                 f"ATAR {len(az)} hits (muon {n_mu}, e+ {n_pos})",fontsize=9)
    handles=[Line2D([0],[0],marker='o',ls='',mfc=c,mec='k',label=l) for c,l in
             [("tab:red","muon hit"),("tab:blue","e+ hit"),("dimgray","pion hit"),("tab:cyan","electron"),("tab:green","other")]]
    handles+=[Line2D([0],[0],marker='*',ls='',mfc='k',mec='k',label='pion stop (truth)'),
              Line2D([0],[0],color='tab:red',ls='--',label='muon track'),Line2D([0],[0],color='tab:blue',ls='--',label='e+ track')]
    axes[0].legend(handles=handles,fontsize=7,loc="best",ncol=2)
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(path,dpi=130); plt.close(fig); return True

def key(a,b): return np.round(np.asarray(a),4).astype(str)+"|"+np.round(np.asarray(b),4).astype(str)
pr=pd.read_parquet(f"{P}/tail_reveal_eval10/predictions.parquet",columns=["is_pidif","pion_dif_score","deposited_energy","dead_E","acceptance"])
pr=pr[pr.is_pidif==1]; pr["k"]=key(pr.deposited_energy,pr.dead_E); pr=pr.drop_duplicates("k")
cols=["atar_x","atar_y","atar_z","atar_E","atar_view","atar_pdg","event_type","live_E","dead_E","truth_acceptance",
      "truth_pion_stop_x","truth_pion_stop_y","truth_pion_stop_z","truth_muon_start_x","truth_muon_start_y","truth_muon_start_z",
      "truth_muon_stop_x","truth_muon_stop_y","truth_muon_stop_z","truth_positron_start_x","truth_positron_start_y","truth_positron_start_z",
      "truth_positron_stop_x","truth_positron_stop_y","truth_positron_stop_z","truth_positron_energy","truth_theta","muon_decay_ke"]
ev=pd.read_parquet(f"{P}/tail10/eval_pidif_osl.parquet",columns=cols)
ev["k"]=key(ev.live_E,ev.dead_E); ev=ev.drop_duplicates("k")
d=ev.merge(pr[["k","pion_dif_score"]],on="k",how="inner")
dep=d.live_E.to_numpy()+d.dead_E.to_numpy()
sel=d[(d.truth_acceptance.to_numpy()>=1)&(dep<56)&(d.pion_dif_score.to_numpy()<NOTVETO_CUT)].sort_values("pion_dif_score").reset_index(drop=True)
print(f"accepted low-bin piDIF NOT vetoed (score<{NOTVETO_CUT}): {len(sel)}")
pick=sel.iloc[np.linspace(0,len(sel)-1,20).astype(int)]   # spread across the not-vetoed score range
made=0
for _,row in pick.iterrows():
    if plot_event(row,os.path.join(OUT,f"pidif_notvetoed_{made:02d}.png"),made):
        print(f"  wrote pidif_notvetoed_{made:02d}.png  score={row['pion_dif_score']:.4f}  E_dep={row['live_E']+row['dead_E']:.1f}  theta={np.degrees(row['truth_theta']):.0f}deg  hits={len(row['atar_z'])}")
        made+=1
print(f"\nwrote {made} displays to {OUT}/")
