"""PURE-michel tail-reveal cut-flow on origin-resliced in-window michel.
kPienu (natural pi->e nu in the unforced pool, event_type & 0x1) REMOVED via positional
alignment with the mixed inputs (exact PURITYDataset filter: 0 < atar+lyso hits <= 250),
alignment verified per piece by corr(pred.deposited_energy, input.live_E)."""
import glob, os, numpy as np, pandas as pd, pyarrow.parquet as pq, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
P="/data/raid3/eliza7/PIONEER/data/ML_TEST/scratch_pipeline"
OUT="/home/obeesley/pioneerML/unified_reco/updated_plots/07_cutflow_michel_rs.png"
MAXH=250
COLS=["acceptance","deposited_energy","muon_score","pileup_score","topo_score","pie_score"]
preds=[]; ets=[]
for pf in sorted(glob.glob(f"{P}/michel_eval_inwin_rs/*/predictions.parquet")):
    piece=pf.split("/")[-2]
    inp=f"{P}/michel_inwin/{piece}.parquet"
    if not os.path.exists(inp): continue
    pr=pd.read_parquet(pf,columns=COLS)
    t=pq.read_table(inp,columns=["event_type","live_E","atar_x","lyso_x"]).to_pandas()
    nh=t.atar_x.apply(lambda x:0 if x is None else len(x)).to_numpy()+t.lyso_x.apply(lambda x:0 if x is None else len(x)).to_numpy()
    keep=(nh>0)&(nh<=MAXH)
    if keep.sum()!=len(pr): continue                      # alignment impossible -> skip piece
    et=t.event_type.to_numpy().astype(np.int64)[keep]
    le=t.live_E.to_numpy()[keep]
    c=np.corrcoef(pr.deposited_energy.to_numpy(),le)[0,1]
    if c<0.999: print(f"  [skip] {piece} align corr={c:.4f}"); continue
    preds.append(pr); ets.append(et)
d=pd.concat(preds,ignore_index=True); et=np.concatenate(ets)
print(f"aligned pieces: {len(preds)}  events: {len(d):,}  kPienu removed: {int(((et&1)>0).sum()):,}")
mask=((et&1)==0)                                          # PURE michel (drop natural pi->e)
d=d[mask].reset_index(drop=True)
d=d[d.deposited_energy<=75].reset_index(drop=True)
E=d.deposited_energy.values; acc=(d.acceptance==1).values
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":150,"font.size":10.5,
                     "axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True})
stages=[("(0) after acceptance",np.ones(len(d),bool)),
        ("(1) + muon veto",(d.muon_score<0.5).values),
        ("(2) + pileup veto",(d.pileup_score<0.5).values),
        ("(3) + pie topo",(d.topo_score>0.5).values),
        ("(4) + pie score",(d.pie_score>0.5).values)]
bins=np.linspace(0,75,51); n0=acc.sum()
fig,axes=plt.subplots(1,5,figsize=(21,4.7),sharex=True,sharey=True)
keep=acc.copy()
for ax,(name,m) in zip(axes,stages):
    keep=keep&m
    ax.hist(E[keep],bins=bins,histtype="step",lw=1.9,color="#1f77b4")
    ax.axvline(56,color="k",ls=":",lw=0.9)
    ax.set_yscale("log"); ax.set_title(name,fontweight="bold")
    n=int(keep.sum()); hi=int((keep&(E>=56)).sum())
    ax.text(0.03,0.03,f"michel: {n:,}\nsupp: {n0/max(n,1):,.0f}$\\times$\nE$\\geq$56: {hi:,}",
            transform=ax.transAxes,va="bottom",fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="0.7",alpha=0.9))
    ax.set_xlabel("deposited energy [MeV]")
axes[0].set_ylabel(r"accepted $\pi\to\mu\to e$ / bin")
fig.tight_layout(); fig.savefig(OUT); plt.close(fig)
print(f"wrote {OUT}")
keep=acc.copy()
for name,m in stages:
    keep=keep&m
    print(f"  {name:22} n={keep.sum():>9,}  supp={n0/max(keep.sum(),1):>10,.0f}x  hi={int((keep&(E>=56)).sum()):,}")
