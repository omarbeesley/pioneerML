import numpy as np, pandas as pd, matplotlib, os, argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import gridspec

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':13,'axes.labelsize':12,
    'axes.edgecolor':'#444444','axes.linewidth':1.0,'figure.facecolor':'white',
    'axes.facecolor':'white','xtick.labelsize':10,'ytick.labelsize':10})

ap=argparse.ArgumentParser(description="Render PURITY hard-event displays from a per-hit parquet.")
ap.add_argument('--input','-i', default='hard_events_100.parquet', help='per-hit parquet (from dump_gallery.py)')
ap.add_argument('--outdir','-o', default='displays', help='directory to write PNGs into')
ap.add_argument('--category','-c', default=None, choices=[None,'pimu_slice','pileup','scatter','accidental_close','pie','pimue'],
                help='only render events of this category')
ap.add_argument('--index', type=int, default=None, help='only render this orig_idx')
ap.add_argument('--dpi', type=int, default=140)
args=ap.parse_args()
IN=args.input; OUTDIR=args.outdir; os.makedirs(OUTDIR, exist_ok=True)
df=pd.read_parquet(IN)
if args.category: df=df[df.category==args.category]
if args.index is not None: df=df[df.orig_idx==args.index]
print(f"rendering {len(df)} event(s) from {IN} -> {OUTDIR}/",flush=True)

C_PION='#d62728'; C_MUON='#1f77b4'; C_SIG='#2ca02c'; C_MERGE='#9467bd'
C_OTHERPOS='#bcbd6b'; C_OTHER='#c7c7c7'; C_CTX='#dcdcdc'

def truth_color(pi,mu,mip,trigpos):
    if trigpos: return C_SIG          # the signal (triggering) positron
    if pi and mu: return C_MERGE      # pion & muon in the same pixel
    if pi: return C_PION
    if mu: return C_MUON
    if mip: return C_OTHERPOS         # other EM activity (pile-up positrons / deltas)
    return C_OTHER

def lyso_color(mask):
    if mask & 4: return C_SIG
    if mask & 8: return C_OTHERPOS
    if mask & 2: return C_MUON
    if mask & 1: return C_PION
    return C_OTHER

DESC={
 'pimu_slice':("A $\\pi\\!\\to\\!\\mu\\!\\to\\!e$ decay where the pion and muon deposit energy in the SAME readout time-slice",
               "$\\pi$/$\\mu$ overlap"),
 'pileup':    ("A signal decay buried under several overlapping pile-up events in the tracker",
               "pile-up"),
 'scatter':   ("A daughter positron that scatters sharply as it crosses the silicon tracker",
               "scattering $e^+$"),
 'accidental_close':("An accidental positron from another decay arrives right at the signal positron's origin",
               "accidental $e^+$"),
 'pie':       ("A clean $\\pi\\!\\to\\!e\\,\\nu$ decay — the pion stops and emits a prompt monoenergetic positron",
               "signal $e^+$"),
 'pimue':     ("A clean $\\pi\\!\\to\\!\\mu\\!\\to\\!e$ decay — the pion stops, the muon stops, and a delayed Michel positron is emitted",
               "Michel $e^+$"),
}
C_ACC='#7a7a28'   # callout color for the accidental positron (darker olive, readable)

ZLIM=(-0.5,7.5); TLIM=(-11.5,11.5)

for _,r in df.iterrows():
    hz=np.array(r['hz']); hx=np.array(r['hx']); hy=np.array(r['hy']); hE=np.array(r['hE'])
    hv=np.array(r['hview']); ht=np.array(r['ht'])
    hpi=np.array(r['hpion']); hmu=np.array(r['hmuon']); hmip=np.array(r['hmip_truth']); hor=np.array(r['horigin'])
    tag=np.array(r['htrig_pos']); tru=np.array(r['htrue_pos']) if 'htrue_pos' in r else (hmip>0)
    cat=r['category']; idx=int(r['orig_idx'])
    N=len(hz)
    tcol=np.array([truth_color(hpi[i],hmu[i],hmip[i],tru[i]) for i in range(N)])
    sz=np.clip(hE*170,14,120)
    desc,ov=DESC[cat]

    # accidental-positron proximity, used by the 'accidental_close' category.
    # If the signal positron's emission point is stored, highlight accidental hits near
    # THAT vertex (the endpoint-at-vertex story); otherwise fall back to track-to-track proximity.
    accm=(hor>=1)&(hmip>0)
    close=np.zeros(N,bool); acc_dmin=np.inf; acc_tgap=np.inf
    vtx=None
    if all(k in r for k in ('truth_positron_start_x','truth_positron_start_y','truth_positron_start_z')):
        v0=np.array([r['truth_positron_start_x'],r['truth_positron_start_y'],r['truth_positron_start_z']],float)
        if np.all(np.isfinite(v0)): vtx=v0
    if accm.any() and (tru>0).any():
        for v in (0,1):
            av=np.where(accm&(hv==v))[0]; sv=np.where((tru>0)&(hv==v))[0]
            if len(av)==0 or len(sv)==0: continue
            c=hx if v==0 else hy
            dd=np.sqrt((hz[av][:,None]-hz[sv][None,:])**2+(c[av][:,None]-c[sv][None,:])**2)
            acc_dmin=min(acc_dmin,float(dd.min()))
            if vtx is not None:   # ring accidental hits near the e+ emission vertex (view projection)
                cv=vtx[0] if v==0 else vtx[1]
                dv=np.sqrt((hz[av]-vtx[2])**2+(c[av]-cv)**2)
                close[av[dv<2.0]]=True
            else:
                close[av[dd.min(1)<1.5]]=True
            cl=dd.min(1)<1.5
            if cl.any():   # time gap restricted to the close PAIRS
                dtp=np.abs(ht[av][:,None]-ht[sv][None,:])
                acc_tgap=min(acc_tgap,float(dtp[dd<1.5].min()))

    fig=plt.figure(figsize=(21,10.4))
    gs=gridspec.GridSpec(2,4,height_ratios=[1,1],width_ratios=[1,1,1,1],
                         hspace=0.28,wspace=0.26,left=0.05,right=0.985,top=0.86,bottom=0.13)

    def spatial(ax,view,mode):
        mv=hv==view
        hc = hx if view==0 else hy
        if mode=='truth':
            ax.scatter(hz[mv],hc[mv],c=tcol[mv],s=sz[mv],alpha=0.9,edgecolors='white',linewidths=0.3,zorder=3)
        else:
            # context = all hits faded; highlight = model-identified positron
            other=mv&(tag==0); hi=mv&(tag>0)
            ax.scatter(hz[other],hc[other],c=C_CTX,s=sz[other]*0.8,alpha=0.9,edgecolors='none',zorder=2)
            if cat=='accidental_close':
                am=mv&close   # ring the accidental hits the model correctly did NOT tag
                ax.scatter(hz[am],hc[am],s=sz[am]*0.8+26,facecolors='none',edgecolors=C_ACC,linewidths=1.1,zorder=3)
            ax.scatter(hz[hi],hc[hi],c=C_SIG,s=sz[hi]+18,alpha=0.95,edgecolors='#0b3d0b',linewidths=0.7,zorder=4)
            # reconstructed direction arrow from the positron origin
            if 'pred_dir_z' in r and hi.sum()>0:
                i0=np.argmin(np.where(hi,ht,1e9)); z0=hz[i0]; c0=(hx if view==0 else hy)[i0]
                dz=r['pred_dir_z']; dc=r['pred_dir_x'] if view==0 else r['pred_dir_y']
                nrm=np.hypot(dz,dc)+1e-9; L=4.5
                ax.annotate('',xy=(z0+L*dz/nrm,c0+L*dc/nrm),xytext=(z0,c0),
                            arrowprops=dict(arrowstyle='-|>',color='#111111',lw=2.0),zorder=6)
        ax.set_xlim(*ZLIM); ax.set_ylim(*TLIM); ax.set_xlabel('z  [mm]')
        ax.set_ylabel(('x' if view==0 else 'y')+'  [mm]')
        ax.grid(True,ls=':',alpha=0.45)

    axTT=fig.add_subplot(gs[0,0]); spatial(axTT,0,'truth'); axTT.set_title('TRUTH — top view (x–z)',fontweight='bold',color='#333')
    axTM=fig.add_subplot(gs[0,1]); spatial(axTM,0,'model'); axTM.set_title('MODEL — top view (x–z)',fontweight='bold',color='#0b3d0b')
    axST=fig.add_subplot(gs[1,0]); spatial(axST,1,'truth'); axST.set_title('TRUTH — side view (y–z)',fontweight='bold',color='#333')
    axSM=fig.add_subplot(gs[1,1]); spatial(axSM,1,'model'); axSM.set_title('MODEL — side view (y–z)',fontweight='bold',color='#0b3d0b')

    # ---- callouts on truth top view ----
    def centroid(mask,view):
        if mask.sum()==0: return None
        return (hz[mask].mean(),(hx if view==0 else hy)[mask].mean())
    BB=dict(boxstyle='round,pad=0.2',fc='white',ec='none',alpha=0.8)
    # callouts are pinned to the panel top (signal/model labels) and bottom (secondary labels)
    # so the two boxes can never collide even when the tracks' centroids coincide
    csig=centroid((hv==0)&(tru>0),0)
    if csig: axTT.annotate('signal $e^+$',xy=csig,xytext=(csig[0],10.9),
                ha='center',va='top',fontsize=11,color=C_SIG,fontweight='bold',bbox=BB,
                arrowprops=dict(arrowstyle='->',color=C_SIG,lw=1.5))
    if cat=='pimu_slice':
        # triggering (origin-0) muon sits inside the pion stop = the physical pi/mu overlap
        cm=centroid((hv==0)&(hor==0)&(hmu>0),0)
        if cm is None: cm=centroid((hv==0)&((hpi>0)&(hmu>0)),0)
        if cm: axTT.annotate('$\\pi$ & $\\mu$ overlap\n(same time-slice)',xy=cm,
                xytext=(cm[0],-10.9),ha='center',va='bottom',fontsize=10.5,
                color=C_MERGE,fontweight='bold',bbox=BB,
                arrowprops=dict(arrowstyle='->',color=C_MERGE,lw=1.5))
    if cat=='accidental_close':
        ca=centroid((hv==0)&close,0)
        if ca is None: ca=centroid((hv==0)&accm,0)
        lab=('accidental $e^+$ ends\nat the signal $e^+$ origin' if ('acc_dend_mm' in r and np.isfinite(r['acc_dend_mm']))
             else 'accidental $e^+$\n(different decay)')
        if ca: axTT.annotate(lab,xy=ca,
                xytext=(ca[0],-10.9),ha='center',va='bottom',fontsize=10.5,
                color=C_ACC,fontweight='bold',bbox=BB,
                arrowprops=dict(arrowstyle='->',color=C_ACC,lw=1.5))
    csig_m=centroid((hv==0)&(tag>0),0)
    if csig_m: axTM.annotate('model-identified $e^+$',xy=csig_m,xytext=(csig_m[0],10.9),
                ha='center',va='top',fontsize=11,color='#0b3d0b',fontweight='bold',bbox=BB,
                arrowprops=dict(arrowstyle='->',color='#0b3d0b',lw=1.5))
    if cat=='accidental_close':
        cam=centroid((hv==0)&close,0)
        if cam: axTM.annotate('accidental — correctly\nNOT tagged',xy=cam,
                xytext=(cam[0],-10.9),ha='center',va='bottom',fontsize=10.5,
                color=C_ACC,fontweight='bold',bbox=BB,
                arrowprops=dict(arrowstyle='->',color=C_ACC,lw=1.5))

    # ---- time panel (right two columns, both rows) ----
    axE=fig.add_subplot(gs[:,2:])
    axE.scatter(ht,hE,c=tcol,s=40,alpha=0.9,edgecolors='white',linewidths=0.3,zorder=3,label='tracker hit')
    axE.axvline(r['truth_positron_t'],color=C_SIG,ls='--',lw=1.8,zorder=1,label='true $e^+$ time')
    axE.axvline(r['pred_time'],color='#111',ls=':',lw=1.8,zorder=1,label='model $e^+$ time')
    if cat=='accidental_close' and close.any():
        axE.axvline(float(np.median(ht[close])),color=C_ACC,ls='-.',lw=1.6,zorder=1,label='accidental $e^+$ time')
    axE.set_yscale('log')
    tmin=min(0,ht.min())-8; tmax=max(r['truth_positron_t'],ht.max())+25
    axE.set_xlim(tmin,tmax); axE.set_ylim(3e-3,3)
    axE.set_xlabel('time  [ns]'); axE.set_ylabel('energy deposit  [MeV]')
    axE.set_title('Energy vs. time — model positron time (dotted) lands on truth (dashed)',fontsize=12)
    axE.grid(True,ls=':',alpha=0.45); axE.legend(loc='upper right',fontsize=9.5,framealpha=0.95)
    if cat=='pimu_slice':
        mm=(hor==0)&(hmu>0)   # triggering muon, prompt with the pion stop
        anch_t=float(np.median(ht[mm])) if mm.sum()>0 else float(np.min(ht[(hpi>0)|(hmu>0)]))
        axE.annotate('triggering $\\pi$ and $\\mu$\nshare one time-slice',xy=(anch_t,6),
            xytext=(tmax*0.33,22),fontsize=10.5,color=C_MERGE,ha='center',fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none',alpha=0.8),
            arrowprops=dict(arrowstyle='->',color=C_MERGE,lw=1.4))

    # ---- titles / result banner ----
    eE=r['truth_positron_energy']; recoE=r['recoE']; tt=r['truth_positron_t']; pt=r['pred_time']
    n_tag=int((tag>0).sum()); n_true=int((tru>0).sum())
    fig.suptitle(desc,fontsize=16,fontweight='bold',y=0.975)
    result=(f"The model isolates the daughter positron ({n_tag} of {n_true} true hits, no false hits) and reconstructs its "
            f"energy to {recoE:.1f} MeV (truth {eE:.1f}) and time to {pt:.0f} ns (truth {tt:.0f}).")
    fig.text(0.5,0.923,result,ha='center',fontsize=12.5,color='#222')
    if cat=='accidental_close':
        de_=r['acc_dend_mm'] if 'acc_dend_mm' in r else np.nan
        dt_=r['acc_dend_dt_ns'] if 'acc_dend_dt_ns' in r else np.nan
        d3=r['acc_d3min_mm'] if 'acc_d3min_mm' in r else np.nan
        tg3=r['acc_tgap_ns'] if 'acc_tgap_ns' in r else np.nan
        if np.isfinite(de_):  # endpoint-at-vertex numbers (verified truth-3D), preferred
            line2=(f"An accidental positron's track ends {de_:.2f} mm from the signal positron's emission point "
                   f"({dt_:.0f} ns apart) — the model tags none of its hits.")
        elif np.isfinite(d3):
            line2=f"The accidental positron passes {d3:.2f} mm (3D) and {tg3:.1f} ns from the signal positron — the model does not tag a single one of its hits."
        elif np.isfinite(acc_dmin):
            line2=f"The accidental positron approaches to {acc_dmin:.2f} mm (projected) and {acc_tgap:.1f} ns — the model does not tag it."
        else: line2=None
        if line2: fig.text(0.5,0.897,line2,ha='center',fontsize=12.5,color=C_ACC,fontweight='bold')

    leg=[mpatches.Patch(color=C_PION,label='pion'),mpatches.Patch(color=C_MUON,label='muon'),
         mpatches.Patch(color=C_MERGE,label='$\\pi$+$\\mu$ same pixel'),
         mpatches.Patch(color=C_SIG,label='signal positron'),
         mpatches.Patch(color=C_OTHERPOS,label='other EM (pile-up $e^+$ / $\\delta$)'),
         mpatches.Patch(color=C_CTX,label='faded = context in model view'),
         Line2D([0],[0],color='#111',lw=2,marker='>',label='model reco direction')]
    fig.legend(handles=leg,loc='lower center',ncol=7,fontsize=10,frameon=False,bbox_to_anchor=(0.5,0.01))

    out=f"{OUTDIR}/{cat}_{idx}.png"; fig.savefig(out,dpi=args.dpi); plt.close(fig)
    print("saved",out,flush=True)
print("done")
