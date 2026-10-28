"""Original 20:1 deposited-energy muDIF contamination versus signal-tail-bias plot.

Source: scratch_pipeline/mudif_tradeoff.py. Original inputs were pie5M_pie_full.parquet
and tail_reveal_eval10/predictions.parquet (plain filename, not predictions_drop).
Explicitly name the desired inputs; this command does not substitute later recipes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ._runtime import AnalysisRun

R_PIE=1e-4; RATIO=20.0; REAL_TAIL=0.006; ES,EM=56.0,75.0


def calculate(pie, md):
    Ep=(pie.deposited_energy+pie.dead_E+pie.Epure).to_numpy(); Em=(md.deposited_energy+md.dead_E).to_numpy()
    pa=((pie.acceptance>=1)&(pie.is_pie==1)).to_numpy()&np.isfinite(Ep)&(Ep<=EM)
    ma=(md.acceptance>=1).to_numpy()&np.isfinite(Em)&(Em<=EM)
    psc=pie.muon_dif_score.to_numpy(); msc=md.muon_dif_score.to_numpy()
    wpie=R_PIE/pa.sum(); wdif=(R_PIE/RATIO)/ma.sum()
    ts=REAL_TAIL/((Ep[pa]<ES).sum()/pa.sum())
    rlo=(pa&(Ep<ES)).sum(); rhi=(pa&(Ep>=ES)).sum()
    effs=[99,95,90,80,70,50,30]
    X=[];Y=[];YE=[]
    for e in effs:
        c=np.quantile(psc[pa],e/100.0)
        pm=pa&(psc<c); mm=ma&(msc<c)
        plo=ts*wpie*(pm&(Ep<ES)).sum(); mlo=wdif*(mm&(Em<ES)).sum()
        contam=100*mlo/(mlo+plo)
        slo=(pm&(Ep<ES)).sum(); shi=(pm&(Ep>=ES)).sum()
        el=slo/rlo; eh=shi/rhi; r=el/eh
        X.append(contam); Y.append(100*(r-1)); YE.append(100*r*np.sqrt((1-el)/max(slo,1)+(1-eh)/max(shi,1)))
    return {'pie_eff_pct': effs, 'contamination_pct': X, 'bias_pct': Y,
            'bias_uncertainty_pct': YE, 'accepted_pie': int(pa.sum()),
            'accepted_mudif': int(ma.sum()), 'tail_rescale': float(ts)}


def render(result, output_dir):
    plt.rcParams.update({"figure.dpi":130,"savefig.dpi":200,"font.size":15,
        "axes.labelsize":17,"axes.titlesize":17,"legend.fontsize":14,
        "xtick.labelsize":14,"ytick.labelsize":14,"xtick.direction":"in","ytick.direction":"in",
        "xtick.top":True,"ytick.right":True,"xtick.minor.visible":True,"ytick.minor.visible":True,
        "axes.grid":False})
    effs=result['pie_eff_pct']; X=result['contamination_pct']; Y=result['bias_pct']; YE=result['bias_uncertainty_pct']
    fig,ax=plt.subplots(figsize=(9,6.2))
    ax.errorbar(X,Y,yerr=YE,fmt="o-",color="#9467bd",lw=2.0,ms=6,capsize=3)
    for x,y,e in zip(X,Y,effs):
        ax.annotate(f"{e}%",(x,y),textcoords="offset points",xytext=(8,7),fontsize=12.5,fontweight="bold")
    ax.axhline(0,color="0.6",ls="--",lw=1.0)
    ax.set_xlabel("Muon-DIF Contamination in Low Bin [%]")
    ax.set_ylabel(r"$\pi\to e\nu$ Tail Bias [%]")
    ax.set_xlim(left=0)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'22_mudif_bias_vs_contam.{ext}')
    plt.close(fig)
    print("points (eff, contam, bias+/-err):")
    for e,x,y,ye in zip(effs,X,Y,YE): print(f"  {e:>3}%  {x:6.2f}%  {y:+6.2f}+/-{ye:.2f}%")
    print("DONE")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pie-pred', type=Path, required=True, help='Signal prediction table with Epure correction')
    parser.add_argument('--mudif-pred', type=Path, required=True, help='Prediction table containing is_mudif; original used plain predictions.parquet')
    parser.add_argument('--output-dir', type=Path, required=True, help='New or empty output directory')
    parser.add_argument('--hash-inputs', action='store_true')
    args=parser.parse_args(argv)
    for path in (args.pie_pred,args.mudif_pred):
        if not path.is_file(): parser.error(f'Input does not exist: {path}')
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f'Output directory must be new or empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    record=AnalysisRun('mudif_tradeoff',args.output_dir,{
        'pie_pred':str(args.pie_pred.resolve()),'mudif_pred':str(args.mudif_pred.resolve()),
        'output_dir':str(args.output_dir.resolve()),'pie_to_mudif_at_acceptance':RATIO,
        'signal_tail_fraction_target':REAL_TAIL,'energy_split_MeV':ES,'energy_max_MeV':EM,
        'original_prediction_filename':'predictions.parquet, not predictions_drop.parquet'},args.hash_inputs)
    try:
        pie=record.read_parquet(args.pie_pred,columns=['deposited_energy','dead_E','Epure','acceptance','is_pie','muon_dif_score'])
        md=record.read_parquet(args.mudif_pred,columns=['is_mudif','muon_dif_score','deposited_energy','dead_E','acceptance'])
        md=md[md.is_mudif==1]
        if pie.empty or md.empty: raise ValueError('Both signal and muDIF populations are required')
        with np.errstate(divide='raise',invalid='raise'):
            result=calculate(pie,md)
        if not all(np.isfinite(result[k]).all() for k in ('contamination_pct','bias_pct','bias_uncertainty_pct')):
            raise ValueError('Nonfinite scan values; check populated signal tail, peak and background bins')
        render(result,args.output_dir)
        (args.output_dir/'22_mudif_bias_vs_contam.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    except BaseException as exc:
        record.finish(error=exc)
        raise
    else:
        record.finish()
    return result


if __name__ == '__main__':
    main()
