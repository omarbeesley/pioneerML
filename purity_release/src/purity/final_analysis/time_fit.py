"""Reproduce the final binned time-fit figure from explicit combined event summaries.

This preserves the historical central fit and separately labels its historical
Gaussian-toy plot-error recipe. Input rows and mixture provenance determine the
result; the command does not reconstruct the frozen combined input samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pyarrow.parquet as pq

from purity.measurement import io as pio, measurement as pm


def _ratio(numerator, denominator):
    return numerator / denominator if denominator != 0 else float("nan")


def _input_record(path):
    path = Path(path).resolve()
    parquet = pq.ParquetFile(path)
    return {"path": str(path), "bytes": path.stat().st_size, "rows": parquet.metadata.num_rows}


def _json_ready(value, path="", undefined=None):
    if isinstance(value, dict):
        return {key: _json_ready(item, path + "." + key, undefined) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item, path + "[" + str(index) + "]", undefined) for index, item in enumerate(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        undefined.append(path)
        return None
    return value


def run(pie_path, michel_path, output_dir):
    """Return JSON-ready metrics and write the historical PDF/PNG and metrics JSON."""
    pie_path, michel_path, output_dir = Path(pie_path), Path(michel_path), Path(output_dir)
    required = {"pred_positron_energy", "pred_dead_energy", "pred_accepted",
                "pred_positron_time_consensus_ns", "truth_positron_t"}
    for path, is_michel in ((pie_path, False), (michel_path, True)):
        columns = set(pq.ParquetFile(path).schema_arrow.names)
        missing = required - columns
        if not ({"gen_weight", "truth_gen_weight"} & columns):
            missing.add("gen_weight or truth_gen_weight")
        if is_michel:
            missing |= {"truth_positron_energy"} - columns
            if not ({"truth_event_type", "event_type"} & columns):
                missing.add("truth_event_type or event_type")
        if missing:
            raise ValueError(str(path) + ": missing required columns: " + ", ".join(sorted(missing)))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty to preserve prior analysis results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi":130,"savefig.dpi":300,"savefig.bbox":"tight","font.size":11,
     "font.family":"serif","mathtext.fontset":"cm","axes.grid":False,"legend.frameon":False,
     # ticks inside, on all four sides, with minors (log axis)
     "xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
     "xtick.minor.visible":True,"ytick.minor.visible":True,
     "xtick.major.size":6,"xtick.minor.size":3,"ytick.major.size":6,"ytick.minor.size":3,
     "xtick.labelsize":13,"ytick.labelsize":12,"axes.linewidth":0.9})
    cfg=pio.AnalysisConfig(mode="reco",energy="deposited",
                           time_col="pred_positron_time_consensus_ns",clean_tol=5.0)
    df=pio.load_all(pie_path, michel_path, cfg)
    if any(float(df.loc[df.cls == c, "w"].sum()) <= 0 for c in ("pie", "mue")):
        raise ValueError("Both input pools need positive total weights after Michel filtering")
    sel=pio.selected(df,require_window=True); sel=sel[sel.E<=cfg.e_max]
    sel=sel.assign(ebin=pio.energy_bin(sel.E.to_numpy(),cfg.e_split))
    # tail fraction of the SELECTED pie: the fraction reconstructing below the split, which the
    # high-bin fit cannot see and which measure() does not restore.
    _p=sel[sel.cls=="pie"]; _E=_p.E.to_numpy(); _w=_p.w.to_numpy()
    _lo=float(_w[_E<cfg.e_split].sum()); _hi=float(_w[_E>=cfg.e_split].sum())
    C_TAIL=_lo/(_lo+_hi) if (_lo+_hi)>0 else 0.0
    print(f"C_T (selected pie below {cfg.e_split} MeV) = {1e2*C_TAIL:.4f}%",flush=True)
    # Historical final choice: fit the low-bin fractional sideband slope; keep
    # the high-bin accidental flat. Each bin anchors its own sideband level.
    _lb=sel[sel.ebin=="low"]; _tl=np.asarray(_lb.t,float)-4.0
    # same shifted binning measure() uses, else the slope is fitted on a different grid than the
    # one it is applied to (worth 2.6 %/800ns and 0.05 pp on the bias)
    _Bf=np.linspace(cfg.window[0]-4.0,cfg.window[1]-4.0,201); _cf=0.5*(_Bf[:-1]+_Bf[1:])
    _ok=np.isfinite(_tl)
    _hf,_=np.histogram(_tl[_ok],bins=_Bf,weights=_lb.w.to_numpy()[_ok])
    _neg=_cf<0
    _cof=np.polyfit(_cf[_neg],_hf[_neg],1,w=1.0/np.sqrt(np.maximum(_hf[_neg],1.0)))
    _midf=float(_cf[_neg].mean())
    FRAC=float(_cof[0]/max(np.polyval(_cof,_midf),1e-30))
    print(f"low-bin sideband fractional slope = {FRAC:+.4g}/ns  ({1e2*FRAC*800:+.1f}%/800ns)",flush=True)
    # Anchor: the slope is FITTED in the low-bin sideband, where ~4000 accidental events per bin
    # determine it, and the HIGH bin is left FLAT.  Importing the low-bin slope into the high bin was
    # tried and is worse on every measure: truth says the high-bin accidental RISES across the window
    # (fit-region/sideband 1.18) while the imported slope makes it fall (0.904), driving the high-bin
    # accidental to 0.77 of truth against 0.84 when flat.  Letting the high bin fit its own sideband is
    # worse still (0.56): 137 events give a 1.7 sigma slope, i.e. noise levered across the window.
    res=pm.measure(df,cfg,n_time_bins=200,fit_pileup=True,
                   pileup_slope_frac={'low':FRAC,'high':0.0})
    bins=res['bins']; ctr=0.5*(bins[:-1]+bins[1:])   # time_fit_bin's own coordinate: already t-t_stop
    # proper weighted errors: sqrt(sum w^2) per bin, per energy bin
    ERR={}
    for b in ("high","low"):
        s=sel[sel.ebin==b]
        t=np.asarray(s.t,float)-4.0   # shift events; bin edges unchanged (matches time_fit_bin)
        w=np.asarray(s.w,float)
        ok=np.isfinite(t)
        h2,_=np.histogram(t[ok],bins=bins,weights=w[ok]**2)
        ERR[b]=np.sqrt(h2)
    # per-bin pie-yield errors (bootstrap, same procedure as ep19_closure_err.py) and prompt S/B
    from scipy.optimize import nnls
    neg=ctr<0.0; pos=ctr>=0.0; fitreg=ctr>=5.0
    pie_b,mic_b=pm._decay_bases(bins); nrm=lambda x:(x/x[pos].sum() if x[pos].sum()>0 else x)
    pie_n,mic_n=nrm(pie_b),nrm(mic_b)
    OBS={b:res['per_bin'][b]['obs'] for b in ('high','low')}
    def _fit(obs,inc_pie):
        """returns (N_pie, N_mue, N_acc) exactly as time_fit_bin does."""
        B=float(obs[neg].mean())
        A=np.vstack([pie_n[fitreg],mic_n[fitreg]]).T if inc_pie else mic_n[fitreg].reshape(-1,1)
        c,_=nnls(A,obs[fitreg]-B)
        npie=c[0] if inc_pie else 0.0
        nmue=c[1] if inc_pie else c[0]
        return npie,nmue,B*len(ctr)
    # TRUE per-component yields in each energy bin: clean pie / clean michel / non-clean (accidental)
    # NB the fit returns the EXTRAPOLATED amplitude (analytic law over t'>=0, i.e. including the
    # merge-deleted early decays), NOT the survivor count -- so the reference for pie/michel is
    # safe-region survivors / analytic safe fraction, else the ratio carries a spurious ~+4.7%.
    # EVERY component is compared as an INTEGRAL OVER THE FIT REGION ONLY (bins with centre
    # >= fit_lo).  The near-stop region is deliberately excluded from the fit, so it must not
    # contribute to any fitted/expected ratio.  (For pie the fit-region ratio equals the
    # extrapolated-amplitude ratio identically, since both divide by the same analytic fraction,
    # so nothing is lost by quoting the fit-region form for all three.)
    _fitreg=ctr>=5.0
    TRUE={}; FITTED_FR={}
    for b in ("high","low"):
        s=sel[sel.ebin==b]
        cl=s.clean.to_numpy(); cls=s.cls.to_numpy(); w=s.w.to_numpy()
        tt_=np.asarray(s.t,float)-4.0; okt=np.isfinite(tt_)
        _H=lambda m: np.histogram(tt_[m&okt],bins=bins,weights=w[m&okt])[0]
        TRUE[b]={'pie':float(_H((cls=='pie')&cl)[_fitreg].sum()),
                 'mue':float(_H((cls=='mue')&cl)[_fitreg].sum()),
                 'acc':float(_H(~cl)[_fitreg].sum())}
        _r=res['per_bin'][b]
        FITTED_FR[b]={'pie':float(_r['fit_pie'][_fitreg].sum()) if _r['include_pie'] else 0.0,
                      'mue':float(_r['fit_mue'][_fitreg].sum()),
                      'acc':float(_r['fit_pileup'][_fitreg].sum())}
        TRUE[b+'_err']={'pie':float(np.sqrt((w[(cls=='pie')&cl]**2).sum())),
                        'mue':float(np.sqrt((w[(cls=='mue')&cl]**2).sum())),
                        'acc':float(np.sqrt((w[~cl]**2).sum()))}
    rng=np.random.default_rng(11); ERRY={}
    for b in ('high','low'):
        inc=(b=='high')
        toys=np.array([_fit(OBS[b]+rng.normal(0,ERR[b]),inc) for _ in range(1500)])
        ERRY[b]={'pie':float(toys[:,0].std()),'mue':float(toys[:,1].std()),'acc':float(toys[:,2].std())}
    fig,ax=plt.subplots(1,2,figsize=(12.5,4.6))
    PANELS=[(ax[0],'low',r"low bin  ($E<56$ MeV)"),(ax[1],'high',r"high bin  ($E\geq56$ MeV)")]
    for a,b,ttl in PANELS:
        r=res['per_bin'][b]; obs=r['obs']; err=ERR[b]
        a.errorbar(ctr,obs,yerr=err,fmt='.',ms=3.0,color='k',ecolor='0.6',alpha=0.85,lw=0.7,
                   elinewidth=0.5,zorder=1,label="data (reconstructed sim)")
        if r['include_pie']:
            a.plot(ctr,r['fit_pie'],'-',color="#1f77b4",lw=2.0,zorder=6,
                   path_effects=[pe.Stroke(linewidth=2.6,foreground='white'),pe.Normal()],
                   label=r"$\pi\to e$  (fixed $\tau_\pi$)")
        a.plot(ctr,r['fit_mue'],'-',color="#d62728",lw=2.0,zorder=6,
               path_effects=[pe.Stroke(linewidth=2.6,foreground='white'),pe.Normal()],
               label=r"$\pi\to\mu\to e$  (Bateman)")
        a.plot(ctr,r['fit_pileup'],'--',color="#2ca02c",lw=1.8,zorder=6,
               path_effects=[pe.Stroke(linewidth=2.5,foreground='white'),pe.Normal()],
               label="accidental (sideband)")
        a.plot(ctr,r['fit_total'],'-',color='k',lw=1.6,zorder=7,
               path_effects=[pe.Stroke(linewidth=2.3,foreground='white'),pe.Normal()],
               label="total fit")
        a.axvline(0,color='0.55',lw=0.8); a.axvline(5,color='0.55',ls=':',lw=1.0)
        a.set_yscale('log'); a.set_xlim(-300,500)
        pos=obs[obs>0]
        lo=max(np.percentile(pos,2)*0.4,1e-3) if len(pos) else 1e-3
        a.set_ylim(lo, max(float(np.nanmax(obs))*4, 10*lo))
        a.set_xlabel(r"$t_\mathrm{reco}-t_{\pi\,\mathrm{stop}}$ [ns]",fontsize=15)
        a.set_ylabel("weighted events / 4 ns",fontsize=13)
        a.set_title(ttl,fontsize=12)
        a.legend(fontsize=8.5,loc='upper right',framealpha=0.92,frameon=True,edgecolor='none')
        # ---- per-bin summary box: fitted / true per component ----
        fitted=FITTED_FR[b]   # fit-region integrals, matching TRUE[b]
        lines=[r"$\bf{fitted/true}$ (fit region)"]
        for key,sym in [('pie',r'\pi e'),('mue',r'\mu e'),('acc',r'\rm acc')]:
            if key=='pie' and not r['include_pie']: continue
            T=TRUE[b][key]
            if T<=0: continue
            ratio=fitted[key]/T
            rerr=ERRY[b][key]/T          # fit-statistics error on the ratio
            lines.append(rf"$N_{{{sym}}}:\ {ratio:.3f} \pm {rerr:.3f}$")
        if r['include_pie']:
            # The high-bin fit returns the pie yield ABOVE the split, while channel_efficiency
            # normalises over the whole selected sample, so the raw ratio is short by exactly the
            # fraction of pie that reconstructed BELOW it.  That fraction is C_T, the correction the
            # master equation already carries, so it is restored here before the closure is quoted.
            # The uncertainty assumes C_T is known exactly: it is the fit-statistics error alone,
            # from the high-bin pion amplitude and the low-bin muon-chain amplitude that normalises
            # it.  The two are fitted in different energy bins, so they add in quadrature.
            _clc=1e2*(_ratio(_ratio(res['remu'], 1.0-C_TAIL), res['remu_true'])-1)
            _rp=ERRY['high']['pie']/max(FITTED_FR['high']['pie'],1e-30)
            _rm=ERRY['low']['mue']/max(FITTED_FR['low']['mue'],1e-30)
            _cle=1e2*np.hypot(_rp,_rm)
            # anchor systematic: the bias lever from the high-bin slope-anchor choice
            # (+-0.04 pp, see the variant-(d) comment above), quoted as a second error
            _anc=0.04
            # kept to short lines: one long line pushes the box under the falling pion curve
            lines.append(rf"$R_{{e/\mu}}$ bias:")
            lines.append(rf"$\quad {_clc:+.2f} \pm {_cle:.2f}\%$ (stat)")
            lines.append(rf"$\quad \pm\ {_anc:.2f}\%$ (anchor)")
        a.text(0.025,0.97,"\n".join(lines),transform=a.transAxes,fontsize=9.5,va='top',ha='left',
               zorder=20,   # above the fit curves, which otherwise draw through the box
               bbox=dict(boxstyle='round,pad=0.45',facecolor='white',edgecolor='0.7',alpha=0.95))
        for key in ('pie','mue','acc'):
            if key=='pie' and not r['include_pie']: continue
            print(f"        {b:>4} {key:>4}: fitted={fitted[key]:.5g} +- {ERRY[b][key]:.4g}   "
                  f"true={TRUE[b][key]:.5g} +- {TRUE[b+'_err'][key]:.4g}   ratio={fitted[key]/max(TRUE[b][key],1e-30):.4f}",flush=True)
        print(f"{b:>5}: N_pie={r['N_pie']:.4g} N_mue={r['N_mue']:.4g} N_acc={r['N_pileup']:.4g}  "
              f"sideband mean={obs[ctr<0].mean():.4g} +- {err[ctr<0].mean():.3g}",flush=True)
    for e in ("pdf","png"): fig.savefig(output_dir / f"fig8_time_fit_fixed.{e}")
    print(f"\nR_e/mu fit = {res['remu']:.6e}   POOL TRUTH = {res['remu_true']:.6e}   "
          f"closure {1e2*(_ratio(res['remu'], res['remu_true'])-1):+.3f}%   [SM {res['remu_sm']:.4e}; this sample is "
          f"norad, so SM is NOT the reference]")
    print("TIMEFIG_DONE",flush=True)
    plt.close(fig)
    metrics = {
        "schema_version": 1,
        "recipe": "fig8_timefit_fixed",
        "inputs": {"pie": _input_record(pie_path), "michel": _input_record(michel_path)},
        "configuration": {"mode": "reco", "time_column": cfg.time_col,
            "energy_split_mev": cfg.e_split, "energy_max_mev": None,
            "readout_window_ns": list(cfg.window), "time_stop_ns": 4.0,
            "fit_minimum_bin_center_ns": 5.0, "time_bins": 200,
            "clean_tolerance_ns": cfg.clean_tol,
            "pileup_slope_fraction_per_ns": {"low": FRAC, "high": 0.0}},
        "selected_rows": {c: int((sel.cls == c).sum()) for c in ("pie", "mue")},
        "selected_pie_tail_fraction": C_TAIL,
        "remu_fit": res["remu"], "remu_pool_truth": res["remu_true"],
        "remu_sm_reference_only": res["remu_sm"],
        "closure_percent_before_tail_correction": 100*(_ratio(res["remu"], res["remu_true"])-1),
        "closure_percent_after_tail_correction": _clc,
        "historical_statistical_error_percentage_points": _cle,
        "historical_anchor_error_percentage_points": _anc,
        "fit_region_truth": TRUE, "fit_region_fitted": FITTED_FR,
        "weighted_histogram_errors": ERR,
        "historical_toy_yield_errors": ERRY,
        "measurement": res,
        "uncertainty_recipe": {
            "histogram_errors": "sqrt(sum of squared event weights)",
            "toys": 1500, "random_seed": 11,
            "resampling": "Independent additive Gaussian bin fluctuations; not event bootstrap.",
            "toy_fit": "Flat sideband mean subtraction in both bins, followed by NNLS in the safe region.",
            "limitations": [
                "The historical central fit uses a low-bin sideband slope; toy error refits use flat subtraction.",
                "Historical amplitude errors are divided directly by fit-region truth integrals for plotted ratios.",
                "Historical truth errors sum squared weights across the full selected time window, while central component truth integrals use the fit region.",
                "Tail fraction is treated as exact in the closure statistical error.",
                "The 0.04 percentage-point anchor error is a fixed historical estimate, not derived from the supplied inputs."
            ]},
        "data_contents_hashed": False,
    }
    undefined = []
    metrics = _json_ready(metrics, undefined=undefined)
    metrics["undefined_metrics"] = undefined
    metrics["status"] = "completed" if not undefined else "completed_with_undefined_metrics"
    (output_dir / "fig8_time_fit_fixed.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pie", type=Path, required=True, help="Combined pion-to-positron benchmark event summary parquet")
    parser.add_argument("--michel", type=Path, required=True, help="Combined Michel benchmark event summary parquet")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PDF, PNG and metrics JSON")
    args = parser.parse_args()
    run(args.pie, args.michel, args.output_dir)


if __name__ == "__main__":
    main()
