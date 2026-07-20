"""The PIONEER R_e/mu measurement on the standardized event frame (see io.py).

Chain:  acceptance/window efficiency (per channel)  ->  fill gen_weight-weighted energy
spectrum  ->  56 MeV high/low split  ->  per-bin time-shape fit (prompt pi->e vs delayed
pi->mu->e) for N_pie / N_mue  ->  efficiency-correct  ->  R_e/mu.

Time fit = template fit: the pure pie pool gives the PROMPT time shape, the pure michel
pool gives the DELAYED shape; the combined (gen_weight-weighted) time histogram in each
energy bin is fit to a * prompt + b * delayed (non-negative least squares) -> a=N_pie,
b=N_mue. In truth mode this closes ~exactly; in reco mode it tests time separation under
smearing.  Truth N_pie/N_mue (from the class labels) are carried alongside for closure.
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from . import io


def channel_efficiency(df, t_stop=4.0, fit_lo=5.0):
    """Per-class selection efficiency.

    'eff'        = Sigma w(accepted & in-window & clean) / Sigma w(all).  Its denominator
                   includes the pie that the merge-window deletes, so it FOLDS IN the
                   early-time/merge survival loss.
    'eff_notime' = eff / frac_safe, where frac_safe = fraction of the clean-selected sample
                   whose (stop-shifted) reco time lands in the SAFE fit region t' >= fit_lo.
                   The safe-region time fit (time_fit_bin) EXTRAPOLATES the analytic decay law
                   back through the near-stop loss, so the extrapolated yields must be paired
                   with this TIME-EXCLUDED efficiency -- dividing them by 'eff' would
                   double-count the loss the extrapolation already recovers.
    """
    eff = {}
    for c in ("pie", "mue"):
        d = df[df.cls == c]
        tot = float(d.w.sum())
        # signal efficiency counts only correctly-reconstructed (clean) triggers: a wrong-
        # positron event is 'selected' but its signal is lost to the pileup component.
        selm = (d.accepted.to_numpy() & d.in_window.to_numpy() & d.clean.to_numpy())
        selw = float(d.w[selm].sum())
        tt = d.t.to_numpy() - t_stop                      # shift to the pion stop
        safe = selm & np.isfinite(tt) & (tt >= fit_lo)    # clean decays in the safe fit region
        frac_safe = float(d.w[safe].sum()) / selw if selw > 0 else 1.0
        e = selw / tot if tot > 0 else 0.0
        eff[c] = dict(total_w=tot, sel_w=selw, eff=e,
                      frac_safe=frac_safe,
                      eff_notime=(e / frac_safe if frac_safe > 0 else e),
                      n_total=int(len(d)), n_sel=int(selm.sum()))
    return eff


def _whist(t, w, bins):
    t = np.asarray(t, float); w = np.asarray(w, float)
    ok = np.isfinite(t)
    h, _ = np.histogram(t[ok], bins=bins, weights=w[ok])
    return h


# --- Physical decay time spectra: FIXED lifetimes, no empirical templates ---
TAU_PI = 26.033      # ns, charged-pion mean lifetime
TAU_MU = 2196.981    # ns, muon mean lifetime


def _decay_bases(bins):
    """Per-bin-integrated analytic decay time distributions for t >= 0 (zero for t < 0).

    pie (pi->e nu)     : prompt single pion-lifetime exponential, (1/tau_pi) e^{-t/tau_pi}.
    michel (pi->mu->e) : the two-step decay chain (pion then muon lifetime),
                         f(t) = [e^{-t/tau_mu} - e^{-t/tau_pi}] / (tau_mu - tau_pi) -- 0 at t=0,
                         rises on ~tau_pi, falls on ~tau_mu.

    Normalized over the t>0 fit window (sum = 1) so a fitted amplitude is that channel's
    in-window yield.  Detector time resolution (~100 ps) << the lifetimes (26 ns, 2197 ns),
    so the analytic shapes are used directly -- these are NOT empirical templates.
    """
    lo, hi = bins[:-1], bins[1:]
    ctr = 0.5 * (lo + hi)
    pos = ctr >= 0.0
    a, b = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
    iexp = lambda tau: np.exp(-a / tau) - np.exp(-b / tau)   # int (1/tau) e^{-t/tau} over [a,b]
    pie = np.where(pos, iexp(TAU_PI), 0.0)
    mic = np.where(pos, (TAU_MU * iexp(TAU_MU) - TAU_PI * iexp(TAU_PI)) / (TAU_MU - TAU_PI), 0.0)
    pie = pie / pie.sum() if pie.sum() > 0 else pie
    mic = mic / mic.sum() if mic.sum() > 0 else mic
    return pie, mic


def time_fit_bin(sel_bin, bins, fit_pileup=True, include_pie=True, t_stop=4.0, fit_lo=5.0):
    """Safe-region time fit of one energy bin -- analytic decay laws, extrapolated to the stop.

    Times are shifted to the pion stop (t' = t - t_stop) so the analytic decay laws start at
    t'=0.  The near-stop region t' in [0, fit_lo] is a mess (merge-window truncation, time-
    slicing artifacts), so it is EXCLUDED from the fit.  The fit uses only the SAFE region
    t' >= fit_lo, and the analytic law -- normalized over its FULL t' >= 0 support -- then
    EXTRAPOLATES back to the true t'=0 population.  So N_pie / N_mue are the TRUE in-acceptance
    yields (including the merge-deleted early decays), NOT the survivor counts; pair them with
    the TIME-EXCLUDED efficiency (channel_efficiency eff_notime) in measure().  Needs fine time
    bins so the fit_lo cut and the ~6 ns onset are resolved (coarse bins bias the extrapolation).

    Step 1  the pre-stop sideband (t' < 0) is PURE flat accidental -> per-bin mean fixes B.
    Step 2  fit the SAFE region t' >= fit_lo to fixed-B flat + the analytic pion (high bin
            only) and muon-chain laws (non-negative yields); include_pie=False in the low bin.
    """
    ctr = 0.5 * (bins[:-1] + bins[1:])
    t = np.asarray(sel_bin.t, float) - t_stop           # shift to the pion stop
    obs = _whist(t, sel_bin.w, bins)
    neg, pos = ctr < 0.0, ctr >= 0.0
    fit = ctr >= fit_lo                                  # SAFE region (past the near-stop mess)

    # Step 1: flat accidental level from the pre-stop (t' < 0) sideband
    B = float(obs[neg].mean()) if (fit_pileup and neg.any()) else 0.0
    npile = B * len(ctr)

    # Step 2: analytic laws normalized over the FULL t' >= 0 support so the fitted amplitude
    # is the EXTRAPOLATED total population; fit only the safe region t' >= fit_lo.
    pie_b, mic_b = _decay_bases(bins)
    def _norm(x):
        s = x[pos].sum()
        return x / s if s > 0 else x
    pie_n, mic_n = _norm(pie_b), _norm(mic_b)
    comps, names = [], []
    if include_pie:
        comps.append(pie_n[fit]); names.append("pie")
    comps.append(mic_n[fit]); names.append("mue")
    tgt = obs[fit] - B                                   # subtract the fixed accidental level
    A = np.vstack(comps).T
    coef = np.zeros(len(comps))
    if A.sum() > 0 and obs[fit].sum() > 0:
        coef, _ = nnls(A, tgt)
    d = dict(zip(names, coef))
    npie, nmue = float(d.get("pie", 0.0)), float(d.get("mue", 0.0))

    return dict(N_pie=npie, N_mue=nmue, N_pileup=npile, include_pie=include_pie,
                true_pie=float(sel_bin.w[(sel_bin.cls == "pie") & sel_bin.clean].sum()),
                true_mue=float(sel_bin.w[(sel_bin.cls == "mue") & sel_bin.clean].sum()),
                true_pileup=float(sel_bin.w[sel_bin.toy_wrong].sum()) if "toy_wrong" in sel_bin.columns else 0.0,
                obs=obs, fit_pie=npie * pie_n, fit_mue=nmue * mic_n,
                fit_pileup=np.full(len(ctr), B), fit_total=npie * pie_n + nmue * mic_n + B)


def measure(df, cfg, n_time_bins=200, fit_pileup=True, t_stop=4.0, fit_lo=5.0):
    """Run the full measurement. Returns a results dict (numbers + per-bin fit histograms).

    Safe-region extrapolation fit (see time_fit_bin): times are shifted to the pion stop, the
    near-stop mess (t' < fit_lo) is excluded, and the analytic pion/muon decay laws fit the
    safe region and extrapolate to the true t'=0 population.  The pie (prompt) component is fit
    ONLY in the high bin; the low bin fits michel + flat accidental only (pi->e is monoenergetic
    ~70 MeV, no signal below 56 MeV).  Extrapolated yields are corrected by the TIME-EXCLUDED
    efficiency (eff_notime) so the merge/early-time loss is not double-counted.  Fine time bins
    (default 200 -> 4 ns) are required to resolve the fit_lo cut and the ~6 ns onset.
    """
    sel = io.selected(df, require_window=True)
    sel = sel[sel.E <= cfg.e_max]                     # drop unphysical high-E
    sel = sel.assign(ebin=io.energy_bin(sel.E.to_numpy(), cfg.e_split))
    bins = np.linspace(cfg.window[0], cfg.window[1], n_time_bins + 1)

    per_bin, N_pie_sel, N_mue_sel, N_pileup_sel = {}, 0.0, 0.0, 0.0
    for b in ("high", "low"):
        r = time_fit_bin(sel[sel.ebin == b], bins, fit_pileup=fit_pileup,
                         include_pie=(b == "high"), t_stop=t_stop, fit_lo=fit_lo)
        per_bin[b] = r
        N_pie_sel += r["N_pie"]; N_mue_sel += r["N_mue"]; N_pileup_sel += r["N_pileup"]

    eff = channel_efficiency(df, t_stop=t_stop, fit_lo=fit_lo)
    # TIME-EXCLUDED efficiency: the safe-region fit already extrapolates through the
    # early-time/merge loss, so correcting by 'eff' (which folds that loss in) would double-count.
    ep, em = eff["pie"]["eff_notime"], eff["mue"]["eff_notime"]
    N_pie_corr = N_pie_sel / ep if ep > 0 else np.nan
    N_mue_corr = N_mue_sel / em if em > 0 else np.nan
    remu = N_pie_corr / N_mue_corr if N_mue_corr else np.nan
    remu_true = eff["pie"]["total_w"] / eff["mue"]["total_w"]

    return dict(cfg_mode=cfg.mode, bins=bins, per_bin=per_bin, eff=eff, fit_pileup=fit_pileup,
                N_pie_sel=N_pie_sel, N_mue_sel=N_mue_sel, N_pileup_sel=N_pileup_sel,
                N_pie_corr=N_pie_corr, N_mue_corr=N_mue_corr,
                remu=remu, remu_true=remu_true,
                remu_sm=1.2352e-4)


def energy_spectrum(df, cfg, n_e_bins=80, e_range=(0.0, 80.0)):
    """gen_weight-weighted energy spectra for pie, mue, and combined (selected events)."""
    sel = io.selected(df, require_window=True)
    edges = np.linspace(*e_range, n_e_bins + 1)
    out = {"edges": edges}
    for c in ("pie", "mue"):
        d = sel[sel.cls == c]
        out[c], _ = np.histogram(d.E, bins=edges, weights=d.w)
    out["combined"] = out["pie"] + out["mue"]
    return out
