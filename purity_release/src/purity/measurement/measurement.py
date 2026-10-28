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


def _analytic_safe_fraction(cls, bins, fit_lo=5.0):
    """Fraction of the ANALYTIC decay law lying in the safe fit region, on the fit's own binning.

    The safe-region fit determines an amplitude A for a law normalised over t' >= 0, from the
    observed safe-region yield: A = w_safe / f_analytic.  Converting A back to the channel total
    therefore needs eff = A/w_tot = (w_safe / f_analytic) / w_tot = eff0 * frac_safe / f_analytic
    -- NOT eff0 / frac_safe.  The two agree only when frac_safe ~ f_analytic ~ 1, which is true
    for the muon chain (0.9993) and false for the short-lived pion (0.86 vs 0.93), so the old
    form was invisible in michel-dominated truth closure and biased the pie leg by ~2.6%.
    """
    ctr = 0.5 * (bins[:-1] + bins[1:])
    pie_b, mic_b = _decay_bases(bins)
    b = pie_b if cls == "pie" else mic_b
    pos, safe = ctr >= 0.0, ctr >= fit_lo
    tot = b[pos].sum()
    return float(b[safe].sum() / tot) if tot > 0 else 1.0


def channel_efficiency(df, t_stop=4.0, fit_lo=5.0, bins=None):
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
        # The safe region must be the SAME set of events the fit uses.  time_fit_bin selects
        # HISTOGRAM BINS whose centre >= fit_lo, so with 4 ns bins the effective event-level
        # threshold is the left EDGE of the first such bin, not fit_lo itself.  Using fit_lo
        # here mismatches the fit by one bin (~3.5% of the pie), which the eff_notime ratio
        # then propagates straight into the pion leg.
        if bins is not None:
            ctr_ = 0.5 * (bins[:-1] + bins[1:])
            keep = np.nonzero(ctr_ >= fit_lo)[0]
            lo_edge = float(bins[keep[0]]) if keep.size else fit_lo
        else:
            lo_edge = fit_lo
        safe = selm & np.isfinite(tt) & (tt >= lo_edge)   # clean decays in the safe fit region
        frac_safe = float(d.w[safe].sum()) / selw if selw > 0 else 1.0
        e = selw / tot if tot > 0 else 0.0
        # eff_notime converts the safe-region FIT AMPLITUDE back to the channel total.
        f_an = _analytic_safe_fraction(c, bins, fit_lo) if bins is not None else 1.0
        eff_nt = (e * frac_safe / f_an) if f_an > 0 else e
        eff[c] = dict(total_w=tot, sel_w=selw, eff=e,
                      frac_safe=frac_safe, f_analytic=f_an,
                      eff_notime=eff_nt,
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


def time_fit_bin(sel_bin, bins, fit_pileup=True, include_pie=True, t_stop=4.0, fit_lo=5.0,
                 pileup_slope=False, pileup_free=False, pileup_slope_frac=None):
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

    # Step 1: accidental level from the pre-stop (t' < 0) sideband.  The sideband is the only
    # region that determines it: past the stop a flat accidental is nearly degenerate with the
    # Bateman, which falls only exp(-500/2197) = 0.80 across the whole window, so letting the
    # level float in the global fit hands most of the muon chain to the accidental (measured:
    # michel component collapses to 0.278 and the closure goes from -1.75% to -6.85%).
    #
    # Constant by default; with pileup_slope=True a straight line is fitted to the sideband and
    # EXTRAPOLATED across the window.  The accidental rate is NOT flat: on the stub-fixed,
    # RMD-purged sample it RISES by (11.3 +- 1.8)% across 800 ns (6.3 sigma, measured in the low
    # bin), so a constant anchored on the early sideband UNDER-predicts the in-window level.  The
    # in-window/sideband ratio is 1.069 (low bin) and 1.221 (high bin), and the flat fit duly
    # reports the accidental component low by exactly those factors: 0.936 and 0.821.  Fitting the
    # slope recovers 0.998 and 0.902 and moves the closure from -1.75% to -1.37%.
    # (An earlier comment here recorded a FALLING rate and a 0.855 ratio.  That was measured on the
    # pre-RMD-purge sample, whose high-energy accidentals were radiative-muon-decay pileup donors
    # entering unweighted; it does not describe the current samples.)
    #
    # CAVEAT, high bin: its sideband averages 0.65 events per 4 ns bin, so the line is fitted to a
    # field of zeros and ones, its slope is only 1.7 sigma, and it is then levered 500 ns past the
    # fit region.  Prefer pileup_slope_frac to import the well-measured low-bin fractional slope.
    if fit_pileup and neg.any():
        if pileup_slope_frac is not None and neg.sum() >= 1:
            # Shape imported from elsewhere (normally the low bin); the level is still anchored on
            # THIS bin's own sideband, so only the shape is shared.
            # NB the justification once written here -- that an accidental's arrival time cannot
            # depend on its energy, so the fractional slope must be common to both bins -- is FALSE
            # of the SELECTED accidental, and measurably so.  Fit-region/sideband level ratio on the
            # ep19fl sample: 0.912 +- 0.003, flat across every 10 MeV slice from 10 to 56 MeV, but
            # 1.18 +- 0.12 above the 56 MeV split, i.e. the opposite sign at 2.2 sigma on 410
            # events.  Selection, not arrival, sets the shape.  The import is still the best
            # available choice for the high bin, whose own sideband slope is 1.7 sigma noise, but it
            # is a choice to be quoted with a systematic, not an identity: it drives the high-bin
            # accidental to 0.77 of truth (0.904/1.179 reproduces it to four decimals), which the
            # Bateman then absorbs, and it is worth 0.025 pp on R_e/mu.
            lvl = float(obs[neg].mean())
            mid = float(ctr[neg].mean())
            Barr = np.clip(lvl * (1.0 + pileup_slope_frac * (ctr - mid)), 0.0, None)
        elif pileup_slope and neg.sum() >= 4:
            yb = obs[neg]
            wgt = 1.0 / np.sqrt(np.maximum(yb, 1.0))
            co = np.polyfit(ctr[neg], yb, 1, w=wgt)
            Barr = np.polyval(co, ctr)
            Barr = np.clip(Barr, 0.0, None)
        else:
            Barr = np.full(len(ctr), float(obs[neg].mean()))
    else:
        Barr = np.zeros(len(ctr))
    B = Barr                       # per-bin accidental level (constant array if no slope)
    npile = float(Barr.sum())

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
    if pileup_free:
        # Let the accidental NORMALISATION float in the fit instead of fixing it from the
        # sideband.  The three shapes (prompt exponential / Bateman / flat) are distinguishable
        # and the late-time bins constrain the flat term, so no extrapolation from the pre-stop
        # region is required -- neither a constant (which over-predicts by ~18%) nor a linear
        # extrapolation of the sideband slope (which under-predicts by ~15%).
        flat_n = np.ones(len(ctr)) / max(len(ctr), 1)
        comps.append(flat_n[fit]); names.append("acc")
        tgt = obs[fit].copy()
    else:
        tgt = obs[fit] - B[fit]                          # subtract the accidental level
    A = np.vstack(comps).T
    coef = np.zeros(len(comps))
    if A.sum() > 0 and obs[fit].sum() > 0:
        coef, _ = nnls(A, tgt)
    d = dict(zip(names, coef))
    npie, nmue = float(d.get("pie", 0.0)), float(d.get("mue", 0.0))
    if pileup_free:
        nacc = float(d.get("acc", 0.0))
        Barr = nacc * (np.ones(len(ctr)) / max(len(ctr), 1))
        npile = nacc

    return dict(N_pie=npie, N_mue=nmue, N_pileup=npile, include_pie=include_pie,
                true_pie=float(sel_bin.w[(sel_bin.cls == "pie") & sel_bin.clean].sum()),
                true_mue=float(sel_bin.w[(sel_bin.cls == "mue") & sel_bin.clean].sum()),
                true_pileup=float(sel_bin.w[sel_bin.toy_wrong].sum()) if "toy_wrong" in sel_bin.columns else 0.0,
                obs=obs, fit_pie=npie * pie_n, fit_mue=nmue * mic_n,
                fit_pileup=Barr, fit_total=npie * pie_n + nmue * mic_n + Barr)


def measure(df, cfg, n_time_bins=200, fit_pileup=True, t_stop=4.0, fit_lo=5.0,
            pileup_slope=False, pileup_free=False, pileup_slope_frac=None):
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
    # Bin edges live in the SHIFTED coordinate t' = t - t_stop, the same one time_fit_bin
    # histograms in.  Binning the raw window instead leaves the last bin structurally unfillable
    # (selection keeps t <= window[1], so t' <= window[1] - t_stop) while the accidental line is
    # still summed over it, and drops the first t_stop ns of the sideband below the range.  On the
    # 4 ns binning that phantom bin inflated the fitted low-bin accidental by ~3400 events and
    # masked a real 1.3% under-prediction as an apparent 0.6% one.
    bins = np.linspace(cfg.window[0] - t_stop, cfg.window[1] - t_stop, n_time_bins + 1)

    per_bin, N_pie_sel, N_mue_sel, N_pileup_sel = {}, 0.0, 0.0, 0.0
    for b in ("high", "low"):
        r = time_fit_bin(sel[sel.ebin == b], bins, fit_pileup=fit_pileup,
                         include_pie=(b == "high"), t_stop=t_stop, fit_lo=fit_lo,
                         pileup_slope=pileup_slope, pileup_free=pileup_free,
                         pileup_slope_frac=(pileup_slope_frac.get(b)
                                            if isinstance(pileup_slope_frac, dict)
                                            else pileup_slope_frac))
        per_bin[b] = r
        N_pie_sel += r["N_pie"]; N_mue_sel += r["N_mue"]; N_pileup_sel += r["N_pileup"]

    eff = channel_efficiency(df, t_stop=t_stop, fit_lo=fit_lo, bins=bins)
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
