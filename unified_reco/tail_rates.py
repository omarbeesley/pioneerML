"""Single source of truth for the tail-analysis channel rates and the tail rescale.

Every rate is quoted per stopped pion, and every entry records the STAGE at which its
external input is defined, because imposing a ratio before or after the acceptance changes
it by the ratio of the two acceptance efficiencies.

  pie    BR = 1.2352e-4                       per stopped pion, pre-selection
  piDIF  1.6e-3                               per stopped pion, pre-selection, measured from
                                              the generation itself: 42.0% of the 100 ps-capped
                                              decays occur in flight -> t_stop = 42 ps ->
                                              P = t_stop/tau_pi = 1.61e-3 (= gen weight
                                              3.836e-3 x 0.420); piDIF outnumbers pie 13:1
  michel F_MIC_WIN = 0.194                    per stopped pion, in the readout window; when
                                              radiative decays are removed by analysis policy
                                              their share leaves with them (do not redistribute)
  muDIF  pie / 18                             AT ACCEPTANCE; measured the same way: 30.4% of
                                              the 50 ps-capped decays occur in flight ->
                                              t_stop(mu) = 15.2 ps -> P = 6.9e-6 per stopped
                                              pion = pie/17.9, and the mu/pie acceptances
                                              (59.8/60.7%) leave the ratio unchanged

REAL_TAIL is the physical pie tail fraction, larger than the simulated one because the
non-radiative samples carry no radiative pie decays; contamination numbers rescale the
simulated pie low bin to it (the convention of mudif_tradeoff.py).
"""
import numpy as np

BR_PIE = 1.2352e-4
R_PIDIF_ABS = 1.6e-3
TAU_PI, TAU_MU = 26.033, 2196.981
F_MIC_WIN = 1 - (TAU_MU * np.exp(-500 / TAU_MU) - TAU_PI * np.exp(-500 / TAU_PI)) / (TAU_MU - TAU_PI)
R_MUDIF_AT_ACC = 1.0 / 18.0
REAL_TAIL = 0.006
ES = 56.0
PIDIF_OVER_PIE = R_PIDIF_ABS / BR_PIE          # 13.0


def _shape(gw, n):
    """Generator weights if they carry shape, flat otherwise."""
    if gw is None:
        return np.ones(n)
    g = np.asarray(gw, dtype=float)
    g = np.where(np.isfinite(g), g, 0.0)
    return g if np.unique(g).size > 1 else np.ones(n)


def weights(n, total_rate, gw=None, mask=None):
    """Per-event weights whose sum over `mask` (default: all rows) is `total_rate`."""
    w = _shape(gw, n)
    m = np.ones(n, bool) if mask is None else mask
    s = w[m].sum()
    return w * (total_rate / s) if s > 0 else np.zeros(n)


def tail_rescale(E_pie_acc):
    """Factor bringing the simulated pie low-bin fraction up to the physical tail."""
    frac = (np.asarray(E_pie_acc) < ES).mean()
    return REAL_TAIL / frac if frac > 0 else 1.0


def contamination(bkg_lo, pie_lo, ts):
    """Background share of the low bin, with the pie low bin rescaled to the physical tail."""
    denom = bkg_lo + ts * pie_lo
    return 100.0 * bkg_lo / denom if denom > 0 else 0.0
