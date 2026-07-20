"""Pedagogical figure: quantile regression with the scale-attenuated pinball loss.

Two panels:
(a) The pinball loss rho_q(e) for q = 0.16, 0.50, 0.84. The asymmetric hinge is
    what makes the minimizer sit at the requested quantile: for q = 0.84 an
    under-prediction costs 0.84 per unit while an over-prediction costs 0.16,
    so the optimum sits where 84% of the targets lie below the prediction.
(b) The scale-attenuated loss L(sigma) = rho/sigma + log(sigma) as a function
    of the predicted interval width sigma, for three fixed pinball errors rho.
    In the implementation sigma is not a separate output: it is the gap
    between the predicted quantiles (median minus 16%, 84% minus median), so
    this term acts directly on the interval width. The minimum sits at
    sigma* = rho: the rho/sigma wall punishes an interval narrower than the
    actual errors, the log(sigma) climb punishes indiscriminate width, and a
    calibrated interval is the loss minimum rather than a constraint.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import PAL, save

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))

# (a) pinball hinges ------------------------------------------------------
ax = axes[0]
e = np.linspace(-2.5, 2.5, 400)
for q, c in [(0.16, PAL["blue"]), (0.50, PAL["truth"]), (0.84, PAL["red"])]:
    rho = np.maximum(q * e, (q - 1) * e)
    ax.plot(e, rho, color=c, lw=2, label=f"$q={q:.2f}$")
ax.set_ylim(0, 2.6)
ax.set_xlabel(r"residual  $y - \hat{y}_q$")
ax.set_ylabel(r"$\rho_q$")
ax.set_title("(a) pinball loss: asymmetric hinge", fontsize=10.5)
ax.legend(fontsize=9, loc="upper center")

# (b) attenuated loss vs predicted interval width -------------------------
ax = axes[1]
sig = np.linspace(0.12, 5.0, 500)
for rho0, c in [(0.5, PAL["green"]), (1.0, PAL["blue"]), (2.0, PAL["orange"])]:
    L = rho0 / sig + np.log(sig)
    ax.plot(sig, L, lw=2, color=c, label=rf"$\rho={rho0:g}$")
    ax.plot(rho0, 1 + np.log(rho0), "o", ms=6, color=c, zorder=5)
ax.set_ylim(0, 4.4)
ax.set_xlabel(r"interval width $\sigma$ (gap between predicted quantiles)")
ax.set_ylabel(r"$\mathcal{L} = \rho/\sigma + \log\sigma$")
ax.set_title("(b) attenuation: calibrated width is the minimum", fontsize=10.5)
ax.legend(fontsize=9, loc="upper right")

fig.tight_layout()
print(save(fig, "fig_pinball_attenuated"))
