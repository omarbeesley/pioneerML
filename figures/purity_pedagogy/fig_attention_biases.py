"""Pedagogical figure: the two physics biases steering the event builder.

(a) The temporal bias as the attention factor exp(B^t) against the track-to-
    cluster time difference. The window peaks at the expected time of flight
    tau_TOF = 0.5 ns, not at zero: a cluster is expected LATER than its track
    by the flight time. The width is energy-dependent, sigma(E) = sigma_0 +
    sigma_1/sqrt(E): a low-energy cluster has poorer photostatistics timing
    and earns a wider window, a high-energy one a sharper gate. Widths use
    the initialization values (sigma_A = 1.0 ns, sigma_0 = 0.5 ns, sigma_1 =
    2.0 ns MeV^1/2); the trained values are learned from data.
(b) The directional bias B^d = cos(alpha)/sigma_theta(E) against the alignment
    between the pion-stop-to-cluster direction and the reconstructed positron
    direction. Higher cluster energy means less multiple scattering and a
    sharper-pointing shower, so sigma_theta shrinks and the preference for
    aligned clusters steepens. Illustrative initialization widths
    (sigma_theta0 = 0.5, sigma_theta1 = 1.0).
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import PAL, save

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
ECOL = [(2, PAL["red"]), (10, PAL["orange"]), (60, PAL["blue"])]

# (a) temporal coincidence window ----------------------------------------
ax = axes[0]
dt = np.linspace(-6, 7, 600)
TOF, sA = 0.5, 1.0
for E, c in ECOL:
    sC = 0.5 + 2.0 / np.sqrt(E)
    s2 = sA**2 + sC**2
    ax.plot(dt, np.exp(-(dt - TOF) ** 2 / (2 * s2)), lw=2, color=c,
            label=rf"$E={E}$ MeV")
ax.axvline(TOF, color="0.35", lw=1.0, ls="--")
ax.text(TOF + 0.18, 1.015, r"$\tau_\mathrm{TOF}$", fontsize=9, color="0.3")
ax.set_xlabel(r"$t_\mathrm{cluster} - t_\mathrm{track}$ (ns)")
ax.set_ylabel(r"attention factor  $e^{B^{t}}$")
ax.set_title("(a) temporal coincidence gate", fontsize=10.5)
ax.legend(fontsize=9, loc="upper left")

# (b) directional preference ----------------------------------------------
ax = axes[1]
cosa = np.linspace(-1, 1, 200)
for E, c in ECOL:
    st = 0.5 + 1.0 / np.sqrt(E)
    ax.plot(cosa, cosa / st, lw=2, color=c, label=rf"$E={E}$ MeV")
ax.axhline(0, color="0.5", lw=0.9)
ax.set_xlabel(r"alignment  $\cos\alpha = \hat{\mathbf{u}}_j\cdot\hat{\mathbf{p}}$")
ax.set_ylabel(r"directional bias  $B^{d}$")
ax.set_title("(b) directional cone", fontsize=10.5)
ax.legend(fontsize=9, loc="lower right")

fig.tight_layout()
print(save(fig, "fig_attention_biases"))
