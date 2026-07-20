"""Pedagogical figure: the two simple regression losses and their robustness.

(a) The Huber loss against the squared error. The two agree in the quadratic
    core |Delta| < delta, where the shrinking gradient lets the head settle
    precisely on the truth. Beyond delta the Huber loss grows linearly, so its
    gradient (dashed) saturates at +-delta: a badly mismeasured outlier event
    pulls on the weights with a bounded force instead of the unbounded pull of
    the squared error. One pathological event cannot dominate a batch update.
(b) The cosine loss for the positron direction, 1 - cos(psi). Near alignment
    the loss is locally quadratic in psi (gentle, precision-friendly), and it
    saturates at 2 for an anti-aligned prediction. It measures only the angle:
    the prediction is normalized to a unit vector, so no gradient is spent on
    a meaningless magnitude.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import PAL, save

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

# (a) Huber vs squared error ----------------------------------------------
ax = axes[0]
d = np.linspace(-3.2, 3.2, 500)
delta = 1.0
H = np.where(np.abs(d) < delta, 0.5 * d**2, delta * (np.abs(d) - 0.5 * delta))
ax.plot(d, 0.5 * d**2, lw=2, color="0.55", label=r"squared error $\frac{1}{2}\Delta^2$")
ax.plot(d, H, lw=2, color=PAL["blue"], label=r"Huber $H(\Delta)$, $\delta=1$")
ax.plot(d, np.clip(d, -delta, delta), lw=1.8, ls="--", color=PAL["red"],
        label=r"Huber gradient $H'(\Delta)$")
for s in (-1, 1):
    ax.axvline(s * delta, color="0.6", lw=0.9, ls=":")
ax.text(1.0, -1.25, r"$+\delta$", ha="center", fontsize=9, color="0.4")
ax.text(-1.0, -1.25, r"$-\delta$", ha="center", fontsize=9, color="0.4")
ax.set_ylim(-1.4, 5.2)
ax.set_xlabel(r"residual  $\Delta = \hat{r} - r$")
ax.set_ylabel("loss / gradient")
ax.set_title("(a) Huber loss: precise core, bounded tails", fontsize=10.5)
ax.legend(fontsize=8.5, loc="lower right")

# (b) cosine loss ----------------------------------------------------------
ax = axes[1]
psi = np.linspace(0, np.pi, 400)
ax.plot(np.degrees(psi), 1 - np.cos(psi), lw=2, color=PAL["green"])
ax.plot(np.degrees(psi), psi**2 / 2, lw=1.6, ls="--", color="0.5",
        label=r"small-angle limit $\psi^2/2$")
ax.set_ylim(0, 2.15)
ax.set_xlabel(r"angle between predicted and true direction  $\psi$ (deg)")
ax.set_ylabel(r"$1-\cos\psi$")
ax.set_title("(b) cosine loss for the positron direction", fontsize=10.5)
ax.legend(fontsize=9, loc="upper left")

fig.tight_layout()
print(save(fig, "fig_robust_losses"))
