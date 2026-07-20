"""Pedagogical figure: the pointwise activation functions used in the model.

ReLU and its smooth counterpart GELU drive the feed-forward blocks (GELU
throughout the transformer MLPs, ReLU in a few heads); the sigmoid squashes a
score into a per-hit probability. ReLU has a hard corner at the origin and is
unbounded above; GELU dips slightly negative before rising to meet it; the
sigmoid saturates to 0 and 1, where its gradient vanishes.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from figstyle import PAL, save

x = np.linspace(-4, 4, 600)
relu = np.maximum(0.0, x)
gelu = 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
sigmoid = 1.0 / (1.0 + np.exp(-x))

fig, ax = plt.subplots(figsize=(6.4, 4.1))

ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
ax.axvline(0.0, color="0.6", lw=0.8, zorder=1)

ax.plot(x, relu, color=PAL["blue"], lw=2.2, label=r"ReLU:  $\max(0,\,x)$")
ax.plot(x, gelu, color=PAL["green"], lw=2.2, label=r"GELU:  $x\,\Phi(x)$")
ax.plot(x, sigmoid, color=PAL["red"], lw=2.2,
        label=r"sigmoid:  $1/(1+e^{-x})$")

ax.set_xlim(-4, 4)
ax.set_ylim(-1.0, 4.0)
ax.set_xlabel("input  $x$")
ax.set_ylabel(r"activation  $\sigma(x)$")
ax.legend(fontsize=10, loc="upper left", framealpha=1.0)

fig.tight_layout()
paths = save(fig, "fig_activation_functions")
print("wrote:", *paths, sep="\n  ")
