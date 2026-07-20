"""Pedagogical figure: exploding and vanishing gradients, and the fix.

During backpropagation the gradient reaching an early layer is the product of the
local derivatives of every layer above it. If those factors are on average below
one the product shrinks toward zero (vanishing); above one it grows without bound
(exploding). Normalization layers and residual connections hold the per-layer
factor near one, so the gradient stays order-one all the way back to the first
layer. Each curve is the running product of per-layer factors drawn from a
distribution with the stated mean.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import PAL, save

rng = np.random.default_rng(11)
L = 40                                   # number of layers
layers = np.arange(1, L + 1)             # 1 = earliest (input) ... L = output


def grad_to_layer(mean, spread):
    """Gradient magnitude reaching each layer: product of the per-layer factors
    of all layers above it. Backprop starts at the output with magnitude 1."""
    factors = rng.normal(mean, spread, L)
    factors[-1] = 1.0                    # output layer: gradient seeded at 1
    g = np.ones(L)
    for l in range(L - 2, -1, -1):       # accumulate downward toward the input
        g[l] = g[l + 1] * factors[l + 1]
    return g


curves = [
    ("exploding  (factor $\\approx 1.25$)", grad_to_layer(1.25, 0.05), PAL["red"]),
    ("stabilized  (norm + skip, factor $\\approx 1$)", grad_to_layer(1.00, 0.05), PAL["green"]),
    ("vanishing  (factor $\\approx 0.75$)", grad_to_layer(0.75, 0.05), PAL["blue"]),
]

fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.axhline(1.0, color="0.5", lw=0.9, ls="--", zorder=1)
for label, g, c in curves:
    ax.plot(layers, g, color=c, lw=2.2, marker="o", ms=3.0, label=label)

ax.set_yscale("log")
ax.set_xlabel("layer  (input  $\\rightarrow$  output)")
ax.set_ylabel("gradient magnitude reaching layer")
ax.set_xlim(1, L)
ax.legend(fontsize=9, loc="upper right", framealpha=1.0)

fig.tight_layout()
paths = save(fig, "fig_gradient_stability")
print("input-layer gradient  explode=%.2e  stable=%.2e  vanish=%.2e"
      % (curves[0][1][0], curves[1][1][0], curves[2][1][0]))
print("wrote:", *paths, sep="\n  ")
