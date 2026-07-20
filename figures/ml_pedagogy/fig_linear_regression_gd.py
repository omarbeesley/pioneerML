"""Pedagogical figure: fitting y = wx + b by iterative gradient descent.

Left panel  : noisy data with the model line drawn at a sequence of training
              iterations, coloured from the random initialisation to the
              converged best fit, with the closed-form (OLS) line for reference.
Right panel : mean-squared-error loss versus iteration, with the same
              iterations marked so the two panels can be read together.

The gradient descent is real: the analytic gradients of the MSE with respect to
w and b are computed at each step and used to update the parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import PAL, save

# ----- synthetic data from a known line y = m x + b + noise --------------------
rng = np.random.default_rng(7)
N = 40
m_true, b_true = 2.0, 1.0
# x is centred on zero so the fit is well conditioned: the slope and intercept
# then converge at the same rate, and the loss plateaus once the line is found,
# rather than crawling for hundreds of extra steps along a slow intercept mode.
x = np.sort(rng.uniform(-2.5, 2.5, N))
y = m_true * x + b_true + rng.normal(0.0, 1.3, N)

# closed-form (ordinary least squares) reference
A = np.vstack([x, np.ones_like(x)]).T
w_ols, b_ols = np.linalg.lstsq(A, y, rcond=None)[0]

# ----- gradient descent --------------------------------------------------------
w, b = -1.0, 6.0          # deliberately poor random initialisation
lr = 0.04
n_iter = 100
w_hist, b_hist, loss_hist = [], [], []
for _ in range(n_iter + 1):
    yhat = w * x + b
    err = yhat - y
    loss_hist.append(np.mean(err**2))
    w_hist.append(w)
    b_hist.append(b)
    grad_w = 2.0 * np.mean(err * x)
    grad_b = 2.0 * np.mean(err)
    w -= lr * grad_w
    b -= lr * grad_b

w_hist, b_hist, loss_hist = map(np.asarray, (w_hist, b_hist, loss_hist))

# the descent must actually land on the closed-form line for the figure to be
# honest; assert it before drawing
assert abs(w_hist[-1] - w_ols) < 0.01 and abs(b_hist[-1] - b_ols) < 0.02, \
    "gradient descent has not converged onto the closed-form line"

# iterations to highlight: the poor start, two intermediate lines, and the
# converged fit. Extra late marks would just pile onto the converged line, so
# only one is kept at the end.
marks = [0, 8, 25, n_iter]
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.08, 0.92, len(marks)))

# ----- figure ------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.9),
                               gridspec_kw=dict(width_ratios=[1.25, 1.0]))

# left: data + evolving fit lines
axL.scatter(x, y, s=22, color="0.25", zorder=3, label="data")
xs = np.array([x.min() - 0.2, x.max() + 0.2])
for c, k in zip(colors, marks):
    axL.plot(xs, w_hist[k] * xs + b_hist[k], color=c, lw=1.9,
             label=f"iter {k}", zorder=2)
axL.plot(xs, w_ols * xs + b_ols, color=PAL["truth"], ls="--", lw=1.4,
         zorder=4, label="closed-form fit")
axL.set_xlabel("$x$")
axL.set_ylabel("$y$")
axL.set_title("(a)  model line during training", fontsize=11)
axL.legend(fontsize=7.6, loc="upper left", ncol=2)

# right: loss curve
axR.plot(np.arange(n_iter + 1), loss_hist, color=PAL["blue"], lw=2.0, zorder=2)
for c, k in zip(colors, marks):
    axR.scatter([k], [loss_hist[k]], color=c, s=34, zorder=4,
                edgecolor="white", linewidth=0.6)
axR.set_yscale("log")
axR.set_xlabel("iteration")
axR.set_ylabel("MSE loss")
axR.set_title("(b)  loss versus iteration", fontsize=11)
axR.set_xlim(-8, n_iter + 8)

fig.tight_layout(w_pad=4.0)
paths = save(fig, "fig_linear_regression_gd")
print("converged  w=%.3f b=%.3f  |  OLS  w=%.3f b=%.3f  |  final MSE=%.3f"
      % (w_hist[-1], b_hist[-1], w_ols, b_ols, loss_hist[-1]))
print("wrote:", *paths, sep="\n  ")
