"""Pedagogical figure: the chain-matching loss over two candidate positron slices.

Setting: slice 1 is the TRUE chain positron and slice 2 is truly unrelated.
p1 and p2 are the role head's positron probabilities for the two slices (the
muon role is taken as negligible, so slice 2's "none" probability is 1 - p2).
All four panels share one color scale. Coefficients are the trained values
lambda_comp = 0.2 and lambda_excl = 10 of the chain loss equation.

(a) The cross-entropy, (1/2)[-ln p1 - ln(1 - p2)]. It diverges as p1 -> 0, so
    predicting nothing is NOT free: the model must commit probability to the
    true slice. It also punishes probability on the unrelated slice.
(b) The weighted composition penalty 0.2*ReLU(p1 + p2 - 1)^2, nearly invisible
    on the trained scale (contours mark its shape): a guard against jointly
    over-committed slices.
(c) The weighted exclusivity penalty 10*min(p1, p2): any positron probability
    on the second slice is charged heavily.
(d) The total. The minimum sits at the decisive correct assignment
    (p1, p2) = (1, 0), marked with a star. Point A = indecision (0.55, 0.50):
    the exclusivity term makes it expensive even though both slices pass a 0.5
    threshold. Point B = over-commitment (0.95, 0.40). Guessing (0, 0) is
    excluded by the cross-entropy wall on the left edge.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import PAL, save

eps = 1e-3
p = np.linspace(eps, 1 - eps, 300)
P1, P2 = np.meshgrid(p, p)
L_COMP, L_EXCL = 0.2, 10.0

ce = 0.5 * (-np.log(P1) - np.log(1 - P2))
comp_w = L_COMP * np.maximum(P1 + P2 - 1, 0) ** 2
excl_w = L_EXCL * np.minimum(P1, P2)
constr = comp_w + excl_w
total = ce + constr
VMAX = 8.0

pts = [("A", 0.55, 0.50), ("B", 0.95, 0.40)]

fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), constrained_layout=True)
panels = [
    (axes[0], ce, r"(a) cross-entropy: $\frac{1}{2}[-\ln p_1 - \ln(1-p_2)]$"),
    (axes[1], constr, r"(b) constraints: $\lambda_\mathrm{comp}\,\mathrm{ReLU}(p_1{+}p_2{-}1)^2 + \lambda_\mathrm{excl}\min(p_1,p_2)$"),
    (axes[2], total, r"(c) total: minimum at the decisive truth"),
]
for ax, Z, title in panels:
    im = ax.pcolormesh(P1, P2, Z, cmap="Blues", vmin=0, vmax=VMAX, shading="auto")
    ax.plot([0, 1], [1, 0], ls="--", lw=1.2, color="0.4")
    if Z is constr:
        ax.text(0.14, 0.80, "$p_1+p_2=1$", rotation=-38, fontsize=8.5, color="0.35")
    if Z is total:
        for lab, x, y in pts:
            ax.plot(x, y, "o", ms=7, mfc=PAL["red"], mec="white", mew=1.2, zorder=5)
            ax.text(x - 0.055, y + 0.045, lab, fontsize=10, color=PAL["red"],
                    fontweight="bold", ha="center",
                    bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="none", alpha=0.8))
        ax.plot(1 - eps, eps, "*", ms=17, mfc=PAL["green"], mec="white",
                mew=1.0, zorder=6, clip_on=False)
        ax.plot(eps, eps, "s", ms=8, mfc="0.25", mec="white", mew=1.0,
                zorder=6, clip_on=False)
    ax.set_xlabel("$p_1$  (positron prob., true positron slice)")
    ax.set_ylabel("$p_2$  (positron prob., unrelated slice)")
    ax.set_title(title, fontsize=10.5)
    ax.set_aspect("equal")
    ax.grid(False)

cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
cb.set_label("loss (shared scale, saturated above 8)", fontsize=9)

print(save(fig, "fig_chain_constraints"))
