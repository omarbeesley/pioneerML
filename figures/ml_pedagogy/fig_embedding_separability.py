"""Pedagogical figure: an embedding turns a non-linearly separable problem
into a linearly separable one.

Signal is defined inside a radius R of the origin (x1^2 + x2^2 < R^2) and
background uniformly outside it. No straight line separates the two classes in
the plane (left). The map phi(x) = [x1, x2, x1^2 + x2^2] lifts the points into
three dimensions, where the single flat plane h3 = R^2 separates them (right).
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from figstyle import PAL, save

R = 1.0
rng = np.random.default_rng(3)

# signal: uniform inside the disk of radius R
n_sig = 65
rr = R * np.sqrt(rng.uniform(0.0, 1.0, n_sig))
th = rng.uniform(0.0, 2 * np.pi, n_sig)
sx1, sx2 = rr * np.cos(th), rr * np.sin(th)

# background: uniform in the square, kept only where it lies outside the disk
L = 1.9
cand = rng.uniform(-L, L, (900, 2))
outside = cand[:, 0] ** 2 + cand[:, 1] ** 2 > R ** 2
bg = cand[outside][:85]
bx1, bx2 = bg[:, 0], bg[:, 1]

C_SIG, C_BG = PAL["signal"], PAL["background"]

fig = plt.figure(figsize=(9.6, 4.2))

# ---- left: input space (not linearly separable) ------------------------------
axL = fig.add_subplot(1, 2, 1)
axL.scatter(bx1, bx2, s=28, color=C_BG, edgecolor="white", linewidth=0.4,
            label="background  ($r > R$)", zorder=2)
axL.scatter(sx1, sx2, s=28, color=C_SIG, edgecolor="white", linewidth=0.4,
            label="signal  ($r < R$)", zorder=3)
circ = plt.Circle((0, 0), R, fill=False, ls="--", lw=1.8, color=PAL["truth"],
                  zorder=4)
axL.add_patch(circ)
axL.annotate("$r = R$", xy=(R * np.cos(0.9), R * np.sin(0.9)),
             xytext=(1.05, 1.4), fontsize=13,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"),
             arrowprops=dict(arrowstyle="->", lw=1.0))
axL.set_xlim(-L, L)
axL.set_ylim(-L, L)
axL.set_aspect("equal")
axL.set_xlabel("$x_1$")
axL.set_ylabel("$x_2$")
axL.set_title("(a)  input space:  no line separates the classes", fontsize=10.5)
axL.legend(fontsize=8.5, loc="lower right", framealpha=1.0)

# ---- right: embedded space (linearly separable by a plane) -------------------
axR = fig.add_subplot(1, 2, 2, projection="3d")
axR.scatter(bx1, bx2, bx1 ** 2 + bx2 ** 2, s=30, color=C_BG, depthshade=False,
            edgecolor="white", linewidth=0.4, label="background")
axR.scatter(sx1, sx2, sx1 ** 2 + sx2 ** 2, s=30, color=C_SIG, depthshade=False,
            edgecolor="white", linewidth=0.4, label="signal")
# separating plane h3 = R^2
gg = np.linspace(-L, L, 2)
P1, P2 = np.meshgrid(gg, gg)
axR.plot_surface(P1, P2, np.full_like(P1, R ** 2), alpha=0.5, color="0.45",
                 edgecolor="0.3", lw=0.6, zorder=1)
# label placement chosen by a 3-agent review (unanimous): the plane's front-left
# corner reads as bottom-left, anchors to the plane, and clears the point cloud
axR.text(-1.75, -1.75, R ** 2 + 0.02, "$h_3 = R^2$", fontsize=12.5, color="0.1",
         ha="center", zorder=10,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9))
axR.set_ylabel("$h_2 = x_2$", labelpad=6)
axR.set_zlabel("$h_3 = x_1^2 + x_2^2$", labelpad=8)
# fewer ticks on the two horizontal axes so the h1 and h2 tick labels do not
# collide where the axes meet at the near corner
axR.set_xticks([-2, 0, 2])
axR.set_yticks([-2, 0, 2])
axR.set_zticks([0, 2, 4, 6])
axR.tick_params(labelsize=9)
axR.set_title("(b)  embedded space:  the plane $h_3 = R^2$ separates them",
              fontsize=10.5)
axR.view_init(elev=12, azim=-84)
# zoom the 3D cube to fill its cell; matplotlib leaves a wide margin otherwise
axR.set_box_aspect((1, 1, 0.85), zoom=1.4)
axR.grid(True)
# the zoom clips the auto x-axis label off the bottom, so place h1 by hand in
# axes-fraction coordinates, below the tick row
axR.text2D(0.5, -0.12, "$h_1 = x_1$", transform=axR.transAxes,
           ha="center", va="top", fontsize=11)

fig.tight_layout()
paths = save(fig, "fig_embedding_separability")
print("signal pts=%d  background pts=%d  R=%.2f  R^2=%.2f"
      % (n_sig, len(bx1), R, R ** 2))
print("max signal h3=%.3f (should be < %.2f);  min background h3=%.3f (should be > %.2f)"
      % ((sx1**2 + sx2**2).max(), R**2, (bx1**2 + bx2**2).min(), R**2))
print("wrote:", *paths, sep="\n  ")
