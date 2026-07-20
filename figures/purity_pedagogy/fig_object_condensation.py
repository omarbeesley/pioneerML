"""Pedagogical figure: object condensation for the calorimeter.

(a) Physical space vs the learned clustering space (illustrative positions).
    In the calorimeter, shower A has thrown an albedo fragment far from its
    core, shower B sits nearby, and radioactivity hits are scattered across
    the detector. A proximity-based clustering would split A and could absorb
    the radioactivity. In the learned clustering space, hits are placed by
    ORIGIN: the fragment sits with its shower's blob because the loss pulls
    every hit toward its own condensation point regardless of physical
    distance; the two seeds are pushed at least m_c = 2 apart; radioactivity
    is expelled beyond m_r = 3 from every seed. Marker size tracks the
    condensation weight beta: one large seed per shower, everything else
    driven small.
(b) The potential terms as a function of clustering-space distance. The
    attraction q*d^2 is drawn for a committed seed-like hit (q = 1.1) and for
    a hit at the charge floor (q = q_min = 0.1): the floor keeps ordinary
    members bound even though the beta loss drives their charge down. The two
    hinge repulsions vanish beyond their margins, so resolved pairs stop
    generating gradient; the blast term reaches further (m_r = 3) and is
    stronger (w_r = 2) than the seed-seed term (m_c = 2, w_c = 1.5), holding
    radioactivity farther from a seed than two genuine showers are held from
    each other.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from figstyle import PAL, save

rng = np.random.default_rng(7)

fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))

# (a1) physical space -----------------------------------------------------
ax = axes[0]
A_core = rng.normal([0, 0], 9, size=(16, 2))
A_frag = rng.normal([52, 38], 4, size=(4, 2))
B_core = rng.normal([34, -26], 8, size=(12, 2))
radio = np.array([[-48, 40], [55, -48], [-35, -42], [-16, 57], [-55, -8], [45, 8]])
ax.scatter(*A_core.T, s=26, color=PAL["blue"], label="shower A")
ax.scatter(*A_frag.T, s=26, facecolors="none", edgecolors=PAL["blue"], lw=1.6)
ax.annotate("albedo fragment\nof shower A", xy=A_frag.mean(0), xytext=(8, 52),
            fontsize=8.5, color=PAL["blue"],
            arrowprops=dict(arrowstyle="->", color=PAL["blue"], lw=1.0))
ax.scatter(*B_core.T, s=26, color=PAL["green"], label="shower B")
ax.scatter(*radio.T, s=34, marker="x", color="0.45", label="radioactivity")
ax.set_xlim(-70, 70); ax.set_ylim(-62, 66)
ax.set_xlabel("calorimeter $x$ (mm)"); ax.set_ylabel("calorimeter $y$ (mm)")
ax.set_title("(a) physical space: origin, not proximity", fontsize=10.5)
ax.legend(fontsize=8, loc="lower left")

# (a2) clustering space ---------------------------------------------------
ax = axes[1]
sA, sB = np.array([-1.4, 0.0]), np.array([1.1, 0.4])   # seeds, |sA-sB| >= m_c
A_blob = sA + rng.normal(0, 0.30, size=(19, 2))        # core + fragment together
B_blob = sB + rng.normal(0, 0.30, size=(11, 2))
ang = np.linspace(0.3, 2 * np.pi, 7)[:-1]
radio_cs = np.array([3.9 * np.array([np.cos(a), np.sin(a)]) + (sA + sB) / 2 for a in ang])[:6]
ax.scatter(*A_blob.T, s=22, color=PAL["blue"], alpha=0.8)
ax.scatter(*B_blob.T, s=22, color=PAL["green"], alpha=0.8)
ax.scatter(*sA, s=200, marker="*", color=PAL["blue"], ec="white", lw=0.8, zorder=5)
ax.scatter(*sB, s=200, marker="*", color=PAL["green"], ec="white", lw=0.8, zorder=5)
ax.scatter(*radio_cs.T, s=34, marker="x", color="0.45")
for s, c in [(sA, PAL["blue"]), (sB, PAL["green"])]:
    ax.add_patch(Circle(s, 3.0, fill=False, ls=":", lw=1.0, ec=c, alpha=0.7))
ax.text(-1.4, 1.05, "seed ($\\beta\\to 1$)", fontsize=8.5, color=PAL["blue"], ha="center")
ax.set_xlim(-5.4, 5.4); ax.set_ylim(-4.7, 4.8)
ax.set_xlabel("clustering-space axis 1"); ax.set_ylabel("clustering-space axis 2")
ax.set_title("(b) learned space: fragment reunited,\nradioactivity expelled", fontsize=10.5)
ax.set_aspect("equal")

# (b) potential terms vs distance ----------------------------------------
ax = axes[2]
d = np.linspace(0, 4.2, 400)
q_seed, q_floor, q_min = 1.1, 0.1, 0.1
ax.plot(d, q_seed * d**2, lw=2, color=PAL["blue"],
        label="hit to its own seed:\nattraction $q\,d^2$, committed ($q=1.1$)")
ax.plot(d, q_floor * d**2, lw=2, color=PAL["blue"], ls="--",
        label="hit to its own seed:\nattraction at floor ($q=q_{\\min}=0.1$)")
ax.plot(d, 1.5 * q_seed * q_seed * np.maximum(2.0 - d, 0), lw=2, color=PAL["green"],
        label="seed to another seed:\nrepulsion ($w_c=1.5$, $m_c=2$)")
ax.plot(d, 2.0 * q_seed * np.maximum(3.0 - d, 0), lw=2, color=PAL["red"],
        label="seed to radioactivity:\nblast ($w_r=2$, $m_r=3$)")
for m, c, lab in [(2.0, PAL["green"], "$m_c$"), (3.0, PAL["red"], "$m_r$")]:
    ax.axvline(m, color=c, lw=1.0, ls=":", alpha=0.8)
    ax.text(m, 6.95, lab, ha="center", fontsize=9, color=c)
ax.set_xlim(0, 4.2)
ax.set_ylim(0, 7.4)
ax.set_xlabel("clustering-space distance $d$")
ax.set_ylabel("loss contribution")
ax.set_title("(c) the potential terms", fontsize=10.5)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.02), borderaxespad=0)

fig.tight_layout()
print(save(fig, "fig_object_condensation"))
