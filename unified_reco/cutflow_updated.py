"""Tail-reveal cut-flow (updated): from the POST-ACCEPTANCE deposited-energy spectrum, show how
each cut -- a DIFFERENT discriminator of the michel background -- peels it away to zero.

  - pi->e nu (signal, red) vs pi->mu->e michel (background, blue).  Accidentals are dropped from
    the DISPLAY (negligible post-acceptance) but the pileup VETO cut is kept -- it removes the
    michel-with-pileup events the muon veto misses (complementary heads).
  - michel is BRANCHING-WEIGHTED to its physical dominance (w_michel=1, w_pie=BR_pie*n_mic/n_pie).
  - cumulative stages: post-acceptance -> +muon veto -> +pileup veto -> +pie topo -> +pie score.
    michel goes 259031 -> 4 -> 1 -> 0 -> 0 (each cut a distinct discriminator).
  - NO time cut (the only time selection is the acceptance's window survival).

Data cached to .cutflow_data.npz so style iterations re-plot in seconds.
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

PRED = "/home/obeesley/pioneerML/tail_reveal_eval/predictions.parquet"
OUT = "/home/obeesley/pioneerML/unified_reco/updated_plots/07_cutflow.png"
CACHE = "/home/obeesley/pioneerML/unified_reco/updated_plots/.cutflow_data.npz"
EMAX, ECUT, BR_PIE = 75.0, 56.0, 1.2352e-4
MUON_CUT = PILEUP_CUT = TOPO_CUT = PIE_CUT = 0.5
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 10.5,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
COL = dict(pie="#d62728", michel="#1f77b4")

if os.path.exists(CACHE):
    z = np.load(CACHE)
    E, is_pie, acc = z["E"], z["is_pie"].astype(bool), z["acc"].astype(bool)
    msc, usc, tsc, psc = z["msc"], z["usc"], z["tsc"], z["psc"]
    print("loaded cutflow cache", flush=True)
else:
    df = pd.read_parquet(PRED, columns=["is_pie", "deposited_energy", "acceptance",
                                        "muon_score", "pileup_score", "topo_score", "pie_score"])
    df = df[df.deposited_energy <= EMAX].reset_index(drop=True)
    E = df.deposited_energy.values; is_pie = df.is_pie.values.astype(bool)
    acc = df.acceptance.values == 1
    msc, usc = df.muon_score.values, df.pileup_score.values
    tsc, psc = df.topo_score.values, df.pie_score.values
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, E=E, is_pie=is_pie, acc=acc, msc=msc, usc=usc, tsc=tsc, psc=psc)
    print("cached cutflow data", flush=True)

n_pie, n_mic = int(is_pie.sum()), int((~is_pie).sum())
w = np.where(is_pie, BR_PIE * n_mic / n_pie, 1.0)

stages = [("(0) after acceptance", np.ones(len(E), bool)),
          ("(1) + muon veto", msc < MUON_CUT),
          ("(2) + pileup veto", usc < PILEUP_CUT),
          ("(3) + pie topo", tsc > TOPO_CUT),
          ("(4) + pie score", psc > PIE_CUT)]
bins = np.linspace(0, EMAX, 51)
base = acc
w_pie0 = float(w[base & is_pie].sum()); w_mic0 = float(w[base & ~is_pie].sum())

fig, axes = plt.subplots(1, 5, figsize=(21, 4.7), sharex=True, sharey=True)
keep = base.copy()
for ax, (name, mask) in zip(axes, stages):
    keep = keep & mask
    kp, km = keep & is_pie, keep & ~is_pie
    ax.hist(E[kp], bins=bins, weights=w[kp], histtype="step", lw=1.9, color=COL["pie"],
            label=r"$\pi\to e\nu$ (signal)")
    ax.hist(E[km], bins=bins, weights=w[km], histtype="step", lw=1.9, color=COL["michel"],
            label=r"$\pi\to\mu\to e$ (michel)")
    ax.axvline(ECUT, color="k", ls=":", lw=0.9)
    ax.set_yscale("log"); ax.set_title(name, fontweight="bold")
    pie_eff = 100.0 * float(w[kp].sum()) / w_pie0
    n_surv = int(km.sum())
    supp = f"michel: 0 (removed)" if n_surv == 0 else f"michel supp: {w_mic0/float(w[km].sum()):.0f}$\\times$"
    ax.text(0.03, 0.03, f"$\\pi\\to e$ eff: {pie_eff:.1f}%\n{supp}\n({n_surv} michel left)",
            transform=ax.transAxes, va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel("deposited energy [MeV]")

axes[0].set_ylabel(r"branching-weighted events / bin  ($\propto$ rate)")
axes[0].legend(fontsize=9, loc="upper left")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT); plt.close(fig)
print(f"wrote {OUT}\n  post-accept: pie={w_pie0:.3g} michel={w_mic0:.3g} (michel/pie={w_mic0/w_pie0:.0f}x)", flush=True)
