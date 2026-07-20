"""Counting R_e/mu -- result figure + statistical error.

Figure: the branching-weighted, accepted deposited-energy spectrum of pi->e (red) and
pi->mu->e (blue), split at 56 MeV.  The four gen_weight-weighted counts are the measurement;
R_e/mu = (N_pie/eff_pie)/(N_mue/eff_mue).  Inset panel: reco-vs-truth acceptance efficiency
(the +0.15% bias source).

Stat error: R_e/mu is a ratio of two independent weighted counts, so
  sigma(R)/R = sqrt(1/Neff_pie + 1/Neff_mue),  Neff = (sum w)^2 / sum(w^2)  (effective # events).
"""
import argparse, numpy as np, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

COLS = ['truth_acceptance', 'pred_accepted', 'pred_positron_energy', 'pred_dead_energy',
        'truth_positron_t', 'pred_positron_time_consensus_ns', 'truth_gen_weight',
        'truth_event_type', 'truth_positron_energy']
COL = dict(pie="#d62728", mue="#1f77b4")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def load(path, is_pie):
    import glob, pandas as pd
    files = sorted(glob.glob(path)) if "*" in path else [path]
    d = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in files], ignore_index=True)
    if not is_pie:
        et = d.truth_event_type.astype('int64')
        d = d[((et & 1) == 0) & ((et & 0x200) == 0) & (d.truth_positron_energy <= 55)].reset_index(drop=True)
    return d


def run(pie_path, michel_path, out, ecut=56.0, emax=float("inf")):
    # emax: NO cap by default -- the 75 MeV cap cut pileup-merged over-reconstructed pie
    # asymmetrically (-0.40% R_e/mu bias); uncapped -0.04%. Display range stays fixed below.
    P, M = load(pie_path, True), load(michel_path, False)
    n_pie, n_mic = len(P), len(M)
    D = {}
    for name, d, is_pie in [("pie", P, True), ("mue", M, False)]:
        wlum = (n_mic / n_pie) if is_pie else 1.0           # luminosity match
        gw = d.truth_gen_weight.values.astype(float) * wlum   # gen_weight IS the BR -> already branching-weighted
        tr = d.pred_positron_time_consensus_ns.values
        Er = np.clip(d.pred_positron_energy.values, 0, None) + np.clip(d.pred_dead_energy.values, 0, None)
        clean = np.abs(tr - d.truth_positron_t.values) < 5.0
        acc_r = (d.pred_accepted.values >= 0.5) & (tr >= -300) & (tr <= 500) & clean & (Er <= emax)
        acc_t = d.truth_acceptance.values > 0.5
        w_acc = gw[acc_r]; E_acc = Er[acc_r]
        Wt = float(gw[acc_t].sum()); Wr = float(gw[acc_r].sum())
        fn = acc_t & ~acc_r; fp = (~acc_t) & acc_r            # acceptance flips drive the ratio error
        sr = np.sqrt(float((gw[fn] ** 2).sum()) + float((gw[fp] ** 2).sum())) / Wt
        D[name] = dict(E=E_acc, w=w_acc,
                       hi=float(gw[acc_r & (Er >= ecut)].sum()), lo=float(gw[acc_r & (Er < ecut)].sum()),
                       tot=float(gw.sum()), eff_t=Wt / gw.sum(), eff_r=Wr / gw.sum(),
                       r=Wr / Wt, sr=sr,
                       neff=float(w_acc.sum()) ** 2 / float((w_acc ** 2).sum()))
    p, m = D["pie"], D["mue"]
    remu = (p["hi"] + p["lo"]) / p["eff_t"] / ((m["hi"] + m["lo"]) / m["eff_t"])
    rel = np.sqrt(1.0 / p["neff"] + 1.0 / m["neff"])
    bias = (p["eff_r"] / p["eff_t"]) / (m["eff_r"] / m["eff_t"]) - 1.0
    print(f"R_e/mu = {remu:.4e} +/- {remu*rel:.2e}  ({rel*100:.2f}% stat)   "
          f"Neff: pie={p['neff']:.0f} michel={m['neff']:.0f}   accept-bias={bias*100:+.2f}%")

    # --- figure --- (fixed display range; selection is uncapped -- overflow folds into last bin)
    E_DISP = 85.0
    bins = np.linspace(0, E_DISP, 69)
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[2.4, 1]))
    for name, lab in [("mue", r"$\pi\to\mu\to e$ (michel)"), ("pie", r"$\pi\to e\nu$ (signal)")]:
        d = D[name]
        ax.hist(np.clip(d["E"], None, E_DISP - 1e-6), bins=bins, weights=d["w"],
                histtype="step", lw=1.9, color=COL[name], label=lab)
    ax.axvline(ecut, color="k", ls=":", lw=1.0)
    ax.set_yscale("log"); ax.set_xlabel("reconstructed deposited energy [MeV]")
    ax.set_ylabel(r"branching-weighted events / bin  ($\propto$ rate)")
    ax.legend(loc="upper left", fontsize=9.5)
    sbias = (1.0 + bias) * np.sqrt((p["sr"] / p["r"]) ** 2 + (m["sr"] / m["r"]) ** 2)  # error on the ratio (~1)
    ax.text(0.035, 0.40,                                     # mid-left empty band (between pie floor & michel)
            f"$N_{{\\pi e}}$: {p['hi']:.3g} (>56) | {p['lo']:.2g} (<56)\n"
            f"$N_{{\\mu}}$: {m['hi']:.2g} (>56) | {m['lo']:.3g} (<56)\n"
            f"$R_{{e/\\mu}} = {remu:.4e}$\n"
            f"$\\pm{rel*100:.2f}\\%$ (stat)\n"
            f"accept. bias ${bias*100:+.2f}\\pm{sbias*100:.2f}\\%$   (no reco-E cap)",
            transform=ax.transAxes, va="center", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.95))

    # acceptance reco/truth: per-channel (black) + their RATIO = R_e/mu bias (green)
    axr.axhline(1.0, color="0.55", ls="--", lw=1.0, zorder=1)
    axr.errorbar([0, 1], [p["r"], m["r"]], yerr=[p["sr"], m["sr"]], fmt="o", ms=8, capsize=5,
                 color="k", lw=1.3, zorder=3, label="channel reco/truth")
    ratio = p["r"] / m["r"]
    axr.errorbar([2], [ratio], yerr=[sbias], fmt="s", ms=9, capsize=5, color="#2ca02c",
                 lw=1.7, zorder=4, label=r"ratio $=R_{e/\mu}$ bias")
    for i, nm in enumerate(("pie", "mue")):
        axr.annotate(f"{D[nm]['r']:.4f}\n$\\pm${D[nm]['sr']:.4f}", (i, D[nm]["r"]),
                     textcoords="offset points", xytext=(11, 0), va="center", fontsize=8.5)
    axr.annotate(f"{ratio:.4f}\n$\\pm${sbias:.4f}", (2, ratio), textcoords="offset points",
                 xytext=(11, 0), va="center", fontsize=8.5, color="#2ca02c", fontweight="bold")
    axr.set_xticks([0, 1, 2]); axr.set_xticklabels([r"$\pi\to e$", r"$\pi\to\mu\to e$", "ratio"])
    axr.set_xlim(-0.5, 2.9); axr.set_ylabel("acceptance efficiency   reco / truth")
    axr.legend(fontsize=8, loc="center left"); axr.set_title("acceptance bias  (1 = unbiased)", fontsize=10)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie", required=True); ap.add_argument("--michel", required=True)
    ap.add_argument("--out", default="/home/obeesley/pioneerML/unified_reco/updated_plots/03_remu_counting.png")
    a = ap.parse_args()
    run(a.pie, a.michel, a.out)
