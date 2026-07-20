"""Talk version of the counting R_e/mu figure: TWO separate plots (energy spectrum, acceptance
efficiency), thesis style (inner ticks on all sides, no grid, large labels), no overflow bin,
no numbers box, capitalized axes, title stating the time window."""
import argparse, numpy as np, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

COLS = ['truth_acceptance', 'pred_accepted', 'pred_positron_energy', 'pred_dead_energy',
        'truth_positron_t', 'pred_positron_time_consensus_ns', 'truth_gen_weight',
        'truth_event_type', 'truth_positron_energy']
COL = dict(pie="#d62728", mue="#1f77b4")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 15,
    "axes.labelsize": 17, "axes.titlesize": 17, "legend.fontsize": 14,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.grid": False})


def load(path, is_pie):
    import glob, pandas as pd
    files = sorted(glob.glob(path)) if "*" in path else [path]
    d = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in files], ignore_index=True)
    if not is_pie:
        et = d.truth_event_type.astype('int64')
        d = d[((et & 1) == 0) & ((et & 0x200) == 0) & (d.truth_positron_energy <= 55)].reset_index(drop=True)
    return d


def run(pie_path, michel_path, out_prefix, ecut=56.0):
    P, M = load(pie_path, True), load(michel_path, False)
    n_pie, n_mic = len(P), len(M)
    D = {}
    for name, d, is_pie in [("pie", P, True), ("mue", M, False)]:
        wlum = (n_mic / n_pie) if is_pie else 1.0
        gw = d.truth_gen_weight.values.astype(float) * wlum
        tr = d.pred_positron_time_consensus_ns.values
        Er = np.clip(d.pred_positron_energy.values, 0, None) + np.clip(d.pred_dead_energy.values, 0, None)
        clean = np.abs(tr - d.truth_positron_t.values) < 5.0
        acc_r = (d.pred_accepted.values >= 0.5) & (tr >= -300) & (tr <= 500) & clean
        acc_t = d.truth_acceptance.values > 0.5
        Wt = float(gw[acc_t].sum()); Wr = float(gw[acc_r].sum())
        fn = acc_t & ~acc_r; fp = (~acc_t) & acc_r
        sr = np.sqrt(float((gw[fn] ** 2).sum()) + float((gw[fp] ** 2).sum())) / Wt
        D[name] = dict(E=Er[acc_r], w=gw[acc_r], r=Wr / Wt, sr=sr)
    p, m = D["pie"], D["mue"]
    ratio = p["r"] / m["r"]
    sbias = ratio * np.sqrt((p["sr"] / p["r"]) ** 2 + (m["sr"] / m["r"]) ** 2)
    print(f"channel reco/truth: pie={p['r']:.4f}+-{p['sr']:.4f}  michel={m['r']:.4f}+-{m['sr']:.4f}  "
          f"ratio={ratio:.4f}+-{sbias:.4f}")

    # ---------- 1) energy spectrum (no overflow, no numbers box) ----------
    E_DISP = 85.0
    bins = np.linspace(0, E_DISP, 69)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for name, lab in [("mue", r"$\pi\to\mu\to e$"), ("pie", r"$\pi\to e\nu$")]:
        d = D[name]
        ax.hist(d["E"], bins=bins, weights=d["w"], histtype="step", lw=2.2, color=COL[name], label=lab)
    ax.axvline(ecut, color="k", ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Reconstructed Energy [MeV]")
    ax.set_ylabel("Weighted Counts")
    ax.set_title(r"Time Window: $-300 \leq t \leq 500$ ns")
    ax.set_xlim(0, E_DISP)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_spectrum.png"); plt.close(fig)
    print(f"wrote {out_prefix}_spectrum.png")

    # ---------- 2) acceptance efficiency ----------
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.axhline(1.0, color="0.55", ls="--", lw=1.2, zorder=1)
    ax.errorbar([0, 1], [p["r"], m["r"]], yerr=[p["sr"], m["sr"]], fmt="o", ms=10, capsize=6,
                color="k", lw=1.6, zorder=3, label="Channel reco / truth")
    ax.errorbar([2], [ratio], yerr=[sbias], fmt="s", ms=11, capsize=6, color="#2ca02c",
                lw=2.0, zorder=4, label=r"Ratio $= R_{e/\mu}$ bias")
    for i, nm in enumerate(("pie", "mue")):
        ax.annotate(f"{D[nm]['r']:.4f}\n$\\pm${D[nm]['sr']:.4f}", (i, D[nm]["r"]),
                    textcoords="offset points", xytext=(14, 0), va="center", fontsize=13)
    ax.annotate(f"{ratio:.4f}\n$\\pm${sbias:.4f}", (2, ratio), textcoords="offset points",
                xytext=(-14, 0), va="center", ha="right", fontsize=13, color="#2ca02c", fontweight="bold")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"$\pi\to e\nu$", r"$\pi\to\mu\to e$", "Ratio"])
    ax.set_xlim(-0.6, 2.7)
    vals = [p["r"], m["r"], ratio]
    span = max(vals) - min(vals) + 2 * max(p["sr"], m["sr"], sbias)
    ax.set_ylim(min(vals) - 0.9 * span, max(vals) + 0.9 * span)   # headroom so the legend clears the markers
    ax.set_ylabel("Acceptance Efficiency (Reco / Truth)")
    ax.set_title("Acceptance Bias  (1 = Unbiased)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_acceptance.png"); plt.close(fig)
    print(f"wrote {out_prefix}_acceptance.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie", required=True); ap.add_argument("--michel", required=True)
    ap.add_argument("--out_prefix", required=True)
    a = ap.parse_args()
    run(a.pie, a.michel, a.out_prefix)
