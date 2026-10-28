"""Part 1 (reco vs truth) — the gap plots for the main PURITY model.

plot_benchmarks.py already covers acceptance / positron-angle / pion-stop / energy / slice-role /
endpoints / IoU.  This module adds the reco-vs-truth plots the *augmented* benchmark.py events
parquet now enables (see benchmark.py: event_id, truth_gen_weight, truth_positron_t, truth_is_pie,
truth_has_muon ...):

  * time residual  (pred_positron_time_ns - truth_positron_t)  <- validates the Part-2 time input
  * HTP performance (pred_htp vs truth_htp: ROC + efficiency)
  * time-spread distribution (pred_positron_time_spread_ns)
  * dead-energy resolution (pred_dead_energy - truth_dead_E)
  * energy spectrum per supplied input table (no truth-class purification)

Portable supplement to purity.diagnostics.benchmark_plots.

  python -m purity.diagnostics.part1 --events pie_events.parquet michel_events.parquet \
      --tags pie_eval pimu_eval --output-dir part1
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .benchmark_plots import watermark, SENTINEL
from ._runtime import prepare_output, record_run
from pathlib import Path
import json
import hashlib
import pandas as pd

COLORS = {"pie_eval": "C3", "pimu_eval": "C0", "pie": "C3", "pimu": "C0", "michel": "C0"}
T_FLOOR = -900.0   # anything below is a time sentinel (truth -1000, reco ~ -999)


def _color(tag):
    return COLORS.get(tag, "C2")


def _valid_time(*arrs):
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        a = np.asarray(a, float)
        m &= np.isfinite(a) & (a > T_FLOOR)
    return m


def plot_time_residual(datasets, out_dir):
    """pred_positron_time_ns - truth_positron_t : the Part-2 time-fit input quality."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    fig2, ax2s = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    for j, (tag, ev) in enumerate(datasets):
        ax, ax2 = axes[0][j], ax2s[0][j]
        if ev is None or not {"pred_positron_time_ns", "truth_positron_t"} <= set(ev.columns):
            ax.set_title(f"{tag}: no time columns"); continue
        rp = ev["pred_positron_time_ns"].to_numpy(float)
        rt = ev["truth_positron_t"].to_numpy(float)
        m = _valid_time(rp, rt)
        res = rp[m] - rt[m]
        ax.hist(res, bins=100, range=(-60, 60), histtype="step", color=_color(tag), lw=1.5)
        med, rms = np.median(res), res.std()
        ax.set_title(f"{tag}: time residual  med={med:.1f}  RMS={rms:.1f} ns  (n={m.sum()})")
        ax.set_xlabel("pred − truth positron time [ns]"); ax.set_ylabel("events"); watermark(ax)
        ax2.hist2d(rt[m], rp[m], bins=80, range=[[-50, 500], [-50, 500]], cmin=1)
        ax2.plot([-50, 500], [-50, 500], "r--", lw=1)
        ax2.set_title(f"{tag}: reco vs truth time"); ax2.set_xlabel("truth t [ns]"); ax2.set_ylabel("reco t [ns]")
    for f, name in ((fig, "time_residual.png"), (fig2, "time_residual_2d.png")):
        f.tight_layout(); f.savefig(os.path.join(out_dir, name), dpi=120); plt.close(f)


def plot_htp_performance(datasets, out_dir):
    """ROC + efficiency of the has-trigger-positron head (gates the whole acceptance chain)."""
    fig, (axr, axe) = plt.subplots(1, 2, figsize=(12, 4.5))
    for tag, ev in datasets:
        if ev is None or not {"pred_htp", "truth_htp"} <= set(ev.columns):
            continue
        score = ev["pred_htp"].to_numpy(float)
        y = ev["truth_htp"].to_numpy(float) > 0.5
        good = np.isfinite(score) & (score != SENTINEL)
        score, y = score[good], y[good]
        thr = np.linspace(0, 1, 101)
        tpr = [(y & (score >= t)).sum() / max(1, y.sum()) for t in thr]
        fpr = [((~y) & (score >= t)).sum() / max(1, (~y).sum()) for t in thr]
        axr.plot(fpr, tpr, label=f"{tag}", color=_color(tag))
        axe.plot(thr, tpr, label=f"{tag} eff", color=_color(tag))
        axe.plot(thr, fpr, "--", label=f"{tag} fake", color=_color(tag), alpha=0.6)
    axr.plot([0, 1], [0, 1], "k:", lw=0.8); axr.set_xlabel("false positive"); axr.set_ylabel("true positive")
    axr.set_title("HTP ROC"); axr.legend(fontsize=8); watermark(axr)
    axe.set_xlabel("pred_htp threshold"); axe.set_ylabel("rate"); axe.set_title("HTP eff / fake vs cut")
    axe.legend(fontsize=8); watermark(axe)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "htp_performance.png"), dpi=120); plt.close(fig)


def plot_time_spread(datasets, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, ev in datasets:
        if ev is None or "pred_positron_time_spread_ns" not in ev.columns:
            continue
        s = ev["pred_positron_time_spread_ns"].to_numpy(float)
        s = s[np.isfinite(s) & (s != SENTINEL) & (s >= 0)]
        ax.hist(s, bins=80, range=(0, 40), histtype="step", label=f"{tag} (med {np.median(s):.1f})", color=_color(tag))
    ax.set_xlabel("pred positron time spread [ns]"); ax.set_ylabel("events")
    ax.set_title("Reco positron time-spread"); ax.legend(fontsize=8); watermark(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "time_spread.png"), dpi=120); plt.close(fig)


def plot_dead_energy_resolution(datasets, out_dir):
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    for j, (tag, ev) in enumerate(datasets):
        ax = axes[0][j]
        if ev is None or not {"pred_dead_energy", "truth_dead_E"} <= set(ev.columns):
            ax.set_title(f"{tag}: no dead-E columns"); continue
        pr = ev["pred_dead_energy"].to_numpy(float)
        tr = ev["truth_dead_E"].to_numpy(float)
        m = np.isfinite(pr) & (pr != SENTINEL) & np.isfinite(tr)
        res = np.clip(pr[m], 0, None) - tr[m]
        ax.hist(res, bins=80, range=(-3, 3), histtype="step", color=_color(tag))
        ax.set_title(f"{tag}: dead-E residual  med={np.median(res):.3f} MeV")
        ax.set_xlabel("pred − truth dead energy [MeV]"); watermark(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "dead_energy_resolution.png"), dpi=120); plt.close(fig)


def plot_energy_by_class(datasets, out_dir):
    """gen_weight-weighted reco energy spectrum split by truth class (pi->e vs pi->mu->e)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    edges = np.linspace(0, 80, 81)
    for tag, ev in datasets:
        if ev is None or "pred_positron_energy" not in ev.columns:
            continue
        acc = (ev["pred_accepted"].to_numpy(float) != SENTINEL) & (ev["pred_accepted"].to_numpy(float) >= 0.5)
        E = np.clip(ev["pred_positron_energy"].to_numpy(float), 0, None)
        if "pred_dead_energy" in ev.columns:
            E = E + np.clip(ev["pred_dead_energy"].to_numpy(float), 0, None)
        w = ev["truth_gen_weight"].to_numpy(float) if "truth_gen_weight" in ev.columns else np.ones(len(ev))
        ax.hist(E[acc], bins=edges, weights=w[acc], histtype="step", label=f"{tag}", color=_color(tag))
    ax.axvline(56.0, ls="--", color="k", lw=1, label="56 MeV")
    ax.set_yscale("log"); ax.set_xlabel("reco energy [MeV]"); ax.set_ylabel(r"$\Sigma$ gen_weight")
    ax.set_title("Reco energy spectrum (accepted, gen_weight-weighted)"); ax.legend(fontsize=8); watermark(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "energy_by_class.png"), dpi=120); plt.close(fig)



REQUIRED_COLUMNS = {
    "pred_positron_time_ns", "truth_positron_t", "pred_htp", "truth_htp",
    "pred_positron_time_spread_ns", "pred_dead_energy", "truth_dead_E",
    "pred_positron_energy", "pred_accepted",
}


def run_part1(event_files, tags, output_dir):
    """Render all six supplemental figures from explicitly named event tables."""
    if not event_files or len(event_files) != len(tags):
        raise ValueError("Supply one tag per input event table")
    datasets = []
    for path, tag in zip(event_files, tags):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        ev = pd.read_parquet(path)
        missing = sorted(REQUIRED_COLUMNS - set(ev.columns))
        if missing:
            raise ValueError(f"{path}: missing required columns: {missing}")
        if ev.empty:
            raise ValueError(f"{path}: empty event table")
        datasets.append((tag, ev))
    out = prepare_output(output_dir)
    plot_time_residual(datasets, out)
    plot_htp_performance(datasets, out)
    plot_time_spread(datasets, out)
    plot_dead_energy_resolution(datasets, out)
    plot_energy_by_class(datasets, out)
    record_run(out, event_files, {"tags": tags, "rows": [len(ev) for _, ev in datasets],
                                "timing_column": "pred_positron_time_ns",
                                "energy_class_rule": "one curve per input table; no truth-class purification",
                                "weight_rule": "truth_gen_weight when available, otherwise one per event"},
               "purity.diagnostics.part1")
    report = out / 'analysis_run.json'
    record = json.loads(report.read_text())
    record['source_sha256'] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ('part1.py', 'benchmark_plots.py', '_runtime.py')}
    record['outputs'] = [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in sorted(out.glob('*.png'))]
    report.write_text(json.dumps(record, indent=2) + '\n')
    print(f"Part-1 supplements -> {out}", flush=True)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--events', type=Path, nargs='+', help='Explicit benchmark event parquet files')
    group.add_argument('--results-dir', type=Path, help='Directory containing TAG_events.parquet')
    ap.add_argument('--tags', nargs='+', help='Labels per input; with --results-dir also selects TAG_events.parquet')
    ap.add_argument('--output-dir', type=Path, required=True, help='New or empty output directory')
    args = ap.parse_args(argv)
    if args.events is not None:
        paths = args.events
        tags = args.tags or [p.stem.removesuffix('_events') for p in paths]
    else:
        tags = args.tags or ['pie_eval', 'pimu_eval']
        paths = [args.results_dir / f'{tag}_events.parquet' for tag in tags]
    if len(tags) != len(paths) or len(set(tags)) != len(tags):
        ap.error('Supply one distinct --tags label per event file')
    return run_part1(paths, tags, args.output_dir)


if __name__ == '__main__':
    main()
