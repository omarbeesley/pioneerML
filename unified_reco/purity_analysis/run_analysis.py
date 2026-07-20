"""CLI driver + plots for the PIONEER R_e/mu analysis.

Truth validation (no model needed):
  python -m purity_analysis.run_analysis --mode truth \
     --pie   .../mixed_standard/pie_benchmark_5_11/data.parquet \
     --michel .../mixed_standard/pimu_benchmark_5_11/data.parquet \
     --out_dir purity_analysis_out/truth

Reco (after benchmark.py writes {tag}_events.parquet):
  python -m purity_analysis.run_analysis --mode reco \
     --pie .../pie_eval_events.parquet --michel .../pimu_eval_events.parquet \
     --out_dir purity_analysis_out/reco
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import io, measurement


def plot_energy_spectrum(spec, cfg, path):
    edges = spec["edges"]; ctr = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(ctr, spec["mue"], where="mid", label=r"$\pi\to\mu\to e$ (michel)", color="C0")
    ax.step(ctr, spec["pie"], where="mid", label=r"$\pi\to e\nu$", color="C3")
    ax.axvline(cfg.e_split, ls="--", color="k", lw=1, label=f"{cfg.e_split:.0f} MeV cut")
    ax.set_yscale("log"); ax.set_xlabel("energy [MeV]")
    ax.set_ylabel(r"$\Sigma\,\mathrm{gen\_weight}$ / bin")
    ax.set_title(f"PIONEER energy spectrum ({cfg.mode}, gen_weight-normalized)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_time_fits(res, path):
    bins = res["bins"]; ctr = 0.5 * (bins[:-1] + bins[1:])
    has_pu = res.get("fit_pileup", False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, b in zip(axes, ("high", "low")):
        r = res["per_bin"][b]
        ax.step(ctr, r["obs"], where="mid", color="k", lw=1.4, label="observed")
        ax.step(ctr, r["fit_pie"], where="mid", color="C3",
                label=fr"$\pi\to e$: {r['N_pie']:.2f} (true {r['true_pie']:.2f})")
        ax.step(ctr, r["fit_mue"], where="mid", color="C0",
                label=fr"$\pi\to\mu\to e$: {r['N_mue']:.1f} (true {r['true_mue']:.1f})")
        if has_pu:
            ax.step(ctr, r["fit_pileup"], where="mid", color="C1",
                    label=fr"pileup: {r['N_pileup']:.2f} (true {r['true_pileup']:.2f})")
        ax.step(ctr, r["fit_total"], where="mid", color="C2", ls="--", lw=1.1, label="fit total")
        top = max(r["obs"].max(), 1e-3)
        ax.set_yscale("log"); ax.set_ylim(3e-4 * top, 3 * top)
        cut = res.get("e_split", 56.0)
        ax.set_title(f"{b} energy bin  (E {'>=' if b == 'high' else '<'} {cut:.0f} MeV)")
        ax.set_xlabel("positron time [ns]"); ax.set_ylabel(r"$\Sigma$ gen_weight / bin")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Per-bin time fits ({res['cfg_mode']} mode)" + ("  +pileup component" if has_pu else ""))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write_summary(res, path):
    e = res["eff"]
    lines = [
        f"=== PIONEER R_e/mu analysis ({res['cfg_mode']} mode) ===",
        "",
        "Channel selection efficiency (accepted & in-window, gen_weight-weighted):",
        f"  pi->e   : eff={e['pie']['eff']:.4f}  (Sw_sel {e['pie']['sel_w']:.3f} / Sw_tot {e['pie']['total_w']:.3f}, n_sel {e['pie']['n_sel']})",
        f"  pi->mu->e: eff={e['mue']['eff']:.4f}  (Sw_sel {e['mue']['sel_w']:.1f} / Sw_tot {e['mue']['total_w']:.1f}, n_sel {e['mue']['n_sel']})",
        "",
        "Per-bin time-fit yields (fit vs truth):",
    ]
    pu = res.get("fit_pileup", False)
    for b in ("high", "low"):
        r = res["per_bin"][b]
        line = (f"  {b:4}: N_pie fit={r['N_pie']:.3f} true={r['true_pie']:.3f}  |  "
                f"N_mue fit={r['N_mue']:.1f} true={r['true_mue']:.1f}")
        if pu:
            line += f"  |  N_pileup fit={r['N_pileup']:.3f} true={r['true_pileup']:.3f}"
        lines.append(line)
    lines += [
        "",
        f"Summed selected yields:  N_pie={res['N_pie_sel']:.3f}   N_mue={res['N_mue_sel']:.1f}"
        + (f"   N_pileup={res['N_pileup_sel']:.3f}" if pu else ""),
        f"Efficiency-corrected:    N_pie={res['N_pie_corr']:.3f}   N_mue={res['N_mue_corr']:.1f}",
        "",
        f"  R_e/mu (measured)  = {res['remu']:.4e}",
        f"  R_e/mu (MC truth)  = {res['remu_true']:.4e}   [Sw_pie/Sw_mue]",
        f"  R_e/mu (SM value)  = {res['remu_sm']:.4e}",
        f"  closure  measured/truth = {res['remu']/res['remu_true']:.4f}",
    ]
    txt = "\n".join(lines)
    with open(path, "w") as f:
        f.write(txt + "\n")
    print(txt, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["truth", "reco"], default="truth")
    ap.add_argument("--pie", required=True, help="pie eval parquet (mixed_standard pie_benchmark, or reco events)")
    ap.add_argument("--michel", required=True, help="michel eval parquet (pimu_benchmark, or reco events)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--e_split", type=float, default=io.E_SPLIT_MEV)
    ap.add_argument("--energy", choices=["deposited", "true_ke"], default="deposited")
    ap.add_argument("--n_time_bins", type=int, default=50)
    ap.add_argument("--fit_pileup", action="store_true",
                    help="add a flat pileup component to the per-bin time fit")
    ap.add_argument("--consensus_time", action="store_true",
                    help="use pred_positron_time_consensus_ns (dominant-slice by SUM of trig*mip)")
    ap.add_argument("--meanconf_time", action="store_true",
                    help="use pred_positron_time_consensus_meanconf_ns (dominant-slice by MEAN trig*mip)")
    ap.add_argument("--argmax_role_time", action="store_true",
                    help="use pred_positron_time_argmax_role_ns (single slice = argmax role e-in-chain prob)")
    ap.add_argument("--clean_tol", type=float, default=5.0,
                    help="ns; |reco - truth_trigger| < clean_tol counts as the reco finding the trigger")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    time_col = ("pred_positron_time_argmax_role_ns" if args.argmax_role_time else
                "pred_positron_time_consensus_meanconf_ns" if args.meanconf_time else
                "pred_positron_time_consensus_ns" if args.consensus_time else
                "pred_positron_time_ns")
    cfg = io.AnalysisConfig(mode=args.mode, e_split=args.e_split, energy=args.energy,
                            time_col=time_col, clean_tol=args.clean_tol)
    df = io.load_all(args.pie, args.michel, cfg)

    res = measurement.measure(df, cfg, n_time_bins=args.n_time_bins, fit_pileup=args.fit_pileup)
    res["e_split"] = cfg.e_split
    spec = measurement.energy_spectrum(df, cfg)

    plot_energy_spectrum(spec, cfg, os.path.join(args.out_dir, "energy_spectrum.png"))
    plot_time_fits(res, os.path.join(args.out_dir, "time_fits.png"))
    write_summary(res, os.path.join(args.out_dir, "summary.txt"))
    print(f"\nwrote plots + summary -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
