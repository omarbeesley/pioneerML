"""
Event displays of muDIF events from a flattened parquet, for intuition.

Two stacked panels per event sharing the z (beam/depth) axis:
  top  = x vs z  (x-z view hits)
  bot  = y vs z  (y-z view hits)
ATAR hits are colored by PDG (muon=red, positron=blue, pion=gray, ...) and sized by
deposited energy (heavier ionization = bigger). The truth pion-stop (star) and the
muon / positron trajectories are overlaid, projected into each view. Selects muDIF
events (event_type & kMudif) that have visible muon + positron hits.

Usage:
  python plot_mudif_events.py --data mudif_event_displays/_mudif_sample.parquet \
     --out_dir mudif_event_displays --n_events 6
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PION, MUON, POSITRON, ELECTRON, GAMMA, OTHER = 1, 2, 4, 8, 16, 32
kMudif = 0x1000


def hit_color(pdg):
    if pdg & MUON:     return "tab:red"
    if pdg & POSITRON: return "tab:blue"
    if pdg & PION:     return "dimgray"
    if pdg & ELECTRON: return "tab:cyan"
    if pdg & GAMMA:    return "tab:orange"
    return "tab:green"


def _finite(*vals):
    return all(np.isfinite(v) for v in vals)


def plot_event(row, path, idx):
    az = np.array(row["atar_z"], dtype=float)
    if len(az) == 0:
        return False
    ax_ = np.array(row["atar_x"], dtype=float)
    ay_ = np.array(row["atar_y"], dtype=float)
    aE = np.array(row["atar_E"], dtype=float)
    aview = np.array(row["atar_view"], dtype=float)
    apdg = np.array(row["atar_pdg"], dtype=int)

    Emax = max(float(aE.max()), 1e-6)
    sizes = 15.0 + 280.0 * (aE / Emax)
    colors = np.array([hit_color(int(p)) for p in apdg])
    is_yz = aview > 0.5
    is_xz = ~is_yz

    fig, axes = plt.subplots(2, 1, figsize=(6, 12), sharex=True)
    T = dict(
        pion=(row["truth_pion_stop_x"], row["truth_pion_stop_y"], row["truth_pion_stop_z"]),
        mu0=(row["truth_muon_start_x"], row["truth_muon_start_y"], row["truth_muon_start_z"]),
        mu1=(row["truth_muon_stop_x"], row["truth_muon_stop_y"], row["truth_muon_stop_z"]),
        e0=(row["truth_positron_start_x"], row["truth_positron_start_y"], row["truth_positron_start_z"]),
        e1=(row["truth_positron_stop_x"], row["truth_positron_stop_y"], row["truth_positron_stop_z"]),
    )
    transv = {0: ax_, 1: ay_}
    for ax, sel, tc, tlab in [(axes[0], is_xz, 0, "x"), (axes[1], is_yz, 1, "y")]:
        if sel.any():
            ax.scatter(az[sel], transv[tc][sel], s=sizes[sel], c=list(colors[sel]),
                       edgecolors="k", linewidths=0.3, alpha=0.85, zorder=3)
        px, py, pz = T["pion"]
        if _finite(pz, px, py):
            ax.scatter([pz], [px if tc == 0 else py], marker="*", s=240, c="k", zorder=4)

        def seg(a, b, color):
            if _finite(a[2], b[2], a[tc], b[tc]):
                ax.plot([a[2], b[2]], [a[tc], b[tc]], color=color, ls="--", lw=1.6,
                        alpha=0.75, zorder=2)
        seg(T["mu0"], T["mu1"], "tab:red")      # muon flight (truth)
        seg(T["e0"], T["e1"], "tab:blue")       # positron (truth)
        ax.set_ylabel(f"{tlab} (mm)")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("z (mm)")
    # Fixed ATAR view so events are directly comparable (sharex -> z set once).
    axes[1].set_xlim(-0.2, 7.0)
    for ax in axes:
        ax.set_ylim(-11.5, 11.5)

    n_mu = int(((apdg & MUON) > 0).sum())
    n_pos = int(((apdg & POSITRON) > 0).sum())
    et = int(row.get("event_type", 0))
    fig.suptitle(
        f"muDIF #{idx}   KE_mu@decay={float(row.get('muon_decay_ke', 0)):.2f} MeV   "
        f"E(e+)={float(row['truth_positron_energy']):.1f} MeV   "
        f"theta={np.degrees(float(row['truth_theta'])):.0f} deg\n"
        f"type=0x{et:x} (kMudif {'SET' if et & kMudif else 'unset'})   "
        f"ATAR {len(az)} hits (muon {n_mu}, e+ {n_pos})",
        fontsize=9)
    handles = [Line2D([0], [0], marker='o', ls='', mfc=c, mec='k', label=l) for c, l in
               [("tab:red", "muon hit"), ("tab:blue", "e+ hit"),
                ("dimgray", "pion hit"), ("tab:green", "other")]]
    handles += [
        Line2D([0], [0], marker='*', ls='', mfc='k', mec='k', label='pion stop (truth)'),
        Line2D([0], [0], color='tab:red', ls='--', label='muon track (truth)'),
        Line2D([0], [0], color='tab:blue', ls='--', label='e+ track (truth)'),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="best", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="mudif_event_displays")
    p.add_argument("--n_events", type=int, default=6)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.data)
    et = df["event_type"].astype(int) if "event_type" in df.columns else pd.Series(np.zeros(len(df), int))

    def count_bit(a, bit):
        if a is None or len(a) == 0:
            return 0
        return int(((np.array(a, dtype=int) & bit) > 0).sum())

    n_mu = df["atar_pdg"].apply(lambda a: count_bit(a, MUON))
    n_pos = df["atar_pdg"].apply(lambda a: count_bit(a, POSITRON))
    is_mudif = (et & kMudif) > 0
    sel = df[is_mudif & (n_mu >= 1) & (n_pos >= 1)].copy()
    sel["_nmu"] = n_mu[sel.index]
    print(f"{len(df)} events total | {int(is_mudif.sum())} muDIF | "
          f"{len(sel)} muDIF with visible muon + e+ hits", flush=True)
    if len(sel) == 0:
        print("no suitable muDIF events found", flush=True)
        return

    # clearest topology first (most muon hits = a visible muon segment)
    sel = sel.sort_values("_nmu", ascending=False)
    made = 0
    for idx, row in sel.head(args.n_events).iterrows():
        path = os.path.join(args.out_dir, f"mudif_event_{made:02d}.png")
        if plot_event(row, path, idx):
            print(f"  wrote {path}  (n_muon_hits={int(row['_nmu'])}, "
                  f"muon_decay_ke={float(row.get('muon_decay_ke', 0)):.2f} MeV)", flush=True)
            made += 1
    print(f"\nwrote {made} muDIF event displays to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
