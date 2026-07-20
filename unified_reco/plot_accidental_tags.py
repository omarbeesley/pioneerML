"""
Event displays of the fake-pie events: an ACCIDENTAL (pileup) positron gets tagged as
the trigger positron alongside the real (delayed) michel one.

Three stacked panels per event sharing the z (beam/depth) axis:
  top  = x vs z  (x-z view hits)
  mid  = y vs z  (y-z view hits)
  bot  = hit TIME vs z  (both views -- shows the two time-separated positrons directly)
Hits are colored by truth origin+species (trigger pion=gray, trigger muon=red, trigger
e+=blue, PILEUP e+=magenta, other pileup=orange) and sized by deposited energy. A bold
black ring marks hits the model tags as trigger-positron (P_trig>0.5 & P_mip>0.5) --
the readout averages the ringed hits' times. Truth pion stop (star) + truth muon/e+
trajectories overlaid. Horizontal lines in the time panel: truth michel time (blue),
accidental time (magenta), model plain-mean time (black), meanconf time (green).

Usage:
  python plot_accidental_tags.py --data accidental_tag_displays/_fakes_sample.parquet \
     --out_dir accidental_tag_displays --n_events 10
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def hit_color(origin, pion, muon, pos):
    if origin == 0:                      # trigger event
        if pion > 0.5:  return "dimgray"
        if muon > 0.5:  return "tab:red"
        if pos > 0.5:   return "tab:blue"
        return "tab:cyan"
    else:                                # pileup / accidental
        if pos > 0.5:   return "magenta"
        return "tab:orange"


def _finite(*vals):
    return all(np.isfinite(v) for v in vals)


def plot_event(row, path, made):
    hz = np.array(row["hz"], float); hx = np.array(row["hx"], float)
    hy = np.array(row["hy"], float); hE = np.array(row["hE"], float)
    ht = np.array(row["ht"], float); view = np.array(row["hview"], int)
    org = np.array(row["horigin"], int)
    pion = np.array(row["hpion"], float); muon = np.array(row["hmuon"], float)
    pos = np.array(row["hpos"], float)
    trig = np.array(row["htrig"], float); mip = np.array(row["hmip"], float)
    if len(hz) == 0:
        return False

    tagged = (trig > 0.5) & (mip > 0.5)
    Emax = max(float(hE.max()), 1e-6)
    sizes = 15.0 + 280.0 * (hE / Emax)
    colors = np.array([hit_color(o, p, m, e) for o, p, m, e in zip(org, pion, muon, pos)])
    is_yz = view > 0.5
    is_xz = ~is_yz

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 13.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 2.6]})
    T = dict(
        pion=(row["truth_pion_stop_x"], row["truth_pion_stop_y"], row["truth_pion_stop_z"]),
        mu0=(row["truth_muon_start_x"], row["truth_muon_start_y"], row["truth_muon_start_z"]),
        mu1=(row["truth_muon_stop_x"], row["truth_muon_stop_y"], row["truth_muon_stop_z"]),
        e0=(row["truth_positron_start_x"], row["truth_positron_start_y"], row["truth_positron_start_z"]),
        e1=(row["truth_positron_stop_x"], row["truth_positron_stop_y"], row["truth_positron_stop_z"]),
    )
    transv = {0: hx, 1: hy}
    for ax, sel, tc, tlab in [(axes[0], is_xz, 0, "x"), (axes[1], is_yz, 1, "y")]:
        if sel.any():
            ax.scatter(hz[sel], transv[tc][sel], s=sizes[sel], c=list(colors[sel]),
                       edgecolors="k", linewidths=0.3, alpha=0.85, zorder=3)
        rs = sel & tagged
        if rs.any():                     # bold ring = model tags as trigger positron
            ax.scatter(hz[rs], transv[tc][rs], s=sizes[rs] * 1.9, facecolors="none",
                       edgecolors="k", linewidths=1.7, zorder=4)
        px, py, pz = T["pion"]
        if _finite(pz, px, py):
            ax.scatter([pz], [px if tc == 0 else py], marker="*", s=240, c="k", zorder=5)

        def seg(a, b, color):
            if _finite(a[2], b[2], a[tc], b[tc]):
                ax.plot([a[2], b[2]], [a[tc], b[tc]], color=color, ls="--", lw=1.6,
                        alpha=0.75, zorder=2)
        seg(T["mu0"], T["mu1"], "tab:red")
        seg(T["e0"], T["e1"], "tab:blue")
        ax.set_ylabel(f"{tlab} (mm)")
        ax.grid(alpha=0.3)
        ax.set_ylim(-11.5, 11.5)

    # --- time panel: the money plot for the blend ---
    ax = axes[2]
    ax.scatter(hz, ht, s=sizes * 0.7, c=list(colors), edgecolors="k",
               linewidths=0.3, alpha=0.85, zorder=3)
    if tagged.any():
        ax.scatter(hz[tagged], ht[tagged], s=sizes[tagged] * 1.4, facecolors="none",
                   edgecolors="k", linewidths=1.7, zorder=4)
    tt = float(row["truth_positron_t"]); at = float(row["truth_accidental_positron_t"])
    ax.axhline(tt, color="tab:blue", ls="-", lw=1.2, alpha=0.8)
    if at > -999:
        ax.axhline(at, color="magenta", ls="-", lw=1.2, alpha=0.8)
    ax.axhline(float(row["plain_t"]), color="k", ls="-.", lw=1.3)
    ax.axhline(float(row["meanconf_t"]), color="tab:green", ls=":", lw=1.6)
    ax.set_ylabel("hit time (ns)")
    ax.set_xlabel("z (mm)")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.2, 7.0)

    et = int(row.get("event_type", 0))
    fig.suptitle(
        f"accidental-tag fake #{int(row['event_idx'])}   "
        f"truthKE={float(row['truth_positron_energy']):.1f} MeV   recoE={float(row['recoE']):.1f} MeV\n"
        f"truth michel t={tt:.0f} ns   accidental t={at:.0f} ns   "
        f"reco plain={float(row['plain_t']):.0f} ns   meanconf={float(row['meanconf_t']):.0f} ns   "
        f"tagged slices={int(row['n_tagged'])}   type=0x{et:x}",
        fontsize=9)
    handles = [Line2D([0], [0], marker='o', ls='', mfc=c, mec='k', label=l) for c, l in
               [("dimgray", "trigger pion"), ("tab:red", "trigger muon"),
                ("tab:blue", "trigger e+ (michel)"), ("magenta", "PILEUP e+ (accidental)"),
                ("tab:orange", "other pileup")]]
    handles += [
        Line2D([0], [0], marker='o', ls='', mfc='none', mec='k', mew=1.7,
               label='model tags as trigger e+'),
        Line2D([0], [0], marker='*', ls='', mfc='k', mec='k', label='pion stop (truth)'),
        Line2D([0], [0], color='k', ls='-.', label='reco time (plain mean)'),
        Line2D([0], [0], color='tab:green', ls=':', label='reco time (meanconf)'),
    ]
    axes[0].legend(handles=handles, fontsize=6.5, loc="best", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="accidental_tag_displays")
    p.add_argument("--n_events", type=int, default=10)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.data)
    # clearest first: accidental has a truth time and the model tagged >=2 slices
    df = df.sort_values("n_tagged", ascending=False)
    made = 0
    for _, row in df.iterrows():
        if made >= args.n_events:
            break
        path = os.path.join(args.out_dir, f"accidental_tag_{made:02d}.png")
        if plot_event(row, path, made):
            print(f"  wrote {path}  (evt {int(row['event_idx'])}, "
                  f"truth t={float(row['truth_positron_t']):.0f}, "
                  f"acc t={float(row['truth_accidental_positron_t']):.0f}, "
                  f"plain={float(row['plain_t']):.0f})", flush=True)
            made += 1
    print(f"\nwrote {made} displays to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
