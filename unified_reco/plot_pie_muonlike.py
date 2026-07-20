"""
Event displays for understanding the muDIF tail-fraction bias: it shows the MARGINAL
events near the veto cut (where the bias is made), comparing pi->e nu vs real muDIF at
the SAME near-threshold score. Runs the model for per-event muDIF/piDIF scores, then draws
ATAR displays from the SAME rows (ds.df stays aligned with the scores under shuffle=False).

Pass BOTH the pie eval and the muDIF eval parquets to --data.

Categories (default 6 events each), all near the cut threshold (pie_eff working point):
  marg_tail   - pi->e nu, tail (E_tot<e_split)        <- the biased population
  marg_peak   - pi->e nu, peak (e_split<=E<=e_max)
  marg_mudif  - real muDIF near the same score        <- what the head calls "muDIF-like"

E_tot = live_E + dead_E (live already includes atar_posE). Each panel pair is x-z (top) /
y-z (bottom); ATAR hits colored by PDG (pion=gray, e+=blue, muon=red), sized by energy;
pileup hits (origin>0) get a magenta edge; truth pion-stop (hollow star) + muon/positron
tracks overlaid. Title: scores + CUT/kept verdict + energies + hit multiplicity.

Run inside pytorch.sif on the GPU node, e.g.:
  apptainer exec --nv --cleanenv --contain \
    --bind /home/obeesley/pioneerML:/pioneerML --bind /data/nvme0/prod_ml_data:/data \
    /data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif \
    python3 /pioneerML/unified_reco/plot_pie_muonlike.py \
      --checkpoint /pioneerML/model_weights/<ckpt>.pth \
      --data /data/tail_reveal_pie_eval.parquet /data/tail_reveal_mudif_eval.parquet \
      --out_dir /pioneerML/pie_muonlike_displays --pie_eff 0.9 --max_rows 200000
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from torch_geometric.loader import DataLoader
from unified_reco.dataset import PURITYDataset
from unified_reco.models_tail import PURITYTailModel

PION, MUON, POSITRON, ELECTRON, GAMMA = 1, 2, 4, 8, 16


def hit_color(pdg):
    # PION before POSITRON: a pixel at the pion stop carries both bits (Bragg peak +
    # nascent positron); showing it as the structural pion avoids a fake "e+ blob".
    if pdg & MUON:     return "tab:red"
    if pdg & PION:     return "dimgray"
    if pdg & POSITRON: return "tab:blue"
    if pdg & ELECTRON: return "tab:cyan"
    if pdg & GAMMA:    return "tab:orange"
    return "tab:green"


def _fin(*v):
    return all(np.isfinite(x) for x in v)


@torch.inference_mode()
def score_in_order(model, ds, device, batch_size=128):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    mud, pid = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.batch)
        n = out["pie_logit"].shape[0]
        mud.append(torch.sigmoid(out.get("muon_dif_logit", torch.zeros(n, device=device))).cpu().numpy())
        pid.append(torch.sigmoid(out.get("pion_dif_logit", torch.zeros(n, device=device))).cpu().numpy())
    return np.concatenate(mud), np.concatenate(pid)


def plot_event(row, info, path):
    az = np.array(row["atar_z"], dtype=float)
    if len(az) == 0:
        return False
    ax_ = np.array(row["atar_x"], dtype=float)
    ay_ = np.array(row["atar_y"], dtype=float)
    aE = np.array(row["atar_E"], dtype=float)
    aview = np.array(row["atar_view"], dtype=float)
    apdg = np.array(row["atar_pdg"], dtype=int)
    aorig = np.array(row.get("atar_origin", np.zeros(len(az))), dtype=int)   # >0 = pileup
    Emax = max(float(aE.max()), 1e-6)
    sizes = 15.0 + 280.0 * (aE / Emax)
    colors = [hit_color(int(p)) for p in apdg]
    # pileup (origin>0) hits get a bright magenta edge so signal vs pileup is obvious
    edgec = np.where(aorig > 0, "magenta", "k")
    edgelw = np.where(aorig > 0, 1.4, 0.3)
    is_yz = aview > 0.5
    is_xz = ~is_yz

    fig, axes = plt.subplots(2, 1, figsize=(6, 12), sharex=True)
    transv = {0: ax_, 1: ay_}
    T = {k: (row.get(f"truth_{k}_x", np.nan), row.get(f"truth_{k}_y", np.nan),
             row.get(f"truth_{k}_z", np.nan)) for k in
         ("pion_stop", "muon_start", "muon_stop", "positron_start", "positron_stop")}
    for ax, sel, tc, tlab in [(axes[0], is_xz, 0, "x"), (axes[1], is_yz, 1, "y")]:
        if sel.any():
            si = np.where(sel)[0]
            ax.scatter(az[sel], transv[tc][sel], s=sizes[sel],
                       c=[colors[i] for i in si],
                       edgecolors=[edgec[i] for i in si],
                       linewidths=[edgelw[i] for i in si], alpha=0.85, zorder=3)
        px, py, pz = T["pion_stop"]
        if _fin(pz, px, py):
            # hollow, smaller star so it marks the stop without covering the hits there
            ax.scatter([pz], [px if tc == 0 else py], marker="*", s=150,
                       facecolors="none", edgecolors="k", linewidths=1.0, zorder=5)

        def seg(a, b, color):
            if _fin(a[2], b[2], a[tc], b[tc]):
                ax.plot([a[2], b[2]], [a[tc], b[tc]], color=color, ls="--", lw=1.6,
                        alpha=0.75, zorder=2)
        seg(T["muon_start"], T["muon_stop"], "tab:red")
        ax.set_ylabel(f"{tlab} (mm)"); ax.grid(alpha=0.3); ax.set_ylim(-11.5, 11.5)
    axes[1].set_xlabel("z (mm)"); axes[1].set_xlim(-0.2, 7.0)

    n_mu = int(((apdg & MUON) > 0).sum())
    n_pos = int(((apdg & POSITRON) > 0).sum())
    verdict = "CUT (vetoed)" if info.get("cut") else "kept"
    fig.suptitle(
        f"{info['cat']}   muDIF={info['mud']:.3f} [{verdict} @ thr={info.get('thr', float('nan')):.3f}]"
        f"   piDIF={info['pid']:.3f}\n"
        f"E_tot={info['E']:.1f} (live {info['live']:.1f} + dead {info['dead']:.1f}) MeV   "
        f"theta={info['theta']:.0f} deg\n"
        f"ATAR {len(az)} hits (e+ {n_pos}, mu {n_mu}, max hitE {aE.max():.2f} MeV)",
        fontsize=9)
    handles = [Line2D([0], [0], marker='o', ls='', mfc=c, mec='k', label=l) for c, l in
               [("tab:blue", "e+ hit"), ("dimgray", "pion hit"), ("tab:red", "muon hit"),
                ("tab:orange", "gamma"), ("tab:green", "other")]]
    handles += [Line2D([0], [0], marker='o', ls='', mfc='tab:gray', mec='magenta', mew=1.4,
                       label='pileup hit (origin>0)'),
                Line2D([0], [0], marker='*', ls='', mfc='none', mec='k', label='pion stop (truth)'),
                Line2D([0], [0], color='tab:red', ls='--', label='muon track (truth)')]
    axes[0].legend(handles=handles, fontsize=7, loc="best", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130); plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", nargs="+", required=True,
                   help="raw eval parquet(s) with ATAR hits — pass BOTH the pie and muDIF "
                        "eval files to compare marginal pie vs marginal muDIF at the same score")
    p.add_argument("--out_dir", default="pie_muonlike_displays")
    p.add_argument("--n_events", type=int, default=6, help="events per category")
    p.add_argument("--max_rows", type=int, default=None, help="cap rows read (speed)")
    p.add_argument("--max_hits", type=int, default=250)
    p.add_argument("--e_split", type=float, default=56.0)
    p.add_argument("--e_max", type=float, default=75.0)
    p.add_argument("--accept_min", type=float, default=0.5)
    p.add_argument("--pie_eff", type=float, default=0.9,
                   help="working point: the muDIF cut threshold = this pie-eff quantile. "
                        "The bias is made by events NEAR this threshold, so that's what we show.")
    p.add_argument("--score_lo", type=float, default=None, help="optional explicit score band low")
    p.add_argument("--score_hi", type=float, default=None, help="optional explicit score band high")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PURITYTailModel(dropout=0.0).to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd, strict=False)

    # Load + score each eval file, keeping ds.df aligned with the scores (shuffle=False),
    # then combine. Pass the pie eval AND the muDIF eval so both populations are present.
    dfs, muds, pids = [], [], []
    for path in args.data:
        dfr = pd.read_parquet(path)
        if args.max_rows:
            dfr = dfr.head(args.max_rows)
        ds = PURITYDataset(dataframe=dfr, max_hits=args.max_hits)
        m, p_ = score_in_order(model, ds, device)
        assert len(m) == len(ds.df), f"score/row misalignment {len(m)} vs {len(ds.df)} ({path})"
        dfs.append(ds.df.reset_index(drop=True)); muds.append(m); pids.append(p_)
    d = pd.concat(dfs, ignore_index=True)
    mud = np.concatenate(muds); pid = np.concatenate(pids)

    live = d["live_E"].to_numpy().astype(float)
    dead = d["dead_E"].to_numpy().astype(float)
    E = live + dead
    theta = np.degrees(d["truth_theta"].to_numpy().astype(float)) if "truth_theta" in d.columns else np.full(len(d), np.nan)
    if "truth_acceptance" not in d.columns:
        raise ValueError("parquet has no 'truth_acceptance' column — cannot apply the acceptance "
                         "cut. Use the mixed eval parquets, not unmixed ones.")
    acc = d["truth_acceptance"].to_numpy() >= args.accept_min
    is_pie = (d["truth_is_pie"].to_numpy() == 1) if "truth_is_pie" in d.columns else np.zeros(len(d), bool)
    is_mudif = (d["truth_is_mudif"].to_numpy() == 1) if "truth_is_mudif" in d.columns else np.zeros(len(d), bool)

    # ACCEPTANCE FIRST: every downstream mask (thr, tail/peak/mudif, selection) uses acc.
    pie = is_pie & acc & np.isfinite(E) & (E <= args.e_max)
    mudif_pop = is_mudif & acc & np.isfinite(E)          # muDIF: NOT e_max-capped (not pie)
    if int(pie.sum()) == 0:
        raise ValueError("no accepted pie — pass the pie eval parquet in --data")
    print(f"events={len(d)}  accepted pie={int(pie.sum())}  accepted muDIF={int(mudif_pop.sum())} "
          f"(>={args.accept_min:g} acceptance)", flush=True)
    thr = float(np.quantile(mud[pie], args.pie_eff))     # the actual cut at this working point
    tail = pie & (E < args.e_split)
    peak = pie & (E >= args.e_split)
    # The bias-introducing events are MARGINAL: near the cut. The cut removes score > thr.
    if args.score_lo is not None or args.score_hi is not None:
        lo = args.score_lo if args.score_lo is not None else 0.0
        hi = args.score_hi if args.score_hi is not None else 1.0
        in_band = (mud >= lo) & (mud <= hi)
        center = 0.5 * (lo + hi)
        band_desc = f"score in [{lo:.3f},{hi:.3f}]"
    else:
        in_band = np.ones(len(d), bool)
        center = thr
        band_desc = f"closest to thr={thr:.3f}"
    cats = {"marg_tail":  np.where(tail & in_band)[0],       # pie, tail, near cut
            "marg_peak":  np.where(peak & in_band)[0],       # pie, peak, near cut
            "marg_mudif": np.where(mudif_pop & in_band)[0]}  # real muDIF, near cut (compare!)
    print(f"cut thr(pie_eff={args.pie_eff:g})={thr:.3f}   selecting {band_desc}   "
          f"tail={int(tail.sum())} peak={int(peak.sum())} mudif={int(mudif_pop.sum())}", flush=True)
    for cat, idx in cats.items():
        order = idx[np.argsort(np.abs(mud[idx] - center))]    # most marginal first
        made = 0
        for i in order:
            if made >= args.n_events:
                break
            row = d.iloc[int(i)]
            info = dict(cat=cat, mud=float(mud[i]), pid=float(pid[i]), E=float(E[i]),
                        live=float(live[i]), dead=float(dead[i]), theta=float(theta[i]),
                        thr=thr, cut=bool(mud[i] > thr))
            path = os.path.join(args.out_dir, f"{cat}_{made:02d}.png")
            if plot_event(row, info, path):
                made += 1
        print(f"  {cat}: {made} displays (of {len(idx)} candidates)", flush=True)
    print(f"\nwrote displays to {args.out_dir}/  "
          f"(marg_tail vs marg_peak at the same near-threshold scores)", flush=True)


if __name__ == "__main__":
    main()
