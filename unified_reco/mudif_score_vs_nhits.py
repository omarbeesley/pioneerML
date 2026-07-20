"""
Overlay muDIF-veto score distributions for muDIF events, one curve per
NUMBER OF IN-FLIGHT MUON (muDIF) HITS in the event. Shows how the head's score
depends on how much muon signal is actually visible: events with 0 muDIF hits have
no muon signature at all (un-vetoable -> pile up at low score and leak through),
while more hits -> the head fires harder.

"muDIF hit" = an in-flight muon hit = (atar_pdg & MUON) AND (atar_origin == 0), the
same definition the dataset uses for muon_dif_hit_target. (Pileup muons, origin>0,
are NOT counted.) The predictions parquet lacks per-hit info, so this runs the model
on the RAW muDIF eval parquet and counts the hits from the truth arrays.

Run inside pytorch.sif on the GPU node, e.g.:
  apptainer exec --nv --cleanenv --contain \
    --bind /home/obeesley/pioneerML:/pioneerML --bind /data/nvme0/prod_ml_data:/data \
    /data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif \
    python3 /pioneerML/unified_reco/mudif_score_vs_nhits.py \
      --checkpoint /pioneerML/model_weights/<ckpt>.pth \
      --data /data/tail_reveal_mudif_eval.parquet \
      --out /pioneerML/mudif_score_vs_nhits.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from torch_geometric.loader import DataLoader
from unified_reco.dataset import PURITYDataset
from unified_reco.models_tail import PURITYTailModel

MUON_BIT = 2
GROUPS = ["0", "1", "2", "3", "4-6", "7+"]


def group_of(n):
    if n <= 3:
        return str(n)
    return "4-6" if n <= 6 else "7+"


@torch.inference_mode()
def mudif_scores(model, ds, device, bs=128):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    out = []
    for batch in loader:
        batch = batch.to(device)
        o = model(batch.x, batch.batch)
        n = o["pie_logit"].shape[0]
        out.append(torch.sigmoid(o.get("muon_dif_logit",
                                       torch.zeros(n, device=device))).cpu().numpy())
    return np.concatenate(out)


def n_inflight_muon_hits(row):
    pdg = np.asarray(row["atar_pdg"], dtype=int)
    orig = np.asarray(row["atar_origin"], dtype=int)
    if len(pdg) == 0:
        return 0
    return int((((pdg & MUON_BIT) > 0) & (orig == 0)).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", nargs="+", required=True, help="raw muDIF eval parquet(s)")
    p.add_argument("--out", default="mudif_score_vs_nhits.png")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_hits", type=int, default=250)
    p.add_argument("--accept_min", type=float, default=0.5,
                   help="restrict to truth_acceptance>=this (set <0 to disable)")
    p.add_argument("--n_bins", type=int, default=40, help="score histogram bins")
    p.add_argument("--thr", type=float, default=None,
                   help="optional veto-cut line; prints per-group leakage (score<thr)")
    p.add_argument("--logy", action="store_true", help="log y axis")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PURITYTailModel(dropout=0.0).to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd, strict=False)

    dfs, scrs = [], []
    for path in args.data:
        dfr = pd.read_parquet(path)
        if args.max_rows:
            dfr = dfr.head(args.max_rows)
        ds = PURITYDataset(dataframe=dfr, max_hits=args.max_hits)
        s = mudif_scores(model, ds, device)
        assert len(s) == len(ds.df), f"misalignment {len(s)} vs {len(ds.df)} ({path})"
        dfs.append(ds.df.reset_index(drop=True)); scrs.append(s)
    d = pd.concat(dfs, ignore_index=True)
    score = np.concatenate(scrs)

    is_mudif = (d["truth_is_mudif"].to_numpy() == 1) if "truth_is_mudif" in d.columns \
        else np.ones(len(d), bool)
    sel = is_mudif.copy()
    if args.accept_min >= 0 and "truth_acceptance" in d.columns:
        sel &= (d["truth_acceptance"].to_numpy() >= args.accept_min)
    nhit = np.array([n_inflight_muon_hits(d.iloc[i]) for i in np.where(sel)[0]])
    sc = score[sel]
    grp = np.array([group_of(int(n)) for n in nhit])
    acc_note = f" (truth_acceptance>={args.accept_min:g})" if args.accept_min >= 0 else ""
    print(f"selected muDIF events: {int(sel.sum())}{acc_note}", flush=True)

    bins = np.linspace(0.0, 1.0, args.n_bins + 1)
    cmap = plt.get_cmap("viridis")
    present = [g for g in GROUPS if (grp == g).any()]
    fig, ax = plt.subplots(figsize=(8, 5))
    print(f"\n{'n_muDIF_hits':>12}  {'N':>7}  {'meanScore':>9}" +
          (f"  {'leak(score<thr)':>15}" if args.thr is not None else ""))
    for i, g in enumerate(present):
        m = grp == g
        s_g = sc[m]
        col = cmap(i / max(len(present) - 1, 1))
        lk = f"  {float((s_g < args.thr).mean()):15.4f}" if args.thr is not None else ""
        print(f"{g:>12}  {m.sum():7d}  {s_g.mean():9.4f}{lk}", flush=True)
        ax.hist(s_g, bins=bins, density=True, histtype="step", lw=2.0, color=col,
                label=f"{g} hits (N={m.sum()}, <s>={s_g.mean():.2f})")
    if args.thr is not None:
        ax.axvline(args.thr, color="k", ls="--", lw=1.2, label=f"cut thr={args.thr:g}")
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("muDIF-veto score (sigmoid)")
    ax.set_ylabel("density (per n-hits group)")
    ax.set_title("muDIF score vs number of in-flight muon (muDIF) hits")
    ax.legend(fontsize=8, title="in-flight muon hits")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=130); plt.close(fig)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
