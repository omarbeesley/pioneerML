"""
Hit-level / head-internal diagnosis of WHY the muDIF head scores some pi->e nu events
high (the source of the tail-fraction bias). Per-event aggregates in predictions.parquet
don't explain it, so this runs the model itself and exposes MuonDIFVetoHead's internal
features (return_parts=True) plus raw ATAR hit structure, then profiles them for:
  - high-muDIF pie (top decile by score)  vs  low-muDIF pie (bottom half)
  - tail pie (E<e_split)                  vs  peak pie (E>=e_split)
where E = live (lyso+atar_posE) + dead_E (the complete deposited energy).

Whichever internal feature (max_g, kink_diff, corr_dist_ion, ...) or hit observable
(n_hits, max hit E, ...) is elevated in BOTH the high-score set AND the tail is the
mechanism driving the bias.

Run inside pytorch.sif on the GPU node, e.g.:
  apptainer exec --nv --cleanenv --contain \
    --bind /home/obeesley/pioneerML:/pioneerML --bind /data/nvme0/prod_ml_data:/data \
    /data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif \
    python3 /pioneerML/unified_reco/mudif_diagnose.py \
      --checkpoint /pioneerML/model_weights/<ckpt>.pth \
      --data /data/tail_reveal_pie_eval.parquet --max_events 200000
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_reco.dataset import PURITYDataset
from unified_reco.models_tail import (PURITYTailModel, assemble_tail_features,
                                      scatter_max_dense, MuonDIFVetoHead)


@torch.inference_mode()
def collect(model, loader, device):
    SCN = list(MuonDIFVetoHead.SCALAR_NAMES)
    acc = {k: [] for k in (["score", "n_hits", "sum_E", "max_hitE",
                            "E_tot", "atar_posE", "theta", "is_pie", "acceptance"] + SCN)}
    for batch in loader:
        batch = batch.to(device)
        output = model.backbone(batch.x, batch.batch)
        if "h_atar" not in output:
            continue
        f = assemble_tail_features(output, batch.x, batch.batch, detach=model.freeze_trunk)
        mlogit, _node, parts = model.mudif_veto(f, return_parts=True)
        B = output["num_graphs_in_batch"]
        scal = parts["scalars"]                       # [B, 6]
        for i, nm in enumerate(SCN):
            acc[nm].append(scal[:, i].cpu().numpy())
        acc["score"].append(torch.sigmoid(mlogit).cpu().numpy())
        # raw ATAR hit structure per event
        is_atar = output["is_atar"]
        he = batch.x[is_atar, 3].float()
        ba = batch.batch[is_atar].long()
        nh = torch.zeros(B, device=device).index_add_(0, ba, torch.ones_like(he))
        se = torch.zeros(B, device=device).index_add_(0, ba, he)
        mx = scatter_max_dense(he, ba, B)
        acc["n_hits"].append(nh.cpu().numpy())
        acc["sum_E"].append(se.cpu().numpy())
        acc["max_hitE"].append(mx.cpu().numpy())
        live = batch.live_E_target.view(-1).float()
        dead = batch.dead_E_target.view(-1).float()
        acc["E_tot"].append((live + dead).cpu().numpy())
        acc["atar_posE"].append(batch.atar_posE_target.view(-1).cpu().numpy())
        acc["theta"].append(torch.rad2deg(batch.positron_theta_target.view(-1)).cpu().numpy()
                            if hasattr(batch, "positron_theta_target")
                            else np.zeros(B))
        acc["is_pie"].append(batch.is_pie_target.view(-1).cpu().numpy())
        acc["acceptance"].append(batch.acceptance_target.view(-1).cpu().numpy())
    return {k: (np.concatenate(v) if v else np.empty(0)) for k, v in acc.items()}


def profile(d, names, sel_a, sel_b, label_a, label_b):
    print(f"\n{'feature':18s} {label_a:>12s} {label_b:>12s} {'ratio':>8s}")
    for nm in names:
        a, b = d[nm][sel_a].mean(), d[nm][sel_b].mean()
        r = (b / a) if abs(a) > 1e-9 else float("nan")
        print(f"{nm:18s} {a:12.4f} {b:12.4f} {r:8.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", nargs="+", required=True, help="pie eval parquet(s)")
    p.add_argument("--max_events", type=int, default=None)
    p.add_argument("--max_hits", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--e_split", type=float, default=56.0)
    p.add_argument("--e_max", type=float, default=75.0)
    p.add_argument("--accept_min", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PURITYTailModel(dropout=0.0).to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.checkpoint} ({len(miss)} missing / {len(unexp)} unexpected)", flush=True)

    arrs = []
    for path in args.data:
        ds = PURITYDataset(path, max_hits=args.max_hits, max_events=args.max_events)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        arrs.append(collect(model, loader, device))
    d = {k: np.concatenate([a[k] for a in arrs]) for k in arrs[0]}

    # accepted pie, complete energy, drop >e_max artifacts
    pie = (d["is_pie"] == 1) & (d["acceptance"] >= args.accept_min) \
        & np.isfinite(d["E_tot"]) & (d["E_tot"] <= args.e_max)
    for k in d:
        d[k] = d[k][pie]
    s = d["score"]
    lo = s < np.quantile(s, 0.5)
    hi = s >= np.quantile(s, 0.9)
    peak = d["E_tot"] >= args.e_split
    tail = d["E_tot"] < args.e_split
    feats = list(MuonDIFVetoHead.SCALAR_NAMES) + \
        ["n_hits", "sum_E", "max_hitE", "atar_posE", "theta", "E_tot"]

    print(f"\naccepted pie N={len(s)}  (peak {int(peak.sum())} / tail {int(tail.sum())})")
    print("\n### muDIF head INTERNALS: high-score pie (top10%) vs low-score pie (bottom50%)")
    profile(d, feats, lo, hi, "low-muDIF", "hi-muDIF")
    print("\n### same INTERNALS: tail pie vs peak pie (this is the bias)")
    profile(d, feats + ["score"], peak, tail, "peak", "tail")
    print("\nThe feature elevated in BOTH columns is what drives the tail bias.")


if __name__ == "__main__":
    main()
