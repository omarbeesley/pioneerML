"""
Quick feature-importance probe for the muDIF veto head: which of its inputs is the
trained model ACTUALLY using to flag muDIF?

For a checkpoint, runs the backbone+head ONCE on a muDIF + pi->e nu eval set, caches
the head's per-event pieces (the 6 engineered scalars + the attention-pool feature),
then — with NO retraining and only cheap MLP re-runs on the cached tensors:

  - baseline muDIF-vs-pi->e nu AUC of the full head,
  - PERMUTATION IMPORTANCE: shuffle each scalar (and the whole attention-pool branch)
    across events, recompute the logit, report the AUC drop. Big drop = the head
    relies on that input. This measures what the model USES, not just what could
    discriminate.
  - STANDALONE AUC of each scalar on its own (which features individually separate).

ROC/AUC are numpy (sklearn not in pytorch.sif); inference needs torch (run in the
container). Pass the muDIF eval parquet plus the pi->e nu (or michel) one; classes
come from is_mudif_target / is_pie_target.

Usage:
  python mudif_importance.py \
    --checkpoint /pioneerML/model_weights/PURITY_TAIL_MUDIF_recon_best.pth \
    --data /data/tail_reveal_mudif_eval.parquet /data/tail_reveal_pie_eval.parquet \
    --shard_size 200000
"""
import argparse
import math
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_reco.dataset import PURITYDataset
from unified_reco.models_tail import PURITYTailModel, assemble_tail_features
from unified_reco.eval_tail import roc_auc


@torch.inference_mode()
def cache_parts(model, loader, device):
    """Run backbone + muDIF head; cache per-event scalars, pool feature, labels."""
    S, P, ISM, ISP = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model.backbone(batch.x, batch.batch)
        if "h_atar" not in out:                      # batch had no ATAR hits
            continue
        f = assemble_tail_features(out, batch.x, batch.batch, detach=True)
        _, _, parts = model.mudif_veto(f, return_parts=True)
        S.append(parts["scalars"].cpu().numpy())
        P.append(parts["pool_feat"].cpu().numpy())
        ISM.append(batch.is_mudif_target.view(-1).cpu().numpy())
        ISP.append(batch.is_pie_target.view(-1).cpu().numpy())
    if not S:
        return None
    return (np.concatenate(S), np.concatenate(P),
            np.concatenate(ISM), np.concatenate(ISP))


def head_logit(head, scalars_t, pool_t):
    """Recompute the muDIF logit from cached pieces (scalar_mlp + fuse only)."""
    sf = head.scalar_mlp(scalars_t)
    return head.fuse(torch.cat([pool_t, sf], dim=-1)).squeeze(-1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", nargs="+", required=True,
                   help="Eval parquets; classes from is_mudif_target / is_pie_target.")
    p.add_argument("--vs", choices=["pie", "michel"], default="pie",
                   help="Negative class for the AUC (default: pi->e nu).")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_hits", type=int, default=250)
    p.add_argument("--shard_size", type=int, default=0,
                   help="Stream each parquet in row-chunks of this many events (0 = whole file).")
    p.add_argument("--n_shuffles", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    model = PURITYTailModel().to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.checkpoint} ({len(miss)} missing, {len(unexp)} unexpected)", flush=True)
    names = list(model.mudif_veto.SCALAR_NAMES)

    chunks = []
    for path in args.data:
        if args.shard_size > 0:
            pf = pq.ParquetFile(path)
            n_sh = math.ceil(pf.metadata.num_rows / args.shard_size)
            for rb in tqdm(pf.iter_batches(batch_size=args.shard_size),
                           total=n_sh, desc=os.path.basename(path), unit="shard"):
                ds = PURITYDataset(dataframe=rb.to_pandas(), max_hits=args.max_hits)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
                r = cache_parts(model, loader, device)
                if r is not None:
                    chunks.append(r)
        else:
            ds = PURITYDataset(path, max_hits=args.max_hits)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
            r = cache_parts(model, loader, device)
            if r is not None:
                chunks.append(r)
    if not chunks:
        print("[error] no events collected — check --data", flush=True)
        return

    S = np.concatenate([c[0] for c in chunks])     # [N, 6] scalars
    P = np.concatenate([c[1] for c in chunks])     # [N, H] pool feature
    ISM = np.concatenate([c[2] for c in chunks])
    ISP = np.concatenate([c[3] for c in chunks])

    neg = (ISP == 1) if args.vs == "pie" else ((ISP == 0) & (ISM == 0))
    sel = (ISM == 1) | neg
    S, P, y = S[sel], P[sel], ISM[sel].astype(np.float64)   # y=1 muDIF, 0 the negative class
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"\nmuDIF={n_pos}  {args.vs}={n_neg}", flush=True)
    if n_pos == 0 or n_neg == 0:
        print("[error] need both muDIF and the negative class present", flush=True)
        return

    St = torch.tensor(S, dtype=torch.float32, device=device)
    Pt = torch.tensor(P, dtype=torch.float32, device=device)
    head = model.mudif_veto

    with torch.inference_mode():
        base_auc = roc_auc(head_logit(head, St, Pt).cpu().numpy(), y)[2]
    print(f"baseline muDIF-vs-{args.vs} AUC (full head) = {base_auc:.4f}\n", flush=True)

    rng = np.random.default_rng(0)
    N = len(y)
    rows = []
    for i, nm in enumerate(names):                  # scalar features
        stand = roc_auc(S[:, i], y)[2]
        stand = max(stand, 1.0 - stand)             # direction-agnostic separability
        drops = []
        for _ in range(args.n_shuffles):
            Sp = S.copy(); Sp[:, i] = S[rng.permutation(N), i]
            with torch.inference_mode():
                lp = head_logit(head, torch.tensor(Sp, dtype=torch.float32, device=device), Pt)
            drops.append(base_auc - roc_auc(lp.cpu().numpy(), y)[2])
        rows.append((nm, stand, float(np.mean(drops)), float(np.std(drops))))

    pdrops = []                                     # attention-pool branch
    for _ in range(args.n_shuffles):
        Pp = P[rng.permutation(N)]
        with torch.inference_mode():
            lp = head_logit(head, St, torch.tensor(Pp, dtype=torch.float32, device=device))
        pdrops.append(base_auc - roc_auc(lp.cpu().numpy(), y)[2])
    rows.append(("[attention pool]", float("nan"), float(np.mean(pdrops)), float(np.std(pdrops))))

    rows.sort(key=lambda r: -r[2])                  # most-used first
    print(f"{'component':>22}  {'standalone_AUC':>14}  {'perm_importance (dAUC)':>24}")
    print("  " + "-" * 62)
    for nm, st, imp, sd in rows:
        st_s = f"{st:.4f}" if st == st else "  -   "
        print(f"{nm:>22}  {st_s:>14}  {imp:>+13.4f} +/- {sd:.4f}", flush=True)
    print(f"\nperm_importance = drop in muDIF-vs-{args.vs} AUC when that input is shuffled\n"
          f"across events (large => the head relies on it). standalone_AUC = that\n"
          f"scalar's own separability (0.5 = useless).", flush=True)


if __name__ == "__main__":
    main()
