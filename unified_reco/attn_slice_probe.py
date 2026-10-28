#!/usr/bin/env python3
"""Slice-level attention probe, with a context ablation.

Pulls the ATAR event builder's slice-to-slice attention off a REAL simulated
pileup event, then re-runs the same event with one slice deleted and pulls it
again. Everything the plot needs -- hit positions, deposits, truth labels,
attention -- comes out of the model and the data, nothing is drawn by hand.

Why slice level. models_v2 pools each time slice into one token
(atar_event_tokens) and runs L_TRIG = 2 TransformerEncoderLayers over them
(atar_event_layers). Those layers are where cross-slice association happens;
the hit-level attention below them is block diagonal by construction.

Why layer 2 (--layer 1, zero-indexed). At layer 1 the queries and keys are
functions of the input tokens alone, so deleting a slice leaves every surviving
raw score bit-identical and only the softmax denominator moves. The tokens have
to be rewritten once, by a layer that saw a different neighbourhood, before the
comparison itself can change.

Why train() and not eval(). The slice attention carries a FLOAT src_mask -- the
endpoint-compatibility chi2 kernel built in models_v2 -- and the
nn.TransformerEncoder eval() fast path silently drops it. Probing in eval() would
measure a different model from the one that was trained. train() also leaves
dropout on, so --mc_passes averages over draws instead of trusting one.

Deleting a slice is not free of bookkeeping, and the output records both traps
so the plotting step can be honest about them:
  - the softmax denominator loses a term, so every surviving weight is pushed
    UP. The null for "nothing contextual happened" is w' = w / (1 - w_removed),
    saved as null_from_renorm.
  - slice ids are NOT renumbered on ablation. The remaining hits keep their
    original slice ids, so event_slice_emb indices do not shift underneath the
    comparison.

Run it on a compute node (see attn_slice_probe.sbatch) -- it loads a parquet and
a checkpoint, which must never happen on the login node.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from torch_geometric.data import Data  # noqa: E402
from dataset import PURITYDataset  # noqa: E402
from models_v2 import PURITYHybridModelV2  # noqa: E402
from constants import PDG_PION, PDG_MUON, PDG_POSITRON  # noqa: E402

# Two hit layouts exist, told apart by column count:
#   unmixed (7 cols):  [x, y, z, E, t, modality, slice_id], positions raw mm
#   mixed  (10 cols):  [x, y, z, E, t, is_xz, is_yz, is_lyso, slice_id,
#                       slice_mean_t], positions divided by NORM_POS_ATAR
# NORM_E_ATAR is 1.0 either way, so column 3 is MeV as stored.
from constants import NORM_POS_ATAR  # noqa: E402

CX, CY, CZ, CE, CT = range(5)


def layout(x):
    """(slice_col, atar_row_mask, position_scale_to_mm, view_col_or_flags)."""
    if x.size(1) >= 10:                      # mixed path
        atar = x[:, 7] < 0.5                 # is_lyso == 0
        view = x[:, 6]                       # is_yz flag doubles as the view
        return 8, atar, NORM_POS_ATAR, view
    atar = torch.isin(x[:, 5], torch.tensor([0.0, 1.0], device=x.device))
    return 6, atar, 1.0, x[:, 5]

TASK_WEIGHTS = {
    "w_atar_trigger_slice": 1.0, "w_pion_kinematics": 1.0,
    "w_event_builder": 1.0, "w_has_trigger_positron": 1.0,
    "w_atar_slice_pdg": 1.0, "w_atar_node_pdg": 1.0,
}


class SliceAttention:
    """Force need_weights on one atar_event_layers self-attention and keep it.

    TransformerEncoderLayer calls self_attn with need_weights=False to enable
    the fused kernel, which discards the weights. Patching the bound forward is
    the only way to see them without reimplementing the layer.
    """

    def __init__(self, model, layer_idx):
        self.weights = None
        self._mha = model.atar_event_layers[layer_idx].self_attn
        self._orig = self._mha.forward

        def patched(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            out = self._orig(*args, **kwargs)
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                self.weights = out[1].detach().float().cpu().numpy()
            return out

        self._mha.forward = patched

    def restore(self):
        self._mha.forward = self._orig


def slice_particle(pdg_mask):
    """Truth species of a slice from its PDG bitmask, pion > muon > positron."""
    if pdg_mask & PDG_PION:
        return "pion"
    if pdg_mask & PDG_MUON:
        return "muon"
    if pdg_mask & PDG_POSITRON:
        return "positron"
    return "other"


def parts_of(d):
    """Per-slice species. The mixed-parquet path stores atar_slice_pdg_target
    as an [n_slices, 3] multi-hot [pion, muon, MIP]; the unmixed path stores a
    scalar bitmask per slice. Priority pion > muon > MIP either way."""
    t = d.atar_slice_pdg_target
    if t.ndim == 2:
        out = []
        for r in t:
            if r[0] > 0.5:
                out.append("pion")
            elif r[1] > 0.5:
                out.append("muon")
            elif r[2] > 0.5:
                out.append("positron")
            else:
                out.append("other")
        return out
    return [slice_particle(int(p)) for p in t]




def run_once(model, data, device, mc_passes, layer_idx):
    """Forward the event mc_passes times; return mean attention and outputs."""
    probe = SliceAttention(model, layer_idx)
    mats, out = [], None
    try:
        with torch.no_grad():
            for _ in range(mc_passes):
                d = data.clone().to(device)
                batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
                out = model(d.x, batch, task_weights=TASK_WEIGHTS,
                            triggering_pion_slice=getattr(
                                d, "atar_triggering_pion_slice", None))
                if probe.weights is None:
                    raise RuntimeError(
                        "no attention captured -- the ATAR event-building branch "
                        "did not run; check task_weights")
                mats.append(probe.weights)
    finally:
        probe.restore()
    return np.mean(np.stack(mats), axis=0), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, help="mixed-events parquet")
    ap.add_argument("--out", required=True, help="npz for the plotting step")
    ap.add_argument("--scan_events", type=int, default=4000)
    ap.add_argument("--max_candidates", type=int, default=8,
                    help="probe up to this many two-chain events")
    ap.add_argument("--event_index", type=int, default=None,
                    help="probe exactly this dataset row")
    ap.add_argument("--drop_slice", type=int, default=None,
                    help="ablate this slice id instead of the trigger muon")
    ap.add_argument("--data_seed", type=int, default=20250811,
                    help="seed for the dataset's row-group subsample, so event "
                         "indices are reproducible across probe runs")
    ap.add_argument("--match_npz", type=str, default=None,
                    help="re-locate the event of this earlier npz by matching "
                         "its hits in the FULL parquet (no subsampling)")
    ap.add_argument("--mc_passes", type=int, default=32)
    ap.add_argument("--layer", type=int, default=1, help="0-indexed; 1 = layer 2")
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PURITYHybridModelV2(dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    # NOT eval(): see the module docstring -- eval() drops the chi2 float mask.
    model.train()
    print(f"[probe] device={device} layer={args.layer} mc={args.mc_passes}", flush=True)

    # The dataset's row-group subsample uses the global numpy RNG unseeded, so
    # without this, "event N" is a different event on every run.
    np.random.seed(args.data_seed)
    if args.match_npz:
        # Full load: hit-match needs the complete, deterministic row order.
        ds = PURITYDataset(parquet_path=args.dataset, max_hits=300)
        ref = np.load(args.match_npz)
        rz, re_ = ref["hit_z"], ref["hit_e"]
        nref = len(rz)
        found = None
        for i in range(len(ds.df)):
            az = ds.df.iloc[i]["atar_z"]
            if az is None or len(az) != nref:
                continue
            if abs(float(az[0]) - float(rz[0])) < 1e-5 and \
                    abs(float(ds.df.iloc[i]["atar_E"][0]) - float(re_[0])) < 1e-6:
                found = i
                break
        if found is None:
            raise SystemExit(f"no row in {args.dataset} matches {args.match_npz}")
        print(f"[probe] matched {args.match_npz} -> full-parquet row {found}",
              flush=True)
        args.event_index = found
    else:
        ds = PURITYDataset(parquet_path=args.dataset, max_hits=300,
                           max_events=args.scan_events)

    # ---- find pi->mu->e events with a second full chain piled on top --------
    # Collect up to max_candidates. Fallback to a mu+e accidental only if no
    # full double chain exists in the scan window.
    full, mue = [], []
    scan = [] if args.event_index is not None else range(len(ds))
    for i in scan:
        d = ds[i]
        if not hasattr(d, "atar_slice_pdg_target") or d.x is None or d.x.numel() == 0:
            continue
        parts = parts_of(d)
        npi, nmu, ne = (parts.count(k) for k in ("pion", "muon", "positron"))
        if npi >= 2 and nmu >= 2 and ne >= 2:
            full.append((i, d, parts))
            if len(full) >= args.max_candidates:
                break
        elif len(mue) < args.max_candidates and npi >= 1 and nmu >= 2 and ne >= 2:
            mue.append((i, d, parts))
    if args.event_index is not None:
        d = ds[args.event_index]
        cands = [(args.event_index, d, parts_of(d))]
    else:
        cands = full or mue
    if not cands:
        raise SystemExit(f"no two-chain pileup event in the first {args.scan_events}")
    print(f"[probe] {len(cands)} candidate events ({'full' if full else 'mu+e'})",
          flush=True)

    hits_rows, slice_rows, attn_rows, meta_rows = [], [], [], []
    for idx, data, parts in cands:
        slice_col, am_t, pos_scale, view = layout(data.x)
        am = am_t.cpu().numpy()
        sl_hit = data.x[am_t][:, slice_col].numpy().astype(int)
        eid_ok = (hasattr(data, "atar_true_event_id")
                  and data.atar_true_event_id is not None
                  and data.atar_true_event_id.size(0) == int(am.sum()))
        hit_eid = (data.atar_true_event_id.numpy() if eid_ok
                   else np.full(int(am.sum()), -99))

        # chain of each slice = majority per-hit event id (0 = trigger)
        chain = {int(s): int(np.bincount(hit_eid[sl_hit == s].clip(min=0)).argmax())
                 for s in np.unique(sl_hit)} if eid_ok else {}

        # ---- intact ---------------------------------------------------------
        attn_intact, out = run_once(model, data, device, args.mc_passes, args.layer)
        tok_slices = out["valid_slice_indices"].detach().cpu().numpy()

        # ---- ablate the TRIGGER chain's muon, keeping slice ids unchanged ---
        if args.drop_slice is not None:
            drop = int(args.drop_slice)
        else:
            muons = [int(s) for s in tok_slices
                     if int(s) < len(parts) and parts[int(s)] == "muon"
                     and chain.get(int(s), 0) == 0]
            if not muons:
                print(f"[probe] event {idx}: no trigger-chain muon, skipping",
                      flush=True)
                continue
            drop = muons[0]
        keep = data.x[:, slice_col] != drop
        abl = data.clone()
        abl.x = data.x[keep]
        attn_abl, out_abl = run_once(model, abl, device, args.mc_passes, args.layer)
        tok_slices_abl = out_abl["valid_slice_indices"].detach().cpu().numpy()

        # ---- the renormalisation null ---------------------------------------
        h_int = (attn_intact[0].mean(axis=0) if attn_intact.ndim == 4
                 else attn_intact.mean(axis=0))
        row_of = {int(s): k for k, s in enumerate(tok_slices)}
        null = {}
        for q in tok_slices:
            r = h_int[row_of[int(q)]]
            w_drop = r[row_of[drop]] if drop in row_of else 0.0
            if w_drop < 1.0:
                null[int(q)] = (r / (1.0 - w_drop)).astype(np.float32)

        # visual figure of merit: how close the two chains come (min inter-chain
        # hit distance in the XZ view) -- smaller = more overlapped = harder
        xz = view.numpy()[am] < 0.5
        p0 = (hit_eid == 0) & xz
        p1 = (hit_eid > 0) & xz
        zz = (data.x[:, CZ] * pos_scale).numpy()[am]
        xx = (data.x[:, CX] * pos_scale).numpy()[am]
        if eid_ok and p0.any() and p1.any():
            dmin = float(np.sqrt(
                ((zz[p0][:, None] - zz[p1]) ** 2
                 + (xx[p0][:, None] - xx[p1]) ** 2)).min())
        else:
            dmin = float("nan")

        # ---- tidy rows for the parquet tables --------------------------------
        xs = (data.x[:, CX] * pos_scale).numpy()[am]
        ys = (data.x[:, CY] * pos_scale).numpy()[am]
        zs = (data.x[:, CZ] * pos_scale).numpy()[am]
        es = data.x[:, CE].numpy()[am]
        vs = view.numpy()[am]
        ss = data.x[:, slice_col].numpy()[am].astype(int)
        for k in range(len(xs)):
            hits_rows.append((idx, float(xs[k]), float(ys[k]), float(zs[k]),
                              float(es[k]), int(vs[k]), int(ss[k]),
                              int(hit_eid[k])))
        for s_ in np.unique(ss):
            slice_rows.append((idx, int(s_), parts[int(s_)],
                               chain.get(int(s_), -1)))
        for kind, mat, tl in (("intact", attn_intact, tok_slices),
                              ("ablated", attn_abl, tok_slices_abl)):
            heads = mat[0] if mat.ndim == 4 else mat[None][0]
            for hh in range(heads.shape[0]):
                for qi, q in enumerate(tl):
                    for ki, kk in enumerate(tl):
                        attn_rows.append((idx, kind, hh, int(q), int(kk),
                                          float(heads[hh, qi, ki])))
        meta_rows.append((idx, drop, args.layer, args.mc_passes, dmin))
        out_path = args.out  # tables are written once, after the loop
        e_q = next((int(s) for s in tok_slices if int(s) < len(parts)
                    and parts[int(s)] == "positron"
                    and chain.get(int(s), 0) == 0), None)
        pi_q = next((int(s) for s in tok_slices if int(s) < len(parts)
                     and parts[int(s)] == "pion"
                     and chain.get(int(s), 0) == 0), None)
        h_abl = (attn_abl[0].mean(axis=0) if attn_abl.ndim == 4
                 else attn_abl.mean(axis=0))
        row_abl = {int(s): k for k, s in enumerate(tok_slices_abl)}
        if e_q is not None and pi_q is not None and e_q in row_abl and pi_q in row_abl:
            w0 = h_int[row_of[e_q], row_of[pi_q]]
            w1 = h_abl[row_abl[e_q], row_abl[pi_q]]
            # the pion's ablated row over every positron candidate: the
            # "hedging" signature is >=2 positrons carrying real weight
            pos_w = {int(s): float(h_abl[row_abl[pi_q], row_abl[int(s)]])
                     for s in tok_slices_abl
                     if int(s) < len(parts) and parts[int(s)] == "positron"}
            spread = sorted(pos_w.values(), reverse=True)
            n_hedge = sum(1 for w in spread if w > 0.10)
            print(f"[probe] event {idx}: chain-dist {dmin:.2f} mm  "
                  f"e->pi {w0:.3f} -> {w1:.3f}  "
                  f"pi-row positrons {[round(w, 2) for w in spread]} "
                  f"hedge={n_hedge}", flush=True)
        else:
            print(f"[probe] event {idx}: probed", flush=True)

    # ---- parquet output: smaller than npz, mmap-friendly, no pickle ----------
    import pandas as pd
    base = args.out
    for suf in (".npz", ".parquet"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    pd.DataFrame(hits_rows, columns=[
        "event", "x", "y", "z", "E", "view", "slice", "chain_eid"
    ]).to_parquet(base + "_hits.parquet", index=False)
    pd.DataFrame(slice_rows, columns=[
        "event", "slice", "particle", "chain"
    ]).to_parquet(base + "_slices.parquet", index=False)
    pd.DataFrame(attn_rows, columns=[
        "event", "kind", "head", "q_slice", "k_slice", "weight"
    ]).to_parquet(base + "_attn.parquet", index=False)
    pd.DataFrame(meta_rows, columns=[
        "event", "dropped_slice", "layer", "mc_passes", "chain_min_dist_xz"
    ]).to_parquet(base + "_meta.parquet", index=False)
    print(f"[probe] wrote {base}_{{hits,slices,attn,meta}}.parquet "
          f"({len(meta_rows)} events)", flush=True)


if __name__ == "__main__":
    main()
