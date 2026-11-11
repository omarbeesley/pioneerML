"""Save what PURITYModel's positron-direction head reads, to train and score heads on a frozen network.

    python -m purity.training.direction_heads.dump_inputs --config PROFILE.json --checkpoint CKPT.pth --data FILE.parquet --out DIR
        [--chunk K] [--row_groups G ...] [--max_events N] [--batch_size 200] [--workers 2]
        [--heads DIR] [--no_features]

The direction head is dir_attn_pool + positron_dir_head. It reads the network's per-hit features
(450 numbers) of the ATAR hits tagged as positron, pooled per strip view, and the predicted pion
stop. A hit is tagged when its slice's trigger probability exceeds PURITY_POS_TRIG_THR and its MIP
probability exceeds PURITY_POS_MIP_THR (both from PROFILE's environment; a low trigger threshold
saves a superset, and the saved probabilities let any higher threshold be applied later). Forward
hooks on the two modules capture those tensors; the model code is not changed. The model runs as
in purity.benchmark (train mode, dropout from the profile, no gradients, events in file order), so
event i of a part is row i of the benchmark's output for the same file.

One part (a directory) is written per call, or one per row group with --chunk / --row_groups
(DIR/rg_XXXX). --chunk K takes the row groups of chunk K (0-based) of the trainer's chunked-epoch
plan (chunk_seed 12345, 1M events). Per part:

    H.npy          [hits, 450] float16   hit features, hits of an event contiguous, x view then y
    hit_ev.npy     [hits] int32          event index within the part
    hit_view.npy   [hits] uint8          0 = x strips, 1 = y strips
    hit_trig.npy, hit_mip.npy  float32   the hit's trigger and MIP probabilities
    hit_true.npy   [hits] uint8          truth: a hit of the trigger positron
    ev_pion.npy    [events, 3] float32   predicted pion stop, as the head receives it
    ev_dir.npy     [events, 3] float32   the checkpoint's own predicted direction
    ev_theta.npy, ev_acc.npy, ev_phtp.npy   the checkpoint's polar angle, acceptance, positron claim
    ev_tar.npy     [events, 3] float32   truth direction (the training target)
    ev_htp.npy     [events] uint8        truth: the event has a trigger positron
    ev_ntrue.npy   [events] int32        truth: number of trigger-positron ATAR hits
    ev_row.npy     [events] int64        row of the input before the hit filter (-1 if undefined)
    meta.json                            counts and settings, written last

--heads DIR runs every trained head DIR/NAME/head_XX.pt (train.py) on the captured inputs
at full precision and adds ev_headtheta.npy [events, K] float32, each head's polar angle (rad),
columns as meta.json's "heads". --no_features then leaves out H.npy and the hit_* files, which is
how a sample too large to cache is scored.

Truth columns are for the training loss and for scoring. Runs with the
release on PYTHONPATH. The network runs without its ensemble heads (PURITY_DIR_HEADS = 0) whatever
the profile says; a checkpoint that carries them is read without them.
"""
import argparse
import json
import os
import time

import numpy as np

DEFAULT_ENV = {"PURITY_GATE_MODE": "3d", "PURITY_BEST_XY_BETA": "0.5", "PURITY_ROLE_FILM": "0",
               "PURITY_ENERGY_TIME_GATE_NS": "0", "PURITY_GATE_DEADZONE_Z_MM": "0",
               "PURITY_GATE_DEADZONE_XY_MM": "0", "PURITY_HIT_ENDPOINTS": "0",
               "PURITY_POS_TRIG_THR": "0.5", "PURITY_POS_MIP_THR": "0.5", "PURITY_HTP_MIN_HITS": "2",
               "PURITY_DIR_HEADS": "0"}
HIT = ("ev", "view", "trig", "mip", "true")
HIT_DTYPE = {"ev": np.int32, "view": np.uint8, "trig": np.float32, "mip": np.float32, "true": np.uint8}
EVT_DTYPE = {"pion": np.float32, "dir": np.float32, "theta": np.float32, "acc": np.float32,
             "phtp": np.float32, "tar": np.float32, "htp": np.uint8, "ntrue": np.int32, "row": np.int64}


def chunk_row_groups(path, k, seed=12345, size=1_000_000):
    """Row groups of chunk k of training/main.py's chunked-epoch plan."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    order = np.random.default_rng(seed).permutation(len(sizes))
    plan, cur, acc = [], [], 0
    for g in order:
        cur.append(int(g)); acc += sizes[g]
        if acc >= size:
            plan.append(cur); cur, acc = [], 0
    if cur:
        plan.append(cur)
    return plan[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--row_groups", type=int, nargs="*")
    ap.add_argument("--max_events", type=int)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--heads")
    ap.add_argument("--no_features", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    env = DEFAULT_ENV | {k: str(v) for k, v in cfg["environment"].items()}
    env["PURITY_MAX_HITS"] = str(cfg["inference"]["max_hits"])
    env["PURITY_DIR_HEADS"] = "0"               # the network alone: these are the inputs of its heads
    os.environ.update(env)                      # before the model import, which reads it
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from purity.benchmark import TASK_WEIGHTS
    from purity.dataset import PURITYDataset
    from purity.models import PURITYModel
    from purity.train_utils import format_targets_from_batch

    trig_thr, mip_thr = float(env["PURITY_POS_TRIG_THR"]), float(env["PURITY_POS_MIP_THR"])
    max_hits = int(env["PURITY_MAX_HITS"])
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PURITYModel(dropout=float(cfg["model_kwargs"]["dropout"])).to(device)
    sd = torch.load(args.checkpoint, map_location=device)["model"]
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("dir_ensemble.")})
    model.train()                               # as purity.benchmark: never eval()
    print(f"device {device}; thresholds trigger {trig_thr} MIP {mip_thr}; checkpoint {args.checkpoint}", flush=True)

    cap = {"pool": [], "head": None}

    def pool_hook(_mod, a, kw):                 # dir_attn_pool(h_v, b_v, dim_size=B)
        cap["pool"].append((a[0], kw["index"] if "index" in kw else a[1]))

    def head_hook(_mod, a):                     # positron_dir_head(cat[pool_x, pool_y, pion_stop])
        cap["head"] = a[0]

    model.dir_attn_pool.register_forward_pre_hook(pool_hook, with_kwargs=True)
    model.positron_dir_head.register_forward_pre_hook(head_hook)

    heads, head_names = [], []
    if args.heads:                              # trained heads to run on the captured inputs
        import glob
        from purity.training.direction_heads.dirhead import head_from_checkpoint
        for f in sorted(glob.glob(f"{args.heads}/*/head_*.pt")):
            heads.append(head_from_checkpoint(f, device).eval())
            head_names.append(f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)[5:-3]}")
        assert heads, f"no heads under {args.heads}"
        # the checkpoint's own head run stand-alone: must reproduce the network's angle
        heads.append(head_from_checkpoint(args.checkpoint, device).eval()); head_names.append("reference/00")
        print(f"{len(heads)} heads: " + ", ".join(sorted({n.split('/')[0] for n in head_names})), flush=True)
    keep = not args.no_features

    def dump(ds, out, row_groups):
        os.makedirs(out, exist_ok=True)
        n = len(ds) if args.max_events is None else min(len(ds), args.max_events)
        it = ds if n == len(ds) else Subset(ds, list(range(n)))
        dl = DataLoader(it, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=(device.type == "cuda"))
        hs, hit, evt = [], {k: [] for k in HIT}, {k: [] for k in EVT_DTYPE if k != "row"}
        head_theta, nh = [], 0
        i0, hmax, t0 = 0, 0.0, time.time()
        with torch.inference_mode():
            for nb, batch in enumerate(dl):
                batch = batch.to(device, non_blocking=True)
                B = batch.num_graphs
                cap["pool"].clear(); cap["head"] = None
                o = model(batch.x, batch.batch, task_weights=TASK_WEIGHTS,
                          triggering_pion_slice=getattr(batch, "atar_triggering_pion_slice", None),
                          truth_positron_mask=None)
                assert cap["head"] is not None, "the direction head did not run for this batch"
                is_x, is_y = batch.x[:, 5] > 0.5, batch.x[:, 6] > 0.5
                is_atar = is_x | is_y
                ba, vx, vy = batch.batch[is_atar], is_x[is_atar], is_y[is_atar]
                trig, mip = o["atar_hit_trigger_prob"].view(-1), o["atar_hit_mip_prob"].view(-1)
                assert trig.numel() == ba.numel() == mip.numel()
                tagged = (trig > trig_thr) & (mip > mip_thr)
                true = torch.zeros_like(tagged)
                if hasattr(batch, "is_trigger_target") and hasattr(batch, "atar_node_pdg_target"):
                    true = (batch.atar_node_pdg_target[:, 2] > 0.5) & batch.is_trigger_target.bool()[is_atar]
                calls, part = iter(cap["pool"]), []
                for view, m in ((0, tagged & vx), (1, tagged & vy)):
                    if not m.any():
                        continue
                    h, b = next(calls)
                    assert h.shape[0] == int(m.sum()) and torch.equal(b, ba[m]), "hook / mask mismatch"
                    part.append((h, b, torch.full_like(b, view), trig[m], mip[m], true[m]))
                assert next(calls, None) is None, "unexpected extra pooling call"
                hb = torch.zeros(0, (cap["head"].size(1) - 3) // 2, device=device)
                bb = vb = torch.zeros(0, dtype=torch.long, device=device)
                if part:
                    h, b, v, tg, mp, tr = (torch.cat(x) for x in zip(*part))
                    order = torch.argsort(b, stable=True)        # hits of an event contiguous, x then y
                    hb, bb, vb = h[order], b[order], v[order]
                    hmax = max(hmax, float(h.abs().max()))
                    nh += len(order)
                    if keep:
                        hs.append(hb.half().cpu().numpy())
                        hit["ev"].append((bb + i0).cpu().numpy())
                        hit["view"].append(vb.cpu().numpy())
                        hit["trig"].append(tg[order].cpu().numpy()); hit["mip"].append(mp[order].cpu().numpy())
                        hit["true"].append(tr[order].cpu().numpy())
                if heads:                                        # full-precision inputs, no float16 step
                    pion = cap["head"][:, -3:]
                    head_theta.append(torch.stack(
                        [torch.acos(hd(hb, bb, vb, pion)[:, 2].clamp(-1.0, 1.0)) for hd in heads], dim=1).cpu().numpy())
                es = o["event_summary"]
                tar = format_targets_from_batch(batch).get("tar_angle_vec_per_graph")
                t = torch.zeros(B, 3, device=device)
                if tar is not None:
                    t[:tar.size(0)] = tar
                evt["pion"].append(cap["head"][:, -3:].cpu().numpy())
                evt["dir"].append(o["atar_positron_dir"].cpu().numpy())
                evt["theta"].append(es["positron_polar_angle"].cpu().numpy())
                evt["acc"].append(es["accepted"].cpu().numpy())
                evt["phtp"].append(es["has_trigger_positron"].cpu().numpy() if "has_trigger_positron" in es
                                   else np.full(B, np.nan))
                evt["tar"].append(t.cpu().numpy())
                evt["htp"].append(batch.has_trigger_positron.view(-1).cpu().numpy())
                evt["ntrue"].append(torch.zeros(B, device=device).index_add_(0, ba, true.float()).cpu().numpy())
                i0 += B
                if nb % 100 == 0:
                    print(f"  batch {nb}/{len(dl)}  events {i0}  hits {nh}  {time.time() - t0:.0f}s", flush=True)
        assert i0 == n
        if keep:
            assert hmax < 6.0e4, f"feature magnitude {hmax} does not fit float16"
            H = np.lib.format.open_memmap(f"{out}/H.npy", mode="w+", dtype=np.float16, shape=(nh, hs[0].shape[1]))
            p = 0
            for x in hs:
                H[p:p + len(x)] = x; p += len(x)
            H.flush(); del H
            for k in HIT:
                np.save(f"{out}/hit_{k}.npy", np.concatenate(hit[k]).astype(HIT_DTYPE[k]))
        if heads:
            np.save(f"{out}/ev_headtheta.npy", np.concatenate(head_theta).astype(np.float32))
        for k, v in evt.items():
            np.save(f"{out}/ev_{k}.npy", np.concatenate(v).astype(EVT_DTYPE[k]))
        row = getattr(ds, "source_row", None)
        np.save(f"{out}/ev_row.npy", np.asarray(row[:n], np.int64) if row is not None else np.full(n, -1, np.int64))
        meta = {"status": "complete", "n_events": n, "n_hits": nh, "trig_thr": trig_thr, "mip_thr": mip_thr,
                "checkpoint": args.checkpoint, "data": args.data, "row_groups": row_groups,
                "max_abs_feature": hmax, "environment": env, "heads": head_names, "features_saved": keep,
                "seconds": round(time.time() - t0, 1)}
        json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
        print(f"wrote {out}: {n} events, {nh} hits ({nh / max(n, 1):.1f} per event), "
              f"max |feature| {hmax:.2f}, {time.time() - t0:.0f}s", flush=True)

    groups = chunk_row_groups(args.data, args.chunk) if args.chunk is not None else args.row_groups
    if groups:
        print(f"row groups {groups}", flush=True)
        for g in groups:
            if os.path.exists(f"{args.out}/rg_{g:04d}/meta.json"):
                print(f"rg {g}: done"); continue
            dump(PURITYDataset(args.data, max_hits=max_hits, row_groups=[g]), f"{args.out}/rg_{g:04d}", [g])
    else:
        dump(PURITYDataset(args.data, max_hits=max_hits), args.out, None)


if __name__ == "__main__":
    main()
