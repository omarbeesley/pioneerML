"""Train positron-direction heads on a frozen network, from its cached head inputs.

    python -m purity.training.direction_heads.train --cache DIR --val DIR --reference CKPT --out DIR --variants FILE.json
        [--only NAME ...] [--pool_trig 0.5]

FILE.json lists the variants: {"name", "heads": [first, last], "mask_frac", "init": "random" |
"checkpoint", "fixed_init", "fixed_mask", "schedule"}. Each head is a DirHead (dirhead.py) trained
with the trainer's angle loss and optimiser settings (training/main.py, train_utils.py): batches of
100 events; loss 0.25 * mean(1 - cos) over the events that pass the match gate + 0.05 * the
moment-decorrelation term (10-batch running mean of the polar-angle residual and its first two
moments in the true cos(theta)); AdamW, weight decay 1e-3, gradient norm clipped at 2. A schedule
is a list of learning rate : steps; the default mirrors the model's own decay and ends at 6.25e-5,
the rate of epochs 15-19.

What differs between heads of a variant:
    starting weights   init "random": PyTorch's default initialisation with seed 1000 + i (the
                       model uses the same default); init "checkpoint": REFERENCE's own head.
                       fixed_init gives every head seed 1000.
    events seen        a random fraction mask_frac of the cached events, drawn with seed 2000 + i
                       (fixed_mask: seed 2000 for every head). 1.0 = all events.
    batch order        a fresh shuffle of the head's events each pass, seed 3000 + i.

The heads of a variant are trained together: their weights are stacked along a leading axis and
every step runs all of them, each on its own batch. The heads do not interact: the loss is a sum
of per-head losses, the gradient norm is clipped per head, and AdamW acts element-wise. Before
training, the stacked forward pass and its gradient are checked against the plain module.

--pool_trig is the trigger-probability threshold of the hits the heads pool (0.5 = as the model was
trained). At start the stand-alone head is checked against the network: REFERENCE's head applied
to the validation cache at the dump's own threshold must reproduce the directions the network
itself produced. Per head, OUT/NAME/head_XX.pt (state dict + record) and one line of
OUT/NAME/summary.jsonl: the training loss, and on the validation cache the angle loss, the mean
dz error and the polar-angle median error and resolution (interquartile range / 1.349) for true
angles 110-130 degrees, next to REFERENCE's head on the same events. Truth enters the loss and the
scoring only. Runs on a GPU; variants.json holds the sets trained for the release checkpoint.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax as group_softmax

from purity.training.direction_heads.dirhead import JK, Cache, DirHead, angle_deg, head_from_checkpoint, parts

BATCH, W_ANGLE, W_DECORR, DECORR_ALPHA, WEIGHT_DECAY, CLIP = 100, 0.25, 0.05, 0.1, 1e-3, 2.0
SCHEDULE = "5e-4:20000,2.5e-4:10000,1.25e-4:10000,6.25e-5:20000"
G0W, G0B, G2W, G2B, W0, B0, W2, B2, W4, B4 = NAMES = (
    "dir_attn_pool.gate_nn.0.weight", "dir_attn_pool.gate_nn.0.bias",
    "dir_attn_pool.gate_nn.2.weight", "dir_attn_pool.gate_nn.2.bias",
    "positron_dir_head.0.weight", "positron_dir_head.0.bias", "positron_dir_head.2.weight",
    "positron_dir_head.2.bias", "positron_dir_head.4.weight", "positron_dir_head.4.bias")


def score(cache, pred):
    """Angle loss, dz error and near-cut polar-angle error of predictions on a cache."""
    g, tar = cache.gate, cache.tar
    tt = torch.rad2deg(torch.acos(tar[:, 2].clamp(-1, 1)))
    w = g & (tt > 110) & (tt < 130)
    r = (torch.rad2deg(torch.acos(pred[w, 2].clamp(-1, 1))) - tt[w]).cpu().numpy()
    q25, q50, q75 = np.percentile(r, [25, 50, 75])
    return {"val_loss": float((1.0 - F.cosine_similarity(pred[g], tar[g], dim=1)).mean()),
            "val_dz": float((pred[g, 2] - tar[g, 2]).mean()),
            "cut_med": float(q50), "cut_sig": float((q75 - q25) / 1.349), "cut_n": int(w.sum())}


def lin(x, w, b):
    """Per-head linear layer: x [K, n, in], w [K, out, in], b [K, out]."""
    return torch.baddbmm(b[:, None, :], x, w.transpose(1, 2))


def stack_forward(P, cache, E, hk, K):
    """Directions of events E [M] (grouped by head, hk [M] = head of each event) from the stacked
    weights P: the same computation as DirHead.forward, each head on its own events."""
    dev, M = E.device, len(E)
    cnt = torch.bincount(hk, minlength=K)
    epos = torch.arange(M, device=dev) - (torch.cumsum(cnt, 0) - cnt)[hk]        # event's slot in its head
    h, loc, view = cache.gather(E)
    hh = hk[loc]
    hc = torch.bincount(hh, minlength=K)
    hpos = torch.arange(len(hh), device=dev) - (torch.cumsum(hc, 0) - hc)[hh]    # hit's slot in its head
    x = h.new_zeros(K, int(hc.max()), JK)
    x[hh, hpos] = h
    gate = lin(F.gelu(lin(x, P[G0W], P[G0B])), P[G2W], P[G2B])[hh, hpos, 0]
    group = 2 * loc + view.long()                                               # (event, view)
    a = group_softmax(gate, group, num_nodes=2 * M)
    pooled = h.new_zeros(2 * M, JK).index_add_(0, group, a[:, None] * h).view(M, 2 * JK)
    y = h.new_zeros(K, int(cnt.max()), 2 * JK + 3)
    y[hk, epos] = torch.cat([pooled, cache.pion[E]], dim=1)
    y = lin(F.gelu(lin(F.gelu(lin(y, P[W0], P[B0])), P[W2], P[B2])), P[W4], P[B4])
    return F.normalize(y[hk, epos], p=2, dim=-1, eps=1e-6), cnt


def stack_loss(P, cache, E, hk, K, ema):
    """Sum over heads of the trainer's angle loss; returns it with the per-head angle losses and
    the updated running moments."""
    pred, cnt = stack_forward(P, cache, E, hk, K)
    tar = cache.tar[E]
    n = cnt.clamp(min=1).float()
    l_angle = pred.new_zeros(K).index_add_(0, hk, 1.0 - F.cosine_similarity(pred, tar, dim=1)) / n
    eps = 1e-4
    r = torch.acos(pred[:, 2].clamp(-1 + eps, 1 - eps)) - torch.acos(tar[:, 2].clamp(-1 + eps, 1 - eps))
    mom = torch.stack([r, r * tar[:, 2], r * (2.0 * tar[:, 2] ** 2 - 1.0)], dim=1)
    m = pred.new_zeros(K, 3).index_add_(0, hk, mom) / n[:, None]
    ok = cnt >= 10
    ms = DECORR_ALPHA * m + (1.0 - DECORR_ALPHA) * ema
    loss = (W_ANGLE * l_angle + W_DECORR * (ms ** 2).sum(1) / DECORR_ALPHA * ok).sum()
    return loss, l_angle.detach(), torch.where(ok[:, None], ms.detach(), ema)


def stacked(heads):
    return {n: torch.stack([h.state_dict()[n] for h in heads]).clone().requires_grad_(True) for n in NAMES}


def check_stack(cache, heads, events, log):
    """The stacked forward pass and gradient against the plain module, on each head's first batch."""
    dev, K = cache.device, len(heads)
    bs = [torch.as_tensor(e[:BATCH], device=dev) for e in events]
    bs = [b[cache.gate[b]] for b in bs]
    E = torch.cat(bs)
    hk = torch.repeat_interleave(torch.arange(K, device=dev), torch.tensor([len(b) for b in bs], device=dev))
    P = stacked(heads)
    loss, _, _ = stack_loss(P, cache, E, hk, K, torch.zeros(K, 3, device=dev))
    loss.backward()
    with torch.no_grad():
        pred = stack_forward(P, cache, E, hk, K)[0]
    dp = dg = 0.0
    for k, (head, b) in enumerate(zip(heads, bs)):
        head.zero_grad()
        p = head(*cache.gather(b), cache.pion[b])
        tar = cache.tar[b]
        eps = 1e-4
        r = torch.acos(p[:, 2].clamp(-1 + eps, 1 - eps)) - torch.acos(tar[:, 2].clamp(-1 + eps, 1 - eps))
        m = torch.stack([r.mean(), (r * tar[:, 2]).mean(), (r * (2.0 * tar[:, 2] ** 2 - 1.0)).mean()])
        l = W_ANGLE * (1.0 - F.cosine_similarity(p, tar, dim=1)).mean()
        if len(b) >= 10:
            l = l + W_DECORR * ((DECORR_ALPHA * m) ** 2).sum() / DECORR_ALPHA
        l.backward()
        dp = max(dp, float((p.detach() - pred[hk == k]).abs().max()))
        # relative to the head's largest gradient: the gate's output bias has an exactly zero
        # gradient (the softmax ignores a common shift), so a per-tensor ratio compares noise
        scale = max(float(q.grad.abs().max()) for q in head.parameters())
        for (n, q) in head.named_parameters():
            dg = max(dg, float((q.grad - P[n].grad[k]).abs().max()) / scale)
        head.zero_grad()
    log(f"  stacked against plain module, {K} heads: max direction difference {dp:.2e}, "
        f"max relative gradient difference {dg:.2e}")
    assert dp < 1e-4 and dg < 1e-2, "the stacked heads do not reproduce the plain module"


def train_stack(cache, heads, events, schedule, order_seeds, log):
    """Train the heads in place, each on its own events and batch order. Returns the per-head
    angle loss averaged over the last 2000 steps, the step count and the wall time."""
    dev, K = cache.device, len(heads)
    P = stacked(heads)
    opt = torch.optim.AdamW(list(P.values()), lr=schedule[0][0], weight_decay=WEIGHT_DECAY)
    gens = [torch.Generator().manual_seed(s) for s in order_seeds]
    evs = [torch.as_tensor(e, device=dev) for e in events]
    perms, pos = [None] * K, [0] * K
    ema = torch.zeros(K, 3, device=dev)
    ks = torch.arange(K, device=dev)
    hist, step, t0 = [], 0, time.time()
    for lr, n in schedule:
        for pg in opt.param_groups:
            pg["lr"] = lr
        for _ in range(n):
            bs = []
            for k in range(K):
                if perms[k] is None or pos[k] >= len(perms[k]):
                    perms[k], pos[k] = evs[k][torch.randperm(len(evs[k]), generator=gens[k]).to(dev)], 0
                bs.append(perms[k][pos[k]:pos[k] + BATCH]); pos[k] += BATCH
            E = torch.cat(bs)
            hk = torch.repeat_interleave(ks, torch.tensor([len(b) for b in bs], device=dev))
            g = cache.gate[E]                              # only gated events enter the angle loss
            loss, l_angle, ema = stack_loss(P, cache, E[g], hk[g], K, ema)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = sum((p.grad.flatten(1) ** 2).sum(1) for p in P.values()).sqrt()
            coef = torch.where(torch.isfinite(norm), (CLIP / (norm + 1e-6)).clamp(max=1.0), torch.zeros_like(norm))
            for p in P.values():
                p.grad.nan_to_num_(0.0, 0.0, 0.0).mul_(coef.view(-1, *[1] * (p.dim() - 1)))
            opt.step()
            hist.append(l_angle); step += 1
            if step % 5000 == 0:
                log(f"    step {step}: lr {lr:g}, angle loss {float(torch.stack(hist[-2000:]).mean()):.5f} "
                    f"(mean over heads), {(time.time() - t0) / step * 1e3:.2f} ms/step")
            if len(hist) > 4000:
                del hist[:2000]
    for k, head in enumerate(heads):
        head.load_state_dict({n: P[n][k].detach() for n in NAMES})
    return torch.stack(hist[-2000:]).mean(0).cpu().numpy(), step, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--pool_trig", type=float, default=0.5)
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [v for v in json.load(open(args.variants)) if not args.only or v["name"] in args.only]
    assert list(DirHead().state_dict()) == list(NAMES)

    ref = head_from_checkpoint(args.reference, dev).eval()
    dump_thr = json.load(open(f"{parts(args.val)[0]}/meta.json"))["trig_thr"]
    chk = Cache(args.val, dump_thr, dev, log)
    d = angle_deg(chk.predict(ref), chk.own).cpu().numpy()
    log(f"stand-alone head against the network on {len(d)} validation events (degrees between directions): "
        f"median {np.median(d):.5f}, 99% {np.percentile(d, 99):.5f}, 99.9% {np.percentile(d, 99.9):.5f}, "
        f"max {d.max():.4f}")
    assert np.median(d) < 0.05 and np.percentile(d, 99) < 0.5, "the stand-alone head does not reproduce the network"
    del chk
    torch.cuda.empty_cache()

    val = Cache(args.val, args.pool_trig, dev, log)
    ref_score = score(val, val.predict(ref))
    log(f"reference head on validation: {ref_score}")
    cache = Cache(args.cache, args.pool_trig, dev, log)

    for v in variants:
        name, (lo, hi) = v["name"], v["heads"]
        frac, init, sched = v.get("mask_frac", 1.0), v.get("init", "random"), v.get("schedule", SCHEDULE)
        schedule = [(float(a), int(b)) for a, b in (s.split(":") for s in sched.split(","))]
        out = f"{args.out}/{name}"
        os.makedirs(out, exist_ok=True)
        ids = [i for i in range(lo, hi + 1) if not os.path.exists(f"{out}/head_{i:02d}.pt")]
        log(f"\nvariant {name}: heads {lo}-{hi} ({len(ids)} to train), mask {frac}, init {init}"
            f"{' (same for all)' if v.get('fixed_init') else ''}{', same events for all' if v.get('fixed_mask') else ''}, "
            f"pool trigger > {args.pool_trig}, schedule {schedule}")
        if not ids:
            continue
        heads, events, recs = [], [], []
        for i in ids:
            init_seed = 1000 + (0 if v.get("fixed_init") else i)
            mask_seed = 2000 + (0 if v.get("fixed_mask") else i)
            if init == "checkpoint":
                heads.append(head_from_checkpoint(args.reference, dev))
            else:
                torch.manual_seed(init_seed)
                heads.append(DirHead().to(dev))
            events.append(np.flatnonzero(np.random.default_rng(mask_seed).random(cache.n_events) < frac))
            recs.append({"variant": name, "head": i, "init": init, "init_seed": init_seed, "mask_seed": mask_seed,
                         "mask_frac": frac, "order_seed": 3000 + i, "pool_trig": args.pool_trig,
                         "schedule": sched, "n_events": int(len(events[-1]))})
        log(f"  events per head: {min(len(e) for e in events)} .. {max(len(e) for e in events)}")
        check_stack(cache, heads, events, log)
        tl, steps, secs = train_stack(cache, heads, events, schedule, [3000 + i for i in ids], log)
        log(f"  trained {len(ids)} heads, {steps} steps in {secs:.0f}s ({secs / steps * 1e3:.1f} ms/step)")
        for head, rec, t in zip(heads, recs, tl):
            head.eval()
            rec.update({"steps": steps, "train_loss": float(t), **score(val, val.predict(head)),
                        "reference": ref_score})
            torch.save({"head": {k: x.cpu() for k, x in head.state_dict().items()}, "record": rec},
                       f"{out}/head_{rec['head']:02d}.pt")
            with open(f"{out}/summary.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"  head {rec['head']:2d}: train {t:.5f}, val {rec['val_loss']:.5f} (reference "
                f"{ref_score['val_loss']:.5f}), near-cut median {rec['cut_med']:+.3f} sigma {rec['cut_sig']:.3f} "
                f"(reference {ref_score['cut_med']:+.3f} / {ref_score['cut_sig']:.3f})")


if __name__ == "__main__":
    main()
