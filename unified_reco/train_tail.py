"""
Tail-reveal trainer: trains PURITYTailModel (ATAR-only trunk + three veto/tag
heads) on the mixed tail_reveal datasets produced by generate_tail_reveal.py.

This is the training entry point that commit ae7e60d never added — the model
(models_tail.PURITYTailModel), its loss (models_tail.pie_tagger_loss), and the
per-graph / per-hit targets (dataset.py: is_pie_target, muon_present_target,
pileup_present_target, muon_hit_target, pileup_hit_target) all already exist;
this script wires them into a loop.

Two things this script supplies that the rest of the codebase does not:
  1. The `targets` dict pie_tagger_loss expects. The dataset stores the labels
     under *_target attribute names; the loss wants bare keys (is_pie, ...).
     It also wants `muon_hit_mask` ([N_atar] bool), which NOTHING in the repo
     produces. ATAR hits carry no radioactivity (origins are all >= 0), so the
     natural default is "all ATAR hits participate" -> torch.ones(...).bool().
     Override with --aux_mask if you later want to restrict it.
  2. A tail-specific optimizer. train_purity.build_optimizer references
     hybrid-model attributes (sigma_t_atar_ns, slim_event_transformer,
     event_head, ...) that PURITYTailModel does not have, so it cannot be
     reused here.

Training mode (see --freeze_trunk):
  Default is END-TO-END (freeze_trunk=False). Stage-1 frozen-backbone transfer
  is NOT the default because the only saved PURITY checkpoints are POST-stereo
  while PURITYTailBackbone is a PRE-stereo ATAR re-implementation: only ~78% of
  backbone tensors load by name+shape, so a frozen trunk would run ~22% of its
  weights (kinematics/event MLPs, pooling, self-attn, positron-dir head) at
  random init. If you do pass --freeze_trunk you should also pass
  --backbone_ckpt so the trunk is at least partially seeded.

Usage:
    python train_tail.py \
        --train_path /data/tail_reveal_training.parquet \
        --val_path   /data/tail_reveal_validation.parquet \
        --weights_dir /pioneerML/model_weights --run_name PURITY_TAIL_MUDIF --epochs 15
"""
import argparse
import os
import sys
import time
import traceback

import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_reco.models_tail import PURITYTailModel, pie_tagger_loss
from unified_reco.dataset import PURITYDataset
from unified_reco.train_utils import PURITYLoss, format_targets_from_batch


# ---------------------------------------------------------------------------
# Optimizer — tail-specific (the hybrid build_optimizer references attributes
# this model lacks). Single AdamW group over trainable params; with
# freeze_trunk=True only the three heads have requires_grad=True.
# ---------------------------------------------------------------------------
def build_tail_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Target adapter — map dataset *_target attrs to the keys pie_tagger_loss wants
# and synthesize the missing muon_hit_mask.
# ---------------------------------------------------------------------------
def build_targets(batch, device, aux_mask="all"):
    muon_hits     = batch.muon_hit_target.to(device).float()
    pileup_hits   = batch.pileup_hit_target.to(device).float()
    muon_dif_hits = batch.muon_dif_hit_target.to(device).float()
    pion_dif_hits = batch.pidif_hit_target.to(device).float()

    if aux_mask == "muon":          # only hits with a muon label feed aux losses
        mask = muon_hits > 0.5
    elif aux_mask == "pileup":      # only true-pileup hits
        mask = pileup_hits > 0.5
    else:                            # "all" — every ATAR hit (default)
        mask = torch.ones_like(muon_hits, dtype=torch.bool)

    return {
        "is_pie":           batch.is_pie_target.to(device).view(-1).float(),
        "muon_present":     batch.muon_present_target.to(device).view(-1).float(),
        "pileup_present":   batch.pileup_present_target.to(device).view(-1).float(),
        "muon_dif_present": batch.muon_dif_present_target.to(device).view(-1).float(),
        "pion_dif_present": batch.pidif_present_target.to(device).view(-1).float(),
        "is_mudif":         batch.is_mudif_target.to(device).view(-1).float(),
        "muon_travel":      batch.muon_travel_target.to(device).view(-1).float(),
        "muon_hits":        muon_hits,
        "pileup_hits":      pileup_hits,
        "muon_dif_hits":    muon_dif_hits,
        "pion_dif_hits":    pion_dif_hits,
        "muon_hit_mask":    mask,
    }


def load_shape_matching(module, sd, label="model"):
    """strict=False load that ALSO skips shape-mismatched keys. (Plain strict=False does
    NOT tolerate size mismatches — it errors.) New or shape-changed params keep their
    fresh init; returns the list left at init so the caller can decide about the optimizer."""
    msd = module.state_dict()
    keep = {k: v for k, v in sd.items() if k in msd and tuple(v.shape) == tuple(msd[k].shape)}
    mismatched = [k for k, v in sd.items() if k in msd and tuple(v.shape) != tuple(msd[k].shape)]
    module.load_state_dict(keep, strict=False)
    fresh = [k for k in msd if k not in keep]
    print(f"  {label}: loaded {len(keep)}/{len(msd)} tensors; {len(fresh)} kept at init "
          f"({len(mismatched)} shape-changed)", flush=True)
    if mismatched:
        show = mismatched[:6] + (["..."] if len(mismatched) > 6 else [])
        print(f"    shape-changed -> re-init: {show}", flush=True)
    return fresh


def loss_kwargs(args):
    return dict(
        pos_weight_pie=args.pw_pie, pos_weight_muon=args.pw_muon,
        pos_weight_pileup=args.pw_pileup, pos_weight_muon_dif=args.pw_mudif,
        pos_weight_pion_dif=args.pw_pidif,
        pos_weight_aux_muon=args.pw_aux_muon,
        pos_weight_aux_pileup=args.pw_aux_pileup,
        pos_weight_aux_muon_dif=args.pw_aux_mudif,
        pos_weight_aux_pion_dif=args.pw_aux_pidif,
        w_pie=args.w_pie, w_muon=args.w_muon, w_pileup=args.w_pileup,
        w_muon_dif=args.w_mudif, w_pion_dif=args.w_pidif,
        w_aux_muon=args.w_aux_muon, w_aux_pileup=args.w_aux_pileup,
        w_aux_muon_dif=args.w_aux_mudif, w_aux_pion_dif=args.w_aux_pidif,
        w_muon_dist=args.w_mudif_dist, muon_dist_scale=args.mudif_dist_scale,
        pos_weight_pie_topo=args.pw_pie_topo, w_pie_topo=args.w_pie_topo,
    )


# ---------------------------------------------------------------------------
# Optional backbone reconstruction supervision (the PURITY multi-task loss).
# OFF by default; enabling any --w_recon_* trains the backbone's PDG / trigger /
# pion-stop / direction / endpoint heads directly against truth, so the features
# the tail heads consume (mip_prob, hit_trigger_prob = positron slice, pion_stop,
# positron_dir) become physically meaningful instead of random-from-scratch.
# Terms whose output keys the ATAR-only tail model doesn't emit (LYSO, edge, role,
# event-builder) self-skip inside PURITYLoss, so reuse is safe.
# ---------------------------------------------------------------------------
def recon_config(args):
    return {
        'w_slice_trigger':    args.w_recon_trigger,
        'w_node_pdg':         args.w_recon_node_pdg,
        'w_slice_pdg':        args.w_recon_slice_pdg,
        'w_atar_slice_multi': args.w_recon_slice_multi,
        'w_pion_kinematics':  args.w_recon_pion,
        'w_positron_angle':   args.w_recon_angle,
        'w_endpoints':        args.w_recon_endpoints,
        'w_time_spread':      args.w_recon_time_spread,
        'w_angle_decorr':     args.w_recon_angle_decorr,
    }


def make_recon_criterion(args, device):
    cfg = recon_config(args)
    if not any(v > 0.0 for v in cfg.values()):
        return None
    on = ", ".join(f"{k}={v}" for k, v in cfg.items() if v > 0.0)
    print(f"  Backbone reconstruction loss ON: {{ {on} }}", flush=True)
    return PURITYLoss(cfg).to(device)


def add_recon_loss(recon, out, batch):
    """(recon_loss, parts) from the backbone reconstruction terms; no-op if disabled."""
    if recon is None:
        return 0.0, {}
    rt = format_targets_from_batch(batch)
    rloss, rdict = recon(out, rt, batch=batch)
    parts = {f"recon_{k}": v for k, v in rdict.items() if k != "loss_total"}
    return rloss, parts


# ---------------------------------------------------------------------------
# Train / validate passes
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, args, device, desc):
    model.train()
    total_loss, n_batches, n_failed, n_skipped = 0.0, 0, 0, 0
    epoch_parts = {}
    optimizer.zero_grad()
    accum_count = 0
    lk = loss_kwargs(args)

    pbar = tqdm(dataloader, desc=desc, leave=True)
    for i, batch in enumerate(pbar):
        try:
            batch = batch.to(device)
            out = model(batch.x, batch.batch)
            if "pie_logit" not in out:        # whole batch had no ATAR hits
                n_skipped += 1
                continue

            targets = build_targets(batch, device, aux_mask=args.aux_mask)
            # Hit-level logits and labels must align 1:1 over ATAR hits.
            if out["muon_node_logit"].shape[0] != targets["muon_hits"].shape[0]:
                n_failed += 1
                print(f"\n[skip] hit-count mismatch: "
                      f"node_logit={out['muon_node_logit'].shape[0]} "
                      f"vs muon_hits={targets['muon_hits'].shape[0]}", flush=True)
                continue

            loss, parts = pie_tagger_loss(out, targets, **lk)
            rloss, rparts = add_recon_loss(getattr(args, "_recon", None), out, batch)
            if isinstance(rloss, torch.Tensor):
                loss = loss + rloss
            parts.update(rparts)
            if (not loss.requires_grad or torch.isnan(loss) or torch.isinf(loss)):
                n_failed += 1
                continue

            (loss / args.accum_steps).backward()
            accum_count += 1
            is_last = (i + 1) == len(dataloader)
            if accum_count == args.accum_steps or is_last:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.clip_norm)
                # Skip the step on a non-finite gradient norm — otherwise clipping
                # by an inf/nan norm propagates NaN into every weight and corrupts
                # the whole model. Print it (loudly) instead of failing silently.
                if torch.isfinite(gnorm):
                    optimizer.step()
                else:
                    print(f"\n[skip-step] non-finite grad norm ({gnorm}); "
                          f"weights left unchanged", flush=True)
                optimizer.zero_grad()
                accum_count = 0

            total_loss += loss.item()
            n_batches += 1
            for k, v in parts.items():
                epoch_parts[k] = epoch_parts.get(k, 0.0) + float(v)
            pbar.set_postfix_str(
                f"L:{loss.item():.4f} | " +
                " | ".join(f"{k.split('_')[0]}:{float(v):.3f}" for k, v in parts.items()))

        except (RuntimeError, ValueError) as e:
            n_failed += 1
            optimizer.zero_grad()
            accum_count = 0
            print(f"\n[skip] {type(e).__name__}: {e}", flush=True)
            if "CUDA" in str(e) or "device-side assert" in str(e):
                raise

    avg = (total_loss / n_batches) if n_batches else float("nan")
    breakdown = {k: v / max(n_batches, 1) for k, v in epoch_parts.items()}
    return avg, breakdown, n_failed, n_skipped


def validate(model, dataloader, args, device, desc, max_batches=None):
    model.eval()
    total, n_val = 0.0, 0
    parts_sum = {}
    lk = loss_kwargs(args)
    with torch.inference_mode():
        for j, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            if max_batches is not None and j >= max_batches:
                break
            batch = batch.to(device)
            out = model(batch.x, batch.batch)
            if "pie_logit" not in out:
                continue
            targets = build_targets(batch, device, aux_mask=args.aux_mask)
            if out["muon_node_logit"].shape[0] != targets["muon_hits"].shape[0]:
                continue
            loss, parts = pie_tagger_loss(out, targets, **lk)
            rloss, rparts = add_recon_loss(getattr(args, "_recon", None), out, batch)
            if isinstance(rloss, torch.Tensor):
                loss = loss + rloss
            parts.update(rparts)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total += loss.item()
            n_val += 1
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + float(v)
    if n_val == 0:
        return float("nan"), {}
    return total / n_val, {k: v / n_val for k, v in parts_sum.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_path",
                   default="/data/tail_reveal_training.parquet")
    p.add_argument("--val_path",
                   default="/data/tail_reveal_validation.parquet")
    p.add_argument("--weights_dir", default="/pioneerML/model_weights")
    p.add_argument("--run_name", default="PURITY_TAIL_MUDIF",
                   help="Checkpoint prefix ({run_name}_best.pth / _epoch{N}.pth / "
                        "_loss_history.txt). Distinct from the pre-muDIF PURITY_TAIL line "
                        "(incompatible state_dict — the model now has a muDIF head).")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow overwriting existing {run_name}_*.pth checkpoints. Without "
                        "it (and without --resume), training aborts if any already exist.")

    p.add_argument("--freeze_trunk", action="store_true",
                   help="Stage-1: freeze the backbone (only train the 3 heads). "
                        "Pair with --backbone_ckpt or the trunk stays random.")
    p.add_argument("--backbone_ckpt", default=None,
                   help="PURITY checkpoint to seed the backbone (load_state_dict "
                        "strict=False into model.backbone).")

    p.add_argument("--dropout",     type=float, default=0.05)
    p.add_argument("--batch_size",  type=int,   default=100)
    p.add_argument("--accum_steps", type=int,   default=1)
    p.add_argument("--max_hits",    type=int,   default=250)
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--epoch_size",  type=int,   default=None,
                   help="If set, load a fresh random subset of this many train "
                        "events each epoch (bounds RAM).")
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--clip_norm",     type=float, default=2.0)
    p.add_argument("--lr_step_size",  type=int,   default=3,
                   help="StepLR period in epochs: LR decays by --lr_factor every N epochs.")
    p.add_argument("--lr_factor",     type=float, default=0.5)

    p.add_argument("--aux_mask", choices=["all", "muon", "pileup"], default="all",
                   help="Which ATAR hits feed the per-hit aux losses (muon_hit_mask).")
    # loss weights (defaults mirror pie_tagger_loss; tuned to ~20/84/36% balance)
    p.add_argument("--pw_pie",        type=float, default=4.0)
    p.add_argument("--pw_muon",       type=float, default=1.0)
    p.add_argument("--pw_pileup",     type=float, default=2.0)
    p.add_argument("--pw_mudif",      type=float, default=5.0,
                   help="pos_weight for the in-flight-muon (mu-DIF) event BCE.")
    p.add_argument("--pw_pidif",      type=float, default=5.0,
                   help="pos_weight for the in-flight-pion (pi-DIF) event BCE.")
    p.add_argument("--pw_aux_muon",   type=float, default=10.0)
    p.add_argument("--pw_aux_pileup", type=float, default=10.0)
    p.add_argument("--pw_aux_mudif",  type=float, default=10.0)
    p.add_argument("--pw_aux_pidif",  type=float, default=10.0)
    p.add_argument("--w_pie",         type=float, default=1.0)
    p.add_argument("--w_muon",        type=float, default=1.0)
    p.add_argument("--w_pileup",      type=float, default=1.0)
    p.add_argument("--w_mudif",       type=float, default=1.0,
                   help="Weight of the mu-DIF veto head's event BCE (0 disables it).")
    p.add_argument("--w_pidif",       type=float, default=1.0,
                   help="Weight of the pi-DIF veto head's event BCE (0 disables it).")
    p.add_argument("--w_aux_mudif",   type=float, default=0.0,
                   help="Weight of the per-hit in-flight-muon aux (needs --aux_mask all).")
    p.add_argument("--w_mudif_dist",  type=float, default=1.0,
                   help="Weight of the muon travel-distance regression (muDIF events only). "
                        "0 disables it (but the head still consumes the unsupervised pred via FiLM).")
    p.add_argument("--mudif_dist_scale", type=float, default=0.3,
                   help="exp(-d/scale) focus of the distance loss on SHORT muon ranges (mm).")
    p.add_argument("--w_aux_pidif",   type=float, default=0.0,
                   help="Weight of the per-hit in-flight-pion aux (needs --aux_mask all).")
    p.add_argument("--w_aux_muon",    type=float, default=0.3)
    # Per-hit pileup is OFF by default: pileup is an event/time-overlap concept,
    # and with time removed for unbiasedness the trunk can't separate a pileup
    # hit from a trigger hit, so this term stalls and dominates the loss. The
    # event-level pileup veto (--w_pileup) still works. Set >0 to re-enable.
    p.add_argument("--w_aux_pileup",  type=float, default=0.0)
    p.add_argument("--w_pie_topo",    type=float, default=1.0,
                   help="Loss weight for the time-group-graph pie head.")
    p.add_argument("--pw_pie_topo",   type=float, default=4.0,
                   help="pos_weight for the pie_topo BCE (pie is the minority class).")
    # Optional backbone reconstruction supervision (PURITY multi-task; all 0 => OFF).
    # Training from scratch, the backbone's reconstruction heads are otherwise
    # unsupervised, so the features the tail heads consume are noise. Enable these to
    # train them directly against truth. Highest leverage for muDIF: trigger, then
    # node_pdg + pion + angle.
    p.add_argument("--w_recon_trigger",     type=float, default=0.0,
                   help="Backbone: per-slice trigger (positron-slice) BCE. Key one for muDIF.")
    p.add_argument("--w_recon_node_pdg",    type=float, default=0.0,
                   help="Backbone: per-hit PDG BCE (feeds mip/muon/pion probs).")
    p.add_argument("--w_recon_slice_pdg",   type=float, default=0.0,
                   help="Backbone: per-slice PDG BCE.")
    p.add_argument("--w_recon_slice_multi", type=float, default=0.0,
                   help="Backbone: per-slice multi-origin BCE.")
    p.add_argument("--w_recon_pion",        type=float, default=0.0,
                   help="Backbone: pion-stop smooth-L1 (feeds kink/correlation).")
    p.add_argument("--w_recon_angle",       type=float, default=0.0,
                   help="Backbone: positron-direction cosine (feeds the kink).")
    p.add_argument("--w_recon_endpoints",   type=float, default=0.0,
                   help="Backbone: per-slice endpoint pinball loss (feeds exit_dir -> positron_dir).")
    p.add_argument("--w_recon_time_spread", type=float, default=0.0,
                   help="Backbone (regularizer): penalize trigger-positron time spread. "
                        "Anti-pileup; intra-positron, so it does NOT bias the inter-slice spectrum.")
    p.add_argument("--w_recon_angle_decorr", type=float, default=0.0,
                   help="Backbone (regularizer): decorrelate the positron-direction residual "
                        "from theta (needs --w_recon_angle > 0). Aligned with the unbiased measurement.")

    p.add_argument("--val_max_batches", type=int, default=None)
    p.add_argument("--skip_val", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init_ckpt", type=str, default=None,
                   help="Warm-start the FULL model from this checkpoint, loading only "
                        "shape-matching weights (tolerates the new muDIF distance head / "
                        "7-d FiLM); fresh optimizer + epochs. Use instead of --resume after "
                        "an architecture change.")
    args = p.parse_args()

    print("=" * 78)
    print(f"  train: {args.train_path}")
    print(f"  val:   {args.val_path}")
    print(f"  mode:  {'FROZEN trunk (Stage-1)' if args.freeze_trunk else 'end-to-end'}")
    print("=" * 78)

    if not os.path.exists(args.train_path):
        print(f"[error] train parquet not found: {args.train_path}", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)

    train_loader = None
    if args.epoch_size is None:
        train_set = PURITYDataset(args.train_path, max_hits=args.max_hits)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2)
        print(f"  train events: {len(train_set)}, batches: {len(train_loader)}", flush=True)

    val_loader = None
    if args.val_path and os.path.exists(args.val_path):
        val_set = PURITYDataset(args.val_path, max_hits=args.max_hits)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)
        print(f"  val events:   {len(val_set)}, batches: {len(val_loader)}", flush=True)

    model = PURITYTailModel(dropout=args.dropout,
                            freeze_trunk=args.freeze_trunk).to(device)

    if args.backbone_ckpt:
        print(f"\n  Seeding backbone from {args.backbone_ckpt}", flush=True)
        ckpt = torch.load(args.backbone_ckpt, map_location=device)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
        loaded = len(model.backbone.state_dict()) - len(missing)
        print(f"  backbone: loaded {loaded}/{len(model.backbone.state_dict())} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected ckpt keys)", flush=True)
    elif args.freeze_trunk:
        print("  [warn] --freeze_trunk set with no --backbone_ckpt: the frozen "
              "trunk is RANDOM. Heads will learn against noise.", flush=True)

    # Warm-start the FULL model from a prior checkpoint, tolerating architecture changes
    # (e.g. the new muDIF distance regressor / 7-d FiLM). Shape-changed + new params keep
    # their fresh init; fresh optimizer + epoch counter (unlike --resume).
    if args.init_ckpt:
        print(f"\n  Warm-starting full model from {args.init_ckpt}", flush=True)
        ckpt = torch.load(args.init_ckpt, map_location=device)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        load_shape_matching(model, sd, "init")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train:,}", flush=True)

    optimizer = build_tail_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    start_epoch = 1
    if args.resume:
        print(f"\n  Resuming from {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        fresh = load_shape_matching(model, ckpt["model"], "resume")
        if not fresh and "optimizer" in ckpt:   # identical architecture -> safe to resume optimizer
            optimizer.load_state_dict(ckpt["optimizer"])
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
        elif fresh:
            print("  [warn] architecture changed vs checkpoint -> fresh optimizer "
                  "(consider --init_ckpt instead of --resume for a warm start).", flush=True)
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("val_loss", float("inf"))
        if best_val != best_val:
            best_val = ckpt.get("train_loss", float("inf"))

    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_factor,
        last_epoch=(start_epoch - 1 if not args.resume else -1))

    os.makedirs(args.weights_dir, exist_ok=True)
    # Guard: never silently clobber another run's checkpoints. Skipped when
    # resuming (reusing the run_name is intentional) or when --overwrite is set.
    if not args.resume and not args.overwrite:
        import glob
        clash = glob.glob(os.path.join(args.weights_dir, f"{args.run_name}_*.pth"))
        if clash:
            raise SystemExit(
                f"[abort] {len(clash)} checkpoint(s) already exist for run_name "
                f"'{args.run_name}' in {args.weights_dir} "
                f"(e.g. {os.path.basename(clash[0])}).\n"
                f"        Choose a different --run_name, or pass --overwrite / --resume.")
    history_path = os.path.join(args.weights_dir, f"{args.run_name}_loss_history.txt")
    if start_epoch == 1:
        with open(history_path, "w") as f:
            f.write("epoch\ttrain_loss\tval_loss\n")

    # Optional backbone reconstruction supervision (off unless a --w_recon_* > 0).
    args._recon = make_recon_criterion(args, device)

    overall_t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.epoch_size is not None:
            print(f"\n  Loading {args.epoch_size} random events for epoch {epoch}...", flush=True)
            train_set = PURITYDataset(args.train_path, max_hits=args.max_hits,
                                      max_events=args.epoch_size)
            train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                      shuffle=True, num_workers=2)
            print(f"  train events: {len(train_set)}, batches: {len(train_loader)}", flush=True)

        print(f"\n  ===== epoch {epoch}/{args.epochs} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} =====", flush=True)
        t0 = time.time()
        try:
            train_avg, train_breakdown, n_failed, n_skipped = train_one_epoch(
                model, train_loader, optimizer, args, device,
                desc=f"ep{epoch} train")
        except Exception as e:
            print(f"\n[abort] training crashed at epoch {epoch}:", flush=True)
            traceback.print_exc()
            if "CUDA" in str(e) or "device-side assert" in str(e):
                raise
            return

        dt = time.time() - t0
        log = " | ".join(f"{k}:{v:.4f}" for k, v in train_breakdown.items())
        print(f"  TRAIN {dt/60:.1f} min  loss={train_avg:.4f}  "
              f"failed={n_failed} skipped={n_skipped}", flush=True)
        print(f"  {log}", flush=True)

        val_avg = float("nan")
        if val_loader is not None and not args.skip_val:
            val_avg, val_breakdown = validate(
                model, val_loader, args, device, desc=f"ep{epoch} val",
                max_batches=args.val_max_batches)
            vlog = " | ".join(f"{k}:{v:.4f}" for k, v in val_breakdown.items())
            print(f"  VAL   loss={val_avg:.4f}  {vlog}", flush=True)

        with open(history_path, "a") as f:
            f.write(f"{epoch}\t{train_avg:.6f}\t{val_avg:.6f}\n")
        scheduler.step()

        def _ckpt():
            return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "train_loss": train_avg, "val_loss": val_avg,
                    "freeze_trunk": args.freeze_trunk}

        ep_path = os.path.join(args.weights_dir, f"{args.run_name}_epoch{epoch}.pth")
        torch.save(_ckpt(), ep_path)
        print(f"  saved: {ep_path}", flush=True)

        metric = val_avg if (val_loader is not None and not args.skip_val) else train_avg
        if metric < best_val:
            best_val = metric
            best_path = os.path.join(args.weights_dir, f"{args.run_name}_best.pth")
            torch.save(_ckpt(), best_path)
            print(f"  *** new best ({metric:.4f}) saved: {best_path}", flush=True)

    print(f"\nDONE in {(time.time() - overall_t0)/3600:.2f} hr.", flush=True)


if __name__ == "__main__":
    main()
