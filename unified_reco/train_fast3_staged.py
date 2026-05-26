"""
Staged training script: trains PURITY models from scratch with progressive
loss activation to prevent downstream heads from memorizing early noisy
representations.

Stage 1 (epochs 1..stage2_epoch-1):
    Backbone + PDG + endpoints + condensation + kinematics + angle.
    Trigger, event builder, time spread, and composition are zeroed out.

Stage 2 (epochs stage2_epoch..end):
    All losses active — trigger/event-builder heads now build on stable,
    generalizable backbone features.

Outputs: {weights_dir}/PURITY_{coupling}_v3_epoch{N}.pth, plus a
         _best.pth tracking lowest validation loss.

Usage:
    python train_fast3_staged.py --only fast3
    python train_fast3_staged.py --epochs 8 --stage2_epoch 3
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
from unified_reco.models import PURITYHybridModel
from unified_reco.dataset import PURITYDataset
from unified_reco.train_utils import PURITYLoss, format_targets_from_batch


# ---------------------------------------------------------------------------
# Base task weights — Stage 2 uses these as-is; Stage 1 zeros out the
# downstream heads (trigger, event builder, time spread).
# ---------------------------------------------------------------------------
BASE_TASK_WEIGHTS = {
    'w_atar_slice_multi':       0.05,
    'w_node_pdg':               1.0,
    'w_slice_pdg':              0.1,
    'w_endpoints':              0.025,
    'w_lyso_condensation':      0.25,
    'w_atar_trigger_slice':     0.5,
    'w_time_spread':            1.0,
    'time_spread_thresh_ns':    1.0,
    'time_spread_trig_floor':   0.25,
    'time_spread_mip_floor':    0.25,
    'w_pion_kinematics':        50.0,
    'w_positron_angle':         0.5,
    'w_event_builder':          0.1,
    'w_has_trigger_positron':   0.0,
    'w_dead_energy':            0.005,
}

# Keys zeroed out during Stage 1.
STAGE2_KEYS = [
    'w_atar_trigger_slice',
    'w_time_spread',
    'w_event_builder',
    'w_pion_kinematics',
    'w_positron_angle',
    'w_dead_energy'

]


def get_task_weights(epoch, stage2_epoch):
    """Return task weights for the given epoch (1-indexed)."""
    tw = dict(BASE_TASK_WEIGHTS)
    if epoch < stage2_epoch:
        for k in STAGE2_KEYS:
            tw[k] = 0.0
    return tw


# ---------------------------------------------------------------------------
# Optimizer + scheduler setup
# ---------------------------------------------------------------------------
def build_optimizer(model, lr_base, lr_event, lr_bias, weight_decay):
    """Three param groups, AdamW (decoupled weight decay) for all of them."""
    bias_scalar_params = [
        model.sigma_t_atar_ns,
        model.sigma_t_lyso_floor_ns,
        model.sigma_t_lyso_scale_ns,
        model.angle_sigma_floor,
        model.angle_sigma_scale,
    ]
    bias_ids = {id(p) for p in bias_scalar_params}

    event_params = (
        list(model.slim_event_transformer.parameters())
        + list(model.lyso_event_proj.parameters())
        + list(model.atar_event_down.parameters())
        + list(model.event_head.parameters())
        + list(model.event_modality_emb.parameters())
        + list(model.event_slice_emb.parameters())
    )
    event_params = [p for p in event_params if id(p) not in bias_ids]
    event_ids = {id(p) for p in event_params}

    base_params = [
        p for p in model.parameters()
        if id(p) not in event_ids and id(p) not in bias_ids
    ]

    return torch.optim.AdamW([
        {'params': base_params,        'lr': lr_base,  'weight_decay': weight_decay},
        {'params': event_params,       'lr': lr_event, 'weight_decay': weight_decay},
        {'params': bias_scalar_params, 'lr': lr_bias,  'weight_decay': 0.0},
    ])


def linear_warmup_factor(global_step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (global_step + 1) / warmup_steps)


def apply_warmup(optimizer, base_lrs, factor):
    for pg, base in zip(optimizer.param_groups, base_lrs):
        pg['lr'] = base * factor


# ---------------------------------------------------------------------------
# Training and validation passes
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, task_weights,
                    device, desc, *, accum_steps, clip_norm,
                    warmup_steps, base_lrs, global_step_start):
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_failed = 0
    epoch_loss_dict = {}
    global_step = global_step_start
    accum_count = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=desc, leave=True)
    for i, batch in enumerate(pbar):
        try:
            batch = batch.to(device)

            anchor_slice = getattr(batch, 'atar_triggering_pion_slice', None)
            outputs = model(batch.x, batch.batch, task_weights=task_weights,
                            triggering_pion_slice=anchor_slice)
            targets = format_targets_from_batch(batch)
            loss, loss_dict = criterion(outputs, targets, batch=batch)

            if (not isinstance(loss, torch.Tensor) or not loss.requires_grad
                    or torch.isnan(loss) or torch.isinf(loss)):
                n_failed += 1
                continue

            (loss / accum_steps).backward()
            accum_count += 1

            is_last_in_epoch = (i + 1) == len(dataloader)
            if accum_count == accum_steps or is_last_in_epoch:
                factor = linear_warmup_factor(global_step, warmup_steps)
                apply_warmup(optimizer, base_lrs, factor)

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

            total_loss += loss.item()
            n_batches += 1
            for k, v in loss_dict.items():
                if k == 'loss_total':
                    continue
                val = v.item() if hasattr(v, 'item') else v
                epoch_loss_dict[k] = epoch_loss_dict.get(k, 0.0) + val

            log_str = " | ".join(
                f"{k.split('_')[-1]}: {(v.item() if hasattr(v, 'item') else v):.3f}"
                for k, v in loss_dict.items() if k != 'loss_total'
            )
            pbar.set_postfix_str(f"L:{loss.item():.4f} | {log_str}")

        except (RuntimeError, ValueError) as e:
            n_failed += 1
            optimizer.zero_grad()
            accum_count = 0
            print(f"\n[skip] {type(e).__name__}: {e}", flush=True)
            if 'CUDA' in str(e) or 'device-side assert' in str(e):
                raise

    avg = (total_loss / n_batches) if n_batches > 0 else float('nan')
    breakdown = {k: v / max(n_batches, 1) for k, v in epoch_loss_dict.items()}
    return avg, breakdown, n_failed, global_step


def validate(model, dataloader, criterion, task_weights, device, desc,
             max_batches=None):
    model.train()
    val_total, n_val = 0.0, 0
    val_dict = {}
    with torch.inference_mode():
        for j, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            if max_batches is not None and j >= max_batches:
                break
            batch = batch.to(device)
            anchor_slice = getattr(batch, 'atar_triggering_pion_slice', None)
            outputs = model(batch.x, batch.batch, task_weights=task_weights,
                            triggering_pion_slice=anchor_slice)
            targets = format_targets_from_batch(batch)
            loss, loss_dict = criterion(outputs, targets, batch=batch)
            if not isinstance(loss, torch.Tensor) or torch.isnan(loss) or torch.isinf(loss):
                continue
            val_total += loss.item()
            n_val += 1
            for k, v in loss_dict.items():
                if k == 'loss_total':
                    continue
                val = v.item() if hasattr(v, 'item') else v
                val_dict[k] = val_dict.get(k, 0.0) + val
    if n_val == 0:
        return float('nan'), {}
    return val_total / n_val, {k: v / n_val for k, v in val_dict.items()}


# ---------------------------------------------------------------------------
# Per-coupling driver
# ---------------------------------------------------------------------------
def run_one_model(coupling, train_path, val_path, args):
    print("=" * 78)
    print(f"COUPLING: {coupling}")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")
    print(f"  stage2 begins at epoch {args.stage2_epoch}")
    print("=" * 78)

    if not os.path.exists(train_path):
        print(f"  [skip] train parquet not found: {train_path}", flush=True)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device: {device}", flush=True)

    train_set = PURITYDataset(train_path, max_hits=args.max_hits)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"  train events: {len(train_set)}, batches: {len(train_loader)}", flush=True)

    val_loader = None
    if val_path and os.path.exists(val_path):
        val_set = PURITYDataset(val_path, max_hits=args.max_hits)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print(f"  val events:   {len(val_set)}, batches: {len(val_loader)}", flush=True)

    model = PURITYHybridModel().to(device)
    optimizer = build_optimizer(
        model,
        lr_base=args.lr_base,
        lr_event=args.lr_event,
        lr_bias=args.lr_bias,
        weight_decay=args.weight_decay,
    )
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_factor,
    )

    print(f"\n  optimizer LRs (base/event/bias): {base_lrs}", flush=True)
    print(f"  scheduler: StepLR (step_size={args.lr_step_size}, "
          f"gamma={args.lr_factor})", flush=True)

    best_val = float('inf')
    global_step = 0
    os.makedirs(args.weights_dir, exist_ok=True)

    history_path = os.path.join(args.weights_dir, f"PURITY_{coupling}_v3_loss_history.txt")
    with open(history_path, 'w') as f:
        f.write("epoch\tstage\ttrain_loss\tval_loss\n")

    for epoch in range(1, args.epochs + 1):
        task_weights = get_task_weights(epoch, args.stage2_epoch)
        stage = 1 if epoch < args.stage2_epoch else 2
        criterion = PURITYLoss(config=task_weights)

        print(f"\n  ===== {coupling} epoch {epoch}/{args.epochs} | stage {stage} | "
              f"LRs: {[pg['lr'] for pg in optimizer.param_groups]} =====",
              flush=True)
        if stage == 2 and epoch == args.stage2_epoch:
            print(f"  >>> STAGE 2: enabling trigger, event builder, time spread", flush=True)
        print(f"  task_weights: {task_weights}", flush=True)

        t0 = time.time()
        try:
            train_avg, train_breakdown, n_failed, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion, task_weights, device,
                desc=f"{coupling}/ep{epoch} S{stage} train",
                accum_steps=args.accum_steps,
                clip_norm=args.clip_norm,
                warmup_steps=args.warmup_steps if epoch == 1 else 0,
                base_lrs=base_lrs,
                global_step_start=global_step,
            )
        except Exception as e:
            print(f"\n[abort] training crashed for {coupling} epoch {epoch}:", flush=True)
            traceback.print_exc()
            if 'CUDA' in str(e) or 'device-side assert' in str(e):
                raise
            return

        dt = time.time() - t0
        log = " | ".join(f"{k.split('_')[-1]}:{v:.4f}" for k, v in train_breakdown.items())
        print(f"  TRAIN done in {dt/60:.1f} min  loss={train_avg:.4f}  "
              f"failed_batches={n_failed}", flush=True)
        print(f"  {log}", flush=True)

        val_avg = float('nan')
        if val_loader is not None and not args.skip_val:
            val_avg, val_breakdown = validate(
                model, val_loader, criterion, task_weights, device,
                desc=f"{coupling}/ep{epoch} S{stage} val",
                max_batches=args.val_max_batches,
            )
            vlog = " | ".join(f"{k.split('_')[-1]}:{v:.4f}" for k, v in val_breakdown.items())
            print(f"  VAL    loss={val_avg:.4f}  {vlog}", flush=True)

        with open(history_path, 'a') as f:
            f.write(f"{epoch}\t{stage}\t{train_avg:.6f}\t{val_avg:.6f}\n")

        scheduler.step()

        ep_path = os.path.join(args.weights_dir,
                                f"PURITY_{coupling}_v3_epoch{epoch}.pth")
        torch.save({
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch':     epoch,
            'stage':     stage,
            'global_step': global_step,
            'train_loss': train_avg,
            'val_loss':   val_avg,
            'task_weights': task_weights,
        }, ep_path)
        print(f"  saved: {ep_path}", flush=True)

        metric = val_avg if (not args.skip_val and val_loader is not None) else train_avg
        if metric < best_val:
            best_val = metric
            best_path = os.path.join(args.weights_dir,
                                      f"PURITY_{coupling}_v3_best.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'stage': stage,
                'global_step': global_step,
                'train_loss': train_avg,
                'val_loss':   val_avg,
                'task_weights': task_weights,
            }, best_path)
            print(f"  *** new best ({metric:.4f})  saved: {best_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="/data/mixed_parquets")
    parser.add_argument("--training_dir",   default="training_5_07")
    parser.add_argument("--validation_dir", default="validation_5_07")
    parser.add_argument("--weights_dir",
                        default="/pioneerML/model_weights")
    parser.add_argument("--batch_size",  type=int, default=100)
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation: optimizer steps every N micro-batches.")
    parser.add_argument("--max_hits",    type=int, default=250)
    parser.add_argument("--epochs",      type=int, default=4)
    parser.add_argument("--stage2_epoch", type=int, default=2,
                        help="Epoch at which Stage 2 begins (trigger/builder/spread losses activate).")

    parser.add_argument("--lr_base",     type=float, default=1e-4)
    parser.add_argument("--lr_event",    type=float, default=5e-4)
    parser.add_argument("--lr_bias",     type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--clip_norm",     type=float, default=2.0)
    parser.add_argument("--warmup_steps",  type=int,   default=50,
                        help="Linear LR warmup over this many optimizer steps (epoch 1 only).")
    parser.add_argument("--lr_factor",    type=float, default=0.5,
                        help="Multiplicative factor applied to LR each step.")
    parser.add_argument("--lr_step_size", type=int,   default=5,
                        help="Halve LR every this many epochs.")

    parser.add_argument("--val_max_batches", type=int, default=None,
                        help="If set, validate on at most this many batches per epoch (for speed).")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--only", choices=["fast3", "fast3_2x"], default=None)
    args = parser.parse_args()

    couplings = ["fast3", "fast3_2x"] if args.only is None else [args.only]

    overall_t0 = time.time()
    cuda_dead = False
    for coupling in couplings:
        if cuda_dead:
            print(f"\n[skip] {coupling}: CUDA context was poisoned by a previous "
                  f"coupling. Re-launch with `--only {coupling}` after restart.", flush=True)
            continue
        train_path = os.path.join(args.data_root, args.training_dir,   f"data_{coupling}.parquet")
        val_path   = os.path.join(args.data_root, args.validation_dir, f"data_{coupling}.parquet")
        try:
            run_one_model(coupling, train_path, val_path, args)
        except Exception as e:
            print(f"\n[!] coupling {coupling} crashed, moving on:", flush=True)
            traceback.print_exc()
            if 'CUDA' in str(e) or 'device-side assert' in str(e):
                cuda_dead = True

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\nALL DONE in {(time.time() - overall_t0)/3600:.2f} hr.", flush=True)


if __name__ == "__main__":
    main()
