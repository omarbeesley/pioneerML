"""
PURITY reconstruction-model training with staged loss activation.

Stage 1 (epochs 1..stage2_epoch-1):
    Backbone + PDG + endpoints + condensation only.
    Trigger, event builder, time spread, kinematics, angle, dead energy
    are zeroed out.

Stage 2 (epochs stage2_epoch..end):
    All losses active — downstream heads build on stable representations.

This is the trainer that produced main_hybridv2_filmg2 (its source matches the MD5
recorded in that run's log), ported to the release package. The selected epoch-19
checkpoint came from two phases on the same data and seed:

    phase 1, run main_hybridv2_filmg,  epochs 1-4:  --w_angle_decorr 0.5
    phase 2, run main_hybridv2_filmg2, epochs 5-20: resume filmg epoch 4, default 0.05,
             --lr_base 2.5e-4 --lr_event 2.5e-4 --lr_bias 5e-4

Both phases used --dropout 0.0 --weight_decay 1e-3 --epoch_mode chunked
--chunk_seed 12345 --val_max_batches 1000. scripts/slurm/train_main.sbatch runs
them in order. See docs/training.md.

The model reads its configuration from the environment at import time, so the
training profile is set before the import below. Values already exported in the
environment take precedence and are recorded in every checkpoint.

Usage:
    python -m purity.training.main --train_path train.parquet --val_path val.parquet \\
        --weights_dir outputs/weights --run_name my_run --dropout 0.0 --weight_decay 1e-3 \\
        --epoch_mode chunked --chunk_seed 12345 --val_max_batches 1000
"""
import argparse
import hashlib
import os
import sys
import time
import traceback

# Training profile of main_hybridv2_filmg/filmg2. The 5 ns positron-energy time
# gate is an inference setting added after training; it stays off here.
TRAIN_ENVIRONMENT = {
    "PURITY_GATE_MODE": "best_xy",
    "PURITY_BEST_XY_BETA": "0.5",
    "PURITY_ROLE_FILM": "1",
    "PURITY_ENERGY_TIME_GATE_NS": "0",
}
for _k, _v in TRAIN_ENVIRONMENT.items():
    os.environ.setdefault(_k, _v)

import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from purity.models.main import PURITYModel
from purity.dataset import PURITYDataset
from purity.train_utils import PURITYLoss, format_targets_from_batch


def _source_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Base task weights — Stage 2 uses these as-is; Stage 1 zeros out the
# downstream heads (trigger, event builder, time spread, kinematics, etc.).
# ---------------------------------------------------------------------------
BASE_TASK_WEIGHTS = {
    'w_atar_slice_multi':       0.05,
    'w_node_pdg':               1.0,
    'w_slice_pdg':              0.1,
    'w_endpoints':              0.01,
    'w_lyso_condensation':      0.25,
    'w_atar_trigger_slice':     0.5,
    'w_chain_exclusive':        5.0,   # heavy: exactly ONE e+ in the triggering chain
    # ASYMMETRIC positron recall: up-weight the role-CE e-column so missing the
    # trigger positron (unrecoverable acceptance loss) costs 5× predicting the wrong
    # role, while a matched accidental (cheap — time fit subtracts it) is not
    # specially penalized. Pushes P(e) on the true positron slice directly (the
    # measured labelfix failure: P(e) 0.94→0.55 leaking to P(none)); rides inside
    # w_atar_trigger_slice's role CE so it is already stage-2 gated and never touches
    # the µ column. alpha lowered 8->5: alpha=8 (prz) over-matched accidentals 5x
    # (false-htp 5x labelfix) for little transverse gain; 5 keeps recall, less over-match.
    'role_ce_positron_weight':  5.0,
    'w_time_spread':            1.0,
    'time_spread_thresh_ns':    1.0,
    'time_spread_trig_floor':   0.25,
    'time_spread_mip_floor':    0.25,
    'w_pion_kinematics':        50.0,
    'w_positron_angle':         0.25,
    # Match-gated angle loss (2026-07-28): angle terms only on events whose tagged pool
    # overlaps the true positron — kills the wrong-object noise and stops the empty-pool
    # default being trained backward. With the gate, the decorr bias penalty operates on
    # a clean residual population for the first time — re-enabled (was 0.0 = abandoned
    # after it never worked on the contaminated population).
    'angle_match_gate':         True,
    # w_angle_decorr: 0.5 was 40x the angle loss's own restoring force on the bias mode —
    # measured (filmg ep2-4) as underdamped ringing (m0 -3.8 -> +3.2 -> -1.96 deg) whose
    # oscillating gradient degraded the shared trunk: vs the identical-seed/data film run,
    # MeanError +60% (0.43->0.70) and role CE +60-80% (0.034->0.060). Lowered 10x to 0.05
    # (~4x the natural restoring force) — still pins the bias, without the ringing.
    'w_angle_decorr':           0.05,
    'decorr_window':            10,
    'w_event_builder':          0.2,
    'w_has_trigger_positron':   0.0,
    'w_dead_energy':            0.005,
}

# Keys zeroed out during Stage 1.
STAGE2_KEYS = [
    'w_atar_trigger_slice',
    'w_chain_exclusive',    # needs a competent role head first
    'w_time_spread',
    'w_event_builder',
    'w_pion_kinematics',
    'w_positron_angle',
    'w_angle_decorr',
    'w_dead_energy',
]


# --- alpha (positron-recall weight) RAMP ----------------------------------
# A flat high alpha permanently blunts the endpoints: the role gradient shocks the
# shared trunk at the stage-2 onset, and by the time it settles StepLR has decayed the
# LR too far for the endpoint head to re-sharpen (measured: prz alpha=8 stuck at
# MeanError 1.3 through ep13, ra5g alpha=5 at 1.0, vs fix10M alpha=1 at 0.30 by ep4).
# Fix: hold alpha=1 through ROLE_CE_RAMP_START-1 so the endpoints re-sharpen at full LR
# after the shock, then linearly ramp alpha up to the target over the next few epochs.
ROLE_CE_RAMP_START = 5    # first epoch (1-indexed) with alpha > 1
ROLE_CE_RAMP_END   = 9    # alpha reaches the full target here, then holds


def role_ce_alpha(epoch, target):
    """Ramped positron-recall weight: 1.0 until RAMP_START, linear to `target` by RAMP_END."""
    if epoch < ROLE_CE_RAMP_START:
        return 1.0
    if epoch >= ROLE_CE_RAMP_END:
        return target
    frac = (epoch - (ROLE_CE_RAMP_START - 1)) / (ROLE_CE_RAMP_END - (ROLE_CE_RAMP_START - 1))
    return 1.0 + (target - 1.0) * frac


def get_task_weights(epoch, stage2_epoch, overrides=None):
    """Return task weights for the given epoch (1-indexed)."""
    tw = dict(BASE_TASK_WEIGHTS)
    tw.update(overrides or {})
    if epoch < stage2_epoch:
        for k in STAGE2_KEYS:
            tw[k] = 0.0
    tw['role_ce_positron_weight'] = role_ce_alpha(epoch, BASE_TASK_WEIGHTS['role_ce_positron_weight'])
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
                    warmup_steps, base_lrs, global_step_start,
                    use_truth_positron_mask=False):
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
            truth_pos_mask = None
            if use_truth_positron_mask and hasattr(batch, 'is_trigger_target') and hasattr(batch, 'atar_node_pdg_target'):
                is_atar = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                is_trigger = batch.is_trigger_target[is_atar].bool()
                is_mip = batch.atar_node_pdg_target[:, 2].bool()
                truth_pos_mask_atar = is_trigger & is_mip
                truth_pos_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=device)
                truth_pos_mask[is_atar] = truth_pos_mask_atar
            outputs = model(batch.x, batch.batch, task_weights=task_weights,
                            triggering_pion_slice=anchor_slice,
                            truth_positron_mask=truth_pos_mask)
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
                if warmup_steps > 0 and global_step < warmup_steps:
                    factor = linear_warmup_factor(global_step, warmup_steps)
                    apply_warmup(optimizer, base_lrs, factor)

                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                # Guard: a NaN/inf gradient (e.g. a NaN forward output made finite-loss by
                # the BCE nan_to_num, or a degenerate-geometry backward) makes total_norm
                # non-finite; clip_grad_norm_ PROPAGATES it, so stepping would write NaN into
                # every parameter and poison the whole model. Skip the step instead.
                if torch.isfinite(total_norm):
                    optimizer.step()
                else:
                    n_failed += 1
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
             max_batches=None, use_truth_positron_mask=False):
    model.train()
    val_total, n_val = 0.0, 0
    val_dict = {}
    with torch.inference_mode():
        for j, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            if max_batches is not None and j >= max_batches:
                break
            batch = batch.to(device)
            anchor_slice = getattr(batch, 'atar_triggering_pion_slice', None)
            truth_pos_mask = None
            if use_truth_positron_mask and hasattr(batch, 'is_trigger_target') and hasattr(batch, 'atar_node_pdg_target'):
                is_atar = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                n_atar = is_atar.sum().item()
                is_trigger = batch.is_trigger_target[:n_atar].bool()
                is_mip = batch.atar_node_pdg_target[:, 2].bool()
                truth_pos_mask_atar = is_trigger & is_mip
                truth_pos_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)
                truth_pos_mask[is_atar] = truth_pos_mask_atar
            outputs = model(batch.x, batch.batch, task_weights=task_weights,
                            triggering_pion_slice=anchor_slice,
                            truth_positron_mask=truth_pos_mask)
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--run_name", default="PURITY",
                        help="Prefix for checkpoint and history filenames.")
    parser.add_argument("--dropout",     type=float, default=0.1,
                        help="Dropout rate (default 0.1). Safe to change when resuming.")
    parser.add_argument("--batch_size",  type=int, default=100)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--max_hits",    type=int, default=250)
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--epoch_size",  type=int, default=1000000,
                        help="If set, load a fresh random subset of this many events "
                             "from the training parquet each epoch. Bounds RAM usage.")
    parser.add_argument("--epoch_mode", choices=["random", "chunked"], default="random",
                        help="With epoch_size: 'random' reloads a fresh random subset each "
                             "epoch (legacy; events repeat across epochs); 'chunked' walks "
                             "the pool in fixed disjoint row-group chunks so NO event repeats "
                             "until the whole pool has been seen once (then a new pass starts).")
    parser.add_argument("--chunk_seed", type=int, default=12345,
                        help="Seed for the one-time row-group shuffle in chunked mode.")
    parser.add_argument("--stage2_epoch", type=int, default=2,
                        help="Epoch at which Stage 2 begins (all losses activate).")

    parser.add_argument("--lr_base",     type=float, default=5e-4)
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

    parser.add_argument("--val_max_batches", type=int, default=None)
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pth to resume training from.")
    parser.add_argument("--truth_positron_mask", action="store_true",
                        help="Use truth-level hit labels for positron hit selection "
                             "in the direction head (bypasses predicted trigger/MIP probs).")
    parser.add_argument("--w_angle_decorr", type=float, default=None,
                        help="Override the angle-decorrelation weight (default 0.05). "
                             "Phase 1 of filmg2 (filmg, epochs 1-4) used 0.5.")
    args = parser.parse_args()

    weight_overrides = {}
    if args.w_angle_decorr is not None:
        weight_overrides['w_angle_decorr'] = args.w_angle_decorr
    environment = {k: os.environ[k] for k in TRAIN_ENVIRONMENT}
    provenance = {'args': vars(args), 'environment': environment,
                  'source_sha256': _source_sha256()}

    print("=" * 78)
    print(f"  train: {args.train_path}")
    print(f"  val:   {args.val_path}")
    print(f"  stage2 begins at epoch {args.stage2_epoch}")
    print(f"  environment: {environment}")
    if weight_overrides:
        print(f"  task-weight overrides: {weight_overrides}")
    print("=" * 78)

    if not os.path.exists(args.train_path):
        print(f"[error] train parquet not found: {args.train_path}", flush=True)
        raise SystemExit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device: {device}", flush=True)
    torch.manual_seed(7)   # reproducible weight init (batch order still shuffles freely)

    # Chunked-epoch plan: partition the pool's row groups (one fixed shuffle) into
    # disjoint chunks of ~epoch_size events. Epoch k reads chunk (k-1) mod n_chunks,
    # so no event repeats until the whole pool has been consumed once.
    chunk_plan = None
    if args.epoch_size is not None and args.epoch_mode == "chunked":
        import numpy as _np
        import pyarrow.parquet as _pq
        _pf = _pq.ParquetFile(args.train_path)
        _sizes = [_pf.metadata.row_group(i).num_rows
                  for i in range(_pf.metadata.num_row_groups)]
        _order = _np.random.default_rng(args.chunk_seed).permutation(len(_sizes))
        chunk_plan, _cur, _acc = [], [], 0
        for _gi in _order:
            _cur.append(int(_gi)); _acc += _sizes[_gi]
            if _acc >= args.epoch_size:
                chunk_plan.append(_cur); _cur, _acc = [], 0
        if _cur:
            chunk_plan.append(_cur)
        print(f"  [chunked epochs] pool {sum(_sizes):,} events in {len(_sizes)} row-groups "
              f"-> {len(chunk_plan)} disjoint chunks (~{args.epoch_size:,} events each); "
              f"full pass every {len(chunk_plan)} epochs, no repeats within a pass "
              f"(chunk_seed={args.chunk_seed})", flush=True)
        if len(chunk_plan) == 1:
            print("  [chunked epochs][warn] pool fits in ONE chunk -> every epoch sees the "
                  "whole pool (chunking is a no-op).", flush=True)

    # When epoch_size is set, defer dataset loading to the epoch loop
    # so each epoch gets a fresh random sample. Otherwise load once.
    train_loader = None
    if args.epoch_size is None:
        train_set = PURITYDataset(args.train_path, max_hits=args.max_hits)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        print(f"  train events: {len(train_set)}, batches: {len(train_loader)}", flush=True)

    val_loader = None
    if args.val_path and os.path.exists(args.val_path):
        val_set = PURITYDataset(args.val_path, max_hits=args.max_hits)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print(f"  val events:   {len(val_set)}, batches: {len(val_loader)}", flush=True)

    model = PURITYModel(dropout=args.dropout).to(device)
    optimizer = build_optimizer(
        model,
        lr_base=args.lr_base,
        lr_event=args.lr_event,
        lr_bias=args.lr_bias,
        weight_decay=args.weight_decay,
    )
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    best_val = float('inf')
    global_step = 0
    start_epoch = 1

    # Resume from checkpoint if provided.
    if args.resume:
        print(f"\n  Resuming from checkpoint: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f"  [resume] missing keys (will init random): {missing}")
        if unexpected:
            print(f"  [resume] unexpected keys (ignored): {unexpected}")
        if missing or unexpected:
            print(f"  [resume] architecture changed — skipping optimizer state restore")
        elif 'optimizer' not in ckpt:
            # Weights-only checkpoints (the release copies) carry no optimizer state.
            print(f"  [resume] no optimizer state in checkpoint — fresh optimizer at the CLI LRs")
        else:
            optimizer.load_state_dict(ckpt['optimizer'])
            # Override optimizer LRs with CLI args (checkpoint may have different LRs)
            cli_lrs = [args.lr_base, args.lr_event, args.lr_bias]
            for pg, lr in zip(optimizer.param_groups, cli_lrs):
                pg['lr'] = lr
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', 0)
        best_val = ckpt.get('val_loss', float('inf'))
        if best_val != best_val:  # NaN check
            best_val = ckpt.get('train_loss', float('inf'))
        print(f"  Resumed: epoch={ckpt['epoch']}, global_step={global_step}, "
              f"best_val={best_val:.4f}", flush=True)

    # Set initial_lr for the scheduler (required by StepLR).
    # On resume, treat CLI LRs as the desired starting LRs — don't let
    # the scheduler retroactively decay them based on elapsed epochs.
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']

    sched_last_epoch = start_epoch - 1 if not args.resume else -1
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_factor,
        last_epoch=sched_last_epoch,
    )

    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    print(f"\n  optimizer LRs (base/event/bias): {base_lrs}", flush=True)
    print(f"  scheduler: StepLR (step_size={args.lr_step_size}, "
          f"gamma={args.lr_factor})", flush=True)

    os.makedirs(args.weights_dir, exist_ok=True)

    history_path = os.path.join(args.weights_dir, f"{args.run_name}_loss_history.txt")
    if start_epoch == 1:
        with open(history_path, 'w') as f:
            f.write("epoch\tstage\ttrain_loss\tval_loss\n")
    else:
        print(f"  Appending to existing loss history: {history_path}", flush=True)

    overall_t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        # Reload the epoch's events when epoch_size is set: either the next disjoint
        # chunk (no repeats within a pass) or a fresh random subset (legacy).
        if args.epoch_size is not None:
            if chunk_plan is not None:
                ci = (epoch - 1) % len(chunk_plan)
                if ci == 0 and epoch > 1:
                    print("  [chunked] pool exhausted -> starting a new pass", flush=True)
                print(f"\n  Loading chunk {ci + 1}/{len(chunk_plan)} "
                      f"({len(chunk_plan[ci])} row-groups) for epoch {epoch} "
                      f"[chunked: no repeats within a pass]...", flush=True)
                train_set = PURITYDataset(args.train_path, max_hits=args.max_hits,
                                          row_groups=chunk_plan[ci])
            else:
                print(f"\n  Loading {args.epoch_size} random events for epoch {epoch}...",
                      flush=True)
                train_set = PURITYDataset(args.train_path, max_hits=args.max_hits,
                                          max_events=args.epoch_size)
            train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                      shuffle=True, num_workers=2)
            print(f"  train events: {len(train_set)}, batches: {len(train_loader)}",
                  flush=True)

        task_weights = get_task_weights(epoch, args.stage2_epoch, weight_overrides)
        stage = 1 if epoch < args.stage2_epoch else 2
        criterion = PURITYLoss(config=task_weights)

        print(f"\n  ===== epoch {epoch}/{args.epochs} | stage {stage} | "
              f"LRs: {[pg['lr'] for pg in optimizer.param_groups]} =====",
              flush=True)
        if stage == 2 and epoch == args.stage2_epoch:
            print(f"  >>> STAGE 2: enabling trigger, event builder, time spread, "
                  f"kinematics, angle, dead energy", flush=True)
        print(f"  task_weights: {task_weights}", flush=True)

        t0 = time.time()
        try:
            train_avg, train_breakdown, n_failed, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion, task_weights, device,
                desc=f"ep{epoch} S{stage} train",
                accum_steps=args.accum_steps,
                clip_norm=args.clip_norm,
                warmup_steps=args.warmup_steps if epoch == 1 else 0,
                base_lrs=base_lrs,
                global_step_start=global_step,
                use_truth_positron_mask=args.truth_positron_mask,
            )
        except Exception as e:
            print(f"\n[abort] training crashed at epoch {epoch}:", flush=True)
            traceback.print_exc()
            if 'CUDA' in str(e) or 'device-side assert' in str(e):
                raise
            raise SystemExit(1)   # non-zero, so dependent SLURM jobs do not start

        dt = time.time() - t0
        log = " | ".join(f"{k.split('_')[-1]}:{v:.4f}" for k, v in train_breakdown.items())
        print(f"  TRAIN done in {dt/60:.1f} min  loss={train_avg:.4f}  "
              f"failed_batches={n_failed}", flush=True)
        print(f"  {log}", flush=True)

        val_avg = float('nan')
        if val_loader is not None and not args.skip_val:
            criterion.reset_decorr_ema()
            val_avg, val_breakdown = validate(
                model, val_loader, criterion, task_weights, device,
                desc=f"ep{epoch} S{stage} val",
                max_batches=args.val_max_batches,
                use_truth_positron_mask=args.truth_positron_mask,
            )
            vlog = " | ".join(f"{k.split('_')[-1]}:{v:.4f}" for k, v in val_breakdown.items())
            print(f"  VAL    loss={val_avg:.4f}  {vlog}", flush=True)

        with open(history_path, 'a') as f:
            f.write(f"{epoch}\t{stage}\t{train_avg:.6f}\t{val_avg:.6f}\n")

        scheduler.step()

        ep_path = os.path.join(args.weights_dir,
                                f"{args.run_name}_epoch{epoch}.pth")
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
            'provenance': provenance,
        }, ep_path)
        print(f"  saved: {ep_path}", flush=True)

        metric = val_avg if (not args.skip_val and val_loader is not None) else train_avg
        if metric < best_val:
            best_val = metric
            best_path = os.path.join(args.weights_dir,
                                      f"{args.run_name}_best.pth")
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
                'provenance': provenance,
            }, best_path)
            print(f"  *** new best ({metric:.4f})  saved: {best_path}", flush=True)

    print(f"\nDONE in {(time.time() - overall_t0)/3600:.2f} hr.", flush=True)


if __name__ == "__main__":
    main()
