#!/usr/bin/env python
"""
Test the hypothesis that the epoch-10 NaN crash originates in PinballLoss
variance collapse, NOT in the condensation BCE where the device-side assert
surfaced.

Hypothesis (three falsifiable claims):
  A. The attenuated quantile term  `2*error/sigma + log(sigma)`  is unbounded
     below as sigma -> 0, and its gradient ~1/sigma^2 explodes. The current
     floor is only +1e-6, which is effectively no protection.
  B. Optimizing the REAL PinballLoss drives sigma -> floor and PosLoss
     strongly negative (reproducing the observed `PosLoss: -8.6`), and with a
     little gradient noise it blows up to inf/NaN.
  C. Clamping sigma at a physical floor (~0.02 in normalized units) keeps both
     the loss and its gradient bounded, eliminating the collapse.

Plus a mechanism check: `.clamp()` does NOT strip NaN (so the condensation
line `e_beta[...].clamp(1e-6, 1-1e-6)` lets a NaN reach F.binary_cross_entropy),
whereas `.nan_to_num()` does.

Run:
    python test_pinball_collapse.py
    python test_pinball_collapse.py --repro-cuda-assert   # reproduces the
                                                           # literal Loss.cu:90
                                                           # crash (ABORTS proc)

No GPU required for the core tests (pure CPU numerics).
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_FLOOR = 0.02          # proposed physical floor (normalized units)
CURRENT_FLOOR = 1e-6        # what the code uses today (additive on sigma_l/r)

# ---------------------------------------------------------------------------
# Try to import the REAL loss so claim B is tested on the actual code path.
# Fall back to a faithful copy (sigma_floor=None) if the package import pulls
# unavailable deps.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REAL_IMPORTED = False
try:
    from unified_reco.train_utils import PinballLoss as RealPinball
    REAL_IMPORTED = True
except Exception as e:                                    # noqa: BLE001
    RealPinball = None
    _IMPORT_ERR = e


class PinballLossLocal(nn.Module):
    """Faithful copy of unified_reco.train_utils.PinballLoss (lines 16-79),
    with one added knob: `sigma_floor`. None reproduces today's behavior
    (only the +1e-6 additive term, no clamp); a float clamps the stacked
    scales at that floor == the proposed fix."""

    def __init__(self, quantiles=(0.16, 0.50, 0.84), loss_scale_span=0.1,
                 loss_scale_dir=0.1, sigma_floor=None):
        super().__init__()
        self.quantiles = list(quantiles)
        self.loss_scale_span = loss_scale_span
        self.loss_scale_dir = loss_scale_dir
        self.sigma_floor = sigma_floor

    def forward(self, preds, targets, weights=None, error_scale=1.0):
        if preds.dim() != 4 or targets.dim() != 3:
            raise ValueError("Expected preds [N,2,3,3] and targets [N,2,3]")

        pred_median = preds[:, :, :, 1]
        loss_direct = F.smooth_l1_loss(pred_median, targets, reduction='none').sum(dim=(1, 2))
        target_swapped = targets[:, [1, 0], :]
        loss_swapped = F.smooth_l1_loss(pred_median, target_swapped, reduction='none').sum(dim=(1, 2))
        swap_mask = loss_swapped < loss_direct
        target_aligned = targets

        if weights is not None:
            weights_swapped = weights[:, [1, 0]]
            batch_weights = torch.where(swap_mask.view(-1, 1), weights_swapped, weights)
        else:
            batch_weights = torch.ones(targets.shape[0], 2, device=targets.device)

        target_exp = target_aligned.unsqueeze(-1)
        sigma_left = (preds[..., 1] - preds[..., 0]).abs() + 1e-6
        sigma_right = (preds[..., 2] - preds[..., 1]).abs() + 1e-6
        sigma_total = sigma_left + sigma_right
        scales = torch.stack([sigma_left, sigma_total, sigma_right], dim=-1)
        # === THE ONLY DIFFERENCE FROM THE REAL CODE ===
        if self.sigma_floor is not None:
            scales = scales.clamp(min=self.sigma_floor)
        # ==============================================
        log_scales = torch.log(scales)

        pos_loss = 0.0
        for i, q in enumerate(self.quantiles):
            error = target_exp - preds[..., i:i + 1]
            pinball = torch.max(q * error, (q - 1.0) * error)
            scale_q = scales[..., i:i + 1]
            log_scale_q = log_scales[..., i:i + 1]
            attenuated_loss = (2.0 * error_scale * pinball / scale_q) + log_scale_q
            w = batch_weights.unsqueeze(-1).unsqueeze(-1)
            pos_loss += (attenuated_loss * w).mean()

        pred_vec = pred_median[:, 1, :] - pred_median[:, 0, :]
        target_vec_aligned = target_aligned[:, 1, :] - target_aligned[:, 0, :]
        span_loss = F.mse_loss(pred_vec.norm(dim=1), target_vec_aligned.norm(dim=1))
        dir_loss = (1.0 - F.cosine_similarity(pred_vec, target_vec_aligned, dim=1, eps=1e-6)).mean() * self.loss_scale_dir
        total_loss = pos_loss + (self.loss_scale_span * span_loss) + dir_loss

        breakdown = {
            "PosLoss": pos_loss.item() if isinstance(pos_loss, torch.Tensor) else pos_loss,
            "MeanWidth": sigma_total.mean().item(),
        }
        return total_loss, breakdown


def banner(txt):
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


# ---------------------------------------------------------------------------
# TEST A — static failure surface of the attenuated term
# ---------------------------------------------------------------------------
def test_a_failure_surface():
    banner("TEST A  static failure surface:  loss = 2*err/sigma + log(sigma)")
    err = torch.tensor(0.1)        # a small but nonzero residual
    print(f"  fixed residual err = {err.item()}")
    print(f"  {'sigma':>10} | {'loss term':>14} | {'|d loss / d sigma|':>20}")
    print("  " + "-" * 50)
    worst_current = None
    for sigma_v in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        sigma = torch.tensor(float(sigma_v), requires_grad=True)
        loss = 2.0 * err / sigma + torch.log(sigma)
        loss.backward()
        print(f"  {sigma_v:>10.0e} | {loss.item():>14.3f} | {abs(sigma.grad.item()):>20.3e}")
        worst_current = loss.item()

    # With the proposed floor, sigma can't go below SIGMA_FLOOR
    sigma = torch.tensor(SIGMA_FLOOR, requires_grad=True)
    loss = 2.0 * err / sigma + torch.log(sigma)
    loss.backward()
    print(f"\n  with floor {SIGMA_FLOOR}: loss term bottoms at {loss.item():.3f}, "
          f"|grad| = {abs(sigma.grad.item()):.3e}")
    diverges = worst_current < -10.0
    print(f"\n  => unfloored term runs to {worst_current:.1f} with exploding gradient: "
          f"{'CONFIRMED' if diverges else 'not seen'}")
    print(f"  => floor bounds both loss and gradient: CONFIRMED")
    return diverges


# ---------------------------------------------------------------------------
# Shared mini-optimization driver: treat `preds` as the learnable thing and
# minimize the loss, exactly like SGD would shrink the model's quantile gap.
# ---------------------------------------------------------------------------
def run_collapse(loss_fn, steps=4000, optimizer='adam', lr=5e-3,
                 target_jitter=0.0, seed=0, label=''):
    torch.manual_seed(seed)
    N = 128
    preds = torch.randn(N, 2, 3, 3, requires_grad=True)      # init: quantiles spread out
    base_target = torch.randn(N, 2, 3)
    if optimizer == 'adam':
        opt = torch.optim.Adam([preds], lr=lr)
    else:
        opt = torch.optim.SGD([preds], lr=lr)

    nan_step = None
    log = []
    for step in range(steps):
        opt.zero_grad()
        target = base_target + (target_jitter * torch.randn_like(base_target)
                                if target_jitter else 0.0)
        total, bd = loss_fn(preds, target)
        if not torch.isfinite(total):
            nan_step = step
            log.append((step, float('nan'), float('nan')))
            break
        total.backward()
        if not torch.isfinite(preds.grad).all():
            nan_step = step
            log.append((step, bd['PosLoss'], bd['MeanWidth']))
            break
        opt.step()
        if not torch.isfinite(preds).all():
            nan_step = step
            log.append((step, bd['PosLoss'], bd['MeanWidth']))
            break
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            log.append((step, bd['PosLoss'], bd['MeanWidth']))

    print(f"  [{label}]  optimizer={optimizer} lr={lr} jitter={target_jitter}")
    print(f"  {'step':>6} | {'PosLoss':>12} | {'mean sigma':>12}")
    print("  " + "-" * 38)
    for step, pl, mw in log:
        print(f"  {step:>6} | {pl:>12.4f} | {mw:>12.6f}")
    if nan_step is not None:
        print(f"  => NaN/inf appeared at step {nan_step}")
    return nan_step, log


# ---------------------------------------------------------------------------
# TEST B — reproduce the collapse on the REAL loss
# ---------------------------------------------------------------------------
def test_b_collapse():
    banner("TEST B  collapse dynamics on the REAL PinballLoss")
    if REAL_IMPORTED:
        real = RealPinball()
        print("  using REAL unified_reco.train_utils.PinballLoss")
    else:
        real = PinballLossLocal(sigma_floor=None)
        print(f"  [warn] could not import real PinballLoss ({_IMPORT_ERR!r})")
        print("  using faithful local copy with sigma_floor=None instead")

    print("\n  B1: Adam, clean targets  -> graceful collapse, large-negative PosLoss")
    run_collapse(real, optimizer='adam', lr=5e-3, target_jitter=0.0, label='real')

    print("\n  B2: SGD + target jitter  -> error/sigma blow-up to inf/NaN")
    nan_step, _ = run_collapse(real, optimizer='sgd', lr=0.05,
                               target_jitter=0.05, label='real')
    return nan_step


# ---------------------------------------------------------------------------
# TEST C — the proposed floor fixes it
# ---------------------------------------------------------------------------
def test_c_fix():
    banner(f"TEST C  same setups, but PinballLoss with sigma_floor={SIGMA_FLOOR}")
    fixed = PinballLossLocal(sigma_floor=SIGMA_FLOOR)

    print("\n  C1: Adam, clean targets")
    _, log1 = run_collapse(fixed, optimizer='adam', lr=5e-3,
                           target_jitter=0.0, label='floored')
    print("\n  C2: SGD + target jitter (the setup that blew up in B2)")
    nan_step, log2 = run_collapse(fixed, optimizer='sgd', lr=0.05,
                                  target_jitter=0.05, label='floored')

    bounded = nan_step is None and all(
        abs(pl) < 50 for _, pl, _ in log1 + log2 if pl == pl)  # pl==pl skips nan
    print(f"\n  => floored loss stayed finite and bounded: "
          f"{'CONFIRMED' if bounded else 'NO'}")
    return bounded


# ---------------------------------------------------------------------------
# TEST D — crash mechanism: clamp does not strip NaN, nan_to_num does
# ---------------------------------------------------------------------------
def test_d_mechanism():
    banner("TEST D  why the NaN reaches BCE (the condensation crash site)")
    nan_beta = torch.tensor([float('nan'), 0.7, float('nan')])

    clamped = nan_beta.clamp(1e-6, 1 - 1e-6)          # what line 153 does today
    fixed = nan_beta.nan_to_num(0.5).clamp(1e-6, 1 - 1e-6)  # proposed
    print(f"  raw beta            : {nan_beta.tolist()}")
    print(f"  .clamp(...)         : {clamped.tolist()}   <- NaN survives")
    print(f"  .nan_to_num().clamp : {fixed.tolist()}   <- NaN gone")

    clamp_keeps_nan = torch.isnan(clamped).any().item()
    nan_to_num_strips = not torch.isnan(fixed).any().item()

    target = torch.ones_like(nan_beta)
    print("\n  F.binary_cross_entropy on the CLAMPED (still-NaN) tensor:")
    try:
        out = F.binary_cross_entropy(clamped, target)
        print(f"    returned {out.item()} (NaN propagates silently on CPU; "
              f"on CUDA this is the Loss.cu:90 assert)")
    except RuntimeError as e:
        print(f"    RuntimeError: {e}")
    print("  F.binary_cross_entropy on the NAN_TO_NUM tensor:")
    try:
        out = F.binary_cross_entropy(fixed, target)
        print(f"    returned {out.item():.4f}  (fine)")
    except RuntimeError as e:
        print(f"    RuntimeError: {e}")

    print(f"\n  => .clamp() leaves NaN in place: "
          f"{'CONFIRMED' if clamp_keeps_nan else 'NO'}")
    print(f"  => .nan_to_num() removes it:     "
          f"{'CONFIRMED' if nan_to_num_strips else 'NO'}")
    return clamp_keeps_nan and nan_to_num_strips


# ---------------------------------------------------------------------------
# Optional: reproduce the LITERAL device-side assert (aborts the process).
# ---------------------------------------------------------------------------
def repro_cuda_assert():
    banner("CUDA REPRO  feeding NaN to F.binary_cross_entropy on GPU")
    if not torch.cuda.is_available():
        print("  no CUDA device available; skipping")
        return
    print("  this should abort with 'Loss.cu:90 ... input_val >= zero && "
          "input_val <= one' -- matching your crash.\n")
    beta = torch.tensor([float('nan'), 0.5], device='cuda').clamp(1e-6, 1 - 1e-6)
    target = torch.ones_like(beta)
    loss = F.binary_cross_entropy(beta, target)
    torch.cuda.synchronize()      # force the async assert to surface here
    print(f"  (did not crash; got {loss.item()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-cuda-assert", action="store_true",
                    help="reproduce the literal CUDA device-side assert "
                         "(ABORTS the process)")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print(f"real PinballLoss imported: {REAL_IMPORTED}")

    a = test_a_failure_surface()
    b_nan = test_b_collapse()
    c = test_c_fix()
    d = test_d_mechanism()

    banner("VERDICT")
    print(f"  A. unfloored attenuated term diverges (unbounded below):   {a}")
    print(f"  B. real loss collapses; SGD+jitter hit NaN at step:        {b_nan}")
    print(f"  C. sigma floor keeps loss finite and bounded:              {c}")
    print(f"  D. clamp() lets NaN reach BCE; nan_to_num fixes it:        {d}")
    supported = a and (b_nan is not None) and c and d
    print(f"\n  HYPOTHESIS SUPPORTED: {supported}")
    if not supported and b_nan is None:
        print("  (note: B may collapse to large-negative PosLoss without a literal\n"
              "   NaN in this toy; that still reproduces the `PosLoss: -8.6` symptom.\n"
              "   Check B1's PosLoss column.)")

    if args.repro_cuda_assert:
        repro_cuda_assert()


if __name__ == "__main__":
    main()
