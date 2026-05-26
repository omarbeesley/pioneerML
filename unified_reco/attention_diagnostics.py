"""
Attention Diagnostics for PURITY Architecture

Extracts and analyzes attention maps from all key attention layers to identify
where the model is making good vs bad decisions. Designed to help identify
architectural bottlenecks.

Probes five attention mechanisms:
  1. Backbone TransformerConv (per-time-group hit-to-hit attention)
  2. Cross-Attention Bridge (ATAR-to-LYSO / LYSO-to-ATAR)
  3. ATAR Event Builder (slice-to-slice chain reasoning)
  4. Slim Event Builder (unified ATAR+LYSO trigger decision)
  5. AttentionalAggregation gates (pooling importance weights)

Usage:
    python attention_diagnostics.py \
        --checkpoint model_weights/PURITY_best.pth \
        --dataset /path/to/mixed_events.parquet \
        --output_dir diagnostics_output/ \
        --num_events 200
"""

import argparse
import os
import sys
import json

# Add parent directory to path so `unified_reco.constants` resolves when
# running from within the unified_reco/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from torch_geometric.loader import DataLoader


# ========================================================================== #
#  Hook-Based Attention Extraction                                            #
# ========================================================================== #

class AttentionCollector:
    """
    Registers forward hooks on all attention-bearing modules in a PURITY model
    and collects their weights during inference.
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attention_maps = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on all attention mechanisms in the model."""

        # 1. Backbone TransformerConv blocks (ATAR + LYSO)
        for name, blocks in [('atar', self.model.atar_blocks),
                             ('lyso', self.model.lyso_blocks)]:
            for i, block in enumerate(blocks):
                key = f'backbone_{name}_block{i}'
                self._hook_transformer_conv(block.conv, key)

        # 2. Cross-Attention Bridge (ATAR queries LYSO)
        self._hook_multihead_attention(self.model.cross_attention, 'cross_attn_atar_queries_lyso')

        # 3. Reverse Cross-Attention (V2 only: LYSO queries ATAR)
        if hasattr(self.model, 'reverse_cross_attention'):
            self._hook_multihead_attention(self.model.reverse_cross_attention, 'cross_attn_lyso_queries_atar')

        # 4. ATAR Event Builder self-attention layers
        for i, layer in enumerate(self.model.atar_event_layers):
            self._hook_multihead_attention(layer.self_attn, f'atar_event_builder_layer{i}')

        # 5. Slim Event Builder transformer layers
        for i, layer in enumerate(self.model.slim_event_transformer.layers):
            self._hook_multihead_attention(layer.self_attn, f'slim_event_builder_layer{i}')

        # 6. AttentionalAggregation gate scores
        for pool_name in ['pool_all', 'pool_x_shared', 'pool_y_shared',
                          'pool_x_event', 'pool_y_event',
                          'pool_x_multi', 'pool_y_multi',
                          'pool_x_global', 'pool_y_global']:
            if hasattr(self.model, pool_name):
                pool = getattr(self.model, pool_name)
                self._hook_attn_aggregation(pool, f'pool_{pool_name}')

    def _hook_transformer_conv(self, conv_module, key):
        """
        TransformerConv stores attention weights when return_attention_weights=True.
        We hook into the forward to capture them.  Note: TransformerConv doesn't
        support return_attention_weights via a hook cleanly, so we store the
        module reference and query it post-forward.
        """
        # TransformerConv in PyG stores self._alpha after forward if it exists
        def hook_fn(module, input, output):
            if hasattr(module, '_alpha') and module._alpha is not None:
                self.attention_maps[key] = module._alpha.detach().cpu()
        handle = conv_module.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def _hook_multihead_attention(self, mha_module, key):
        """
        Monkey-patch the MHA forward to force need_weights=True so attention
        weights are returned.  TransformerEncoderLayer calls self_attn with
        need_weights=False by default (enabling the fused SDPA fast-path),
        which means the weights would be None without this patch.

        Forcing need_weights=True disables the fused fast-path but produces
        mathematically equivalent results (small FP32 drift at most), which
        is acceptable for diagnostics.
        """
        orig_forward = mha_module.forward

        def patched_forward(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            out = orig_forward(*args, **kwargs)
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                self.attention_maps[key] = out[1].detach().cpu()
            return out

        mha_module.forward = patched_forward
        self.hooks.append((mha_module, orig_forward))

    def _hook_attn_aggregation(self, pool_module, key):
        """
        AttentionalAggregation computes gate = softmax(gate_nn(x)) internally.
        We hook into the gate_nn to capture its output pre-softmax.
        """
        if hasattr(pool_module, 'gate_nn') and pool_module.gate_nn is not None:
            def hook_fn(module, input, output):
                self.attention_maps[key] = output.detach().cpu()
            handle = pool_module.gate_nn.register_forward_hook(hook_fn)
            self.hooks.append(handle)

    def clear(self):
        """Clear collected attention maps for the next batch."""
        self.attention_maps.clear()

    def remove_hooks(self):
        """Remove all hooks and restore patched forwards."""
        for h in self.hooks:
            if isinstance(h, tuple):
                module, original_forward = h
                module.forward = original_forward
            else:
                h.remove()
        self.hooks.clear()


# ========================================================================== #
#  Diagnostic Analysis Functions                                              #
# ========================================================================== #

def classify_event_correctness(outputs, batch):
    """
    Per-event correctness of the ATAR decay chain classification.

    Copied directly from benchmark.py logic: softmax → argmax on role_logits,
    compare to batch.atar_slice_role_target[:n_slices], excluding anchors.

    Returns:
        correct: [B] bool tensor — True if all non-anchor slices match
        pred_roles: list of int tensors per event
        true_roles: list of int tensors per event
    """
    B = int(batch.batch.max().item()) + 1

    role_logits = outputs.get('atar_role_logits')
    slice_event_idx = outputs.get('atar_slice_event_idx')
    anchor_mask = outputs.get('atar_anchor_slice_mask')

    if (role_logits is None or slice_event_idx is None
            or not hasattr(batch, 'atar_slice_role_target')):
        return (torch.zeros(B, dtype=torch.bool),
                [torch.tensor([]) for _ in range(B)],
                [torch.tensor([]) for _ in range(B)])

    # Exactly benchmark.py lines 247-251
    rp = F.softmax(role_logits, dim=-1).cpu()       # [N_valid, 3]
    n_slices = rp.shape[0]
    rt = batch.atar_slice_role_target.cpu().numpy().astype(np.int64)
    rt = rt[:n_slices]                               # truncate to match
    pred = rp.numpy().argmax(axis=-1)                # [N_valid]
    anc = anchor_mask.cpu().numpy() if anchor_mask is not None else np.zeros(n_slices, dtype=bool)
    ev_idx = slice_event_idx.cpu().numpy()

    correct = torch.ones(B, dtype=torch.bool)
    pred_roles_list = []
    true_roles_list = []

    for ev in range(B):
        mask = (ev_idx == ev) & (~anc)
        if mask.any():
            p = pred[mask]
            t = rt[mask]
            correct[ev] = bool(np.all(p == t))
            pred_roles_list.append(torch.tensor(p))
            true_roles_list.append(torch.tensor(t))
        else:
            pred_roles_list.append(torch.tensor([], dtype=torch.long))
            true_roles_list.append(torch.tensor([], dtype=torch.long))

    return correct, pred_roles_list, true_roles_list


def classify_event_builder_correctness(outputs, batch):
    """
    Per-event correctness of the LYSO event builder (trigger hit classification).

    Matches benchmark.py: per-LYSO-hit trigger probability (from assignment-
    weighted mixture) thresholded at 0.5, compared to is_trigger_target.
    An event is "correct" if the energy-weighted IoU > 0.5.

    Returns:
        correct: [B] bool tensor
        iou_per_event: [B] float tensor
    """
    B = int(batch.batch.max().item()) + 1
    device = batch.x.device

    is_lyso = (batch.x[:, 7] > 0.5)
    p_hit = outputs.get('lyso_hit_trigger_prob')

    if p_hit is None or not is_lyso.any() or not hasattr(batch, 'is_trigger_target'):
        return torch.zeros(B, dtype=torch.bool), torch.zeros(B)

    lyso_batch = batch.batch[is_lyso].cpu()
    pred_trig = (p_hit > 0.5).cpu()
    truth_trig = batch.is_trigger_target[is_lyso].cpu().bool()
    lyso_energy = batch.x[is_lyso, 3].cpu()

    iou_per_event = torch.full((B,), float('nan'))
    correct = torch.zeros(B, dtype=torch.bool)

    for k in range(B):
        m = (lyso_batch == k)
        if not m.any():
            continue
        pp = pred_trig[m]
        tp = truth_trig[m]
        e = lyso_energy[m]

        # Energy-weighted IoU
        e_pred = (pp.float() * e).sum()
        e_truth = (tp.float() * e).sum()
        e_match = ((pp & tp).float() * e).sum()
        e_union = e_pred + e_truth - e_match

        if e_union > 1e-6:
            iou = (e_match / e_union).item()
        elif e_truth < 1e-6 and e_pred < 1e-6:
            iou = 1.0  # both empty = correct
        else:
            iou = 0.0

        iou_per_event[k] = iou
        correct[k] = iou > 0.5

    return correct, iou_per_event


def analyze_backbone_attention(attn_maps, batch_data, is_atar, is_lyso):
    """
    Analyze backbone TransformerConv attention patterns.

    Returns dict with:
      - cross_view_attention_frac: how much attention flows between XZ and YZ views
      - energy_attention_correlation: correlation between hit energy and attention received
      - attention_entropy: per-head entropy (uniform = high, focused = low)
    """
    stats = {}
    for key, alpha in attn_maps.items():
        if not key.startswith('backbone_'):
            continue

        # alpha: [num_edges, num_heads] — attention weight per edge per head
        if alpha.numel() == 0:
            continue

        # Per-head entropy (normalized by log(fan_in) for comparability)
        num_heads = alpha.shape[1]
        head_entropies = []
        for h in range(num_heads):
            weights_h = alpha[:, h]
            # Approximate per-node entropy by averaging over all edges
            eps = 1e-8
            ent = -(weights_h * torch.log(weights_h + eps)).mean().item()
            head_entropies.append(ent)

        stats[key] = {
            'mean_attention': alpha.mean().item(),
            'std_attention': alpha.std().item(),
            'per_head_entropy': head_entropies,
            'mean_entropy': float(np.mean(head_entropies)),
        }

    return stats


def _unpack_attn(attn, nhead_hint=None):
    """
    Normalize attention tensor to [B, nhead, T_q, T_k].

    nn.MultiheadAttention with average_attn_weights=False returns
    [B, nhead, T, T] (4D).  With average_attn_weights=True it returns
    [B, T, T] (3D, nhead=1 effectively).  Some older PyTorch versions
    or custom wrappers may return [B*nhead, T, T] (3D).

    Returns (attn_4d, B, nhead, T_q, T_k) or None if shape is unrecognized.
    """
    if attn.dim() == 4:
        # Already [B, nhead, T_q, T_k]
        B, nhead, T_q, T_k = attn.shape
        return attn, B, nhead, T_q, T_k
    elif attn.dim() == 3:
        # Could be [B, T_q, T_k] (averaged) or [B*nhead, T_q, T_k]
        d0, T_q, T_k = attn.shape
        if nhead_hint is not None and d0 % nhead_hint == 0 and d0 // nhead_hint > 1:
            B = d0 // nhead_hint
            return attn.view(B, nhead_hint, T_q, T_k), B, nhead_hint, T_q, T_k
        else:
            # Treat as [B, T_q, T_k] with nhead=1
            return attn.unsqueeze(1), d0, 1, T_q, T_k
    return None


def analyze_cross_attention(attn_maps, outputs, batch_data):
    """
    Analyze the cross-attention bridge (ATAR queries LYSO).

    Returns:
      - attention_to_trigger_lyso: fraction of attention going to trigger LYSO hits
      - attention_concentration: how concentrated the attention is (Gini coefficient)
    """
    stats = {}
    ca_key = 'cross_attn_atar_queries_lyso'
    if ca_key not in attn_maps:
        return stats

    result = _unpack_attn(attn_maps[ca_key], nhead_hint=5)
    if result is None:
        return stats

    attn_4d, B, nhead, N_atar, N_lyso = result
    attn_avg = attn_4d.mean(dim=1)  # [B, N_atar, N_lyso]

    # Concentration: what fraction of attention goes to top-3 LYSO keys
    topk_vals, _ = attn_avg.topk(min(3, N_lyso), dim=-1)
    top3_frac = topk_vals.sum(dim=-1).mean().item()

    stats[ca_key] = {
        'mean_attention': attn_avg.mean().item(),
        'top3_concentration': top3_frac,
        'num_events': B,
    }

    return stats


def analyze_event_builder_attention(attn_maps, outputs, correct_mask):
    """
    Analyze ATAR event builder self-attention patterns for correct vs incorrect
    event builder decisions.

    Key diagnostic: does the model form a clear pi->mu->e+ chain in the attention?
    """
    stats = {'correct': defaultdict(list), 'incorrect': defaultdict(list)}

    for layer_key in sorted(k for k in attn_maps if k.startswith('atar_event_builder')):
        attn = attn_maps[layer_key]
        if attn.dim() < 3:
            continue

        result = _unpack_attn(attn, nhead_hint=4)
        if result is None:
            continue
        attn_4d, B, nhead, N_max, _ = result
        attn_avg = attn_4d.mean(dim=1)  # [B, N, N]

        for b in range(min(B, len(correct_mask))):
            a = attn_avg[b]  # [N, N]

            # Entropy of attention distribution (low = focused chain, high = diffuse)
            eps = 1e-8
            ent = -(a * torch.log(a + eps)).sum(dim=-1).mean().item()

            # Max off-diagonal attention (chain strength)
            diag_mask = torch.eye(N_max, dtype=torch.bool)
            off_diag = a.masked_fill(diag_mask, 0.0)
            max_chain = off_diag.max().item()

            # Attention asymmetry: |A - A^T| measures directional flow
            asymmetry = (a - a.T).abs().mean().item()

            bucket = 'correct' if correct_mask[b] else 'incorrect'
            stats[bucket][f'{layer_key}_entropy'].append(ent)
            stats[bucket][f'{layer_key}_max_chain'].append(max_chain)
            stats[bucket][f'{layer_key}_asymmetry'].append(asymmetry)

    # Average over events
    result = {}
    for bucket in ['correct', 'incorrect']:
        result[bucket] = {}
        for metric, values in stats[bucket].items():
            if len(values) > 0:
                result[bucket][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'n': len(values),
                }
    return result


def analyze_slim_event_builder(attn_maps, outputs, correct_mask):
    """
    Analyze the slim event builder transformer attention.

    Key diagnostic: how do ATAR track tokens and LYSO cluster tokens attend
    to each other? For correct trigger decisions, do LYSO clusters attend
    strongly to their matching ATAR track?
    """
    stats = {'correct': defaultdict(list), 'incorrect': defaultdict(list)}

    n_atar = outputs.get('unified_num_atar_tokens', 0)

    for layer_key in sorted(k for k in attn_maps if k.startswith('slim_event_builder')):
        attn = attn_maps[layer_key]
        if attn.dim() < 3:
            continue

        unpacked = _unpack_attn(attn, nhead_hint=4)
        if unpacked is None:
            continue
        attn_4d, B, nhead, T, _ = unpacked
        attn_avg = attn_4d.mean(dim=1)  # [B, T, T]

        for b in range(min(B, len(correct_mask))):
            a = attn_avg[b]  # [T, T]

            # Cross-modality attention: LYSO tokens attending to ATAR tokens
            # This is approximate since token ordering depends on batch content
            if T > 1:
                # Overall attention entropy
                eps = 1e-8
                ent = -(a * torch.log(a + eps)).sum(dim=-1).mean().item()

                # Self-attention vs cross-attention ratio
                diag_sum = a.diag().sum().item()
                total_sum = a.sum().item()
                self_ratio = diag_sum / max(total_sum, 1e-6)

                bucket = 'correct' if correct_mask[b] else 'incorrect'
                stats[bucket][f'{layer_key}_entropy'].append(ent)
                stats[bucket][f'{layer_key}_self_ratio'].append(self_ratio)

    result = {}
    for bucket in ['correct', 'incorrect']:
        result[bucket] = {}
        for metric, values in stats[bucket].items():
            if len(values) > 0:
                result[bucket][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'n': len(values),
                }
    return result


def analyze_pool_gates(attn_maps, batch_data, is_atar):
    """
    Analyze AttentionalAggregation gate scores: which hits does the model
    consider important for each pooling operation?
    """
    stats = {}
    for key, gate_out in attn_maps.items():
        if not key.startswith('pool_'):
            continue
        if gate_out.numel() == 0:
            continue

        gate_scores = gate_out.squeeze(-1)  # [N_hits]

        # Gate score statistics
        stats[key] = {
            'mean_gate': gate_scores.mean().item(),
            'std_gate': gate_scores.std().item(),
            'max_gate': gate_scores.max().item(),
            'min_gate': gate_scores.min().item(),
            # Effective number of attended hits (exp of entropy)
            'effective_n': float(torch.exp(
                -(F.softmax(gate_scores, dim=0) *
                  F.log_softmax(gate_scores, dim=0)).sum()
            ).item()),
            'total_n': gate_scores.shape[0],
        }

    return stats


# ========================================================================== #
#  Visualization                                                              #
# ========================================================================== #

def plot_correct_vs_incorrect(event_builder_stats, output_dir):
    """Bar chart comparing attention metrics for correct vs incorrect events."""
    if not event_builder_stats.get('correct') and not event_builder_stats.get('incorrect'):
        return

    metrics = set()
    for bucket in ['correct', 'incorrect']:
        metrics.update(event_builder_stats.get(bucket, {}).keys())

    if not metrics:
        return

    metrics = sorted(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        labels = []
        for bucket in ['correct', 'incorrect']:
            if metric in event_builder_stats.get(bucket, {}):
                s = event_builder_stats[bucket][metric]
                means.append(s['mean'])
                stds.append(s['std'])
                labels.append(f'{bucket}\n(n={s["n"]})')
            else:
                means.append(0)
                stds.append(0)
                labels.append(f'{bucket}\n(n=0)')

        x_pos = range(len(labels))
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(x_pos, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_title(metric.replace('_', ' '), fontsize=9)
        ax.set_ylabel('Value')

    plt.suptitle('ATAR Event Builder: Correct vs Incorrect', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'event_builder_correct_vs_incorrect.png'), dpi=150)
    plt.close()


def plot_slim_builder_comparison(slim_stats, output_dir):
    """Compare slim event builder attention for correct vs incorrect."""
    if not slim_stats.get('correct') and not slim_stats.get('incorrect'):
        return

    metrics = set()
    for bucket in ['correct', 'incorrect']:
        metrics.update(slim_stats.get(bucket, {}).keys())

    if not metrics:
        return

    metrics = sorted(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        labels = []
        for bucket in ['correct', 'incorrect']:
            if metric in slim_stats.get(bucket, {}):
                s = slim_stats[bucket][metric]
                means.append(s['mean'])
                stds.append(s['std'])
                labels.append(f'{bucket}\n(n={s["n"]})')
            else:
                means.append(0)
                stds.append(0)
                labels.append(f'{bucket}\n(n=0)')

        x_pos = range(len(labels))
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(x_pos, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_title(metric.replace('_', ' '), fontsize=9)

    plt.suptitle('Slim Event Builder: Correct vs Incorrect', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slim_builder_correct_vs_incorrect.png'), dpi=150)
    plt.close()


def plot_backbone_entropy(backbone_stats, output_dir):
    """Per-head entropy for each backbone block."""
    if not backbone_stats:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, prefix, title in [(axes[0], 'backbone_atar', 'ATAR Backbone'),
                               (axes[1], 'backbone_lyso', 'LYSO Backbone')]:
        block_keys = sorted(k for k in backbone_stats if k.startswith(prefix))
        if not block_keys:
            ax.set_visible(False)
            continue

        for key in block_keys:
            entropies = backbone_stats[key]['per_head_entropy']
            ax.bar(range(len(entropies)), entropies, alpha=0.6,
                   label=key.replace('backbone_', ''))

        ax.set_xlabel('Head Index')
        ax.set_ylabel('Attention Entropy')
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'backbone_attention_entropy.png'), dpi=150)
    plt.close()


def plot_pool_gate_summary(pool_stats, output_dir):
    """Summary of pooling gate effectiveness."""
    if not pool_stats:
        return

    keys = sorted(pool_stats.keys())
    effective_n = [pool_stats[k]['effective_n'] for k in keys]
    total_n = [pool_stats[k]['total_n'] for k in keys]
    ratios = [e / max(t, 1) for e, t in zip(effective_n, total_n)]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(keys))
    ax.bar(x, ratios, color='#3498db', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace('pool_pool_', '') for k in keys],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Effective Hits / Total Hits')
    ax.set_title('Pooling Gate Selectivity (lower = more focused)')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pool_gate_selectivity.png'), dpi=150)
    plt.close()


def plot_single_event_attention(attn_maps, event_idx, output_dir, tag=''):
    """
    Detailed attention heatmap for a single event from the ATAR event builder.
    Useful for visual inspection of chain formation.
    """
    for layer_key in sorted(k for k in attn_maps if k.startswith('atar_event_builder')):
        attn = attn_maps[layer_key]
        if attn.dim() < 3:
            continue

        unpacked = _unpack_attn(attn, nhead_hint=4)
        if unpacked is None:
            continue
        attn_per_head, B, nhead, N, _ = unpacked

        if event_idx >= B:
            continue

        fig, axes = plt.subplots(1, nhead + 1, figsize=(4 * (nhead + 1), 4))

        # Per-head attention
        for h in range(nhead):
            a = attn_per_head[event_idx, h].numpy()
            im = axes[h].imshow(a, cmap='viridis', vmin=0)
            axes[h].set_title(f'Head {h}', fontsize=10)
            axes[h].set_xlabel('Key Slice')
            axes[h].set_ylabel('Query Slice')
            plt.colorbar(im, ax=axes[h], fraction=0.046)

        # Average attention
        a_avg = attn_per_head[event_idx].mean(dim=0).numpy()
        im = axes[nhead].imshow(a_avg, cmap='viridis', vmin=0)
        axes[nhead].set_title('Average', fontsize=10)
        axes[nhead].set_xlabel('Key Slice')
        plt.colorbar(im, ax=axes[nhead], fraction=0.046)

        plt.suptitle(f'{layer_key} — Event {event_idx} {tag}', fontsize=11)
        plt.tight_layout()
        fname = f'event{event_idx}_{layer_key}{tag}.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()


# ========================================================================== #
#  Main Diagnostic Pipeline                                                   #
# ========================================================================== #

def run_diagnostics(model, dataloader, device, output_dir, max_events=200):
    """Run full attention diagnostic analysis."""
    os.makedirs(output_dir, exist_ok=True)

    model.train()

    # Task weights matching benchmark.py
    task_weights = {
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

    # Aggregated statistics
    all_backbone_stats = defaultdict(lambda: defaultdict(list))
    all_cross_attn_stats = defaultdict(list)
    all_event_builder_stats = {'correct': defaultdict(list), 'incorrect': defaultdict(list)}
    all_slim_stats = {'correct': defaultdict(list), 'incorrect': defaultdict(list)}
    all_pool_stats = defaultdict(lambda: defaultdict(list))
    n_correct = 0
    n_incorrect = 0
    n_eb_correct = 0
    n_eb_incorrect = 0
    all_iou = []
    n_events_processed = 0

    # Per-slice role confusion matrix: role_confusion[true_role][pred_role]
    role_names = {0: 'none', 1: 'muon', 2: 'positron'}
    role_confusion = {0: {0: 0, 1: 0, 2: 0},
                      1: {0: 0, 1: 0, 2: 0},
                      2: {0: 0, 1: 0, 2: 0}}

    # Track example events for detailed visualization
    example_correct_idx = None
    example_incorrect_idx = None
    example_correct_maps = None
    example_incorrect_maps = None

    # ================================================================
    # PASS 1: Clean forward pass (no hooks) to get correct outputs
    #         and per-event correctness labels.
    # ================================================================
    print("Pass 1: Clean inference (no hooks)...")
    batch_outputs = []  # store (correct_mask, pred_roles, true_roles, B) per batch
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if n_events_processed >= max_events:
                break

            batch = batch.to(device)
            tps = getattr(batch, 'atar_triggering_pion_slice', None)
            outputs = model(batch.x, batch.batch, task_weights=task_weights,
                           triggering_pion_slice=tps)

            correct_mask, pred_roles, true_roles = classify_event_correctness(outputs, batch)
            eb_correct, eb_iou = classify_event_builder_correctness(outputs, batch)
            B = correct_mask.shape[0]

            batch_outputs.append((correct_mask, pred_roles, true_roles, B, eb_correct))

            n_correct += correct_mask.sum().item()
            n_incorrect += (~correct_mask).sum().item()
            n_eb_correct += eb_correct.sum().item()
            n_eb_incorrect += (~eb_correct).sum().item()
            all_iou.append(eb_iou)

            for pred_ev, true_ev in zip(pred_roles, true_roles):
                if pred_ev.numel() > 0:
                    for p, t in zip(pred_ev.tolist(), true_ev.tolist()):
                        role_confusion[t][p] += 1

            n_events_processed += B

    iou_all = torch.cat(all_iou)
    iou_valid = iou_all[~iou_all.isnan()]
    print(f"Pass 1 done:")
    print(f"  Role chain: {n_correct}/{n_events_processed} correct "
          f"({100*n_correct/max(n_events_processed,1):.1f}%)")
    print(f"  Event builder (IoU>0.5): {n_eb_correct}/{n_events_processed} correct "
          f"({100*n_eb_correct/max(n_events_processed,1):.1f}%)")
    if iou_valid.numel() > 0:
        print(f"  Event builder IoU: mean={iou_valid.mean():.3f}, "
              f"median={iou_valid.median():.3f}")

    # ================================================================
    # PASS 2: Forward pass WITH hooks to capture attention maps.
    #         Outputs are slightly corrupted by need_weights=True but
    #         we only use the attention maps, not the predictions.
    # ================================================================
    print("Pass 2: Hooked inference (capturing attention maps)...")
    collector = AttentionCollector(model)
    batch_idx_2 = 0
    n_processed_2 = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if n_processed_2 >= max_events:
                break

            batch = batch.to(device)
            collector.clear()

            tps = getattr(batch, 'atar_triggering_pion_slice', None)
            outputs = model(batch.x, batch.batch, task_weights=task_weights,
                           triggering_pion_slice=tps)

            is_atar = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
            is_lyso = (batch.x[:, 7] > 0.5)

            # Use correctness labels from Pass 1
            correct_mask = batch_outputs[batch_idx][0]   # role chain correctness
            eb_correct = batch_outputs[batch_idx][4]      # event builder correctness
            B = batch_outputs[batch_idx][3]

            attn_maps = collector.attention_maps

            bb_stats = analyze_backbone_attention(attn_maps, batch, is_atar, is_lyso)
            for key, s in bb_stats.items():
                for metric, value in s.items():
                    if isinstance(value, list):
                        all_backbone_stats[key][metric].extend(value)
                    else:
                        all_backbone_stats[key][metric].append(value)

            ca_stats = analyze_cross_attention(attn_maps, outputs, batch)
            for key, s in ca_stats.items():
                for metric, value in s.items():
                    all_cross_attn_stats[f'{key}_{metric}'].append(value)

            eb_stats = analyze_event_builder_attention(attn_maps, outputs, correct_mask)
            for bucket in ['correct', 'incorrect']:
                for metric, s in eb_stats.get(bucket, {}).items():
                    all_event_builder_stats[bucket][metric].append(s)

            se_stats = analyze_slim_event_builder(attn_maps, outputs, eb_correct)
            for bucket in ['correct', 'incorrect']:
                for metric, s in se_stats.get(bucket, {}).items():
                    all_slim_stats[bucket][metric].append(s)

            pg_stats = analyze_pool_gates(attn_maps, batch, is_atar)
            for key, s in pg_stats.items():
                for metric, value in s.items():
                    all_pool_stats[key][metric].append(value)

            # Save example events for detailed visualization
            if example_correct_maps is None and correct_mask.any():
                idx = correct_mask.nonzero(as_tuple=False)[0].item()
                example_correct_idx = idx
                example_correct_maps = {k: v.clone() for k, v in collector.attention_maps.items()}

            if example_incorrect_maps is None and (~correct_mask).any():
                idx = (~correct_mask).nonzero(as_tuple=False)[0].item()
                example_incorrect_idx = idx
                example_incorrect_maps = {k: v.clone() for k, v in collector.attention_maps.items()}

            n_processed_2 += B

    collector.remove_hooks()
    print(f"Pass 2 done: captured attention maps for {n_processed_2} events.")

    # --- Aggregate and summarize ---
    print(f"\nProcessed {n_events_processed} events:")
    print(f"  Role chain: {n_correct} correct, {n_incorrect} incorrect "
          f"({100*n_correct/max(n_events_processed,1):.1f}%)")
    print(f"  Event builder (IoU>0.5): {n_eb_correct} correct, {n_eb_incorrect} incorrect "
          f"({100*n_eb_correct/max(n_events_processed,1):.1f}%)")

    # Aggregate event builder stats
    aggregated_eb = {}
    for bucket in ['correct', 'incorrect']:
        aggregated_eb[bucket] = {}
        for metric, stat_list in all_event_builder_stats[bucket].items():
            if len(stat_list) > 0:
                means = [s['mean'] for s in stat_list]
                aggregated_eb[bucket][metric] = {
                    'mean': float(np.mean(means)),
                    'std': float(np.std(means)),
                    'n': sum(s['n'] for s in stat_list),
                }

    # Aggregate slim stats
    aggregated_slim = {}
    for bucket in ['correct', 'incorrect']:
        aggregated_slim[bucket] = {}
        for metric, stat_list in all_slim_stats[bucket].items():
            if len(stat_list) > 0:
                means = [s['mean'] for s in stat_list]
                aggregated_slim[bucket][metric] = {
                    'mean': float(np.mean(means)),
                    'std': float(np.std(means)),
                    'n': sum(s['n'] for s in stat_list),
                }

    # Aggregate backbone stats
    aggregated_backbone = {}
    for key, metrics in all_backbone_stats.items():
        aggregated_backbone[key] = {}
        for metric, values in metrics.items():
            if isinstance(values[0], list):
                # Per-head lists: average across batches
                n_heads = len(values[0])
                aggregated_backbone[key][metric] = [
                    float(np.mean([v[h] for v in values if h < len(v)]))
                    for h in range(n_heads)
                ]
            else:
                aggregated_backbone[key][metric] = float(np.mean(values))

    # Aggregate pool stats
    aggregated_pool = {}
    for key, metrics in all_pool_stats.items():
        aggregated_pool[key] = {m: float(np.mean(v)) for m, v in metrics.items()}

    # --- Generate plots ---
    print("Generating visualizations...")
    plot_correct_vs_incorrect(aggregated_eb, output_dir)
    plot_slim_builder_comparison(aggregated_slim, output_dir)
    plot_backbone_entropy(aggregated_backbone, output_dir)
    plot_pool_gate_summary(aggregated_pool, output_dir)

    # Detailed single-event attention maps
    if example_correct_maps is not None and example_correct_idx is not None:
        plot_single_event_attention(example_correct_maps, example_correct_idx,
                                    output_dir, tag='_correct')
    if example_incorrect_maps is not None and example_incorrect_idx is not None:
        plot_single_event_attention(example_incorrect_maps, example_incorrect_idx,
                                    output_dir, tag='_incorrect')

    # --- Save summary report ---
    report = {
        'total_events': n_events_processed,
        'chain_correct_events': n_correct,
        'chain_incorrect_events': n_incorrect,
        'chain_accuracy': n_correct / max(n_events_processed, 1),
        'eb_correct_events': n_eb_correct,
        'eb_incorrect_events': n_eb_incorrect,
        'eb_accuracy': n_eb_correct / max(n_events_processed, 1),
        'eb_iou_mean': float(iou_valid.mean()) if iou_valid.numel() > 0 else None,
        'eb_iou_median': float(iou_valid.median()) if iou_valid.numel() > 0 else None,
        'role_confusion_matrix': role_confusion,
        'backbone_attention': aggregated_backbone,
        'cross_attention': {k: float(np.mean(v)) for k, v in all_cross_attn_stats.items()},
        'atar_event_builder_correct_vs_incorrect': aggregated_eb,
        'slim_builder_correct_vs_incorrect': aggregated_slim,
        'pool_gate_stats': aggregated_pool,
    }

    report_path = os.path.join(output_dir, 'attention_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Role classification confusion matrix
    print(f"\nATAR Decay Chain: {n_correct}/{n_events_processed} events fully correct "
          f"({100*n_correct/max(n_events_processed,1):.1f}%)")
    print("\nPer-slice role confusion matrix (rows=truth, cols=predicted):")
    print(f"  {'':>12s}  {'none':>8s}  {'muon':>8s}  {'positron':>8s}  {'recall':>8s}")
    for t in [0, 1, 2]:
        row = role_confusion[t]
        total = sum(row.values())
        recall = row[t] / max(total, 1)
        print(f"  {role_names[t]:>12s}  {row[0]:>8d}  {row[1]:>8d}  {row[2]:>8d}  {recall:>8.1%}")
    # Per-class precision
    print(f"  {'precision':>12s}", end='')
    for p in [0, 1, 2]:
        col_total = sum(role_confusion[t][p] for t in [0, 1, 2])
        prec = role_confusion[p][p] / max(col_total, 1)
        print(f"  {prec:>8.1%}", end='')
    print()

    if aggregated_eb.get('correct') and aggregated_eb.get('incorrect'):
        print("\nATAR Event Builder (slice-to-slice chain reasoning):")
        for metric in sorted(set(list(aggregated_eb['correct'].keys()) +
                                 list(aggregated_eb['incorrect'].keys()))):
            c = aggregated_eb['correct'].get(metric, {}).get('mean', 0)
            ic = aggregated_eb['incorrect'].get(metric, {}).get('mean', 0)
            diff = c - ic
            arrow = "^" if diff > 0 else "v"
            print(f"  {metric}:  correct={c:.4f}  incorrect={ic:.4f}  "
                  f"delta={diff:+.4f} {arrow}")

    print(f"\nSlim Event Builder (LYSO trigger, bucketed by event builder IoU>0.5):")
    print(f"  {n_eb_correct}/{n_events_processed} events correct "
          f"({100*n_eb_correct/max(n_events_processed,1):.1f}%), "
          f"IoU mean={float(iou_valid.mean()):.3f}" if iou_valid.numel() > 0 else "")
    if aggregated_slim.get('correct') and aggregated_slim.get('incorrect'):
        for metric in sorted(set(list(aggregated_slim['correct'].keys()) +
                                 list(aggregated_slim['incorrect'].keys()))):
            c = aggregated_slim['correct'].get(metric, {}).get('mean', 0)
            ic = aggregated_slim['incorrect'].get(metric, {}).get('mean', 0)
            diff = c - ic
            arrow = "^" if diff > 0 else "v"
            print(f"  {metric}:  correct={c:.4f}  incorrect={ic:.4f}  "
                  f"delta={diff:+.4f} {arrow}")
    elif aggregated_slim.get('correct') or aggregated_slim.get('incorrect'):
        bucket = 'correct' if aggregated_slim.get('correct') else 'incorrect'
        print(f"  (only {bucket} events found — need more data for comparison)")
        for metric, s in aggregated_slim[bucket].items():
            print(f"  {metric}: {s['mean']:.4f} (n={s['n']})")

    if aggregated_pool:
        print("\nPooling Gate Selectivity (effective_n / total_n):")
        for key in sorted(aggregated_pool.keys()):
            s = aggregated_pool[key]
            eff = s.get('effective_n', 0)
            tot = s.get('total_n', 1)
            ratio = eff / max(tot, 1)
            print(f"  {key.replace('pool_pool_', '')}: "
                  f"{ratio:.3f} ({eff:.0f}/{tot:.0f})")

    return report


# ========================================================================== #
#  Entry Point                                                                #
# ========================================================================== #

def main():
    parser = argparse.ArgumentParser(description='PURITY Attention Diagnostics')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to mixed events parquet file')
    parser.add_argument('--output_dir', type=str, default='diagnostics_output',
                        help='Output directory for plots and report')
    parser.add_argument('--num_events', type=int, default=200,
                        help='Number of events to analyze')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--model_version', type=str, default='v1',
                        choices=['v1', 'v2'],
                        help='Which model class to load (v1=PURITYHybridModel, v2=V2)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    if args.model_version == 'v2':
        from models_v2 import PURITYHybridModelV2
        model = PURITYHybridModelV2().to(device)
    else:
        from models import PURITYHybridModel
        model = PURITYHybridModel().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        # train_purity.py / train_fast3.py save under 'model'
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError(f"Checkpoint has keys {list(checkpoint.keys())} "
                           f"— expected 'model' or 'model_state_dict'")
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Load dataset
    from dataset import PURITYDataset
    dataset = PURITYDataset(args.dataset, max_hits=300,
                            max_events=args.num_events)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Run diagnostics
    report = run_diagnostics(model, loader, device, args.output_dir,
                             max_events=args.num_events)

    print(f"\nDone! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
