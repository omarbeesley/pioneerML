"""
PURITY model benchmarking script.

Runs inference on one or more evaluation parquets, collects per-event and
per-slice predictions + truth, and writes summary parquet files for offline
plotting.

Outputs (in --output_dir):
    {tag}_events.parquet   — one row per event (truth + predictions)
    {tag}_slices.parquet   — one row per ATAR slice (role, endpoints, etc.)
    {tag}_lyso_hits.parquet — one row per LYSO hit (clustering debug)

Recommended entry point (pins all inference switches):
    python -m purity --config configs/main_filmg2_ep19.json \
        --data examples/pie.parquet --output outputs/main
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from purity.models import PURITYModel
from purity.dataset import PURITYDataset
from purity.constants import (NORM_POS_ATAR, NORM_E_LYSO, NORM_E_ATAR,
                                    NORM_T_ATAR)

# --- Opt-in ATAR hit-level dump ---------------------------------------------------------
# The model classifies every ATAR hit (trigger / MIP / pion / muon) but only an event-level
# IoU against truth was ever written out. Setting DUMP_ATAR_HITS writes the per-hit
# probabilities themselves, which is what a hit-level topology variable needs. Off by
# default: it is ~50 rows per event, so it multiplies the output size.
DUMP_ATAR_HITS = False
ATAR_HITS = {k: [] for k in ('event', 'x', 'y', 'z', 'E', 't_ns', 'is_xz', 'is_yz',
                             'p_trigger', 'p_mip', 'p_pion', 'p_muon',
                             # truth per hit (2026-08-01, for the confusion matrix and the
                             # misclassification displays; absent columns before this date)
                             't_pion', 't_muon', 't_mip', 't_is_trig')}

BATCH_SIZE = 50
# dataset.py DELETES events with atar+lyso hits outside (0, MAX_HITS]; it does not
# truncate them.  250 matches train_purity.py's --max_hits default, so the model has
# never seen a bigger event.  PURITY_MAX_HITS lifts the cut for the over-250 study
# (what would the model have done with the events we throw away?); leave it unset for
# every normal eval so the default stays exactly 250.
MAX_HITS = int(os.environ.get("PURITY_MAX_HITS", "250"))
NUM_WORKERS = 2
SENTINEL = -999.0

TASK_WEIGHTS = {
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


def run_inference(model, device, parquet_path, tag='', max_events=None,
                  mc_passes=1, use_truth_positron_mask=False):
    ds_full = PURITYDataset(parquet_path, max_hits=MAX_HITS)
    n_total = len(ds_full)
    n = n_total if max_events is None else min(n_total, max_events)

    if n < n_total:
        ds_iter = Subset(ds_full, list(range(n)))
        df = ds_full.df.iloc[:n].reset_index(drop=True)
    else:
        ds_iter = ds_full
        df = ds_full.df

    dl = DataLoader(ds_iter, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))

    # --- Truth arrays ---
    truth = {
        'theta':      df['truth_theta'].to_numpy(dtype=np.float32),
        'phi':        df['truth_phi'].to_numpy(dtype=np.float32),
        'acceptance': df['truth_acceptance'].to_numpy(dtype=np.int32)
                      if 'truth_acceptance' in df.columns else np.full(n, -1, dtype=np.int32),
        'pion_stop':  np.stack([
            df['truth_pion_stop_x'].to_numpy(dtype=np.float32),
            df['truth_pion_stop_y'].to_numpy(dtype=np.float32),
            df['truth_pion_stop_z'].to_numpy(dtype=np.float32),
        ], axis=1),
        'positron_energy': df['truth_positron_energy'].to_numpy(dtype=np.float32)
                           if 'truth_positron_energy' in df.columns
                           else np.full(n, np.nan, dtype=np.float32),
        'event_type': df['event_type'].to_numpy(dtype=np.int32)
                      if 'event_type' in df.columns else np.full(n, -1, dtype=np.int32),
        'calo_E':   (df['live_E'].to_numpy(dtype=np.float32)
                     - df['atar_posE'].to_numpy(dtype=np.float32)),
        'live_E':    df['live_E'].to_numpy(dtype=np.float32),
        'dead_E':    df['dead_E'].to_numpy(dtype=np.float32),
        'atar_posE': df['atar_posE'].to_numpy(dtype=np.float32),
        'total_E':  (df['live_E'].to_numpy(dtype=np.float32)
                     + df['dead_E'].to_numpy(dtype=np.float32)),
        # --- fields consumed by the R_e/mu analysis (purity_analysis/) ---
        'event_id':  np.arange(n, dtype=np.int64),
        # stable event key: input file and its row before the hit filter (join key for
        # main/tail outputs of the same input; event_id is the post-filter position)
        'source_file': np.full(n, str(parquet_path), dtype=object),
        'source_row': ds_full.source_row[:n],
        # atar+lyso hit count -- the quantity dataset.py:224 cuts on.  Recorded so a single
        # eval run at a loose PURITY_MAX_HITS can be re-analysed at any tighter cap by
        # filtering truth_n_hits <= cap, instead of re-running the model per cap.
        'n_hits': (df['atar_x'].map(lambda a: 0 if a is None else len(a)).to_numpy(dtype=np.int32)
                   + df['lyso_x'].map(lambda a: 0 if a is None else len(a)).to_numpy(dtype=np.int32)),
        'gen_weight': (df['gen_weight'].to_numpy(dtype=np.float64)
                       if 'gen_weight' in df.columns else np.ones(n, dtype=np.float64)),
        'positron_t': (df['truth_positron_t'].to_numpy(dtype=np.float32)
                       if 'truth_positron_t' in df.columns else np.full(n, -1000., dtype=np.float32)),
        'is_pie':    (df['truth_is_pie'].to_numpy(dtype=np.int8)
                      if 'truth_is_pie' in df.columns else np.zeros(n, dtype=np.int8)),
        'has_muon':  (df['truth_has_muon'].to_numpy(dtype=np.int8)
                      if 'truth_has_muon' in df.columns else np.zeros(n, dtype=np.int8)),
        'is_mudif':  (df['truth_is_mudif'].to_numpy(dtype=np.int8)
                      if 'truth_is_mudif' in df.columns else np.zeros(n, dtype=np.int8)),
        'is_pidif':  (df['truth_is_pidif'].to_numpy(dtype=np.int8)
                      if 'truth_is_pidif' in df.columns else np.zeros(n, dtype=np.int8)),
        'has_atar_pileup': (df['truth_has_atar_pileup'].to_numpy(dtype=np.int8)
                            if 'truth_has_atar_pileup' in df.columns else np.zeros(n, dtype=np.int8)),
        'has_accidental': (df['truth_has_accidental'].to_numpy(dtype=np.int8)
                           if 'truth_has_accidental' in df.columns else np.zeros(n, dtype=np.int8)),
        'accidental_positron_t': (df['truth_accidental_positron_t'].to_numpy(dtype=np.float32)
                                  if 'truth_accidental_positron_t' in df.columns
                                  else np.full(n, -1000., dtype=np.float32)),
    }

    # --- Prediction arrays ---
    preds = {
        'accepted':        np.full(n, SENTINEL, dtype=np.float32),
        'pion_stop':       np.full((n, 3), SENTINEL, dtype=np.float32),
        'positron_dir':    np.full((n, 3), SENTINEL, dtype=np.float32),
        'polar_angle':     np.full(n, SENTINEL, dtype=np.float32),
        'positron_energy': np.full(n, SENTINEL, dtype=np.float32),
        'htp':             np.full(n, SENTINEL, dtype=np.float32),
        'pos_precision':   np.full(n, np.nan, dtype=np.float32),
        'pos_recall':      np.full(n, np.nan, dtype=np.float32),
        'pos_iou':         np.full(n, np.nan, dtype=np.float32),
        'dead_energy':     np.full(n, SENTINEL, dtype=np.float32),
        # Reconstructed muon ATAR deposit: the ~4.1 MeV a stopped muon leaves and the pi->e
        # chain does not, which is the in-situ analysis's Michel-suppression handle.  Summed
        # from the model's own per-hit muon classification, so it uses no truth.
        'muon_energy':     np.full(n, SENTINEL, dtype=np.float32),
        # --- systematic-study columns (2026-08-01, all reco-only, see chapter Sec. in-situ) ---
        # pion Bragg energy: pion-classified ATAR deposit, the chain's start signature.
        'pion_energy':     np.full(n, SENTINEL, dtype=np.float32),
        # unassociated MIP track: MIP-classified hits NOT tagged as the trigger positron -- the
        # failed-match signature (a positron track the event builder did not claim).
        'mip_orphan_energy': np.full(n, SENTINEL, dtype=np.float32),
        'mip_orphan_nhits':  np.full(n, SENTINEL, dtype=np.float32),
        # slice-level multiplicity + top-2 trigger-slice probabilities: the exclusivity
        # diagnostics (the main model computes them; they were never persisted).
        'n_tagged_pos_slices': np.full(n, SENTINEL, dtype=np.float32),
        'trig_slice_prob_max': np.full(n, SENTINEL, dtype=np.float32),
        'trig_slice_prob_2nd': np.full(n, SENTINEL, dtype=np.float32),
        # leading LYSO cluster, computed INDEPENDENTLY of the positron match: the handle for the
        # in-situ pie matching-efficiency measurement (select on the cluster, ask if the chain
        # was reconstructed).
        'lead_cluster_E':    np.full(n, SENTINEL, dtype=np.float32),
        'lead_cluster_t_ns': np.full(n, SENTINEL, dtype=np.float32),
        'lead_cluster_dt_ns': np.full(n, SENTINEL, dtype=np.float32),
        'n_lyso_clusters':   np.full(n, SENTINEL, dtype=np.float32),
        'positron_time_ns': np.full(n, SENTINEL, dtype=np.float32),
        'positron_time_consensus_ns': np.full(n, SENTINEL, dtype=np.float32),
        'positron_time_consensus_meanconf_ns': np.full(n, SENTINEL, dtype=np.float32),
        'positron_time_argmax_role_ns': np.full(n, SENTINEL, dtype=np.float32),
        'positron_time_spread_ns': np.full(n, np.nan, dtype=np.float32),
        'positron_log_kappa': np.full(n, SENTINEL, dtype=np.float32),
    }
    truth_htp = np.zeros(n, dtype=np.int32)

    # --- Slice-level accumulators ---
    slice_role_truth, slice_role_probs, slice_is_anchor = [], [], []
    slice_pdg_truth, slice_event_id = [], []
    slice_pred_start, slice_pred_stop = [], []
    slice_truth_start, slice_truth_stop = [], []
    slice_start_sigma, slice_stop_sigma = [], []

    # --- LYSO hit-level accumulators ---
    lyso_seed_logits, lyso_seed_betas = [], []
    lyso_hit_event, lyso_hit_E = [], []
    lyso_hit_xyz_n, lyso_hit_p_hit = [], []
    lyso_hit_assign, lyso_hit_beta, lyso_hit_frac = [], [], []
    lyso_hit_t = []  # per-hit raw time (normalized col 4)
    LYSO_K = None

    # --- LYSO cluster-level accumulators (per seed, not per hit) ---
    lyso_cluster_event = []
    lyso_cluster_coinc = []      # Gaussian coincidence window value
    lyso_cluster_dt_ns = []      # TOF-corrected dt from positron (ns)
    lyso_cluster_time_ns = []    # cluster mean time (ns)

    i0 = 0
    t0 = time.time()
    mc_desc = f'infer[{tag}]' if mc_passes <= 1 else f'infer[{tag}] MC×{mc_passes}'
    # KEEP train() mode during eval -- do NOT switch to model.eval().
    # NOT for the reason this comment used to give. There is no BatchNorm in this model: all 26
    # normalization layers are nn.LayerNorm, the model holds zero buffers, and LayerNorm ignores
    # self.training, so there are no running statistics to be "poorly estimated" and no batch
    # dependence of any kind (verified: batch size 200 vs 43 vs 1 gives bitwise identical discrete
    # outputs on the same events).
    # The real reason is the attention fastpath. eval() lets PyTorch select the fused MHA kernel,
    # which accepts only a BOOLEAN padding mask and SILENTLY DISCARDS the float one -- and the float
    # mask is how the event builder's endpoint-compatibility (chi2 / best_xy) bias is injected, so
    # in eval() that physics gate vanishes and trigger-slice logits shift by ~370.
    # models/main.py disables the fastpath at import, so both modes take the same path; train()
    # is retained only as a second guard. Dropout is 0.0 for the production checkpoint, so train()
    # and eval() are bitwise identical there and --mc_passes averaging is not needed.
    model.train()
    with torch.inference_mode():
        for batch in tqdm(dl, desc=mc_desc):
            batch = batch.to(device, non_blocking=True)
            anchor = getattr(batch, 'atar_triggering_pion_slice', None)
            B = batch.num_graphs
            sl = slice(i0, i0 + B)

            # --- MC dropout: run forward mc_passes times, average p_hit ---
            # Pass 0 captures everything. Passes 1..N-1 only accumulate
            # lyso_hit_trigger_prob for averaging.
            p_hit_accum = None
            energy_accum = None

            for mc_i in range(mc_passes):
                truth_pos_mask = None
                if use_truth_positron_mask and hasattr(batch, 'is_trigger_target') and hasattr(batch, 'atar_node_pdg_target'):
                    is_atar = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                    is_trigger = batch.is_trigger_target[is_atar].bool()
                    is_mip = batch.atar_node_pdg_target[:, 2].bool()
                    truth_pos_mask_atar = is_trigger & is_mip
                    truth_pos_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=batch.x.device)
                    truth_pos_mask[is_atar] = truth_pos_mask_atar
                out = model(batch.x, batch.batch, task_weights=TASK_WEIGHTS,
                            triggering_pion_slice=anchor,
                            truth_positron_mask=truth_pos_mask)

                if mc_i == 0:
                    # First pass: capture all outputs
                    out0 = out
                    es = out.get('event_summary', {})

                    def _arr(k):
                        t = es.get(k)
                        return t.float().cpu().numpy() if isinstance(t, torch.Tensor) else None

                    a = _arr('accepted')
                    if a is not None: preds['accepted'][sl] = a
                    ps = _arr('pion_stop')
                    if ps is not None: preds['pion_stop'][sl] = ps * NORM_POS_ATAR
                    pd_ = _arr('positron_dir')
                    if pd_ is not None: preds['positron_dir'][sl] = pd_
                    pa = _arr('positron_polar_angle')
                    if pa is not None: preds['polar_angle'][sl] = pa
                    lk = _arr('positron_log_kappa')
                    if lk is not None: preds['positron_log_kappa'][sl] = lk
                    de = _arr('dead_energy')
                    if de is not None: preds['dead_energy'][sl] = de
                    htp = _arr('has_trigger_positron')
                    if htp is not None: preds['htp'][sl] = htp
                    truth_htp[sl] = batch.has_trigger_positron.view(-1).detach().cpu().numpy().astype(np.int32)

                    # --- Positron hit-level precision/recall/IoU ---
                    hit_trig = out.get('atar_hit_trigger_prob')
                    hit_mip = out.get('atar_hit_mip_prob')
                    if hit_trig is not None and hit_mip is not None \
                            and hasattr(batch, 'is_trigger_target') \
                            and hasattr(batch, 'atar_node_pdg_target'):
                        is_atar = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                        is_atar_cpu = is_atar.cpu()
                        atar_batch = batch.batch[is_atar].cpu()
                        pred_pos = ((hit_trig > 0.5) & (hit_mip > 0.5)).cpu()
                        is_trig_atar = batch.is_trigger_target.cpu().bool()[is_atar_cpu]
                        pdg = batch.atar_node_pdg_target.cpu()
                        pos_bit = pdg[:, 2] > 0.5
                        truth_pos = is_trig_atar & pos_bit
                        for k in range(B):
                            m_k = (atar_batch == k)
                            if not m_k.any():
                                continue
                            pp = pred_pos[m_k]; tp = truth_pos[m_k]
                            n_p = int(pp.sum()); n_t = int(tp.sum())
                            n_m = int((pp & tp).sum())
                            if n_p > 0: preds['pos_precision'][i0 + k] = n_m / n_p
                            if n_t > 0: preds['pos_recall'][i0 + k] = n_m / n_t
                            union = n_p + n_t - n_m
                            if union > 0: preds['pos_iou'][i0 + k] = n_m / union

                    # --- reconstructed muon ATAR energy (no truth) ---
                    hit_muon = out.get('atar_hit_muon_prob')
                    if hit_muon is not None:
                        is_atar_m = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                        if is_atar_m.any():
                            e_m = (batch.x[is_atar_m, 3] * NORM_E_ATAR).detach()
                            pm = hit_muon[is_atar_m].detach() if hit_muon.shape[0] == batch.x.shape[0] \
                                 else hit_muon.detach()
                            bm = batch.batch[is_atar_m].detach()
                            sel_m = (pm > 0.5).float()
                            acc_m = torch.zeros(B, device=e_m.device)
                            acc_m.index_add_(0, bm, e_m * sel_m)
                            preds['muon_energy'][sl] = acc_m.cpu().numpy()

                    # --- pion Bragg energy + orphan MIP track (no truth) ---
                    hit_pion = out.get('atar_hit_pion_prob')
                    if hit_pion is not None:
                        is_atar_p = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                        if is_atar_p.any():
                            e_p = (batch.x[is_atar_p, 3] * NORM_E_ATAR).detach()
                            bp = batch.batch[is_atar_p].detach()
                            pp_ = hit_pion[is_atar_p].detach() if hit_pion.shape[0] == batch.x.shape[0]                                   else hit_pion.detach()
                            acc_p = torch.zeros(B, device=e_p.device)
                            acc_p.index_add_(0, bp, e_p * (pp_ > 0.5).float())
                            preds['pion_energy'][sl] = acc_p.cpu().numpy()
                    if hit_trig is not None and hit_mip is not None:
                        is_atar_o = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                        if is_atar_o.any():
                            e_o = (batch.x[is_atar_o, 3] * NORM_E_ATAR).detach()
                            bo = batch.batch[is_atar_o].detach()
                            ht = hit_trig[is_atar_o].detach() if hit_trig.shape[0] == batch.x.shape[0] else hit_trig.detach()
                            hm2 = hit_mip[is_atar_o].detach() if hit_mip.shape[0] == batch.x.shape[0] else hit_mip.detach()
                            orphan = ((hm2 > 0.5) & (ht <= 0.5)).float()
                            acc_e = torch.zeros(B, device=e_o.device); acc_n = torch.zeros(B, device=e_o.device)
                            acc_e.index_add_(0, bo, e_o * orphan); acc_n.index_add_(0, bo, orphan)
                            preds['mip_orphan_energy'][sl] = acc_e.cpu().numpy()
                            preds['mip_orphan_nhits'][sl] = acc_n.cpu().numpy()
                    # --- slice multiplicity + top-2 trigger-slice probabilities (no truth) ---
                    ntag = out.get('n_tagged_pos_slices')
                    if ntag is not None and ntag.numel() >= B:
                        preds['n_tagged_pos_slices'][sl] = ntag.view(-1)[:B].float().cpu().numpy()
                    tlog = out.get('atar_trigger_logits'); sev = out.get('atar_slice_event_idx')
                    if tlog is not None and sev is not None and tlog.numel() == sev.numel() and tlog.numel() > 0:
                        tp2 = torch.sigmoid(tlog.detach().view(-1)).cpu().numpy()
                        se2 = sev.detach().view(-1).cpu().numpy().astype(int)
                        for k in range(B):
                            pk = tp2[se2 == k]
                            if pk.size >= 1:
                                srt = np.sort(pk)[::-1]
                                preds['trig_slice_prob_max'][i0 + k] = srt[0]
                                preds['trig_slice_prob_2nd'][i0 + k] = srt[1] if pk.size >= 2 else 0.0

                    # --- ATAR hit-level dump (opt-in, no truth used) ---
                    if DUMP_ATAR_HITS and hit_trig is not None and hit_mip is not None:
                        is_atar_d = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                        if is_atar_d.any():
                            xa = batch.x[is_atar_d].detach().cpu().numpy()
                            ba = batch.batch[is_atar_d].detach().cpu().numpy()
                            nh = xa.shape[0]

                            def _hp(name, fallback=None):
                                v = out.get(name) if fallback is None else fallback
                                if v is None:
                                    return np.full(nh, np.nan, dtype=np.float32)
                                return v.detach().cpu().numpy().reshape(-1)[:nh]

                            ATAR_HITS['event'].append((i0 + ba).astype(np.int64))
                            ATAR_HITS['x'].append(xa[:, 0] * NORM_POS_ATAR)
                            ATAR_HITS['y'].append(xa[:, 1] * NORM_POS_ATAR)
                            ATAR_HITS['z'].append(xa[:, 2] * NORM_POS_ATAR)
                            ATAR_HITS['E'].append(xa[:, 3] * NORM_E_ATAR)
                            ATAR_HITS['t_ns'].append(xa[:, 4] * NORM_T_ATAR)
                            ATAR_HITS['is_xz'].append(xa[:, 5])
                            ATAR_HITS['is_yz'].append(xa[:, 6])
                            if hasattr(batch, 'atar_node_pdg_target'):
                                pdg_t = batch.atar_node_pdg_target.detach().cpu().numpy()
                                ATAR_HITS['t_pion'].append(pdg_t[:nh, 0])
                                ATAR_HITS['t_muon'].append(pdg_t[:nh, 1])
                                ATAR_HITS['t_mip'].append(pdg_t[:nh, 2])
                            else:
                                for _k in ('t_pion', 't_muon', 't_mip'):
                                    ATAR_HITS[_k].append(np.full(nh, np.nan, dtype=np.float32))
                            if hasattr(batch, 'is_trigger_target'):
                                itt = batch.is_trigger_target.detach().cpu().float().numpy()
                                ATAR_HITS['t_is_trig'].append(itt[is_atar_d.cpu().numpy() > 0][:nh]
                                                              if itt.shape[0] == batch.x.shape[0] else itt[:nh])
                            else:
                                ATAR_HITS['t_is_trig'].append(np.full(nh, np.nan, dtype=np.float32))
                            ATAR_HITS['p_trigger'].append(_hp(None, hit_trig))
                            ATAR_HITS['p_mip'].append(_hp(None, hit_mip))
                            ATAR_HITS['p_pion'].append(_hp('atar_hit_pion_prob'))
                            ATAR_HITS['p_muon'].append(_hp('atar_hit_muon_prob'))

                    # --- Per-event timing capture ---
                    pos_time_out = out.get('positron_time_per_graph')
                    if pos_time_out is not None:
                        preds['positron_time_ns'][sl] = pos_time_out.cpu().numpy()[:B] * NORM_T_ATAR
                    pos_cons_out = out.get('positron_time_consensus')
                    if pos_cons_out is not None:
                        preds['positron_time_consensus_ns'][sl] = pos_cons_out.cpu().numpy()[:B] * NORM_T_ATAR
                    pos_mc_out = out.get('positron_time_consensus_meanconf')
                    if pos_mc_out is not None:
                        preds['positron_time_consensus_meanconf_ns'][sl] = pos_mc_out.cpu().numpy()[:B] * NORM_T_ATAR
                    pos_ar_out = out.get('positron_time_argmax_role')
                    if pos_ar_out is not None:
                        preds['positron_time_argmax_role_ns'][sl] = pos_ar_out.cpu().numpy()[:B] * NORM_T_ATAR
                    hit_trig_ts = out.get('atar_hit_trigger_prob')
                    hit_mip_ts = out.get('atar_hit_mip_prob')
                    if hit_trig_ts is not None and hit_mip_ts is not None:
                        is_atar_ts = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                        if is_atar_ts.any():
                            t_atar = batch.x[is_atar_ts, 4].cpu()
                            b_atar = batch.batch[is_atar_ts].cpu()
                            w_pos = ((hit_trig_ts > 0.5) & (hit_mip_ts > 0.5)).float().cpu()
                            for k in range(B):
                                m_k = (b_atar == k)
                                w_k = w_pos[m_k]
                                if w_k.sum() > 1.5:
                                    t_k = t_atar[m_k]
                                    wsum = w_k.sum()
                                    mean_t = (w_k * t_k).sum() / wsum
                                    var_t = (w_k * (t_k - mean_t)**2).sum() / wsum
                                    preds['positron_time_spread_ns'][i0 + k] = float(
                                        (var_t.clamp(min=0).sqrt() * 500).item())

                # Accumulate p_hit and positron_energy across MC passes
                p_hit_mc = out.get('lyso_hit_trigger_prob')
                if p_hit_mc is not None:
                    p_mc = p_hit_mc.cpu()
                    if p_hit_accum is None:
                        p_hit_accum = p_mc.clone()
                    else:
                        p_hit_accum += p_mc

                pe_mc = out.get('event_summary', {}).get('positron_energy')
                if pe_mc is not None:
                    pe_np = pe_mc.float().cpu()
                    if energy_accum is None:
                        energy_accum = pe_np.clone()
                    else:
                        energy_accum += pe_np

            # Average MC accumulations
            if p_hit_accum is not None:
                p_hit_accum /= mc_passes
            if energy_accum is not None:
                preds['positron_energy'][sl] = (energy_accum / mc_passes).numpy()

            # Use pass-0 outputs for everything below, but with averaged p_hit
            out = out0
            p_hit_t = p_hit_accum  # averaged across MC passes (or None)

            # --- LYSO clustering capture ---
            beta_lyso = out.get('lyso_seed_beta')
            w_lyso = out.get('lyso_soft_assignments')
            ev_logits = out.get('unified_event_logits')
            n_atar_tok = int(out.get('unified_num_atar_tokens', 0))
            beta_hit_t = out.get('lyso_beta')
            frac_hit_t = out.get('lyso_fractions')

            if w_lyso is not None and beta_lyso is not None and ev_logits is not None:
                if LYSO_K is None:
                    LYSO_K = int(w_lyso.size(1))
                K = LYSO_K
                is_lyso_b = (batch.x[:, 7] > 0.5).cpu()
                batch_lyso_b = batch.batch[is_lyso_b].cpu().numpy()

                seed_logits = np.full((B, K), np.nan, dtype=np.float32)
                seed_betas = np.full((B, K), np.nan, dtype=np.float32)
                has_lyso_g = np.zeros(B, dtype=bool)
                for ev in range(B):
                    has_lyso_g[ev] = (batch_lyso_b == ev).any()
                n_valid = int(has_lyso_g.sum())
                if n_valid > 0:
                    sl_logits = ev_logits[n_atar_tok:].view(n_valid, K).cpu().numpy()
                    sl_betas = beta_lyso.view(n_valid, K).cpu().numpy()
                    seed_logits[has_lyso_g] = sl_logits
                    seed_betas[has_lyso_g] = sl_betas
                for ev in range(B):
                    lyso_seed_logits.append(seed_logits[ev])
                    lyso_seed_betas.append(seed_betas[ev])

                if is_lyso_b.any():
                    e_lyso = (batch.x[is_lyso_b, 3].cpu() * NORM_E_LYSO).numpy()
                    xyz_n = batch.x[is_lyso_b, 0:3].cpu().numpy()
                    w_np = w_lyso.cpu().numpy()
                    p_np = p_hit_t.numpy() if p_hit_t is not None else np.full(w_np.shape[0], np.nan)
                    b_np = beta_hit_t.view(-1).cpu().numpy() if beta_hit_t is not None else np.full(w_np.shape[0], np.nan)
                    f_np = frac_hit_t.view(-1).cpu().numpy() if frac_hit_t is not None else np.full(w_np.shape[0], np.nan)
                    t_lyso_n = batch.x[is_lyso_b, 4].cpu().numpy()  # normalized time
                    for hidx in range(w_np.shape[0]):
                        lyso_hit_event.append(i0 + int(batch_lyso_b[hidx]))
                        lyso_hit_E.append(e_lyso[hidx])
                        lyso_hit_xyz_n.append(xyz_n[hidx])
                        lyso_hit_p_hit.append(p_np[hidx])
                        lyso_hit_assign.append(w_np[hidx])
                        lyso_hit_beta.append(b_np[hidx])
                        lyso_hit_frac.append(f_np[hidx])
                        lyso_hit_t.append(t_lyso_n[hidx] * 500.0)  # to ns

                # Per-cluster timing (coinc_feat, dt_corr_ns, cluster_time)
                coinc_out = out.get('lyso_coinc_feat')        # [Total_K]
                dt_out = out.get('lyso_dt_corr_ns')           # [Total_K] ns
                ct_out = out.get('lyso_cluster_times')        # [Total_K] ns. Affinity-weighted mean cluster time.
                if coinc_out is not None and n_valid > 0:
                    coinc_np = coinc_out.cpu().numpy().reshape(n_valid, K)
                    dt_np = dt_out.cpu().numpy().reshape(n_valid, K) if dt_out is not None else np.full((n_valid, K), np.nan)
                    ct_np = ct_out.cpu().numpy().reshape(n_valid, K) if ct_out is not None else np.full((n_valid, K), np.nan)
                    valid_event_ids = np.where(has_lyso_g)[0]
                    ke_out = out.get('lyso_cluster_energies')     # [Total_K], units of NORM_E_LYSO
                    ke_np = (ke_out.cpu().numpy().reshape(n_valid, K) * NORM_E_LYSO
                             if ke_out is not None and ke_out.numel() == n_valid * K else None)
                    for vi in range(n_valid):
                        ev_global = i0 + int(valid_event_ids[vi])
                        if ke_np is not None:
                            ei = ke_np[vi]
                            li = int(np.argmax(ei))
                            preds['lead_cluster_E'][ev_global] = ei[li]
                            preds['lead_cluster_t_ns'][ev_global] = ct_np[vi, li]
                            preds['lead_cluster_dt_ns'][ev_global] = dt_np[vi, li]
                            preds['n_lyso_clusters'][ev_global] = float((ei > 0.5).sum())
                        for ki in range(K):
                            lyso_cluster_event.append(ev_global)
                            lyso_cluster_coinc.append(float(coinc_np[vi, ki]))
                            lyso_cluster_dt_ns.append(float(dt_np[vi, ki]))
                            lyso_cluster_time_ns.append(float(ct_np[vi, ki]))
            else:
                k = LYSO_K if LYSO_K is not None else 4
                for ev in range(B):
                    lyso_seed_logits.append(np.full(k, np.nan, dtype=np.float32))
                    lyso_seed_betas.append(np.full(k, np.nan, dtype=np.float32))

            # --- Slice-level role/endpoint capture ---
            role_logits = out.get('atar_role_logits')
            endpoints = out.get('atar_endpoints')
            slice_ev_id = out.get('atar_slice_event_idx')
            anchor_mask = out.get('atar_anchor_slice_mask')

            if role_logits is not None and hasattr(batch, 'atar_slice_role_target'):
                rp = F.softmax(role_logits, dim=-1).cpu().numpy()
                rt = batch.atar_slice_role_target.cpu().numpy().astype(np.int64)
                n_slices = rp.shape[0]
                slice_role_probs.append(rp)
                slice_role_truth.append(rt[:n_slices])
                slice_is_anchor.append(
                    anchor_mask.cpu().numpy() if anchor_mask is not None
                    else np.zeros(n_slices, dtype=bool)
                )
                if hasattr(batch, 'atar_slice_pdg_target'):
                    slice_pdg_truth.append(batch.atar_slice_pdg_target[:n_slices].cpu().numpy())
                else:
                    slice_pdg_truth.append(np.full((n_slices, 3), np.nan, dtype=np.float32))
                if slice_ev_id is not None:
                    ev_local = slice_ev_id.cpu().numpy().astype(np.int64)
                    slice_event_id.append(ev_local + i0)
                else:
                    slice_event_id.append(np.full(n_slices, -1, dtype=np.int64))
                if endpoints is not None:
                    ep = endpoints.cpu().numpy()
                    slice_pred_start.append(ep[:, 0, :, 1] * NORM_POS_ATAR)
                    slice_pred_stop.append(ep[:, 1, :, 1] * NORM_POS_ATAR)
                    slice_start_sigma.append(((ep[:, 0, :, 2] - ep[:, 0, :, 0]) / 2) * NORM_POS_ATAR)
                    slice_stop_sigma.append(((ep[:, 1, :, 2] - ep[:, 1, :, 0]) / 2) * NORM_POS_ATAR)
                else:
                    slice_pred_start.append(np.full((n_slices, 3), np.nan))
                    slice_pred_stop.append(np.full((n_slices, 3), np.nan))
                    slice_start_sigma.append(np.full((n_slices, 3), np.nan))
                    slice_stop_sigma.append(np.full((n_slices, 3), np.nan))
                if hasattr(batch, 'atar_slice_start_target'):
                    slice_truth_start.append(batch.atar_slice_start_target[:n_slices].cpu().numpy() * NORM_POS_ATAR)
                    slice_truth_stop.append(batch.atar_slice_stop_target[:n_slices].cpu().numpy() * NORM_POS_ATAR)
                else:
                    slice_truth_start.append(np.full((n_slices, 3), np.nan))
                    slice_truth_stop.append(np.full((n_slices, 3), np.nan))

            i0 += B

    dt = time.time() - t0
    truth['htp'] = truth_htp
    print(f'{tag}: {n}/{n_total} events in {dt:.1f}s ({n/dt:.1f} evt/s)  '
          f'truth htp=1: {int(truth_htp.sum())} ({truth_htp.mean():.1%})')

    return truth, preds, slice_role_truth, slice_role_probs, slice_is_anchor, \
           slice_pdg_truth, slice_event_id, slice_pred_start, slice_pred_stop, \
           slice_truth_start, slice_truth_stop, slice_start_sigma, slice_stop_sigma, \
           lyso_hit_event, lyso_hit_E, lyso_hit_xyz_n, lyso_hit_p_hit, \
           lyso_hit_assign, lyso_hit_beta, lyso_hit_frac, lyso_hit_t, LYSO_K, \
           lyso_cluster_event, lyso_cluster_coinc, lyso_cluster_dt_ns, lyso_cluster_time_ns


def save_event_parquet(truth, preds, output_path):
    """Save per-event truth + predictions to a single parquet."""
    data = {
        'truth_theta': truth['theta'],
        'truth_phi': truth['phi'],
        'truth_acceptance': truth['acceptance'],
        'truth_pion_stop_x': truth['pion_stop'][:, 0],
        'truth_pion_stop_y': truth['pion_stop'][:, 1],
        'truth_pion_stop_z': truth['pion_stop'][:, 2],
        'truth_positron_energy': truth['positron_energy'],
        'truth_event_type': truth['event_type'],
        'truth_calo_E': truth['calo_E'],
        'truth_live_E': truth['live_E'],
        'truth_dead_E': truth['dead_E'],
        'truth_atar_posE': truth['atar_posE'],
        'truth_total_E': truth['total_E'],
        'truth_htp': truth['htp'],
        # --- fields for purity_analysis (R_e/mu) ---
        'event_id': truth['event_id'],
        'source_file': truth['source_file'],
        'source_row': truth['source_row'],
        'n_hits': truth['n_hits'],
        'truth_gen_weight': truth['gen_weight'],
        'truth_positron_t': truth['positron_t'],
        'truth_is_pie': truth['is_pie'],
        'truth_has_muon': truth['has_muon'],
        'truth_is_mudif': truth['is_mudif'],
        'truth_is_pidif': truth['is_pidif'],
        'truth_has_atar_pileup': truth['has_atar_pileup'],
        'truth_has_accidental': truth['has_accidental'],
        'truth_accidental_positron_t': truth['accidental_positron_t'],
        'pred_accepted': preds['accepted'],
        'pred_pion_stop_x': preds['pion_stop'][:, 0],
        'pred_pion_stop_y': preds['pion_stop'][:, 1],
        'pred_pion_stop_z': preds['pion_stop'][:, 2],
        'pred_positron_dir_x': preds['positron_dir'][:, 0],
        'pred_positron_dir_y': preds['positron_dir'][:, 1],
        'pred_positron_dir_z': preds['positron_dir'][:, 2],
        'pred_polar_angle': preds['polar_angle'],
        'pred_positron_energy': preds['positron_energy'],
        'pred_htp': preds['htp'],
        'pred_dead_energy': preds['dead_energy'],
        'pred_muon_energy': preds['muon_energy'],
        'pred_pion_energy': preds['pion_energy'],
        'pred_mip_orphan_energy': preds['mip_orphan_energy'],
        'pred_mip_orphan_nhits': preds['mip_orphan_nhits'],
        'pred_n_tagged_pos_slices': preds['n_tagged_pos_slices'],
        'pred_trig_slice_prob_max': preds['trig_slice_prob_max'],
        'pred_trig_slice_prob_2nd': preds['trig_slice_prob_2nd'],
        'pred_lead_cluster_E': preds['lead_cluster_E'],
        'pred_lead_cluster_t_ns': preds['lead_cluster_t_ns'],
        'pred_lead_cluster_dt_ns': preds['lead_cluster_dt_ns'],
        'pred_n_lyso_clusters': preds['n_lyso_clusters'],
        'pred_pos_precision': preds['pos_precision'],
        'pred_pos_recall': preds['pos_recall'],
        'pred_pos_iou': preds['pos_iou'],
        'pred_positron_time_ns': preds['positron_time_ns'],
        'pred_positron_time_consensus_ns': preds['positron_time_consensus_ns'],
        'pred_positron_time_consensus_meanconf_ns': preds['positron_time_consensus_meanconf_ns'],
        'pred_positron_time_argmax_role_ns': preds['positron_time_argmax_role_ns'],
        'pred_positron_time_spread_ns': preds['positron_time_spread_ns'],
        'pred_positron_log_kappa': preds['positron_log_kappa'],
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"  wrote {output_path} ({len(df)} events)")


def save_slice_parquet(slice_data, output_path):
    """Save per-slice truth + predictions to a parquet."""
    role_truth = np.concatenate(slice_data['role_truth'])
    role_probs = np.concatenate(slice_data['role_probs'])
    is_anchor = np.concatenate(slice_data['is_anchor'])
    pdg_truth = np.concatenate(slice_data['pdg_truth'])
    event_id = np.concatenate(slice_data['event_id'])
    pred_start = np.concatenate(slice_data['pred_start'])
    pred_stop = np.concatenate(slice_data['pred_stop'])
    truth_start = np.concatenate(slice_data['truth_start'])
    truth_stop = np.concatenate(slice_data['truth_stop'])
    start_sigma = np.concatenate(slice_data['start_sigma'])
    stop_sigma = np.concatenate(slice_data['stop_sigma'])

    data = {
        'event_id': event_id,
        'role_truth': role_truth,
        'role_pred': role_probs.argmax(axis=-1),
        'role_prob_none': role_probs[:, 0],
        'role_prob_muon': role_probs[:, 1],
        'role_prob_positron': role_probs[:, 2],
        'is_anchor': is_anchor,
        'pdg_truth_pion': pdg_truth[:, 0] if pdg_truth.ndim > 1 else pdg_truth,
        'pdg_truth_muon': pdg_truth[:, 1] if pdg_truth.ndim > 1 else pdg_truth,
        'pdg_truth_mip': pdg_truth[:, 2] if pdg_truth.ndim > 1 else pdg_truth,
    }
    for axis_i, axis_name in enumerate(['x', 'y', 'z']):
        data[f'pred_start_{axis_name}'] = pred_start[:, axis_i]
        data[f'pred_stop_{axis_name}'] = pred_stop[:, axis_i]
        data[f'truth_start_{axis_name}'] = truth_start[:, axis_i]
        data[f'truth_stop_{axis_name}'] = truth_stop[:, axis_i]
        data[f'start_sigma_{axis_name}'] = start_sigma[:, axis_i]
        data[f'stop_sigma_{axis_name}'] = stop_sigma[:, axis_i]

    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"  wrote {output_path} ({len(df)} slices)")


def save_lyso_parquet(lyso_data, output_path):
    """Save per-LYSO-hit data to a parquet."""
    n_hits = len(lyso_data['event'])
    if n_hits == 0:
        print(f"  no LYSO hits to write, skipping {output_path}")
        return

    data = {
        'event_id': np.asarray(lyso_data['event'], dtype=np.int64),
        'E': np.asarray(lyso_data['E'], dtype=np.float32),
        'x_n': np.stack(lyso_data['xyz_n'])[:, 0] if lyso_data['xyz_n'] else np.zeros(0),
        'y_n': np.stack(lyso_data['xyz_n'])[:, 1] if lyso_data['xyz_n'] else np.zeros(0),
        'z_n': np.stack(lyso_data['xyz_n'])[:, 2] if lyso_data['xyz_n'] else np.zeros(0),
        'p_hit': np.asarray(lyso_data['p_hit'], dtype=np.float32),
        'beta': np.asarray(lyso_data['beta'], dtype=np.float32),
        'frac': np.asarray(lyso_data['frac'], dtype=np.float32),
        't_ns': np.asarray(lyso_data['t_ns'], dtype=np.float32),
    }
    # Add per-seed assignment columns
    K = lyso_data['K']
    if lyso_data['assign']:
        assigns = np.stack(lyso_data['assign'])
        for ki in range(K):
            data[f'assign_seed{ki}'] = assigns[:, ki]

    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"  wrote {output_path} ({len(df)} LYSO hits)")


def save_atar_hit_parquet(output_path):
    """Write the accumulated ATAR hit dump and clear it for the next eval file.

    Positions are in mm as the model receives them, so one transverse coordinate is zero per
    hit: is_xz hits carry x, is_yz hits carry y, and both carry z. Anything built from this
    table therefore sees exactly what the network sees, which is the point of dumping it.
    """
    n = sum(len(a) for a in ATAR_HITS['event'])
    if n == 0:
        print(f"  [atar_hits] nothing to write for {output_path}")
        return
    data = {k: np.concatenate(v) for k, v in ATAR_HITS.items()}
    data['event_id'] = data.pop('event').astype(np.int64)
    df = pd.DataFrame({k: (v.astype(np.int64) if k == 'event_id' else v.astype(np.float32))
                       for k, v in data.items()})
    df.to_parquet(output_path, index=False)
    print(f"  [atar_hits] {len(df):,} hits -> {output_path}")
    for v in ATAR_HITS.values():
        v.clear()


def save_lyso_cluster_parquet(cluster_data, output_path):
    """Save per-LYSO-cluster timing data to a parquet."""
    n = len(cluster_data['event'])
    if n == 0:
        print(f"  no LYSO clusters to write, skipping {output_path}")
        return

    data = {
        'event_id': np.asarray(cluster_data['event'], dtype=np.int64),
        'coinc_feat': np.asarray(cluster_data['coinc'], dtype=np.float32),
        'dt_corr_ns': np.asarray(cluster_data['dt_ns'], dtype=np.float32),
        'cluster_time_ns': np.asarray(cluster_data['time_ns'], dtype=np.float32),
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"  wrote {output_path} ({len(df)} clusters)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint .pth")
    parser.add_argument("--eval_path", required=True, nargs='+',
                        help="One or more evaluation parquet paths.")
    parser.add_argument("--tag", nargs='+', default=None,
                        help="Tag per eval_path (for output filenames). "
                             "Defaults to parquet parent dir name.")
    parser.add_argument("--output_dir", default="./benchmark_results",
                        help="Directory to write result parquets.")
    parser.add_argument("--max_events", type=int, default=None,
                        help="Cap events per eval parquet.")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (must match training, default 0.1)")
    parser.add_argument("--mc_passes", type=int, default=1,
                        help="MC dropout passes for averaging p_hit (default 1 = no averaging)")
    parser.add_argument("--truth_positron_mask", action="store_true",
                        help="Use truth-level hit labels for positron direction head input.")
    parser.add_argument("--dump_atar_hits", action="store_true",
                        help="Also write {tag}_atar_hits.parquet: per-ATAR-hit position, "
                             "energy, time and the model's trigger/MIP/pion/muon "
                             "probabilities. Large (~50 rows per event).")
    args = parser.parse_args()

    global BATCH_SIZE, DUMP_ATAR_HITS
    BATCH_SIZE = args.batch_size
    DUMP_ATAR_HITS = args.dump_atar_hits

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}", flush=True)

    # Load model
    model = PURITYModel(dropout=args.dropout).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint: {args.checkpoint} "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', '?')})", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate tags if not provided
    tags = args.tag
    if tags is None:
        tags = [os.path.basename(os.path.dirname(p)) for p in args.eval_path]
    if len(tags) != len(args.eval_path):
        print(f"[error] {len(tags)} tags for {len(args.eval_path)} eval paths")
        return

    for eval_path, tag in zip(args.eval_path, tags):
        print(f"\n{'='*60}")
        print(f"Benchmarking: {tag}")
        print(f"  parquet: {eval_path}")
        print(f"{'='*60}")

        if not os.path.exists(eval_path):
            print(f"  [skip] not found: {eval_path}")
            continue

        (truth, preds,
         s_role_truth, s_role_probs, s_is_anchor,
         s_pdg_truth, s_event_id, s_pred_start, s_pred_stop,
         s_truth_start, s_truth_stop, s_start_sigma, s_stop_sigma,
         l_hit_event, l_hit_E, l_hit_xyz_n, l_hit_p_hit,
         l_hit_assign, l_hit_beta, l_hit_frac, l_hit_t, l_K,
         lc_event, lc_coinc, lc_dt_ns, lc_time_ns) = \
            run_inference(model, device, eval_path, tag=tag,
                          max_events=args.max_events,
                          mc_passes=args.mc_passes,
                          use_truth_positron_mask=args.truth_positron_mask)

        # Save event-level
        save_event_parquet(truth, preds,
                           os.path.join(args.output_dir, f"{tag}_events.parquet"))

        # Save slice-level
        if s_role_truth:
            save_slice_parquet({
                'role_truth': s_role_truth,
                'role_probs': s_role_probs,
                'is_anchor': s_is_anchor,
                'pdg_truth': s_pdg_truth,
                'event_id': s_event_id,
                'pred_start': s_pred_start,
                'pred_stop': s_pred_stop,
                'truth_start': s_truth_start,
                'truth_stop': s_truth_stop,
                'start_sigma': s_start_sigma,
                'stop_sigma': s_stop_sigma,
            }, os.path.join(args.output_dir, f"{tag}_slices.parquet"))

        # Save LYSO hit-level
        save_lyso_parquet({
            'event': l_hit_event,
            'E': l_hit_E,
            'xyz_n': l_hit_xyz_n,
            'p_hit': l_hit_p_hit,
            'assign': l_hit_assign,
            'beta': l_hit_beta,
            'frac': l_hit_frac,
            't_ns': l_hit_t,
            'K': l_K or 4,
        }, os.path.join(args.output_dir, f"{tag}_lyso_hits.parquet"))

        # Save ATAR hit-level classification (opt-in)
        if args.dump_atar_hits:
            save_atar_hit_parquet(
                os.path.join(args.output_dir, f"{tag}_atar_hits.parquet"))

        # Save LYSO cluster-level timing
        save_lyso_cluster_parquet({
            'event': lc_event,
            'coinc': lc_coinc,
            'dt_ns': lc_dt_ns,
            'time_ns': lc_time_ns,
        }, os.path.join(args.output_dir, f"{tag}_lyso_clusters.parquet"))

    print(f"\nAll benchmarks complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
