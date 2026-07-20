import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PinballLoss(nn.Module):
    """
    Robust Permutation-Invariant Endpoint Loss using Asymmetric Attenuated Pinball Loss.
    Adapted from endpoint_finder.ipynb for the PURITY architecture.
    """
    def __init__(self, quantiles=[0.16, 0.50, 0.84], loss_scale_span=0.1, loss_scale_dir=0.1,
                 sigma_floor=1e-3):
        super().__init__()
        self.quantiles = quantiles
        self.loss_scale_span = loss_scale_span
        self.loss_scale_dir = loss_scale_dir
        # Numerical guardrail on the predicted quantile scale -- NOT an uncertainty
        # target. Without it the attenuated term (2*err/sigma + log sigma) rewards
        # sigma -> 0 and its gradient explodes like 1/sigma^2, NaN-ing the shared
        # trunk over many steps. Keep this 10-100x BELOW the healthy MeanWidth so it
        # only bites during collapse and never inflates normal uncertainty estimates.
        self.sigma_floor = sigma_floor
    def forward(self, preds, targets, weights=None, error_scale=1.0):
        if preds.dim() != 4 or targets.dim() != 3:
            raise ValueError("Expected preds shape [N, 2, 3, 3] and targets shape [N, 2, 3]")
            
        pred_median = preds[:, :, :, 1]
        
        loss_direct = F.smooth_l1_loss(pred_median, targets, reduction='none').sum(dim=(1, 2))
        target_swapped = targets[:, [1, 0], :]
        loss_swapped = F.smooth_l1_loss(pred_median, target_swapped, reduction='none').sum(dim=(1, 2))
        swap_mask = loss_swapped < loss_direct
        target_aligned = targets
        #swap_mask = torch.zeros(targets.shape[0], dtype=torch.bool, device=targets.device)
        
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
        scales = scales.clamp(min=self.sigma_floor)   # block variance collapse / 1/sigma^2 blow-up
        log_scales = torch.log(scales)
        
        pos_loss = 0.0
        for i, q in enumerate(self.quantiles):
            error = target_exp - preds[..., i:i+1]
            pinball = torch.max(q * error, (q - 1.0) * error)
            scale_q = scales[..., i:i+1]
            log_scale_q = log_scales[..., i:i+1]
            attenuated_loss = (2.0 * error_scale * pinball / scale_q) + log_scale_q
            w = batch_weights.unsqueeze(-1).unsqueeze(-1)
            pos_loss += (attenuated_loss * w).mean()
            
        pred_vec = pred_median[:, 1, :] - pred_median[:, 0, :]
        target_vec_aligned = target_aligned[:, 1, :] - target_aligned[:, 0, :]
        
        span_loss = F.mse_loss(pred_vec.norm(dim=1), target_vec_aligned.norm(dim=1))
        dir_loss_raw = 1.0 - F.cosine_similarity(pred_vec, target_vec_aligned, dim=1, eps=1e-6)
        
        if isinstance(self.loss_scale_dir, torch.Tensor):
            dir_loss = (dir_loss_raw * self.loss_scale_dir).mean()
        else:
            dir_loss = dir_loss_raw.mean() * self.loss_scale_dir
            
        total_loss = pos_loss + (self.loss_scale_span * span_loss) + dir_loss
        
        mean_width = sigma_total.mean()
        d_raw = (pred_median - target_aligned).norm(dim=2)
        d_start = torch.where(swap_mask, d_raw[:, 1], d_raw[:, 0])
        d_end = torch.where(swap_mask, d_raw[:, 0], d_raw[:, 1])
        d_align = torch.stack([d_start, d_end], dim=1)
        mean_error = d_align.mean()
        breakdown = {
            "PosLoss": pos_loss.item() if isinstance(pos_loss, torch.Tensor) else pos_loss,
            "SpanLoss": span_loss.item(),
            "DirLoss": dir_loss.item(),
            "MeanWidth": mean_width.item(),
            "MeanError": mean_error.item()
        }
        return total_loss, breakdown


class CondensationLoss(nn.Module):
    """
    Refined Object Condensation Loss with Highlander Penalty and Zero-Object Safety.
    Ensures background rejection is learned even in events without signal tracks.
    """
    def __init__(self, q_min=0.1, s_B=1.0, w_highlander=1.0):
        super().__init__()
        self.q_min = q_min
        self.s_B = s_B             # Background Beta Weight
        self.w_highlander = w_highlander 
        self.w_beta = 1.0
        self.w_potential = 1.0
        self.w_fraction = 1.0
        
    def forward(self, pred_beta, pred_coords, pred_fracs, 
                e_y_fracs, e_obj_targets, e_obj_mask, batch_idx, num_graphs):
                
        loss_beta = torch.tensor(0.0, device=pred_beta.device)
        loss_potential = torch.tensor(0.0, device=pred_beta.device)
        loss_fraction = torch.tensor(0.0, device=pred_beta.device)
        
        # Ensure 1D/2D consistency
        p_beta = pred_beta.view(-1)
        p_fracs = pred_fracs.view(-1)
        
        for b in range(num_graphs):
            event_mask = (batch_idx == b)
            if event_mask.sum() == 0: continue
            
            # --- Event-Level Tensors ---
            e_beta = p_beta[event_mask]
            e_coords = pred_coords[event_mask]
            e_frac = p_fracs[event_mask]
            fracs = e_y_fracs[event_mask] # [N_hits, Max_Objs]
            
            # Calculate object existence for THIS graph
            obj_mask = e_obj_mask[b*e_y_fracs.shape[1] : (b+1)*e_y_fracs.shape[1]]
            valid_obj_mask = (obj_mask > 0.5)
            num_objects = valid_obj_mask.sum().item()

            # Identify Background (Radioactivity/Noise)
            # Hits with no ground-truth object assignment (fracs sum = 0)
            is_background = (fracs.sum(dim=1) == 0.0)
            
            # -------------------------------------------------------------
            # 1. ALWAYS-ACTIVE BACKGROUND PENALTY (Learns the Null Hypothesis)
            # -------------------------------------------------------------
            if is_background.any():
                # YOUR VISION: Force radioactivity hits to form high-beta clusters (1.0)
                # Note: These will be pushed away by repulsion in Step 2.
                #l_bkg = F.binary_cross_entropy(e_beta[is_background], torch.ones_like(e_beta[is_background]))
                l_bkg = e_beta[is_background].mean()
                loss_beta += self.s_B * l_bkg
            
            # -------------------------------------------------------------
            # 2. OBJECT-DEPENDENT LOSSES (Guarded against Zero-Object NaNs)
            # -------------------------------------------------------------
            if num_objects > 0:
                # --- Selection ---
                beta_weighted = e_beta.unsqueeze(1) * fracs
                alpha_indices = torch.argmax(beta_weighted, dim=0) # [Max_Objs]
                
                # --- A. Highlander Suppression (Redundant Signal Hits) ---
                is_seed = torch.zeros_like(e_beta, dtype=torch.bool)
                is_seed[alpha_indices] = True
                is_redundant_signal = (~is_background) & (~is_seed)
                
                if is_redundant_signal.any():
                    # Highlander Penalty: Force 'Loser' signal hits to beta -> 0
                    # BCE preserves gradient at saturation (vs L1 mean which dies as
                    # β→1), so non-alpha hits stuck near 1 actually feel pull back.
                    redundant_betas = e_beta[is_redundant_signal].nan_to_num(0.5).clamp(1e-6, 1 - 1e-6)
                    loss_beta += self.w_highlander * F.binary_cross_entropy(
                        redundant_betas, torch.zeros_like(redundant_betas)
                    )

                # --- B. Signal Alpha Loss (Seeds must be 1.0) ---
                valid_alpha_indices = alpha_indices[valid_obj_mask]
                valid_alpha_betas = e_beta[valid_alpha_indices].nan_to_num(0.5).clamp(1e-6, 1-1e-6)
                loss_beta += F.binary_cross_entropy(valid_alpha_betas, torch.ones_like(valid_alpha_betas))


                # --- C. Fraction Logic ---
                owner_obj_idx = torch.argmax(fracs, dim=1) 
                target_fracs = torch.gather(fracs, 1, owner_obj_idx.unsqueeze(1)).squeeze(1)
                loss_fraction += F.mse_loss(e_frac, target_fracs)

                # --- D. Potential Logic (Variance + Seed Repulsion) ---
                alpha_coords = e_coords[alpha_indices]
                target_coords_per_hit = alpha_coords[owner_obj_idx]
                
                # d^2 with q_min charge floor to block the zero-beta cheating
                dists_sq = torch.sum((e_coords - target_coords_per_hit)**2, dim=1)
                is_signal = (~is_background)
                if is_signal.any():
                    charge_i = e_beta[is_signal]**2 + self.q_min # Charge Floor
                    loss_potential += (dists_sq[is_signal] * charge_i).mean()
                
                # Repel different Positron seeds
                valid_alpha_coords = alpha_coords[valid_obj_mask]
                if num_objects > 1:
                    delta = valid_alpha_coords.unsqueeze(1) - valid_alpha_coords.unsqueeze(0)
                    dist_matrix = torch.sqrt(torch.sum(delta**2, dim=2) + 1e-6)
                    margin_cluster = 2.0
                    repulsion = torch.relu(margin_cluster - dist_matrix)
                    mask_diag = torch.eye(num_objects, device=dist_matrix.device)
                    repulsion = repulsion * (1 - mask_diag)

                    # Calculate Charges: [Num_Objects, 1]
                    q_sig = (valid_alpha_betas**2 + self.q_min).unsqueeze(1) 
                    
                    # Create the Charge Matrix: q_i * q_j -> [Num_Objects, Num_Objects]
                    charge_matrix = torch.matmul(q_sig, q_sig.t())
                    
                    # Apply weights to the repulsion
                    repulsion = repulsion * charge_matrix
                    w_cluster_repulsion = 1.5
                    loss_potential += w_cluster_repulsion * repulsion.mean()

                # --- E. CUSTOM BACKGROUND REPULSION ---
                if is_background.any():
                    bkg_coords = e_coords[is_background]
                    delta_bkg = valid_alpha_coords.unsqueeze(1) - bkg_coords.unsqueeze(0)
                    dist_bkg = torch.sqrt(torch.sum(delta_bkg**2, dim=2) + 1e-6)
                    
                    margin_bkg = 3.0
                    repulsion_bkg = torch.relu(margin_bkg - dist_bkg) # Hinge Loss
                    #loss_potential += repulsion_bkg.mean()

                    q_sig = (valid_alpha_betas**2 + self.q_min).unsqueeze(1) # [Num_Valid_Seeds, 1]
                    w_radio_repulsion = 2.0 # <--- The "Blast" Coefficient
                    q_bkg = torch.ones_like(e_beta[is_background]).unsqueeze(0)
                    # β² weighting restored: flat q_bkg=1 was driving the model to
                    # spawn extra seeds to escape uniform repulsion from every
                    # background hit. With β²+q_min, well-identified noise (low β)
                    # stops pushing seeds apart, so OC converges to fewer clusters.
                    #q_bkg = (e_beta[is_background]**2 + self.q_min).unsqueeze(0)
                    
                    # 4. The Charge Matrix
                    charge_matrix = q_sig * q_bkg # [Num_Valid_Seeds, Num_Bkg_Hits]
                    
                    # 5. The True Physics Repulsion
                    loss_potential += w_radio_repulsion*(charge_matrix * repulsion_bkg).mean()
            
            else:
                # NULL HYPOTHESIS: Empty event. Suppress any accidental signal hits to beta=0
                is_signal_mod = (~is_background)
                if is_signal_mod.any():
                    loss_beta += e_beta[is_signal_mod].mean()

        # Average across graphs and return
        total_loss = (self.w_beta * loss_beta + 
                      self.w_potential * loss_potential + 
                      self.w_fraction * loss_fraction) / max(1, num_graphs)
        
        breakdown = {
            'beta': loss_beta.item() / max(1, num_graphs),
            'potential': loss_potential.item() / max(1, num_graphs),
            'fraction': loss_fraction.item() / max(1, num_graphs)
        }
            
        return total_loss, breakdown



def event_builder_loss(outputs, batch, w_floor=0.05):
    """
    Event-level trigger classification loss.
    LYSO: assignment-weighted mixture per hit, then BCE, with per-graph
          down-weighting for graphs with upstream issues (sentinel pt,
          has_trigger_positron==0 but ATAR fired, or no confident cluster).
    ATAR: hard slice assignment (unchanged).

    w_floor: residual weight on bad-upstream graphs (0.0 = hard mask,
             0.05 keeps a small gradient so the skip connection still
             learns to be cautious when pt is fake).
    """
    event_logits = outputs.get('unified_event_logits')
    if event_logits is None:
        return torch.tensor(0.0, device=batch.x.device, requires_grad=True)

    energy = batch.x[:, 3]
    is_atar = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
    is_lyso = (batch.x[:, 7] > 0.5)
    trigger_targets = batch.is_trigger_target
    num_atar_tokens = outputs.get('unified_num_atar_tokens', 0)
    p_tokens = torch.sigmoid(event_logits.squeeze(-1))

    atar_loss_sum = torch.tensor(0.0, device=batch.x.device)
    lyso_loss_sum = torch.tensor(0.0, device=batch.x.device)
    num_atar_slices = 0
    lyso_weight_sum = torch.tensor(0.0, device=batch.x.device)

    # =============================================
    # ATAR slice loss (UNCHANGED)
    # =============================================
    if is_atar.any() and num_atar_tokens > 0:
        p_atar = p_tokens[:num_atar_tokens]
        atar_energy = energy[is_atar]
        atar_targets = trigger_targets[is_atar]

        valid_slice_mask = outputs['valid_slice_mask']
        num_slices_max = outputs['num_slices_max']
        global_slice_ids = batch.batch[is_atar] * num_slices_max + batch.x[is_atar, 8].long()
        hit_is_valid = valid_slice_mask[global_slice_ids]
        mapped_slice_indices = torch.cumsum(valid_slice_mask.long(), dim=0) - 1
        idx_in_valid = mapped_slice_indices[global_slice_ids[hit_is_valid]]

        p_atar_broadcast = p_atar[idx_in_valid]
        atar_targets_valid = atar_targets[hit_is_valid]
        atar_energy_valid = atar_energy[hit_is_valid]

        bce_atar = F.binary_cross_entropy(
            p_atar_broadcast.nan_to_num(0.5).clamp(1e-6, 1-1e-6), atar_targets_valid, reduction='none')
        weighted_loss_atar = bce_atar * atar_energy_valid

        num_valid_slices = num_atar_tokens
        slice_loss_sum = torch.zeros(num_valid_slices, device=bce_atar.device)
        slice_energy_sum = torch.zeros(num_valid_slices, device=bce_atar.device)
        slice_loss_sum.index_add_(0, idx_in_valid, weighted_loss_atar)
        slice_energy_sum.index_add_(0, idx_in_valid, atar_energy_valid)
        slice_loss_norm = slice_loss_sum / slice_energy_sum.clamp(min=1e-6)

        atar_loss_sum = slice_loss_norm.sum()
        num_atar_slices = num_valid_slices

    # =============================================
    # LYSO: Assignment-weighted mixture loss, per-graph weighted
    # =============================================
    lyso_assignments = outputs.get('lyso_soft_assignments')  # [N_lyso_hits, K]
    if is_lyso.any() and lyso_assignments is not None:
        p_lyso = p_tokens[num_atar_tokens:]
        K = lyso_assignments.size(1)
        p_lyso_matrix = p_lyso.view(-1, K)

        lyso_energy = energy[is_lyso]
        lyso_targets = trigger_targets[is_lyso]
        lyso_batch = batch.batch[is_lyso]
        num_graphs_in_batch = outputs.get(
            'num_graphs_in_batch', batch.batch.max().item() + 1)

        # --- Map graph IDs to contiguous LYSO-graph indices ---
        graph_has_lyso = torch.zeros(num_graphs_in_batch, dtype=torch.bool,
                                      device=lyso_batch.device)
        graph_has_lyso[lyso_batch] = True
        mapped_graph_indices = torch.cumsum(graph_has_lyso.long(), dim=0) - 1
        lyso_mapped_batch = mapped_graph_indices[lyso_batch]
        lyso_graph_ids = torch.nonzero(graph_has_lyso, as_tuple=False).squeeze(1)  # [n_lyso_graphs]

        # --- Broadcast K cluster probs to each hit ---
        p_lyso_broadcast = p_lyso_matrix[lyso_mapped_batch]

        # --- Effective weights: assignment × seed_beta ---
        effective_weights = lyso_assignments
        lyso_seed_beta = outputs.get('lyso_seed_beta')
        if lyso_seed_beta is not None:
            beta_matrix = lyso_seed_beta.view(-1, K)  # [n_lyso_graphs, K]
            beta_broadcast = beta_matrix[lyso_mapped_batch]
            effective_weights = lyso_assignments * beta_broadcast
        else:
            beta_matrix = None

        # --- Hit-level mixture probability ---
        w_sum = effective_weights.sum(dim=1).clamp(min=1e-6)
        p_hit = (effective_weights * p_lyso_broadcast).sum(dim=1) / w_sum

        # --- Per-hit BCE with class weighting ---
        bce_per_hit = F.binary_cross_entropy(
            p_hit.nan_to_num(0.5).clamp(1e-6, 1-1e-6), lyso_targets, reduction='none')
        pos_weight = 3.0
        class_weight = torch.where(lyso_targets > 0.5, pos_weight, 1.0)
        bce_per_hit = bce_per_hit * class_weight

        # --- Energy-weighted per-graph loss ---
        weighted_bce = bce_per_hit * lyso_energy
        n_lyso_graphs = p_lyso_matrix.size(0)
        graph_loss = torch.zeros(n_lyso_graphs, device=bce_per_hit.device)
        graph_energy = torch.zeros(n_lyso_graphs, device=bce_per_hit.device)
        graph_loss.index_add_(0, lyso_mapped_batch, weighted_bce)
        graph_energy.index_add_(0, lyso_mapped_batch, lyso_energy)
        graph_loss_norm = graph_loss / graph_energy.clamp(min=1e-6)

        # ====================================================
        # Per-graph upstream-confidence weight
        # ====================================================
        # pt_ok:    positron time is not the sentinel (-1.0 normalized ≈ -500 ns)
        # trig_ok:  event truly contains a trigger positron (truth label)
        # beta_ok:  at least one cluster has beta > 0.2 (confident seed exists)
        device = graph_loss_norm.device

        pt_per_graph = outputs.get('positron_time_per_graph')  # [num_graphs_in_batch] or [n_lyso_graphs]
        if pt_per_graph is not None:
            if pt_per_graph.shape[0] == num_graphs_in_batch:
                pt_lyso = pt_per_graph[lyso_graph_ids]
            else:
                pt_lyso = pt_per_graph  # already aligned to lyso graphs
            pt_ok = (pt_lyso > -0.9).float()
        else:
            pt_ok = torch.ones(n_lyso_graphs, device=device)

        has_trig = getattr(batch, 'has_trigger_positron', None)
        if has_trig is not None:
            trig_ok = has_trig.float().to(device)[lyso_graph_ids]
        else:
            trig_ok = torch.ones(n_lyso_graphs, device=device)

        if beta_matrix is not None:
            max_beta = beta_matrix.max(dim=1).values  # [n_lyso_graphs]
            beta_ok = (max_beta > 0.2).float()
        else:
            beta_ok = torch.ones(n_lyso_graphs, device=device)

        upstream_ok = pt_ok * trig_ok * beta_ok  # {0, 1}
        graph_weight = w_floor + (1.0 - w_floor) * upstream_ok  # [w_floor, 1]

        lyso_loss_sum = (graph_loss_norm * graph_weight).sum()
        lyso_weight_sum = graph_weight.sum()

    # =============================================
    # Final: weighted mean over LYSO graphs
    # =============================================
    if lyso_weight_sum > 0:
        event_loss = lyso_loss_sum / lyso_weight_sum.clamp(min=1e-8)
    else:
        event_loss = torch.tensor(0.0, device=event_logits.device, requires_grad=True)

    return event_loss




class PURITYLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.pinball = PinballLoss(quantiles=[0.16, 0.50, 0.84])
        self.condensation = CondensationLoss()

        decorr_window = config.get('decorr_window', 10)
        self._decorr_alpha = 1.0 / decorr_window
        self._m0_ema = 0.0
        self._m1_ema = 0.0
        self._m2_ema = 0.0

    def reset_decorr_ema(self):
        self._m0_ema = 0.0
        self._m1_ema = 0.0
        self._m2_ema = 0.0
        
    def forward(self, outputs, targets, batch=None):
        loss_dict = {}
        total_loss = 0.0

        # 0.  Multi-Event Slice Classifier
        w_multi = self.config.get('w_atar_slice_multi', 0.0)
        if w_multi > 0.0 and 'atar_slice_multi' in outputs and 'atar_slice_multi_target' in targets:
            # Compute binary cross entropy for the slice-level pileup flag
            l_multi = self.bce_logits(outputs['atar_slice_multi'], targets['atar_slice_multi_target'])
            loss_dict['loss_slice_multi'] = l_multi
            total_loss += w_multi * l_multi
        
        # 1. Node PDG Splitter
        w_node = self.config.get('w_node_pdg', 0.0)
        if w_node > 0.0 and 'atar_node_pdg' in outputs and 'tar_node_pdg' in targets:
            l_node = self.bce_logits(outputs['atar_node_pdg'], targets['tar_node_pdg'])
            loss_dict['loss_node_pdg'] = l_node
            total_loss += w_node * l_node

        # Triggering node classifier
        w_trigger = self.config.get('w_node_trigger', 0.0)
        if w_trigger > 0.0 and 'atar_hit_trigger' in outputs and 'is_trigger' in targets:
            # 1. Create a boolean mask identifying exactly where the ATAR hits live
            is_atar = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
            
            # 2. Extract those corresponding labels for the BCE calculation!
            l_atar_trigger = self.bce_logits(outputs['atar_hit_trigger'], targets['is_trigger'][is_atar])
            
            total_loss += w_trigger * l_atar_trigger
            loss_dict['loss_atar_hit_trigger'] = l_atar_trigger
            
        # 2. Slice Group Classifiers
        # FIX: Removed the [valid_batch_idx] scrambling. PyG already aligns this!
        w_slice = self.config.get('w_slice_pdg', 0.0)
        if w_slice > 0.0 and 'atar_slice_pdg' in outputs and 'tar_slice_pdg' in targets:
            l_slice = self.bce_logits(outputs['atar_slice_pdg'], targets['tar_slice_pdg'])
            loss_dict['loss_slice_pdg'] = l_slice
            total_loss += w_slice * l_slice

        # 2b. Per-slice trigger (positron-slice) classifier. The ATAR-only tail model
        # emits per-valid-slice `atar_trigger_logits`; supervising it is what makes
        # "focus on the positron slice" real for the muDIF head. Guarded + default-off
        # (+ shape check, same per-slice alignment as slice_pdg) so the full PURITY
        # trainer, which has no 'w_slice_trigger' weight, is unaffected.
        w_slice_trig = self.config.get('w_slice_trigger', 0.0)
        if (w_slice_trig > 0.0 and 'atar_trigger_logits' in outputs
                and 'tar_slice_trigger' in targets
                and outputs['atar_trigger_logits'].shape == targets['tar_slice_trigger'].shape):
            l_slice_trig = self.bce_logits(outputs['atar_trigger_logits'],
                                           targets['tar_slice_trigger'])
            loss_dict['loss_slice_trigger'] = l_slice_trig
            total_loss += w_slice_trig * l_slice_trig
            
        # 3. ATAR Trigger Slice Loss (Phase 9) — role-based attachment.
        # Replaces per-slice BCE: for each non-anchor slice, the head outputs
        # a distribution over {none, μ-in-chain, e-in-chain}. The anchor pion
        # is given (not predicted) and skipped from the CE term. A listwise
        # composition penalty enforces at most one μ and at most one e per
        # event, which is what couples the per-slice predictions at loss time.
        w_trigger_slice = self.config.get('w_atar_trigger_slice', 0.0)
        if (w_trigger_slice > 0.0
                and 'atar_role_logits' in outputs
                and 'tar_slice_role' in targets):
            role_logits = outputs['atar_role_logits']                # [N_valid, 3]
            role_targets = targets['tar_slice_role'].long()          # [N_valid]
            is_anchor = outputs.get('atar_anchor_slice_mask',
                                    torch.zeros(role_logits.size(0),
                                                dtype=torch.bool,
                                                device=role_logits.device))
            slice_event_idx = outputs.get('atar_slice_event_idx')     # [N_valid]

            non_anchor = ~is_anchor
            if non_anchor.any() and role_targets.numel() == role_logits.size(0):
                l_role_ce = F.cross_entropy(
                    role_logits[non_anchor], role_targets[non_anchor]
                )
            else:
                l_role_ce = role_logits.new_zeros(())

            # Composition penalty: at most ONE μ and ONE e per event.
            if slice_event_idx is not None and slice_event_idx.numel() > 0:
                role_probs = F.softmax(role_logits, dim=-1)          # [N_valid, 3]
                num_events = int(slice_event_idx.max().item()) + 1
                mu_sum = role_logits.new_zeros(num_events)
                e_sum  = role_logits.new_zeros(num_events)
                mu_sum.index_add_(0, slice_event_idx, role_probs[:, 1])
                e_sum .index_add_(0, slice_event_idx, role_probs[:, 2])
                l_composition = (
                    F.relu(mu_sum - 1.0).pow(2).mean()
                    + F.relu(e_sum - 1.0).pow(2).mean()
                )
            else:
                l_composition = role_logits.new_zeros(())

            lambda_comp = self.config.get('lambda_composition', 0.2)
            l_trigger_slice = l_role_ce + lambda_comp * l_composition

            loss_dict['loss_role_ce'] = l_role_ce
            loss_dict['loss_composition'] = l_composition
            loss_dict['loss_trigger_slice'] = l_trigger_slice
            total_loss += w_trigger_slice * l_trigger_slice

            # 3.2 Chain-positron EXCLUSIVITY — exactly one e+ in the triggering chain.
            # The relu(sum-1)^2 composition above only fires once the per-event e+ mass
            # EXCEEDS 1, so the observed failure (true slice P~0.55 + accidental P~0.5,
            # BOTH above the 0.5 readout threshold) is barely penalized, and the time
            # readout then averages two positrons (pileup e+ threading the stop region
            # fakes a prompt pi->e). This term is "total minus best": any e+ probability
            # on a SECOND slice is penalized LINEARLY, and it is zero when the mass sits
            # on one slice — the structural prior that the chain has exactly one e+.
            w_excl = self.config.get('w_chain_exclusive', 0.0)
            if (w_excl > 0.0 and slice_event_idx is not None
                    and slice_event_idx.numel() > 0):
                s_e = role_probs[:, 2] * (~is_anchor).float()        # e+ mass, non-anchor
                e_sum_na = role_logits.new_zeros(num_events)
                e_sum_na.index_add_(0, slice_event_idx, s_e)         # per-event total e+
                e_max = role_logits.new_zeros(num_events)
                e_max.index_reduce_(0, slice_event_idx, s_e, 'amax',
                                    include_self=True)               # per-event best e+
                extra_e = (e_sum_na - e_max).clamp(min=0.0)          # mass on 2nd, 3rd...
                l_exclusive = extra_e.mean()
                loss_dict['loss_chain_exclusive'] = l_exclusive
                total_loss += w_excl * l_exclusive

        # 3.5 Trigger-positron time-spread loss
        # Physics: the trigger positron is a single track confined to one slice
        # with intrinsic spread ~1 ns. If the model tags hits across multiple
        # slices as the trigger positron (the pile-up failure mode), the
        # weighted std over those tagged hits blows up.
        #
        # Per event:
        #   w[i]        = P(positron-MIP)[i] × P(slice trigger)[i]   (per-hit weight)
        #   std         = sqrt( Σ w (t - μ)² / Σ w )                  (weighted std in ns)
        #   avg_trig    = mean(P(slice trigger))                      (per-event scalar)
        #   penalty     = log(1 + relu(std - threshold) × avg_trig)
        #
        # avg_trig auto-gates events where the model is uncertain that any
        # trigger chain is present. relu inside log keeps the argument in
        # [0, ∞), so log1p never goes negative or NaN.
        w_spread = self.config.get('w_time_spread', 0.0)
        if (w_spread > 0.0 and batch is not None
                and 'atar_node_pdg' in outputs
                and 'atar_hit_trigger_prob' in outputs):
            x = batch.x
            is_atar = (x[:, 5] > 0.5) | (x[:, 6] > 0.5)
            if is_atar.any():
                t_atar = (x[is_atar, 4] * 500.0)                  # ns (NORM_T_ATAR=500)
                b_atar = batch.batch[is_atar]                      # [N_atar]
                B_total = int(batch.batch.max().item()) + 1

                node_pdg = outputs['atar_node_pdg']                # [N_atar, 3] logits
                # Only the positron-MIP class (column 2). Sigmoid since the
                # head is a 3-bit BCE-style classifier, not a softmax over classes.
                pos_p = torch.sigmoid(node_pdg[:, 2])              # [N_atar]
                hit_trig = outputs['atar_hit_trigger_prob']        # [N_atar]

                # Hard floors on both per-hit trigger probability and per-hit
                # MIP probability. A hit only contributes to the spread if the
                # model is at least mildly confident that BOTH (a) its slice is
                # part of the trigger chain and (b) the hit itself is a MIP/
                # positron. This kills variance contribution from confidently-
                # non-trigger hits and from hits whose residual sigmoid floor
                # P(MIP) on a true-π/μ track was being amplified by (t-mean)².
                trig_floor = float(self.config.get(
                    'time_spread_trig_floor', 0.25))
                mip_floor = float(self.config.get(
                    'time_spread_mip_floor', 0.25))
                trig_gate = (hit_trig > trig_floor).float()        # [N_atar] 0/1
                mip_gate  = (pos_p   > mip_floor).float()          # [N_atar] 0/1
                hit_trig_eff = hit_trig * trig_gate                # [N_atar]
                pos_p_eff    = pos_p   * mip_gate                  # [N_atar]

                # Per-event average slice-trigger probability (gating factor).
                # Same trig gate applied so events with no slice above the floor
                # contribute nothing.
                sum_trig   = torch.zeros(B_total, device=x.device)
                count_atar = torch.zeros(B_total, device=x.device)
                sum_trig  .index_add_(0, b_atar, hit_trig_eff)
                count_atar.index_add_(0, b_atar, torch.ones_like(hit_trig))
                avg_trig_per_event = sum_trig / count_atar.clamp(min=1.0)  # [B_total]

                threshold_ns = float(self.config.get(
                    'time_spread_thresh_ns', 1.0))

                # Per-hit weight = (gated MIP prob) × (gated trigger-slice prob).
                w = pos_p_eff * hit_trig_eff                       # [N_atar]

                # Σw, Σ(w·t) per event
                sum_w  = torch.zeros(B_total, device=x.device)
                sum_wt = torch.zeros(B_total, device=x.device)
                sum_w .index_add_(0, b_atar, w)
                sum_wt.index_add_(0, b_atar, w * t_atar)

                sum_w_safe = sum_w.clamp(min=1e-3)
                mean_t_per_event = sum_wt / sum_w_safe             # [B_total]
                mean_t_per_hit = mean_t_per_event[b_atar]          # [N_atar]

                diff_sq = (t_atar - mean_t_per_hit) ** 2
                sum_w_diff = torch.zeros(B_total, device=x.device)
                sum_w_diff.index_add_(0, b_atar, w * diff_sq)
                var_t  = sum_w_diff / sum_w_safe
                std_t  = torch.sqrt(var_t.clamp(min=0.0) + 1e-6)   # [B_total] ns

                excess = F.relu(std_t - threshold_ns)
                penalty = torch.log1p(excess * avg_trig_per_event)
                l_time_spread = penalty.mean()
                loss_dict['loss_time_spread'] = l_time_spread
                total_loss += w_spread * l_time_spread

        # 3A. Kinematic Pion Stop (Phase 10 — per-graph regression)
        w_pion = self.config.get('w_pion_kinematics', 0.0)
        if w_pion > 0.0 and 'atar_pion_stop' in outputs and 'tar_pion_stop_xyz' in targets:
            l_pion = F.smooth_l1_loss(outputs['atar_pion_stop'], targets['tar_pion_stop_xyz'])
            loss_dict['loss_pion_kinematics'] = l_pion
            total_loss += w_pion * l_pion
            
        # 3B. Endpoints
        w_end = self.config.get('w_endpoints', 0.0)
        if w_end > 0.0 and 'atar_endpoints' in outputs and 'tar_slice_start_x' in targets:
            # Model predictions [N, 2, 3, 3]
            preds_xyz = outputs['atar_endpoints'] * 10.0
            
            # Stack Targets: Start Point [N, 3] and Stop Point [N, 3]
            targets_start = torch.stack([
                targets['tar_slice_start_x'], 
                targets['tar_slice_start_y'], 
                targets['tar_slice_start_z']
            ], dim=1)
            targets_stop = torch.stack([
                targets['tar_slice_stop_x'], 
                targets['tar_slice_stop_y'], 
                targets['tar_slice_stop_z']
            ], dim=1)
            
            # Combine into [N, 2, 3]
            targets_xyz = torch.stack([targets_start, targets_stop], dim=1) * 10.0
            # Calculate Asymmetric Pinball Loss
            l_end, end_breakdown = self.pinball(preds_xyz, targets_xyz)
            for k, v in end_breakdown.items():
                loss_dict[f'end_{k}'] = v
            total_loss += w_end * l_end
            
        # 4. Positron Direction (Phase 11 — per-graph unit vector)
        w_angle = self.config.get('w_positron_angle', 0.0)
        if w_angle > 0.0 and 'atar_positron_dir' in outputs and 'tar_angle_vec_per_graph' in targets:
            has_pos = batch.has_trigger_positron.bool()  # [B]
            if has_pos.any():
                pred_dir = outputs['atar_positron_dir'][has_pos]
                tar_dir = targets['tar_angle_vec_per_graph'][has_pos]
                l_angle = (1.0 - F.cosine_similarity(pred_dir, tar_dir, dim=1)).mean()
                loss_dict['loss_positron_angle'] = l_angle
                total_loss += w_angle * l_angle

                # 4a. Moment decorrelation: penalize θ-dependent bias
                w_decorr = self.config.get('w_angle_decorr', 0.0)
                if w_decorr > 0.0 and has_pos.sum() >= 10:
                    eps = 1e-4
                    theta_pred = torch.acos(pred_dir[:, 2].clamp(-1 + eps, 1 - eps))
                    theta_true = torch.acos(tar_dir[:, 2].clamp(-1 + eps, 1 - eps))
                    residual = theta_pred - theta_true

                    m0_batch = residual.mean()
                    m1_batch = (residual * tar_dir[:, 2]).mean()
                    cos2 = 2.0 * tar_dir[:, 2] ** 2 - 1.0
                    m2_batch = (residual * cos2).mean()

                    a = self._decorr_alpha
                    m0_smooth = a * m0_batch + (1 - a) * self._m0_ema
                    m1_smooth = a * m1_batch + (1 - a) * self._m1_ema
                    m2_smooth = a * m2_batch + (1 - a) * self._m2_ema

                    self._m0_ema = m0_smooth.detach().item()
                    self._m1_ema = m1_smooth.detach().item()
                    self._m2_ema = m2_smooth.detach().item()

                    l_decorr = (m0_smooth ** 2 + m1_smooth ** 2 + m2_smooth ** 2) / a
                    loss_dict['loss_angle_decorr'] = l_decorr.detach()
                    loss_dict['loss_angle_decorr_m0'] = m0_smooth.detach()
                    loss_dict['loss_angle_decorr_m1'] = m1_smooth.detach()
                    loss_dict['loss_angle_decorr_m2'] = m2_smooth.detach()
                    total_loss += w_decorr * l_decorr

        # 4b. Has-trigger-positron (per-graph binary)
        w_htp = self.config.get('w_has_trigger_positron', 0.0)
        if w_htp > 0.0 and 'has_trigger_positron_logits' in outputs:
            tgt = batch.has_trigger_positron.float().view(-1)
            l_htp = F.binary_cross_entropy_with_logits(
                outputs['has_trigger_positron_logits'], tgt
            )
            loss_dict['loss_has_trigger_positron'] = l_htp
            total_loss += w_htp * l_htp


        # --- Edge Classification Loss ---
        w_edge = self.config.get('w_atar_edge', 0.5)
        if w_edge > 0.0 and 'atar_edge_logits' in outputs and 'atar_local_edge_index' in outputs:
            local_ei = outputs['atar_local_edge_index']  # Already ATAR-local from radius_graph
            
            if local_ei.size(1) > 0 and hasattr(batch, 'atar_true_event_id'):
                ev_id = batch.atar_true_event_id.to(local_ei.device)  # [total_N_atar_in_batch]
                
                # Direct index — no remapping needed since radius_graph is ATAR-local
                y_edge = (ev_id[local_ei[0]] == ev_id[local_ei[1]]).float()
                
                n_pos = y_edge.sum().clamp(min=1)
                n_neg = (1 - y_edge).sum().clamp(min=1)
                pos_weight = (n_neg / n_pos).clamp(max=3.0)
                
                loss_edge = F.binary_cross_entropy_with_logits(
                    outputs['atar_edge_logits'], y_edge, pos_weight=pos_weight
                )
                total_loss += w_edge * loss_edge
                loss_dict['loss_atar_edge'] = loss_edge
                    
        # ... LYSO and Energy blocks remain the same ...

        w_lyso = self.config.get('w_lyso_condensation', 0.0)
        
        # Check if the LYSO prediction tensors and dataset targets actually exist in this batch
        if w_lyso > 0.0 and 'lyso_beta' in outputs and 'tar_lyso_fracs' in targets:
            
            # models.py embeds the batch graph count natively into the output dictionary for loss loops
            num_graphs = outputs.get('num_graphs_in_batch', 1)
            
            # Execute the Condensation Loss
            l_cond, cond_breakdown = self.condensation(
                pred_beta=outputs['lyso_beta'],
                pred_coords=outputs['lyso_cluster_coords'],
                pred_fracs=outputs['lyso_fractions'],
                e_y_fracs=targets['tar_lyso_fracs'],
                e_obj_targets=targets['tar_lyso_payload'],
                e_obj_mask=targets['tar_lyso_mask'],
                batch_idx=targets['lyso_batch_idx'],
                num_graphs=num_graphs
            )
            
            # Save the primary loss
            loss_dict['loss_lyso_condensation'] = l_cond
            
            # Unpack the secondary breakdown metrics (beta, potential, fraction) into the logging dict
            for k, v in cond_breakdown.items():
                loss_dict[f'lyso_{k}'] = v
                
            # Add to the global backpropagation total
            total_loss += w_lyso * l_cond

        # --- Dead-material energy regression (post-hoc, detached) ---
        w_dead = self.config.get('w_dead_energy', 0.0)
        if (w_dead > 0.0
                and 'dead_energy_log_pred' in outputs
                and 'tar_dead_E' in targets):
            log_pred = outputs['dead_energy_log_pred']
            log_true = torch.log1p(targets['tar_dead_E'].clamp(min=0.0))
            log6 = math.log(6.0)
            diff = (log_pred - log_true).clamp(min=-log6, max=log6)
            sq = diff.pow(2)

            # Mask: zero loss when predicted polar angle > 130° or no
            # triggering positron (htp_rule == 0).
            mask = torch.ones_like(sq)
            es = outputs.get('event_summary', {})
            polar = es.get('positron_polar_angle')
            if polar is not None:
                mask = mask * (polar < math.radians(130.0)).float()
            htp_rule = outputs.get('has_trigger_positron_rule')
            if htp_rule is not None:
                mask = mask * (htp_rule > 0.5).float()

            denom = mask.sum().clamp(min=1.0)
            l_dead = (sq * mask).sum() / denom

            loss_dict['loss_dead_energy'] = l_dead
            total_loss += w_dead * l_dead

        # --- Event Synthesis Loss ---
        w_event = self.config.get('w_event_builder', 0.0)
        if w_event > 0.0 and 'unified_event_logits' in outputs and batch is not None:
            # Call the new energy-weighted broadcast BCE
            l_event = event_builder_loss(outputs, batch)

            # Always log the value (works in train and inference_mode);
            # only add to total_loss when grad is available (training).
            loss_dict['L_event_builder'] = (
                l_event.item() if hasattr(l_event, 'item') else float(l_event)
            )
            if l_event.requires_grad:
                total_loss += w_event * l_event

        loss_dict['loss_total'] = total_loss
        return total_loss, loss_dict

def format_targets_from_batch(batch):
    """
    Extracts and standardizes targets from the PyG batch object.
    Leaves data in its original normalization scale (e.g., [-1, 1]).
    """
    targets = {}

    # Multi-event slicing
    if hasattr(batch, 'atar_slice_multi_target') and batch.atar_slice_multi_target is not None:
        targets['atar_slice_multi_target'] = batch.atar_slice_multi_target.float()
    
    # Per-slice trigger flag (Phase 9 target)
    if hasattr(batch, 'atar_slice_trigger_target') and batch.atar_slice_trigger_target is not None:
        targets['tar_slice_trigger'] = batch.atar_slice_trigger_target.float()

    # Per-slice role in the triggering chain (0 = none, 1 = μ, 2 = e⁺).
    if hasattr(batch, 'atar_slice_role_target') and batch.atar_slice_role_target is not None:
        targets['tar_slice_role'] = batch.atar_slice_role_target.long()

    # Per-event truth dead-material energy (MeV).
    if hasattr(batch, 'dead_E_target') and batch.dead_E_target is not None:
        targets['tar_dead_E'] = batch.dead_E_target.float().view(-1)
    
    # Node Level Targets
    if hasattr(batch, 'atar_node_pdg_target') and batch.atar_node_pdg_target is not None:
        targets['tar_node_pdg'] = batch.atar_node_pdg_target.float()
        
    # Graph/Slice Level Targets
    if hasattr(batch, 'atar_slice_pdg_target') and batch.atar_slice_pdg_target is not None:
        targets['tar_slice_pdg'] = batch.atar_slice_pdg_target.float()
        
    if hasattr(batch, 'atar_pion_stop_target') and batch.atar_pion_stop_target is not None:
        if batch.atar_pion_stop_target.dim() == 2:
            targets['tar_pion_stop_x'] = batch.atar_pion_stop_target[:, 0]
            targets['tar_pion_stop_y'] = batch.atar_pion_stop_target[:, 1]
            targets['tar_pion_stop_z'] = batch.atar_pion_stop_target[:, 2]
            # Per-graph pion stop: take first slice's target per graph (all copies identical)
            from torch_geometric.utils import scatter
            slice_batch = batch.atar_slice_pdg_target.new_zeros(batch.atar_pion_stop_target.size(0), dtype=torch.long)
            if hasattr(batch, '_slice_dict') and 'atar_pion_stop_target' in batch._slice_dict:
                slices = batch._slice_dict['atar_pion_stop_target']
                for g in range(len(slices) - 1):
                    slice_batch[slices[g]:slices[g+1]] = g
            num_graphs = batch.batch.max().item() + 1
            targets['tar_pion_stop_xyz'] = scatter(batch.atar_pion_stop_target, slice_batch, dim=0, dim_size=num_graphs, reduce='mean')

            
    if hasattr(batch, 'atar_angle_target') and batch.atar_angle_target is not None:
        targets['tar_angle_vec'] = batch.atar_angle_target.float()
        # Per-graph angle: take first slice's target per graph (all copies identical)
        if batch.atar_angle_target.dim() == 2 and batch.atar_angle_target.size(0) > 0:
            angle_batch = batch.atar_angle_target.new_zeros(batch.atar_angle_target.size(0), dtype=torch.long)
            if hasattr(batch, '_slice_dict') and 'atar_angle_target' in batch._slice_dict:
                slices = batch._slice_dict['atar_angle_target']
                for g in range(len(slices) - 1):
                    angle_batch[slices[g]:slices[g+1]] = g
            num_graphs = batch.batch.max().item() + 1
            targets['tar_angle_vec_per_graph'] = scatter(batch.atar_angle_target, angle_batch, dim=0, dim_size=num_graphs, reduce='mean')
        
    #if hasattr(batch, 'atar_endpoint_target') and batch.atar_endpoint_target is not None:
    #    if batch.atar_endpoint_target.dim() == 2:
    #        targets['tar_endpoint_x'] = batch.atar_endpoint_target[:, 0]
    #        targets['tar_endpoint_y'] = batch.atar_endpoint_target[:, 1]
    #        targets['tar_endpoint_z'] = batch.atar_endpoint_target[:, 2]

    if hasattr(batch, 'atar_slice_start_target') and batch.atar_slice_start_target is not None:
        if batch.atar_slice_start_target.dim() == 2:
            targets['tar_slice_start_x'] = batch.atar_slice_start_target[:, 0]
            targets['tar_slice_start_y'] = batch.atar_slice_start_target[:, 1]
            targets['tar_slice_start_z'] = batch.atar_slice_start_target[:, 2]

    if hasattr(batch, 'atar_slice_stop_target') and batch.atar_slice_stop_target is not None:
        if batch.atar_slice_stop_target.dim() == 2:
            targets['tar_slice_stop_x'] = batch.atar_slice_stop_target[:, 0]
            targets['tar_slice_stop_y'] = batch.atar_slice_stop_target[:, 1]
            targets['tar_slice_stop_z'] = batch.atar_slice_stop_target[:, 2]
            
    if hasattr(batch, 'positron_initial_energy_target') and batch.positron_initial_energy_target is not None:
        targets['tar_initial_energy'] = batch.positron_initial_energy_target.float()
        
    is_lyso = (batch.x[:, 7] > 0.5)
    # LYSO Targets
    if hasattr(batch, 'lyso_fracs_target') and batch.lyso_fracs_target is not None:
        targets['tar_lyso_fracs'] = batch.lyso_fracs_target
        targets['tar_lyso_payload'] = batch.lyso_payload_target
        targets['tar_lyso_mask'] = batch.lyso_mask_target
        # Isolate the batch indices specifically for LYSO hits
        targets['lyso_batch_idx'] = batch.batch[is_lyso]

    if hasattr(batch, 'is_trigger_target'):
        targets['is_trigger'] = batch.is_trigger_target
        
    return targets