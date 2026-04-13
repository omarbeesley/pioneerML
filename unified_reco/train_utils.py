import torch
import torch.nn as nn
import torch.nn.functional as F

class PinballLoss(nn.Module):
    """
    Robust Permutation-Invariant Endpoint Loss using Asymmetric Attenuated Pinball Loss.
    Adapted from endpoint_finder.ipynb for the PURITY architecture.
    """
    def __init__(self, quantiles=[0.16, 0.50, 0.84], loss_scale_span=0.1, loss_scale_dir=0.1):
        super().__init__()
        self.quantiles = quantiles
        self.loss_scale_span = loss_scale_span
        self.loss_scale_dir = loss_scale_dir
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
                    loss_beta += self.w_highlander * e_beta[is_redundant_signal].mean()

                # --- B. Signal Alpha Loss (Seeds must be 1.0) ---
                valid_alpha_betas = e_beta[alpha_indices][valid_obj_mask].nan_to_num(0.5).clamp(1e-6, 1-1e-6)
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
            p_atar_broadcast.clamp(1e-6, 1-1e-6), atar_targets_valid, reduction='none')
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
            p_hit.clamp(1e-6, 1-1e-6), lyso_targets, reduction='none')
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
        # Assuming PinballLoss and CondensationLoss are defined elsewhere
        self.pinball = PinballLoss(quantiles=[0.16, 0.50, 0.84]) 
        self.condensation = CondensationLoss()
        
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
            
        # 3. ATAR Trigger Slice Loss (Phase 9)
        w_trigger_slice = self.config.get('w_atar_trigger_slice', 0.0)
        if w_trigger_slice > 0.0 and 'atar_trigger_logits' in outputs and 'tar_slice_trigger' in targets:
            l_trigger_slice = self.bce_logits(outputs['atar_trigger_logits'], targets['tar_slice_trigger'])
            loss_dict['loss_trigger_slice'] = l_trigger_slice
            total_loss += w_trigger_slice * l_trigger_slice
        
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
                l_cosine = (1.0 - F.cosine_similarity(pred_dir, tar_dir, dim=1)).mean()
                #l_mse = F.mse_loss(pred_dir, tar_dir)
                #l_angle = l_cosine + 0.1 * l_mse
                l_angle = l_cosine
                loss_dict['loss_positron_angle'] = l_angle
                total_loss += w_angle * l_angle


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
        
        # --- Event Synthesis Loss ---
        w_event = self.config.get('w_event_builder', 0.0)
        if w_event > 0.0 and 'unified_event_logits' in outputs and batch is not None:
            # Call the new energy-weighted broadcast BCE
            l_event = event_builder_loss(outputs, batch)
            
            # If the loss returned a valid gradient tensor
            if l_event.requires_grad:
                total_loss += w_event * l_event
                loss_dict['L_event_builder'] = l_event.item()
            else:
                loss_dict['L_event_builder'] = 0.0

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