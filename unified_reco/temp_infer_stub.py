def run_inference(parquet_path, tag='', max_events=None):
    """As before, but truncates to the first `max_events` rows of the dataset
    if specified. Useful for time-limited benchmarks."""
    from torch.utils.data import Subset
    ds_full = PURITYDataset(parquet_path, max_hits=MAX_HITS)
    n_total = len(ds_full)
    n = n_total if max_events is None else min(n_total, max_events)

    if n < n_total:
        ds_iter = Subset(ds_full, list(range(n)))
        df      = ds_full.df.iloc[:n].reset_index(drop=True)
    else:
        ds_iter = ds_full
        df      = ds_full.df

    dl = DataLoader(ds_iter, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))

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
    }
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
    }
    truth_htp = np.zeros(n, dtype=np.int32)

    slice_role_truth, slice_role_probs, slice_is_anchor = [], [], []
    slice_pdg_truth, slice_event_id = [], []
    slice_pred_start, slice_pred_stop = [], []
    slice_truth_start, slice_truth_stop = [], []
    slice_start_sigma, slice_stop_sigma = [], []

    # --- new accumulators (place beside slice_role_truth etc.) ---
    lyso_seed_logits = []   # per event, [K] — np.nan if no LYSO
    lyso_seed_betas  = []   # per event, [K]
    lyso_hit_event   = []   # flat global event_idx per LYSO hit
    lyso_hit_E       = []   # flat MeV
    lyso_hit_xyz_n   = []   # flat normalized xyz per hit
    lyso_hit_p_hit   = []   # current model-aggregated p_hit per hit
    lyso_hit_assign  = []   # flat [N_lyso_hits, K] soft-assignments
    lyso_hit_beta    = []   # flat per-hit raw beta (pre-seed selection)
    lyso_hit_frac    = []   # flat per-hit cluster fraction
    LYSO_K = None           # fixed across events; set on first batch


    i0 = 0
    t0 = time.time()
    with torch.inference_mode():
        for batch in tqdm(dl, desc=f'infer[{tag}]'):
            batch = batch.to(device, non_blocking=True)
            anchor = getattr(batch, 'atar_triggering_pion_slice', None)
            out = model(batch.x, batch.batch, task_weights=TASK_WEIGHTS,
                        triggering_pion_slice=anchor)
            es = out.get('event_summary', {})
            B  = batch.num_graphs
            sl = slice(i0, i0 + B)

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
            pe = _arr('positron_energy')
            if pe is not None: preds['positron_energy'][sl] = pe
            de = _arr('dead_energy')
            if de is not None: preds['dead_energy'][sl] = de
            htp = _arr('has_trigger_positron')
            if htp is not None: preds['htp'][sl] = htp
            truth_htp[sl] = batch.has_trigger_positron.view(-1).detach().cpu().numpy().astype(np.int32)

            hit_trig = out.get('atar_hit_trigger_prob')
            hit_mip  = out.get('atar_hit_mip_prob')
            if hit_trig is not None and hit_mip is not None \
                    and hasattr(batch, 'is_trigger_target') \
                    and hasattr(batch, 'atar_node_pdg_target'):
                is_atar     = (batch.x[:, 5] > 0.5) | (batch.x[:, 6] > 0.5)
                is_atar_cpu = is_atar.cpu()
                atar_batch  = batch.batch[is_atar].cpu()
                pred_pos    = ((hit_trig > 0.5) & (hit_mip > 0.5)).cpu()
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
                    if n_t > 0: preds['pos_recall'][i0 + k]    = n_m / n_t
                    union = n_p + n_t - n_m
                    if union > 0: preds['pos_iou'][i0 + k]     = n_m / union

                        # --- LYSO clustering capture (per-hit + per-seed, ragged across events) ---
            beta_lyso  = out.get('lyso_seed_beta')
            w_lyso     = out.get('lyso_soft_assignments')
            ev_logits  = out.get('unified_event_logits')
            n_atar_tok = int(out.get('unified_num_atar_tokens', 0))
            p_hit_t    = out.get('lyso_hit_trigger_prob')
            beta_hit_t = out.get('lyso_beta')
            frac_hit_t = out.get('lyso_fractions')

            if w_lyso is not None and beta_lyso is not None and ev_logits is not None:
                if LYSO_K is None:
                    LYSO_K = int(w_lyso.size(1))
                K = LYSO_K
                is_lyso_b   = (batch.x[:, 7] > 0.5).cpu()
                batch_lyso_b = batch.batch[is_lyso_b].cpu().numpy()

                # Per-event seed quantities — fill nan for events with no LYSO
                seed_logits = np.full((B, K), np.nan, dtype=np.float32)
                seed_betas  = np.full((B, K), np.nan, dtype=np.float32)
                has_lyso_g  = np.zeros(B, dtype=bool)
                for ev in range(B):
                    has_lyso_g[ev] = (batch_lyso_b == ev).any()
                n_valid = int(has_lyso_g.sum())
                if n_valid > 0:
                    sl_logits = ev_logits[n_atar_tok:].view(n_valid, K).cpu().numpy()
                    sl_betas  = beta_lyso.view(n_valid, K).cpu().numpy()
                    seed_logits[has_lyso_g] = sl_logits
                    seed_betas[has_lyso_g]  = sl_betas
                for ev in range(B):
                    lyso_seed_logits.append(seed_logits[ev])
                    lyso_seed_betas.append(seed_betas[ev])

                if is_lyso_b.any():
                    e_lyso = (batch.x[is_lyso_b, 3].cpu() * 70.0).numpy()  # NORM_E_LYSO
                    xyz_n  = batch.x[is_lyso_b, 0:3].cpu().numpy()
                    w_np   = w_lyso.cpu().numpy()
                    p_np   = p_hit_t.cpu().numpy() if p_hit_t is not None else np.full(w_np.shape[0], np.nan)
                    b_np   = beta_hit_t.view(-1).cpu().numpy() if beta_hit_t is not None else np.full(w_np.shape[0], np.nan)
                    f_np   = frac_hit_t.view(-1).cpu().numpy() if frac_hit_t is not None else np.full(w_np.shape[0], np.nan)
                    for hidx in range(w_np.shape[0]):
                        lyso_hit_event.append(i0 + int(batch_lyso_b[hidx]))
                        lyso_hit_E.append(e_lyso[hidx])
                        lyso_hit_xyz_n.append(xyz_n[hidx])
                        lyso_hit_p_hit.append(p_np[hidx])
                        lyso_hit_assign.append(w_np[hidx])
                        lyso_hit_beta.append(b_np[hidx])
                        lyso_hit_frac.append(f_np[hidx])
            else:
                # Maintain length alignment with B
                k = LYSO_K if LYSO_K is not None else 4
                for ev in range(B):
                    lyso_seed_logits.append(np.full(k, np.nan, dtype=np.float32))
                    lyso_seed_betas .append(np.full(k, np.nan, dtype=np.float32))


            role_logits = out.get('atar_role_logits')
            endpoints   = out.get('atar_endpoints')
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
                    slice_pred_stop .append(ep[:, 1, :, 1] * NORM_POS_ATAR)
                    slice_start_sigma.append(((ep[:, 0, :, 2] - ep[:, 0, :, 0]) / 2) * NORM_POS_ATAR)
                    slice_stop_sigma .append(((ep[:, 1, :, 2] - ep[:, 1, :, 0]) / 2) * NORM_POS_ATAR)
                else:
                    slice_pred_start.append(np.full((n_slices, 3), np.nan))
                    slice_pred_stop .append(np.full((n_slices, 3), np.nan))
                    slice_start_sigma.append(np.full((n_slices, 3), np.nan))
                    slice_stop_sigma .append(np.full((n_slices, 3), np.nan))
                if hasattr(batch, 'atar_slice_start_target'):
                    slice_truth_start.append(batch.atar_slice_start_target[:n_slices].cpu().numpy() * NORM_POS_ATAR)
                    slice_truth_stop .append(batch.atar_slice_stop_target [:n_slices].cpu().numpy() * NORM_POS_ATAR)
                else:
                    slice_truth_start.append(np.full((n_slices, 3), np.nan))
                    slice_truth_stop .append(np.full((n_slices, 3), np.nan))

            i0 += B

    dt = time.time() - t0
    truth['htp'] = truth_htp
    print(f'{tag}: {n}/{n_total} events in {dt:.1f}s ({n/dt:.1f} evt/s)  '
          f'truth htp=1: {int(truth_htp.sum())} ({truth_htp.mean():.1%})')

    slices = {}
    if slice_role_truth:
        slices = {
            'role_truth':   np.concatenate(slice_role_truth),
            'role_probs':   np.concatenate(slice_role_probs),
            'is_anchor':    np.concatenate(slice_is_anchor),
            'pdg_truth':    np.concatenate(slice_pdg_truth),
            'event_id':     np.concatenate(slice_event_id),
            'pred_start':   np.concatenate(slice_pred_start),
            'pred_stop':    np.concatenate(slice_pred_stop),
            'truth_start':  np.concatenate(slice_truth_start),
            'truth_stop':   np.concatenate(slice_truth_stop),
            'start_sigma':  np.concatenate(slice_start_sigma),
            'stop_sigma':   np.concatenate(slice_stop_sigma),
        }
        slices['role_pred'] = slices['role_probs'].argmax(axis=-1)

        lyso_debug = {}
    if lyso_seed_logits:
        lyso_debug = {
            'K':            int(LYSO_K) if LYSO_K is not None else 0,
            'seed_logits':  np.stack(lyso_seed_logits),     # [n_event, K]
            'seed_betas':   np.stack(lyso_seed_betas),      # [n_event, K]
            'hit_event':    np.asarray(lyso_hit_event,  dtype=np.int64),
            'hit_E':        np.asarray(lyso_hit_E,      dtype=np.float32),
            'hit_xyz_n':    np.stack(lyso_hit_xyz_n) if lyso_hit_xyz_n else np.zeros((0, 3), np.float32),
            'hit_p_hit':    np.asarray(lyso_hit_p_hit,  dtype=np.float32),
            'hit_assign':   np.stack(lyso_hit_assign)  if lyso_hit_assign  else np.zeros((0, LYSO_K or 4), np.float32),
            'hit_beta':     np.asarray(lyso_hit_beta,   dtype=np.float32),
            'hit_frac':     np.asarray(lyso_hit_frac,   dtype=np.float32),
        }

    del ds_full, ds_iter, dl, df
    return truth, preds, slices, lyso_debug 