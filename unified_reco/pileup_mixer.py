"""
Mix unmixed PURITY parquets (from root_to_parquet.py) into per-event
events with realistic pileup and intrinsic LYSO radioactivity.

Each output row contains:
  - The triggering event's hits (origin=0)
  - Optional pileup overlay events from the Michel pool (origin=1, 2, ...)
  - Optional LYSO self-radioactivity hits (origin=-1, pdg=OTHER)
  - Truth kinematics + per-event scalars (dead_E, atar_posE, live_E) from
    the triggering event ONLY. Pileup and radioactivity do not contribute
    to those scalars.

Usage:
    python pileup_mixer.py --michel /path/to/unmixed_michel.parquet \\
                           --pie    /path/to/unmixed_pie.parquet \\
                           --output /path/to/out.parquet \\
                           --num_events N \\
                           --mode {michel|pie|mixed} \\
                           [--pie_mix_fraction F] [--trigger_gap_ns G] \\
                           [--biased_fraction F] [--biased_sigma S] \\
                           [--cal_only_fraction F] [--radio_rate HZ] \\
                           [--enforce_window]

Examples:
    # 50k mixed events with default Poisson pileup (lambda=0.24) and
    # ¹⁷⁶Lu radioactivity at 20 MHz:
    python pileup_mixer.py --michel unmixed_michel_train.parquet \\
                           --pie    unmixed_pie_train.parquet \\
                           --output mixed_train.parquet \\
                           --num_events 50000 --mode mixed --pie_mix_fraction 0.2

    # Pure Pienu eval set (no biased pileup, default radioactivity):
    python pileup_mixer.py --michel unmixed_michel_eval.parquet \\
                           --pie    unmixed_pie_eval.parquet \\
                           --output pie_eval.parquet \\
                           --num_events 100000 --mode pie

For a higher-level driver that wraps this for the standard train/val/eval
split, see generate_benchmarks.py.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from tqdm import tqdm

PION      = 0b000001
MUON      = 0b000010
POSITRON  = 0b000100
ELECTRON  = 0b001000
GAMMA     = 0b010000
OTHER     = 0b100000

def extract_hits(row, pdg_mask_keep, origin_val, dt_offset=0.0):
    """
    Extracts hits from the Parquet row that match ANY bit in pdg_mask_keep.
    Applies the time offset, and strictly DROPS any hits outside the [-300, 500] ns window.
    """
    # ATAR
    atar_pdgs = np.array(row['atar_pdg'])
    atar_t_shifted = np.array(row['atar_t']) + dt_offset
    
    keep_atar = ((atar_pdgs & pdg_mask_keep) > 0) & (atar_t_shifted >= -300) & (atar_t_shifted <= 500)
    
    atar_dict = {
        'x': np.array(row['atar_x'])[keep_atar].tolist(),
        'y': np.array(row['atar_y'])[keep_atar].tolist(),
        'z': np.array(row['atar_z'])[keep_atar].tolist(),
        't': atar_t_shifted[keep_atar].tolist(),
        'truth_t': np.array(row['atar_truth_t'])[keep_atar].tolist() if 'atar_truth_t' in row else np.zeros(np.sum(keep_atar)).tolist(),
        'E': np.array(row['atar_E'])[keep_atar].tolist(),
        'view': np.array(row['atar_view'])[keep_atar].tolist(),
        'pdg': atar_pdgs[keep_atar].tolist(),
        'slice_id': np.array(row['atar_slice_id'])[keep_atar].tolist(), # Pre-mix placeholder
        'slice_mean_t': [0.0] * np.sum(keep_atar), # Placeholder, recomputed after mixing
        'origin': [origin_val] * np.sum(keep_atar)
    }
    
    # LYSO
    lyso_pdgs = np.array(row['lyso_pdg'])
    lyso_t_shifted = np.array(row['lyso_t']) + dt_offset
    
    keep_lyso = ((lyso_pdgs & pdg_mask_keep) > 0) & (lyso_t_shifted >= -300) & (lyso_t_shifted <= 500)
    
    lyso_dict = {
        'x': np.array(row['lyso_x'])[keep_lyso].tolist(),
        'y': np.array(row['lyso_y'])[keep_lyso].tolist(),
        'z': np.array(row['lyso_z'])[keep_lyso].tolist(),
        't': lyso_t_shifted[keep_lyso].tolist(),
        'E': np.array(row['lyso_E'])[keep_lyso].tolist(),
        'pdg': lyso_pdgs[keep_lyso].tolist(),
        'slice_id': [0] * np.sum(keep_lyso), # Placeholder, recomputed after mixing
        'slice_mean_t': [0.0] * np.sum(keep_lyso), # Placeholder, recomputed after mixing
        'origin': [origin_val] * np.sum(keep_lyso)
    }
    
    return atar_dict, lyso_dict

def merge_extracted(base_hit_dict, new_hit_dict):
    for k in base_hit_dict.keys():
        base_hit_dict[k].extend(new_hit_dict[k])

class PileupMixer:
    def __init__(self, michel_path, pie_path=None):
        print(f"Loading Michel data from {michel_path}...")
        self.michel_df = pd.read_parquet(michel_path)
        if pie_path:
            print(f"Loading PiE data from {pie_path}...")
            self.pie_df = pd.read_parquet(pie_path)
        else:
            self.pie_df = None

        # Load Calorimeter Geometry LUTs for Radioactivity
        lut_dir = "/mnt/c/Users/obbee/research/notebooks/ML/caloRecon/"
        print(f"Loading Geometry LUTs from {lut_dir}...")
        self.geo_lookup = np.load(lut_dir + "crystalLUT.npy")
        self.valid_ids = np.load(lut_dir + "validLUT.npy")
            
    def generate_radioactivity(self, rate_hz, window_ms=[-300, 500]):
        """
        Generates intrinsic LYSO radioactivity hits.
        Uniformly distributed in time across the window and spatially across valid crystals.
        """
        window_size_ns = window_ms[1] - window_ms[0]
        avg_hits = rate_hz * (window_size_ns * 1e-9)
        n_hits = np.random.poisson(avg_hits)
        
        if n_hits == 0:
            return None
            
        # Sample indices from valid LYSO crystals
        bkg_indices = np.random.randint(0, len(self.valid_ids), size=n_hits)
        bkg_ids = self.valid_ids[bkg_indices]
        
        # Uniform time and Beta-decay energy approximation
        bkg_times = np.random.uniform(window_ms[0], window_ms[1], n_hits)
        bkg_energies = np.random.normal(loc=0.6, scale=0.2, size=n_hits)
        bkg_energies = np.maximum(bkg_energies, 0.05) # Clamp
        
        xyz_positions = self.geo_lookup[bkg_ids]
        
        lyso_dict = {
            'x': xyz_positions[:, 0].tolist(),
            'y': xyz_positions[:, 1].tolist(),
            'z': xyz_positions[:, 2].tolist(),
            't': bkg_times.tolist(),
            'E': bkg_energies.tolist(),
            'pdg': [OTHER] * n_hits, # Placeholder PDG (usually beta)
            'slice_id': [0] * n_hits, # Placeholder, recomputed after mixing
            'slice_mean_t': [0.0] * n_hits, # Placeholder, recomputed after mixing
            'origin': [-1] * n_hits
        }
        return lyso_dict

    def generate_batch(self, num_events, mode='michel', biased_fraction=0.0, biased_sigma=2.0,
                       cal_only_fraction=0.0, radio_rate=2e7, enforce_window=False,
                       pie_mix_fraction=0.5, trigger_gap_ns=2.0, max_reroll=16):

        if mode == 'mixed':
            if self.michel_df is None or self.pie_df is None:
                raise ValueError("Mixed mode requires both michel_df and pie_df.")
        else:
            pool = self.michel_df if mode == 'michel' else self.pie_df
            if pool is None:
                raise ValueError(f"Data for mode {mode} not loaded.")

        out_rows = []

        for _ in tqdm(range(num_events), desc=f"Mixing {mode}"):
            # Per-event pool selection for mixed mode
            if mode == 'mixed':
                use_pie = np.random.random() < pie_mix_fraction
                pool = self.pie_df if use_pie else self.michel_df
                event_mode = 'pie' if use_pie else 'michel'
            else:
                event_mode = mode

            # 1. Main Event: reroll until 2 ns trigger-gap cut passes (always on)
            # and, if enforce_window, positron survives the time window.
            main_row = None
            for attempt in range(max_reroll):
                idx = np.random.randint(len(pool))
                cand = pool.iloc[idx]

                # Trigger-gap cut: earliest POSITRON time must be >= trigger_gap_ns
                # after latest PION time. Prefer atar_truth_t; fall back to atar_t
                # (smeared observed time, ~200 ps resolution — fine against 2 ns cut).
                cand_pdgs = np.array(cand['atar_pdg'], dtype=int)
                if 'atar_truth_t' in cand:
                    cand_tt = np.array(cand['atar_truth_t'])
                    if np.all(cand_tt == 0):  # truth_t stored but unfilled
                        cand_tt = np.array(cand['atar_t'])
                else:
                    cand_tt = np.array(cand['atar_t'])
                pos_m = (cand_pdgs & POSITRON) > 0
                pion_m = (cand_pdgs & PION) > 0
                if pos_m.any() and pion_m.any():
                    if cand_tt[pos_m].min() - cand_tt[pion_m].max() < trigger_gap_ns:
                        continue

                if enforce_window:
                    a_temp, _ = extract_hits(cand, POSITRON, 0, 0.0)
                    if len(a_temp['x']) == 0:
                        continue

                main_row = cand
                break

            if main_row is None:
                # Exhausted rerolls: fall back to last candidate (extreme edge case)
                main_row = cand

            # Kinematics from Main Event
            out_row = {k: main_row[k] for k in main_row.keys() if k.startswith('truth_')}
            out_row['event_type'] = 0 if event_mode == 'michel' else 1

            # Per-event truth energy scalars from the triggering event only.
            # Pileup event contributions are NOT aggregated here — these are
            # strictly the triggering positron's (and its daughters') energy.
            for _scalar_col in ('dead_E', 'atar_posE', 'live_E'):
                if _scalar_col in main_row:
                    out_row[_scalar_col] = float(main_row[_scalar_col])
            
            atar_hits = {k: [] for k in ['x', 'y', 'z', 't', 'truth_t', 'E', 'view', 'pdg', 'slice_id', 'slice_mean_t', 'origin']}
            lyso_hits = {k: [] for k in ['x', 'y', 'z', 't', 'E', 'pdg', 'slice_id', 'slice_mean_t', 'origin']}
            
            a_main, l_main = extract_hits(main_row, 0xFFFFFFFF, origin_val=0, dt_offset=0.0)
            
            # Acceptance criteria: Positron survives time window cut, pion stops in ATAR, angle < 120
            pos_survived = np.any((np.array(a_main['pdg'], dtype=int) & POSITRON) > 0)
            fiducial_z = 1.2 < main_row['truth_pion_stop_z'] < 4.8
            fiducial_xy = abs(main_row['truth_pion_stop_x']) < 8 and abs(main_row['truth_pion_stop_y']) < 8
            valid_angle = np.degrees(main_row['truth_theta']) < 120
            
            if pos_survived and fiducial_z and fiducial_xy and valid_angle:
                out_row['truth_acceptance'] = 1
            else:
                out_row['truth_acceptance'] = 0
                
            merge_extracted(atar_hits, a_main)
            merge_extracted(lyso_hits, l_main)

            # Did any positron-tagged hit (ATAR OR LYSO) from the main event survive
            # the [-300, 500] ns mixer window? If not, the triggering positron is
            # effectively invisible in this mixed event, so zero out the truth
            # energy scalars copied above.
            _main_atar_pdg = np.array(a_main['pdg'], dtype=int)
            _main_lyso_pdg = np.array(l_main['pdg'], dtype=int)
            _main_atar_pos_mask = ((_main_atar_pdg & POSITRON) > 0) if len(_main_atar_pdg) > 0 else np.zeros(0, dtype=bool)
            _main_lyso_pos_mask = ((_main_lyso_pdg & POSITRON) > 0) if len(_main_lyso_pdg) > 0 else np.zeros(0, dtype=bool)
            positron_in_window = bool(_main_atar_pos_mask.any() or _main_lyso_pos_mask.any())
            if not positron_in_window:
                for _scalar_col in ('dead_E', 'atar_posE', 'live_E'):
                    if _scalar_col in out_row:
                        out_row[_scalar_col] = 0.0

            # Truth time of the triggering positron — earliest truth-t among
            # the surviving positron-tagged hits (ATAR preferred, LYSO fallback).
            # Stored on the mixed row directly so downstream code doesn't need
            # to re-derive it from per-hit arrays + origin filters.
            _truth_pos_t = -1000.0
            if positron_in_window:
                cands = []
                if _main_atar_pos_mask.any() and len(a_main['truth_t']) > 0:
                    cands.append(np.min(np.array(a_main['truth_t'])[_main_atar_pos_mask]))
                if _main_lyso_pos_mask.any() and len(l_main['t']) > 0:
                    cands.append(np.min(np.array(l_main['t'])[_main_lyso_pos_mask]))
                if cands:
                    _truth_pos_t = float(np.min(cands))
            out_row['truth_positron_t'] = _truth_pos_t

            # Pileup Selection
            rand_val = np.random.random()
            use_biased = (rand_val < biased_fraction)
            use_cal_only = (biased_fraction <= rand_val < biased_fraction + cal_only_fraction)
            
            # Check for primary positron anchoring (Required for both biased types)
            main_atar_pdgs = np.array(main_row['atar_pdg'], dtype=int)
            main_pos_mask = (main_atar_pdgs & POSITRON) > 0
            
            # Fallback if no primary positron exists to anchor the Gaussian
            if (use_biased or use_cal_only) and np.sum(main_pos_mask) == 0:
                use_biased = False 
                use_cal_only = False
                
            origin_counter = 1
                
            if use_biased or use_cal_only:
                # --- Biased Gaussian Pipeline (ATAR+LYSO or LYSO-Only) ---
                idx_bg = np.random.randint(len(self.michel_df))
                bg_row = self.michel_df.iloc[idx_bg]
                
                bg_atar_pdgs = np.array(bg_row['atar_pdg'], dtype=int)
                bg_pos_mask = (bg_atar_pdgs & POSITRON) > 0
                
                # If background lacks a positron, skip adding pileup
                if np.sum(bg_pos_mask) > 0:
                    main_truth_t = np.array(main_row['atar_truth_t']) if 'atar_truth_t' in main_row else np.zeros(len(main_atar_pdgs))
                    bg_truth_t = np.array(bg_row['atar_truth_t']) if 'atar_truth_t' in bg_row else np.zeros(len(bg_atar_pdgs))
                    
                    t_pos_main = np.min(main_truth_t[main_pos_mask])
                    t_pos_bg = np.min(bg_truth_t[bg_pos_mask])
                    
                    dt = np.random.normal(0.0, biased_sigma)
                    dt_shift = (t_pos_main + dt) - t_pos_bg
                    
                    a_bg, l_bg = extract_hits(bg_row, 0xFFFFFFFF, origin_val=origin_counter, dt_offset=dt_shift)
                    
                    # Only merge ATAR if NOT in calorimeter-only mode
                    if not use_cal_only:
                        merge_extracted(atar_hits, a_bg)
                        
                    merge_extracted(lyso_hits, l_bg)
                    origin_counter += 1
            else:
                # --- Standard Poisson Pipeline ---
                # Poisson Expectation for an 800ns window at a 300kHz beam arrival rate
                lam = 0.24

                # 2. Pileup - Old Muon Decays
                num_old_decays = np.random.poisson(lam)
                for _ in range(num_old_decays):
                    idx_bg = np.random.randint(len(self.michel_df))
                    bg_row = self.michel_df.iloc[idx_bg]
                    
                    bg_atar_pdgs = np.array(bg_row['atar_pdg'], dtype=int)
                    pos_mask = (bg_atar_pdgs & POSITRON) > 0
                    if np.sum(pos_mask) == 0: continue
                    
                    t_pos_raw = np.min(np.array(bg_row['atar_t'])[pos_mask])
                    t_decay_target = np.random.uniform(-300, 500)
                    dt_shift = t_decay_target - t_pos_raw
                    
                    a_bg, l_bg = extract_hits(bg_row, 0xFFFFFFFF, origin_val=origin_counter, dt_offset=dt_shift)
                    merge_extracted(atar_hits, a_bg)
                    merge_extracted(lyso_hits, l_bg)
                    origin_counter += 1
                    
                # 3. Pileup - New Entering Pions
                num_new_pions = np.random.poisson(lam)
                for _ in range(num_new_pions):
                    idx_bg = np.random.randint(len(self.michel_df))
                    bg_row = self.michel_df.iloc[idx_bg]
                    
                    t_enter_target = np.random.uniform(-300, 500)
                    
                    a_bg, l_bg = extract_hits(bg_row, 0xFFFFFFFF, origin_val=origin_counter, dt_offset=t_enter_target)
                    merge_extracted(atar_hits, a_bg)
                    merge_extracted(lyso_hits, l_bg)
                    origin_counter += 1
                
            # 4. Self-Radioactivity (Intrinsic LYSO background)
            radio_hits = self.generate_radioactivity(radio_rate)
            if radio_hits:
                merge_extracted(lyso_hits, radio_hits)
                
            # Re-calculate Time-Slice IDs independently per detector
            all_atar_t = np.array(atar_hits['t'])
            all_lyso_t = np.array(lyso_hits['t'])
            all_atar_E = np.array(atar_hits['E'])
            all_lyso_E = np.array(lyso_hits['E'])
            
            def energy_weighted_slicing(times, energies, gap_threshold):
                """
                Clusters hits using an energy-weighted sliding mean.
                A new cluster starts when a hit's time exceeds the current
                cluster's energy-weighted mean time by more than gap_threshold.
                """
                n = len(times)
                if n == 0:
                    return np.zeros(0, dtype=int)
                
                slices = np.zeros(n, dtype=int)
                sort_idx = np.argsort(times)
                
                current_slice = 1
                sum_Et = times[sort_idx[0]] * energies[sort_idx[0]]
                sum_E = energies[sort_idx[0]]
                slices[sort_idx[0]] = current_slice
                
                for i in range(1, n):
                    idx = sort_idx[i]
                    t_i = times[idx]
                    E_i = energies[idx]
                    
                    weighted_mean = sum_Et / max(sum_E, 1e-9)
                    gap = t_i - weighted_mean
                    
                    if gap > gap_threshold:
                        # Start new cluster
                        current_slice += 1
                        sum_Et = t_i * E_i
                        sum_E = E_i
                    else:
                        # Extend current cluster
                        sum_Et += t_i * E_i
                        sum_E += E_i
                    
                    slices[idx] = current_slice
                
                return slices
            
            # ATAR: 1.0 ns gap from energy-weighted mean (200 ps resolution)
            final_atar_slices = energy_weighted_slicing(all_atar_t, all_atar_E, gap_threshold=1.0)
            
            # LYSO: 5.0 ns gap from energy-weighted mean (worse calorimeter timing)
            final_lyso_slices = energy_weighted_slicing(all_lyso_t, all_lyso_E, gap_threshold=5.0)
            
            def compute_slice_mean_t(times, energies, slices):
                """Compute energy-weighted mean time per slice, broadcast back to per-hit."""
                n = len(times)
                if n == 0:
                    return np.zeros(0)
                mean_t = np.zeros(n)
                for sid in np.unique(slices):
                    if sid == 0: continue
                    mask = (slices == sid)
                    E_slice = energies[mask]
                    t_slice = times[mask]
                    total_E = E_slice.sum()
                    if total_E > 0:
                        wt = np.sum(t_slice * E_slice) / total_E
                    else:
                        wt = t_slice.mean()
                    mean_t[mask] = wt
                return mean_t
            
            atar_slice_mean_t = compute_slice_mean_t(all_atar_t, all_atar_E, final_atar_slices)
            lyso_slice_mean_t = compute_slice_mean_t(all_lyso_t, all_lyso_E, final_lyso_slices)
            
            atar_hits['slice_id'] = final_atar_slices.tolist()
            atar_hits['slice_mean_t'] = atar_slice_mean_t.tolist()
            lyso_hits['slice_id'] = final_lyso_slices.tolist()
            lyso_hits['slice_mean_t'] = lyso_slice_mean_t.tolist()
            
            # Identify rare overlaps where multiple MC events fall in same ATAR temporal group
            multi_origin_slice = 0
            if len(final_atar_slices) > 0:
                atar_origins = np.array(atar_hits['origin'])
                for slc_id in np.unique(final_atar_slices):
                    if slc_id == 0: continue
                    mask = (final_atar_slices == slc_id)
                    origins_in_slice = np.unique(atar_origins[mask])
                    if len(origins_in_slice) > 1:
                        multi_origin_slice = 1
                        break
            
            out_row['truth_multi_event_atar_slice'] = multi_origin_slice

            # Pie-tagger truth labels (energy-blind, ATAR-only).
            # is_pie:           main chain came from the pie pool
            # has_muon:         any ATAR hit (main OR pileup) carries the muon bit
            # has_atar_pileup:  any ATAR hit comes from a non-trigger origin
            #                   (LYSO-only / cal-only pileup is excluded by
            #                    design — the head can't see LYSO so the label
            #                    must match the feature space)
            atar_pdg_arr = (np.array(atar_hits['pdg'], dtype=int)
                            if len(atar_hits['pdg']) else np.zeros(0, dtype=int))
            atar_origin_arr = (np.array(atar_hits['origin'], dtype=int)
                               if len(atar_hits['origin']) else np.zeros(0, dtype=int))
            out_row['truth_is_pie']          = int(event_mode == 'pie')
            out_row['truth_has_muon']        = int(((atar_pdg_arr & MUON) > 0).any())
            out_row['truth_has_atar_pileup'] = int((atar_origin_arr > 0).any())

            # Finalize this mixed event into column schema
            for k in atar_hits.keys():
                out_col = 'atar_slice' if k == 'slice_id' else f'atar_{k}'
                out_row[out_col] = atar_hits[k]
                
            for k in lyso_hits.keys():
                out_col = 'lyso_slice' if k == 'slice_id' else f'lyso_{k}'
                out_row[out_col] = lyso_hits[k]
            
            out_rows.append(out_row)
            
        return pd.DataFrame(out_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mix unrolled Parquet events with Pileup")
    parser.add_argument("--michel", type=str, required=True, help="Path to Michel Parquet")
    parser.add_argument("--pie", type=str, default=None, help="Path to PiE Parquet")
    parser.add_argument("--output", type=str, required=True, help="Output mixed Parquet")
    parser.add_argument("--num_events", type=int, default=10)
    parser.add_argument("--mode", type=str, default='michel', choices=['michel', 'pie', 'mixed'])
    parser.add_argument("--pie_mix_fraction", type=float, default=0.5, help="Fraction of pie main events in mixed mode")
    parser.add_argument("--trigger_gap_ns", type=float, default=2.0, help="Minimum allowed gap (ns) between pion stop and triggering positron")
    parser.add_argument("--biased_fraction", type=float, default=0.0, help="Fraction of events to use biased pileup (0.0=Off)")
    parser.add_argument("--biased_sigma", type=float, default=2.0, help="Gaussian spread for biased pileup separation (ns)")
    parser.add_argument("--cal_only_fraction", type=float, default=0.0, help="Fraction of events to use calorimeter-only biased pileup")
    parser.add_argument("--radio_rate", type=float, default=2e7, help="LYSO self-radioactivity rate (Hz)")
    parser.add_argument("--enforce_window", action='store_true', help="Retry sampling (up to 3x) if primary positron is outside window")
    
    args = parser.parse_args()
    
    if args.biased_fraction + args.cal_only_fraction > 1.0:
        parser.error("The sum of --biased_fraction and --cal_only_fraction cannot exceed 1.0")
    
    mixer = PileupMixer(args.michel, args.pie)
    mixed_df = mixer.generate_batch(args.num_events, mode=args.mode,
                                    biased_fraction=args.biased_fraction,
                                    biased_sigma=args.biased_sigma,
                                    cal_only_fraction=args.cal_only_fraction,
                                    radio_rate=args.radio_rate,
                                    enforce_window=args.enforce_window,
                                    pie_mix_fraction=args.pie_mix_fraction,
                                    trigger_gap_ns=args.trigger_gap_ns)
    
    print(f"Writing {args.num_events} mixed events to {args.output}")
    table = pa.Table.from_pandas(mixed_df)
    pq.write_table(table, args.output)
    print("Done!")
