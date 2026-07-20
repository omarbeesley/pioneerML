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
kMudif    = 0x0000001000   # PIEventType muon-decay-in-flight bit (in the event_type column)
kPidif    = 0x0000000010   # PIEventType pion-decay-in-flight bit (in the event_type column)
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
        'truth_t': (np.array(row['atar_truth_t']) + dt_offset)[keep_atar].tolist() if 'atar_truth_t' in row else np.zeros(np.sum(keep_atar)).tolist(),
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
    def __init__(self, michel_path, pie_path=None, mudif_path=None, pidif_path=None,
                 lut_dir="/data/nvme0/test_ml_data/radio_LUTs/", drop_rmd=False):
        print(f"Loading Michel data from {michel_path}...")
        self.michel_df = pd.read_parquet(michel_path)
        if drop_rmd and 'event_type' in self.michel_df.columns:
            # Purify the michel pool of RADIATIVE muon decays (kMurad, 0x200) at load, so
            # RMD reaches neither triggers nor pileup donors nor the calo-only stream.
            # (The analysis-level kMurad filter only protected the counting pools; RMD
            # pileup donors were found feeding the high-energy accidental bin.)
            et = self.michel_df['event_type'].astype('int64')
            n0 = len(self.michel_df)
            self.michel_df = self.michel_df[(et & 0x200) == 0].reset_index(drop=True)
            print(f"[mixer] drop_rmd: removed {n0 - len(self.michel_df):,} kMurad events "
                  f"({100*(n0-len(self.michel_df))/max(n0,1):.2f}%) from the michel pool")
        if pie_path:
            print(f"Loading PiE data from {pie_path}...")
            self.pie_df = pd.read_parquet(pie_path)
        else:
            self.pie_df = None
        if mudif_path:
            print(f"Loading muDIF data from {mudif_path}...")
            self.mudif_df = pd.read_parquet(mudif_path)
            # PURIFY: the lifetime-biased muDIF MC is a mix of true muon-DIF and
            # ordinary mu-DAR. Keep ONLY true muon-DIF (kMudif bit) so that
            # mudif_mix_fraction counts only true muDIF events. The dropped mu-DAR are
            # redundant with the (much larger) michel pool.
            if 'event_type' in self.mudif_df.columns:
                n0 = len(self.mudif_df)
                keep = (self.mudif_df['event_type'].astype('int64') & kMudif) > 0
                self.mudif_df = self.mudif_df[keep].reset_index(drop=True)
                frac = 100.0 * len(self.mudif_df) / max(n0, 1)
                print(f"  purified muDIF pool to kMudif: kept {len(self.mudif_df)}/{n0} "
                      f"({frac:.1f}% true muon-DIF)")
                if len(self.mudif_df) == 0:
                    raise ValueError(
                        f"muDIF pool has 0 true-kMudif events after purification "
                        f"(of {n0}) — check the muDIF MC / its event_type column.")
            else:
                print("  [warn] muDIF parquet has no 'event_type' column — cannot purify "
                      "to kMudif; pool may still contain mu-DAR contamination.")
        else:
            self.mudif_df = None
        if pidif_path:
            print(f"Loading piDIF data from {pidif_path}...")
            self.pidif_df = pd.read_parquet(pidif_path)
            # PURIFY: the lifetime-biased piDIF MC is a mix of true pion-DIF
            # (pi[DIF]->mu[DAR]->e) and ordinary pi-DAR. Keep ONLY true pion-DIF
            # (kPidif bit) so pidif_mix_fraction counts only true piDIF; the dropped
            # pi-DAR are redundant with the (much larger) michel pool.
            if 'event_type' in self.pidif_df.columns:
                n0 = len(self.pidif_df)
                keep = (self.pidif_df['event_type'].astype('int64') & kPidif) > 0
                self.pidif_df = self.pidif_df[keep].reset_index(drop=True)
                frac = 100.0 * len(self.pidif_df) / max(n0, 1)
                print(f"  purified piDIF pool to kPidif: kept {len(self.pidif_df)}/{n0} "
                      f"({frac:.1f}% true pion-DIF)")
                if len(self.pidif_df) == 0:
                    raise ValueError(
                        f"piDIF pool has 0 true-kPidif events after purification "
                        f"(of {n0}) — check the piDIF MC (was it converted with "
                        f"--keep_pidif?) / its event_type column.")
            else:
                print("  [warn] piDIF parquet has no 'event_type' column — cannot purify "
                      "to kPidif; pool may still contain pi-DAR contamination.")
        else:
            self.pidif_df = None

        # Load Calorimeter Geometry LUTs for Radioactivity. lut_dir is
        # overridable (e.g. point at the ML_TEST share when running data-gen in
        # the host conda env). Must contain crystalLUT.npy + validLUT.npy.
        if not lut_dir.endswith("/"):
            lut_dir = lut_dir + "/"
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

        # Apply the LYSO digitizer trigger threshold (0.20 MeV) exactly as real LYSO
        # hits are in root_to_parquet (LYSO_TRIG_E): sub-threshold radioactivity is not
        # recorded by the detector, so DROP those hits rather than clamping them up.
        LYSO_TRIG_E = 0.20
        keep = bkg_energies > LYSO_TRIG_E
        bkg_ids = bkg_ids[keep]
        bkg_times = bkg_times[keep]
        bkg_energies = bkg_energies[keep]
        n_hits = int(keep.sum())
        if n_hits == 0:
            return None

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

    def _sample_pileup_idx(self, trigger_pool, trigger_idx):
        """Draw a michel-pool pileup-donor index, avoiding the trigger row itself when
        the trigger was ALSO drawn from the michel pool — so a pileup donor can never be
        the exact same event as the trigger (which would fake a multi-event coincidence)."""
        n = len(self.michel_df)
        avoid = trigger_idx if (trigger_pool is self.michel_df) else -1
        if n <= 1:
            return 0
        j = np.random.randint(n)
        while j == avoid:
            j = np.random.randint(n)
        return j

    def _sample_calo_only_idx(self, trigger_pool, trigger_idx, max_tries=400):
        """Draw a michel-pool donor whose positron left NO ATAR hits but DID deposit in
        LYSO -- the old-muon-backlog decay that missed the ATAR (the population the
        'old muon decays' stream skips). Rejection sampling (~6% of the pool); -1 if
        no eligible donor found."""
        for _ in range(max_tries):
            j = self._sample_pileup_idx(trigger_pool, trigger_idx)
            row = self.michel_df.iloc[j]
            a_pdg = np.asarray(row['atar_pdg'], dtype=int)
            if a_pdg.size and np.any((a_pdg & POSITRON) > 0):
                continue
            l_pdg = np.asarray(row['lyso_pdg'], dtype=int)
            if l_pdg.size and np.any((l_pdg & POSITRON) > 0):
                return j
        return -1

    def _calo_michel_auto_rate(self, lam, sample=20000):
        """Physical calo-only michel rate = lam x P(donor positron missed the ATAR but
        hit LYSO), estimated once from the michel pool."""
        if getattr(self, "_calo_frac", None) is None:
            n = min(sample, len(self.michel_df))
            hit = 0
            for j in range(n):
                row = self.michel_df.iloc[j]
                a_pdg = np.asarray(row['atar_pdg'], dtype=int)
                if a_pdg.size and np.any((a_pdg & POSITRON) > 0):
                    continue
                l_pdg = np.asarray(row['lyso_pdg'], dtype=int)
                if l_pdg.size and np.any((l_pdg & POSITRON) > 0):
                    hit += 1
            self._calo_frac = hit / max(n, 1)
            print(f"[mixer] calo-only michel fraction = {self._calo_frac:.4f} "
                  f"-> auto rate {lam * self._calo_frac:.5f}/event", flush=True)
        return lam * self._calo_frac

    def generate_batch(self, num_events, mode='michel', biased_fraction=0.0, biased_sigma=2.0,
                       cal_only_fraction=0.0, radio_rate=2e7, enforce_window=False,
                       pie_mix_fraction=0.5, mudif_mix_fraction=0.0, pidif_mix_fraction=0.0,
                       trigger_gap_ns=2.0, max_reroll=16, no_replace=False,
                       calo_michel_rate=0.0):

        if mode == 'mixed':
            if self.michel_df is None or self.pie_df is None:
                raise ValueError("Mixed mode requires both michel_df and pie_df.")
            if mudif_mix_fraction > 0 and self.mudif_df is None:
                raise ValueError("mudif_mix_fraction > 0 requires mudif_df (pass mudif_path).")
            if pidif_mix_fraction > 0 and self.pidif_df is None:
                raise ValueError("pidif_mix_fraction > 0 requires pidif_df (pass pidif_path).")
            if pie_mix_fraction + mudif_mix_fraction + pidif_mix_fraction > 1.0:
                raise ValueError(
                    f"pie_mix_fraction + mudif_mix_fraction + pidif_mix_fraction = "
                    f"{pie_mix_fraction + mudif_mix_fraction + pidif_mix_fraction:.3f} > 1.0 "
                    f"(no room for michel).")
        else:
            pool = {'michel': self.michel_df, 'pie': self.pie_df,
                    'mudif': self.mudif_df, 'pidif': self.pidif_df}.get(mode)
            if pool is None:
                raise ValueError(f"Data for mode {mode} not loaded.")

        out_rows = []

        # Without-replacement primary sampling (single-pool modes only): draw each pool
        # primary AT MOST ONCE via a shuffled cursor, instead of np.random.randint which
        # reuses ~37% of primaries. Every idx draw (incl. rerolls past kPitar/window cuts)
        # advances the cursor; production stops early when the pool is exhausted, so no
        # primary is ever reused. NOTE: with enforce_window this consumes out-of-window
        # primaries, so the unique output count is (pool size) x (in-window fraction).
        _perm = np.random.permutation(len(pool)) if (no_replace and mode != 'mixed') else None
        _cursor = 0

        for _ in tqdm(range(num_events), desc=f"Mixing {mode}"):
            # Per-event pool selection for mixed mode (four-way: pie / mudif / pidif /
            # michel; michel takes the remaining 1 - pie - mudif - pidif fractions).
            if mode == 'mixed':
                r = np.random.random()
                if r < pie_mix_fraction:
                    pool, event_mode = self.pie_df, 'pie'
                elif r < pie_mix_fraction + mudif_mix_fraction:
                    pool, event_mode = self.mudif_df, 'mudif'
                elif r < pie_mix_fraction + mudif_mix_fraction + pidif_mix_fraction:
                    pool, event_mode = self.pidif_df, 'pidif'
                else:
                    pool, event_mode = self.michel_df, 'michel'
            else:
                event_mode = mode

            # 1. Main Event: reroll until 2 ns trigger-gap cut passes (always on)
            # and, if enforce_window, positron survives the time window.
            main_row = None
            last_kpitar = None      # last candidate passing the kPitar (stopped-pion) cut
            for attempt in range(max_reroll):
                if _perm is not None:
                    if _cursor >= len(_perm):
                        break                      # pool exhausted -> no more unique primaries
                    idx = int(_perm[_cursor]); _cursor += 1
                else:
                    idx = np.random.randint(len(pool))
                cand = pool.iloc[idx]

                # Triggering pion must DECAY IN THE TARGET (kPitar = PIEventType BIT6 = 0x40),
                # for ALL channels. Rejects beam-halo pions (large xprime divergence) whose
                # decay/stop vertex is outside the ATAR (downstream, large radius) and thus
                # unlocalizable -> floors the kinematics loss (~0.038 vs ~0.0003) and corrupts
                # the fiducial/angle acceptance. For michel/pie/muDIF this is decay-at-rest in
                # the target; for piDIF it is an IN-FLIGHT decay that still occurs INSIDE the
                # target (kPidif & kPitar, ~85% of the piDIF pool), NOT a halo DIF downstream.
                # HARD cut (also honored by the reroll-exhausted fallback below).
                if (int(cand.get('event_type', 0)) & 0x40) == 0:
                    continue
                # DTAR beam trigger: the TRIGGERING event must have fired the degrader
                # (dtar_triggered==1). Newly-kept DTAR-miss donors (dtar_triggered==0) are
                # calo-pileup material only and must never become the trigger. .get default
                # 1 for legacy parquets predating the column (they hold only triggered events).
                # Placed before last_kpitar so the reroll-exhausted fallback also honors it.
                if int(cand.get('dtar_triggered', 1)) == 0:
                    continue
                last_kpitar = cand

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
                if _perm is not None and _cursor >= len(_perm):
                    break                          # without-replacement: pool exhausted, stop
                # Exhausted rerolls: fall back to the last kPitar-passing candidate so a
                # halo pion never sneaks in as the trigger. Only if NO kPitar candidate was
                # drawn at all (astronomically rare) use the last raw candidate.
                main_row = last_kpitar if last_kpitar is not None else cand

            # Kinematics from Main Event
            out_row = {k: main_row[k] for k in main_row.keys() if k.startswith('truth_')}
            # Forward the full PIEventType bitmask (kMudif=0x1000, kPienu, kPidar|kMudar, ...)
            # plus the lifetime-bias weight and muon KE-at-decay, instead of clobbering
            # event_type with a class index. .get() defaults keep this working for older
            # unmixed parquets that predate these columns in root_to_parquet.py.
            out_row['event_type']    = int(main_row.get('event_type', 0))
            out_row['gen_weight']    = float(main_row.get('gen_weight', 1.0))
            out_row['muon_decay_ke'] = float(main_row.get('muon_decay_ke', 0.0))
            out_row['pion_decay_ke'] = float(main_row.get('pion_decay_ke', 0.0))

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
            # Truth times (readout-window ns) of injected ACCIDENTAL positrons -- old-muon or
            # new-pion decays carrying NO trigger pion. The analysis uses these to separate the
            # trigger pi->mu->e from accidental contamination (a consistent R_e/mu denominator).
            accidental_pos_times = []

            if use_biased or use_cal_only:
                # --- Biased Gaussian Pipeline (ATAR+LYSO or LYSO-Only) ---
                idx_bg = self._sample_pileup_idx(pool, idx)
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
                        _t_acc = float(t_pos_main + dt)   # anchored bg positron lands here
                        if -300.0 <= _t_acc <= 500.0:
                            accidental_pos_times.append(_t_acc)

                    merge_extracted(lyso_hits, l_bg)
                    origin_counter += 1
            else:
                # --- Standard Poisson Pipeline ---
                # Poisson Expectation for an 800ns window at a 300kHz beam arrival rate
                lam = 0.24

                # 2. Pileup - Old Muon Decays
                num_old_decays = np.random.poisson(lam)
                for _ in range(num_old_decays):
                    idx_bg = self._sample_pileup_idx(pool, idx)
                    bg_row = self.michel_df.iloc[idx_bg]
                    
                    bg_atar_pdgs = np.array(bg_row['atar_pdg'], dtype=int)
                    pos_mask = (bg_atar_pdgs & POSITRON) > 0
                    if np.sum(pos_mask) == 0: continue
                    
                    t_pos_raw = np.min(np.array(bg_row['atar_t'])[pos_mask])
                    t_decay_target = np.random.uniform(-300, 500)
                    dt_shift = t_decay_target - t_pos_raw
                    
                    a_bg, l_bg = extract_hits(bg_row, 0xFFFFFFFF, origin_val=origin_counter, dt_offset=dt_shift)
                    accidental_pos_times.append(float(t_decay_target))  # old-muon positron time (in-window by construction)
                    merge_extracted(atar_hits, a_bg)
                    merge_extracted(lyso_hits, l_bg)
                    origin_counter += 1

                # 3. Pileup - New Entering Pions
                num_new_pions = np.random.poisson(lam)
                for _ in range(num_new_pions):
                    idx_bg = self._sample_pileup_idx(pool, idx)
                    bg_row = self.michel_df.iloc[idx_bg]
                    
                    t_enter_target = np.random.uniform(-300, 500)

                    # Record the new-pion positron time if its decay lands in the window.
                    _np_pos = (np.array(bg_row['atar_pdg'], dtype=int) & POSITRON) > 0
                    if np.sum(_np_pos) > 0:
                        _t_np = float(np.min(np.array(bg_row['atar_t'])[_np_pos]) + t_enter_target)
                        if -300.0 <= _t_np <= 500.0:
                            accidental_pos_times.append(_t_np)

                    a_bg, l_bg = extract_hits(bg_row, 0xFFFFFFFF, origin_val=origin_counter, dt_offset=t_enter_target)
                    merge_extracted(atar_hits, a_bg)
                    merge_extracted(lyso_hits, l_bg)
                    origin_counter += 1

                # 4. Pileup - Calo-only michel decays (old-muon backlog whose positron
                #    MISSED the ATAR). Time-INDEPENDENT: decay anchored uniformly in the
                #    readout window. Rate: calo_michel_rate<0 -> physical auto rate
                #    (lam x pool fraction with LYSO-positron & no ATAR-positron);
                #    0 disables (default, preserves old behavior); >0 explicit.
                #    LYSO deposits ONLY are merged -- the donor's stale pion/muon ATAR
                #    stubs belong microseconds in the past, not this window.
                if calo_michel_rate != 0.0:
                    _rate = (self._calo_michel_auto_rate(lam) if calo_michel_rate < 0
                             else calo_michel_rate)
                    for _ in range(np.random.poisson(_rate)):
                        idx_cal = self._sample_calo_only_idx(pool, idx)
                        if idx_cal < 0:
                            break
                        bg_row = self.michel_df.iloc[idx_cal]
                        l_pdg = np.asarray(bg_row['lyso_pdg'], dtype=int)
                        pos_l = (l_pdg & POSITRON) > 0
                        t_ref = float(np.min(np.asarray(bg_row['lyso_t'], float)[pos_l]))
                        t_target = np.random.uniform(-300.0, 500.0)
                        _a_bg, l_bg = extract_hits(bg_row, 0xFFFFFFFF, origin_val=origin_counter,
                                                   dt_offset=t_target - t_ref)
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
            # muDIF vs mu-DAR is decided by the per-event PIEventType bitmask, NOT by
            # which POOL the event came from. The lifetime-biased muDIF MC contains BOTH
            # true muon-DIF (kMudif) AND ordinary mu-DAR events, so a mu-DAR event drawn
            # from the mudif pool must be labeled like a Michel (stopped muon), not muDIF.
            # An in-flight muon hit = a muon-pdg hit (origin 0) in a true-kMudif event;
            # pileup muons (origin>0, from the michel pool) are always mu-DAR -> muon veto.
            is_mudif_evt = (int(main_row.get('event_type', 0)) & kMudif) > 0
            atar_muon_mask = (atar_pdg_arr & MUON) > 0
            if is_mudif_evt:
                inflight_mask = atar_muon_mask & (atar_origin_arr == 0)
            else:
                inflight_mask = np.zeros(len(atar_muon_mask), dtype=bool)
            # piDIF (pi[DIF]->mu[DAR]->e): same bitmask logic. A DIF-pion hit = a
            # pion-pdg hit (origin 0) in a true-kPidif event. Stopping pions (non-kPidif
            # triggers) and pileup pions (origin>0, from the michel pool) are excluded,
            # so this isolates the in-flight pion track that lacks a Bragg stop.
            is_pidif_evt = (int(main_row.get('event_type', 0)) & kPidif) > 0
            atar_pion_mask = (atar_pdg_arr & PION) > 0
            if is_pidif_evt:
                difpion_mask = atar_pion_mask & (atar_origin_arr == 0)
            else:
                difpion_mask = np.zeros(len(atar_pion_mask), dtype=bool)
            out_row['truth_is_pie']          = int(event_mode == 'pie')   # pie pool is clean
            out_row['truth_is_mudif']        = int(is_mudif_evt)
            out_row['truth_has_muon']        = int((atar_muon_mask & ~inflight_mask).any())  # stopped (muDAR)
            out_row['truth_has_muon_dif']    = int(inflight_mask.any())                       # in-flight (muDIF)
            out_row['truth_is_pidif']        = int(is_pidif_evt)
            out_row['truth_has_pidif']       = int(difpion_mask.any())                        # in-flight pion (piDIF)
            out_row['truth_has_atar_pileup'] = int((atar_origin_arr > 0).any())
            # Accidental (old-muon / new-pion) positron truth. These carry NO trigger pion, so
            # the analysis must drop them from BOTH the pi->mu->e count and the selection
            # efficiency (the double-standard that inflated R_e/mu). truth_accidental_positron_t
            # is the earliest in-window accidental positron time (-1000 if none), for reco match.
            _acc = [t for t in accidental_pos_times if -300.0 <= t <= 500.0]
            out_row['truth_has_accidental']        = int(len(_acc) > 0)
            out_row['truth_n_accidental']          = int(len(_acc))
            out_row['truth_accidental_positron_t'] = float(min(_acc)) if _acc else -1000.0

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
    parser.add_argument("--mudif", type=str, default=None, help="Path to muDIF Parquet")
    parser.add_argument("--pidif", type=str, default=None, help="Path to piDIF Parquet")
    parser.add_argument("--output", type=str, required=True, help="Output mixed Parquet")
    parser.add_argument("--num_events", type=int, default=10)
    parser.add_argument("--mode", type=str, default='michel',
                        choices=['michel', 'pie', 'mudif', 'pidif', 'mixed'])
    parser.add_argument("--pie_mix_fraction", type=float, default=0.5, help="Fraction of pie main events in mixed mode")
    parser.add_argument("--mudif_mix_fraction", type=float, default=0.0, help="Fraction of muDIF main events in mixed mode")
    parser.add_argument("--pidif_mix_fraction", type=float, default=0.0, help="Fraction of piDIF main events in mixed mode")
    parser.add_argument("--trigger_gap_ns", type=float, default=2.0, help="Minimum allowed gap (ns) between pion stop and triggering positron")
    parser.add_argument("--biased_fraction", type=float, default=0.0, help="Fraction of events to use biased pileup (0.0=Off)")
    parser.add_argument("--biased_sigma", type=float, default=2.0, help="Gaussian spread for biased pileup separation (ns)")
    parser.add_argument("--cal_only_fraction", type=float, default=0.0, help="Fraction of events to use calorimeter-only biased pileup")
    parser.add_argument("--radio_rate", type=float, default=2e7, help="LYSO self-radioactivity rate (Hz)")
    parser.add_argument("--enforce_window", action='store_true', help="Retry sampling (up to 3x) if primary positron is outside window")
    parser.add_argument("--lut_dir", type=str, default=None,
                        help="Dir with crystalLUT.npy + validLUT.npy (radioactivity LUTs). "
                             "Overrides the built-in default (which is GPU-node-local).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the numpy RNG for a reproducible mix.")

    args = parser.parse_args()

    if args.biased_fraction + args.cal_only_fraction > 1.0:
        parser.error("The sum of --biased_fraction and --cal_only_fraction cannot exceed 1.0")

    if args.seed is not None:
        np.random.seed(args.seed & 0xFFFFFFFF)
        print(f"Seeded numpy RNG with {args.seed & 0xFFFFFFFF}")

    _mk = {} if args.lut_dir is None else {"lut_dir": args.lut_dir}
    mixer = PileupMixer(args.michel, args.pie, mudif_path=args.mudif, pidif_path=args.pidif, **_mk)
    mixed_df = mixer.generate_batch(args.num_events, mode=args.mode,
                                    biased_fraction=args.biased_fraction,
                                    biased_sigma=args.biased_sigma,
                                    cal_only_fraction=args.cal_only_fraction,
                                    radio_rate=args.radio_rate,
                                    enforce_window=args.enforce_window,
                                    pie_mix_fraction=args.pie_mix_fraction,
                                    mudif_mix_fraction=args.mudif_mix_fraction,
                                    pidif_mix_fraction=args.pidif_mix_fraction,
                                    trigger_gap_ns=args.trigger_gap_ns)
    
    print(f"Writing {args.num_events} mixed events to {args.output}")
    table = pa.Table.from_pandas(mixed_df)
    pq.write_table(table, args.output)
    print("Done!")
