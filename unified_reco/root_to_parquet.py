"""
Convert PIONEER ROOT simulation files to a flat per-event Parquet dataset.

Reads `sim` trees from one or many `.root` files, walks each event's
Geant4 tracks, time-bucket-merges deposits per ATAR strip / LYSO crystal,
and writes ragged columnar arrays to a Parquet file. Per-event scalars
include truth kinematics, the dead-material energy lost by the triggering
positron's lineage, the positron-only ATAR ionization, and the
sensitive-material total (live_E = atar_posE + lyso_E).

Usage:
    python root_to_parquet.py --input "/path/to/dir/or/glob/*.root" \\
                              --output /path/to/out.parquet \\
                              [--max_events N] [--max_files M] \\
                              [--shuffle_files] [--seed S]

Examples:
    # Single directory, 100k events, file-shuffled run:
    python root_to_parquet.py \\
        --input  /mnt/e/global_ai_recon/pie/train \\
        --output unmixed_pie_train.parquet \\
        --max_events 100000 --shuffle_files --seed 42

    # Glob pattern across multiple directories:
    python root_to_parquet.py --input "/data/run000*-*.root" \\
                              --output run000.parquet
"""
import ROOT
import sys
import os
import numpy as np
import glob
from collections import Counter
import random
import argparse
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# --- GLOBALS & CONSTANTS ---
kPidif = 0x0000000010

PION      = 0b000001
MUON      = 0b000010
POSITRON  = 0b000100
ELECTRON  = 0b001000
GAMMA     = 0b010000  # Added Gamma (22)
OTHER     = 0b100000

MASK_TO_PDG = {
    PION: 211, MUON: -13, POSITRON: -11, ELECTRON: 11, GAMMA: 22, OTHER: 98105
}

def pdg_to_mask(pdg_id):
    if pdg_id == 211: return PION
    elif pdg_id == -13: return MUON
    elif pdg_id == -11: return POSITRON
    elif pdg_id == 11:  return ELECTRON
    elif pdg_id == 22:  return GAMMA
    else:               return OTHER

# --- ENERGY STRIPPING & CLUSTERING UTILS ---
# Using the same gain suppression logic from `test_gainSatVar.py` for ATAR 
# and smearing logic from `calorimeter_clustering.py` for LYSO
GAIN_PARAMS = {
    #"gain": 13.564, 
    "gain": -1.0, 
    "k": 0.5,        # Birks constant
    "alpha": 1.0,    # Z dependence
    "a": 2.937,      # Saturation A
    "b": -0.239,     # Saturation B
    "z_offset": 0.065,
    "z_scale": 0.055,
    "min_s_angle": np.tan(np.radians(9.0)) * 1000.0
}

def apply_atar_gain_suppression(edep, deds, z, params):
    if params["gain"] <= 0: return edep
    term1 = np.minimum(1.0, (deds / 0.03)**10)
    z_fac = 1.0 - (z / params["z_scale"])
    term2 = np.zeros_like(z_fac)
    valid_z_mask = z_fac > 0
    term2[valid_z_mask] = z_fac[valid_z_mask] ** params["alpha"]
    kappaZ = 1.0 - params["k"] * term1 * term2
    kappaZ = np.nan_to_num(kappaZ, nan=1.0)
    quenched_edep = edep * kappaZ
    sat_calc = params["a"] * (np.abs(deds)**params["b"])
    saturation = np.where(deds > 0, sat_calc, 0.0)
    gain_factor = np.minimum(params["gain"], saturation) / params["gain"]
    return quenched_edep * gain_factor

def smear_atar(energy, energy_resolution=0.15):
    mask = energy > 0
    smeared_energy = np.zeros_like(energy, dtype=np.float64)
    valid_energies = energy[mask]
    stdv = valid_energies * energy_resolution
    noise = np.random.randn(len(valid_energies)) * stdv
    smeared_energy[mask] = valid_energies + noise
    return smeared_energy


PARQUET_NAMES = [
    'event_id',
    'truth_theta', 'truth_phi', 'truth_positron_energy',
    'truth_pion_start_x', 'truth_pion_start_y', 'truth_pion_start_z',
    'truth_pion_stop_x', 'truth_pion_stop_y', 'truth_pion_stop_z',
    'truth_muon_start_x', 'truth_muon_start_y', 'truth_muon_start_z',
    'truth_muon_stop_x', 'truth_muon_stop_y', 'truth_muon_stop_z',
    'truth_positron_start_x', 'truth_positron_start_y', 'truth_positron_start_z',
    'truth_positron_stop_x', 'truth_positron_stop_y', 'truth_positron_stop_z',
    'atar_x', 'atar_y', 'atar_z', 'atar_t', 'atar_truth_t', 'atar_E', 'atar_view', 'atar_pdg', 'atar_slice_id',
    'lyso_x', 'lyso_y', 'lyso_z', 'lyso_t', 'lyso_E', 'lyso_pdg',
    'dead_E',
    'atar_posE',
    'live_E',
]


def _make_accumulator():
    """Return a fresh dict of empty lists for event accumulation."""
    return {name: [] for name in PARQUET_NAMES}


def _build_table(acc):
    """Build a PyArrow Table from an accumulator dict."""
    arrays = []
    for name in PARQUET_NAMES:
        if name in ('dead_E', 'atar_posE', 'live_E'):
            arrays.append(pa.array(acc[name], type=pa.float64()))
        else:
            arrays.append(pa.array(acc[name]))
    return pa.Table.from_arrays(arrays, names=PARQUET_NAMES)


def process_root_file(file_list, geoheader, output_file, max_events=None, shard_size=100000):
    """
    Parses a list of ROOT files and directly flattens EVERY hit sequentially into a Parquet-ready dictionary/list.
    Flushes to disk every `shard_size` events to bound peak RAM.
    """
    chain = ROOT.TChain("sim")
    if isinstance(file_list, str):
        chain.Add(file_list)
    else:
        for f in file_list:
            print(f"Adding {f} to chain...")
            chain.Add(f)
    entries = chain.GetEntries()
    if max_events is not None:
        entries = min(entries, max_events)
        
    print(f"Total entries to process: {entries}")
    print(f"Shard size: {shard_size} events (flush to disk to bound RAM)")

    acc = _make_accumulator()
    writer = None
    schema = None
    n_written = 0

    def _flush():
        nonlocal writer, schema, n_written
        if len(acc['event_id']) == 0:
            return
        table = _build_table(acc)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(output_file, schema)
        else:
            table = table.cast(schema, safe=False)
        writer.write_table(table)
        n_written += len(acc['event_id'])
        print(f"    flushed {len(acc['event_id'])} events (total written: {n_written})", flush=True)
        for k in acc:
            acc[k].clear()

    for i, entry in tqdm(enumerate(chain), total=entries):
        if i >= entries: break
        
        # 1. Base Quality Skims
        eventType = int(entry.info.GetType())
        if eventType & kPidif:
            continue

        triggered = 0
        for upstream in entry.upstream:
            if upstream.GetVID() == 99999:
                if upstream.GetEdep() > 0.5:
                    triggered = 1
                    break
        if not triggered:
            continue

        # 2. Extract Event Truths (Decay Kinematics)
        nD = 0
        thetaInit, phiInit = -1000.0, -1000.0
        positron_initial_energy = 0.0
        for decay in entry.decay:
            nD = decay.GetNDaughters()
            if nD == 3: # Michel
                mom = decay.GetDaughterMomAt(0)
                thetaInit = mom.Theta()
                phiInit = mom.Phi()
                positron_initial_energy = np.sqrt(mom.Mag2())
                break
            elif nD == 2: # Pi-E
                if decay.GetDaughterPDGIDAt(0) == -13: continue
                mom = decay.GetDaughterMomAt(0)
                thetaInit = mom.Theta()
                phiInit = mom.Phi()
                positron_initial_energy = np.sqrt(mom.Mag2())
                break
                
        if (thetaInit < 0) or (phiInit == -1000):
            continue
        #if np.degrees(thetaInit) > 130:
        #    continue

        # Temporary dictionaries for temporal merging within this event
        event_hits_atar = {} # vol_id -> list of dicts: {'x':, 'y':, 'z':, 't':, 'E':, 'view':, 'pdg':}
        event_hits_lyso = {} # vol_id -> list of dicts

        # Truth (unsmeared) LYSO energy per crystal, accumulated across all tracks.
        # Used in pass 4 to subtract from GetTotalEnergyDeposit so the residual
        # captures only genuinely untracked deposits (e.g. gammas without Geant4
        # tracks), not smearing artifacts.
        lyso_truth_per_crystal = {}

        # Accumulated dead-material energy for this event (MeV).
        evt_dead_E = 0.0

        # Positron-only ATAR energy accumulator (summed across positron tracks).
        evt_atar_posE = 0.0

        pionStopX, pionStopY, pionStopZ = 0.0, 0.0, 0.0
        endpoints = {
            211: {'start': [np.nan]*3, 'stop': [np.nan]*3},
            -13: {'start': [np.nan]*3, 'stop': [np.nan]*3},
            -11: {'start': [np.nan]*3, 'stop': [np.nan]*3}
        }

        # Pre-pass: build track ancestry so we can classify deposits by lineage.
        # dead_E attribution should include the signal positron and its descendants
        # (brem γ, pair e±) but exclude pion/muon tracks AND their descendants
        # (δ-rays, nuclear fragments). PDG alone doesn't distinguish a pion δ-ray
        # electron from a positron-shower electron.
        track_parent = {}
        track_pdg    = {}
        for t in entry.track:
            tid = t.GetTrackID()
            track_parent[tid] = t.GetParentID()
            track_pdg[tid]    = t.GetPDGID()

        # Find the signal positron: smallest trackID positron (first created).
        positron_tid = None
        for tid, pdg_i in track_pdg.items():
            if pdg_i == -11 and (positron_tid is None or tid < positron_tid):
                positron_tid = tid

        # Memoized ancestry: is `tid` the positron or a descendant?
        from_positron = {}
        def _is_from_positron(tid):
            if tid in from_positron: return from_positron[tid]
            if positron_tid is None:
                from_positron[tid] = False
                return False
            cur = tid
            visited = []
            while cur in track_parent:
                if cur == positron_tid:
                    result = True
                    break
                visited.append(cur)
                p = track_parent[cur]
                if p == 0 or p == cur:
                    result = False
                    break
                cur = p
            else:
                result = False
            for v in visited + [tid]:
                from_positron[v] = result
            return result

        # 3. Track Hit Processing
        for track in entry.track:
            pdg = track.GetPDGID()
            tid = track.GetTrackID()
            track_from_positron = _is_from_positron(tid)
            # Allow all particles to deposit energy (e.g. Gammas=22). Unrecognized PDGs get mapped to OTHER bitmask.
            
            # Accessors
            post_x = np.frombuffer(track.GetPostX().data(), dtype=np.float32, count=track.GetPostX().size())
            post_y = np.frombuffer(track.GetPostY().data(), dtype=np.float32, count=track.GetPostY().size())
            post_z = np.frombuffer(track.GetPostZ().data(), dtype=np.float32, count=track.GetPostZ().size())
            post_t = np.frombuffer(track.GetPostTime().data(), dtype=np.float32, count=track.GetPostTime().size())
            edep_vec = np.frombuffer(track.GetEdep().data(), dtype=np.float32, count=track.GetEdep().size())
            volumes = np.frombuffer(track.GetVolume().data(), dtype=np.int32, count=track.GetVolume().size())
            
            if len(post_x) > 0 and pdg in endpoints: 
                start_pt = [float(post_x[0]), float(post_y[0]), float(post_z[0])]
                stop_pt = [float(post_x[-1]), float(post_y[-1]), float(post_z[-1])]
                
                # Geant tracks might be fragmented, only record first point if empty
                if np.isnan(endpoints[pdg]['start'][0]):
                    endpoints[pdg]['start'] = start_pt
                
                # Continually update stop backwards so it catches the true end coordinate
                endpoints[pdg]['stop'] = stop_pt
                
                if pdg == 211:
                    pionStopX, pionStopY, pionStopZ = stop_pt

            # 1. Geometry Mask
            unique_vols, inverse = np.unique(volumes, return_inverse=True)
            
            is_atar_lookup = np.array([geoheader.GetDetectorType(int(v)) == ROOT.PIDetectorType.kAtar for v in unique_vols], dtype=bool)
            is_lyso_lookup = np.array([geoheader.GetDetectorType(int(v)) == ROOT.PIDetectorType.kCalo for v in unique_vols], dtype=bool)

            # Mask out empty steps early and apply strict logical digitization readout window
            mask_energy = (edep_vec > 1e-4) & (post_t >= 0) & (post_t <= 10000)
            inv_valid = inverse[mask_energy]
            edep_valid = edep_vec[mask_energy]

            if len(edep_valid) == 0: continue

            # Energy-weighted centroids per volume (for this specific track)
            sum_e = np.bincount(inv_valid, weights=edep_valid, minlength=len(unique_vols))
            safe_div = np.where(sum_e == 0, 1.0, sum_e)

            # Dead-material energy: volumes that are neither kAtar nor kCalo.
            # Include only tracks whose ancestry traces to the signal positron
            # (positron itself + brem γ + pair products). Pion/muon and their
            # δ-ray descendants are excluded.
            is_dead_lookup = ~(is_atar_lookup | is_lyso_lookup)
            if is_dead_lookup.any() and track_from_positron:
                evt_dead_E += float(sum_e[is_dead_lookup].sum())

            avg_x = np.bincount(inv_valid, weights=post_x[mask_energy] * edep_valid, minlength=len(unique_vols)) / safe_div
            avg_y = np.bincount(inv_valid, weights=post_y[mask_energy] * edep_valid, minlength=len(unique_vols)) / safe_div
            avg_z = np.bincount(inv_valid, weights=post_z[mask_energy] * edep_valid, minlength=len(unique_vols)) / safe_div
            avg_t = np.bincount(inv_valid, weights=post_t[mask_energy] * edep_valid, minlength=len(unique_vols)) / safe_div
            
            # Store true unsmeared times before any smearing modifications
            truth_t = avg_t.copy()

            valid_indices = np.where(sum_e > 1e-4)[0]

            atar_indices = valid_indices[is_atar_lookup[valid_indices]]
            lyso_indices = valid_indices[is_lyso_lookup[valid_indices]]

            # Accumulate truth (unsmeared) LYSO energy per crystal BEFORE smearing.
            # This is used in pass 4 to compute residuals against GetTotalEnergyDeposit.
            for idx in lyso_indices:
                v_id = int(unique_vols[idx])
                lyso_truth_per_crystal[v_id] = lyso_truth_per_crystal.get(v_id, 0.0) + float(sum_e[idx])

            # Positron-only ATAR energy: attribute this track's ATAR deposits
            # to positron accumulator iff the track is a positron (pdg -11).
            if pdg == -11 and len(atar_indices) > 0:
                evt_atar_posE += float(sum_e[atar_indices].sum())

            # Legacy Physical Smearing Validation (ATAR uses flat resolution)
            if len(atar_indices) > 0:
                avg_t[atar_indices] += np.random.normal(0, 0.2, size=len(atar_indices))
                sum_e[atar_indices] = smear_atar(sum_e[atar_indices], 0.15)
                
            # Physical Smearing (LYSO uses empirical energy-dependent resolutions).
            # σ_E/E = √((a/√E)² + (b/E)² + c²) with a=6, b=50, c=1.2 (in %)
            # σ_t   = √((a/E)²   + (b/√E)² + c²) with a=300, b=600, c=75 (in ps)
            #
            # Three stages:
            #   1. Pre-cut at LYSO_PRE_SMEAR_E — drop deposits so small that even
            #      a 1σ up-fluctuation couldn't trigger, AND avoid the pathological
            #      σ_t / σ_E divergence at sub-keV truth deposits.
            #   2. Apply σ_E and σ_t smearing using TRUTH energy. Both σs scale
            #      with photon statistics N ∝ truth E, so smearing must be
            #      parametrised by truth, not by the σ_E-noisy "measured" E.
            #   3. Apply the trigger threshold LYSO_TRIG_E to the *smeared*
            #      energy, mimicking what a real digitizer does.
            LYSO_PRE_SMEAR_E = 0.05   # MeV — coarse "real deposit" sanity floor
            LYSO_TRIG_E      = 0.20   # MeV — digitizer trigger threshold (post-smear)
            if len(lyso_indices) > 0:
                keep = sum_e[lyso_indices] > LYSO_PRE_SMEAR_E
                lyso_indices = lyso_indices[keep]

            if len(lyso_indices) > 0:
                lyso_E_truth = sum_e[lyso_indices].copy()   # captured before smearing; used by σ_t

                # Energy smear. σ_E/E from the formula diverges at low E (b/E
                # term); cap at 10% so smeared hits remain physically interpretable
                # and so a single noise draw can't inflate or deflate by orders
                # of magnitude.
                res_frac_raw = np.sqrt((6 / np.sqrt(lyso_E_truth))**2
                                       + (50 / lyso_E_truth)**2
                                       + 1.2**2) / 100.0
                res_frac = np.minimum(res_frac_raw, 0.10)
                sum_e[lyso_indices] = np.maximum(
                    sum_e[lyso_indices] * (1 + np.random.normal(0, res_frac)),
                    0.0,
                )

                # Time smear (ps → ns) using TRUTH energy (photon-statistics-limited
                # timing depends on the real photon count N ∝ E_truth, not on the
                # σ_E-noisy "measured" E — using smeared E would double-count).
                sigma_t = np.sqrt((300 / lyso_E_truth)**2
                                  + (600 / np.sqrt(lyso_E_truth))**2
                                  + 75**2) / 1000.0
                avg_t[lyso_indices] += np.random.normal(0, sigma_t)

                # Trigger threshold applied to smeared (measured) E. Hits whose
                # smeared signal falls below trigger don't get recorded.
                survive = sum_e[lyso_indices] > LYSO_TRIG_E
                lyso_indices = lyso_indices[survive]

            def merge_hit(hit_list, new_hit, merge_window):
                for hit in hit_list:
                    if abs(hit['t'] - new_hit['t']) < merge_window:
                        # Energy-weighted merge
                        e_tot = hit['E'] + new_hit['E']
                        if e_tot > 0:
                            f_old, f_new = hit['E'] / e_tot, new_hit['E'] / e_tot
                            hit['x'] = hit['x'] * f_old + new_hit['x'] * f_new
                            hit['y'] = hit['y'] * f_old + new_hit['y'] * f_new
                            hit['z'] = hit['z'] * f_old + new_hit['z'] * f_new
                            hit['truth_t'] = hit['truth_t'] * f_old + new_hit['truth_t'] * f_new
                            # Energy-weight the smeared time too. Was previously
                            # frozen as the first-in track's smeared time, which
                            # corrupted high-E merged hits when a low-E precursor
                            # (with much larger σ_t under the smearing formula)
                            # was processed first.
                            hit['t'] = hit['t'] * f_old + new_hit['t'] * f_new
                        hit['E'] = e_tot
                        # Bitwise OR for PDGs if multiple particles hit same strip AT EXACT SAME TIME
                        hit['pdg_mask'] |= new_hit['pdg_mask']
                        return
                hit_list.append(new_hit)

            for idx in valid_indices:
                v_id = int(unique_vols[idx])
                ax, ay, az, at, se = avg_x[idx], avg_y[idx], avg_z[idx], avg_t[idx], sum_e[idx]
                pdg_mask = pdg_to_mask(pdg)
                
                # ATAR PROCESSING
                if is_atar_lookup[idx]:
                    # Derive view from GeoHeader rotation: unrotated (psi≈0) → XZ (0),
                    # rotated (psi≈-π/4) → YZ (1). Replaces old AtarHeader.GetChannel().
                    strip_orientation = 0 if abs(geoheader.GetPsi(v_id)) < 0.1 else 1
                    
                    new_hit = {'x': float(ax), 'y': float(ay), 'z': float(az), 't': max(float(at), 0.05), 'truth_t': max(float(truth_t[idx]), 0.05), 'E': float(se), 'view': int(strip_orientation), 'pdg_mask': pdg_mask}
                    if v_id not in event_hits_atar: event_hits_atar[v_id] = []
                    merge_hit(event_hits_atar[v_id], new_hit, merge_window=2.0)
                    
                # LYSO CALORIMETER PROCESSING
                elif is_lyso_lookup[idx]:
                    new_hit = {'x': float(ax), 'y': float(ay), 'z': float(az), 't': float(at), 'truth_t': float(truth_t[idx]), 'E': float(se), 'pdg_mask': pdg_mask}
                    if v_id not in event_hits_lyso: event_hits_lyso[v_id] = []
                    merge_hit(event_hits_lyso[v_id], new_hit, merge_window=10.0)

        # 4. Calorimeter Exclusive Hit Processing (e.g. Gammas)
        # Uncharged particles often don't leave Geant4 'tracks' but DO deposit in entry.calo.
        # crystal.GetTotalEnergyDeposit() returns the FULL deposit for the crystal, including
        # contributions from tracked particles already added in pass 3. Subtract the TRUTH
        # (unsmeared) energy so the residual captures only genuinely untracked deposits
        # (gammas, etc.). Using smeared energy here would create a one-sided bias: downward
        # smearing fluctuations produce positive residuals that get re-added, while upward
        # fluctuations produce negative residuals that are skipped — systematically
        # inflating the total energy.
        for crystal in entry.calo:
            edep_total = crystal.GetTotalEnergyDeposit()
            if edep_total < 1e-4: continue

            c_id = int(crystal.GetCaloID())
            edep_residual = edep_total - lyso_truth_per_crystal.get(c_id, 0.0)
            if edep_residual < 1e-4: continue  # fully accounted for by tracked hits

            # Replicate temporal mask cut and explicitly clip for log-scale plotting
            t = max(crystal.GetTime()[0], 0.05)
            if t > 10000: continue

            pdg = crystal.GetPDGID()
            pdg_mask = pdg_to_mask(pdg)

            xyz = geoheader.GetCentre(c_id)
            # Use same SCALE_FACTOR (1.19) from old script to map front-face to shower max
            x, y, z = xyz.X() * 1.19, xyz.Y() * 1.19, xyz.Z() * 1.19

            new_hit = {'x': float(x), 'y': float(y), 'z': float(z),
                       't': float(t), 'truth_t': float(t),
                       'E': float(edep_residual), 'pdg_mask': pdg_mask}

            if c_id not in event_hits_lyso:
                event_hits_lyso[c_id] = []
            merge_hit(event_hits_lyso[c_id], new_hit, merge_window=10.0)

        # 5. Flatten Merged Hits into Event Lists & Calculate Time Slices
        evt_atar_x, evt_atar_y, evt_atar_z, evt_atar_E, evt_atar_t, evt_atar_truth_t, evt_atar_view, evt_atar_pdg = [], [], [], [], [], [], [], []
        for v_id, hits in event_hits_atar.items():
            for h in hits:
                evt_atar_x.append(h['x'])
                evt_atar_y.append(h['y'])
                evt_atar_z.append(h['z'])
                evt_atar_t.append(h['t'])
                evt_atar_truth_t.append(h['truth_t'])
                evt_atar_E.append(h['E'])
                evt_atar_view.append(h['view'])
                evt_atar_pdg.append(h['pdg_mask'])

        evt_lyso_x, evt_lyso_y, evt_lyso_z, evt_lyso_E, evt_lyso_t, evt_lyso_pdg = [], [], [], [], [], []
        for v_id, hits in event_hits_lyso.items():
            for h in hits:
                evt_lyso_x.append(h['x'])
                evt_lyso_y.append(h['y'])
                evt_lyso_z.append(h['z'])
                evt_lyso_t.append(h['t'])
                evt_lyso_E.append(h['E'])
                evt_lyso_pdg.append(h['pdg_mask'])

        # Algorithm: Highest-Energy Seeded Time Slicing (Global across ATAR and LYSO)
        evt_atar_slice = [-1] * len(evt_atar_t)
        evt_lyso_slice = [-1] * len(evt_lyso_t)
        
        all_times = np.concatenate([evt_atar_t, evt_lyso_t]) if len(evt_atar_t) + len(evt_lyso_t) > 0 else np.array([])
        all_energies = np.concatenate([evt_atar_E, evt_lyso_E]) if len(evt_atar_t) + len(evt_lyso_t) > 0 else np.array([])
        is_lyso = np.concatenate([np.zeros(len(evt_atar_t), dtype=bool), np.ones(len(evt_lyso_t), dtype=bool)]) if len(all_times) > 0 else np.array([])
        
        if len(all_times) > 0:
            unassigned_mask = np.ones(len(all_times), dtype=bool)
            centers = []
            
            # Phase 1: Identify all slice centers (seed times)
            while np.any(unassigned_mask):
                valid_idx = np.where(unassigned_mask)[0]
                max_e_idx = valid_idx[np.argmax(all_energies[valid_idx])]
                seed_time = all_times[max_e_idx]
                centers.append(seed_time)
                
                # Mark hits within the specific detector windows as assigned
                # ATAR window: +/- 1.0 ns, LYSO window: +/- 2.0 ns
                # NOTE: The user requested OR (10ns for LYSO and 2ns for ATAR) in the same volume, 
                # but since we already merged per-volume using `merge_hit` with 2ns/10ns windows respectively, 
                # the global slicer should strictly use 1ns/2ns bounds across the whole topology.
                windows = np.where(is_lyso, 2.0, 1.0)
                in_window = np.abs(all_times - seed_time) <= windows
                unassigned_mask[unassigned_mask & in_window] = False
                
            # Phase 2: Assign hits to the closest valid center
            atar_len = len(evt_atar_t)
            for i in range(len(all_times)):
                hit_time = all_times[i]
                hit_is_lyso = is_lyso[i]
                window = 2.0 if hit_is_lyso else 1.0
                
                valid_centers = []
                for c_idx, c_time in enumerate(centers):
                    if abs(hit_time - c_time) <= window:
                        valid_centers.append(c_idx)
                        
                if not valid_centers:
                    slice_id = -1
                else:
                    # Dispute resolution: assign to the time slice closer in time
                    best_center = valid_centers[0]
                    min_dist = abs(hit_time - centers[best_center])
                    for c_idx in valid_centers[1:]:
                        dist = abs(hit_time - centers[c_idx])
                        if dist < min_dist:
                            min_dist = dist
                            best_center = c_idx
                    slice_id = best_center
                    
                if i < atar_len:
                    evt_atar_slice[i] = slice_id
                else:
                    evt_lyso_slice[i - atar_len] = slice_id

        # We append regardless of length to maintain strict 1:1 event indexing
        acc['event_id'].append(i)
        
        acc['atar_x'].append(evt_atar_x)
        acc['atar_y'].append(evt_atar_y)
        acc['atar_z'].append(evt_atar_z)
        acc['atar_E'].append(evt_atar_E)
        acc['atar_t'].append(evt_atar_t)
        acc['atar_truth_t'].append(evt_atar_truth_t)
        acc['atar_view'].append(evt_atar_view)
        acc['atar_pdg'].append(evt_atar_pdg)
        acc['atar_slice_id'].append(evt_atar_slice)

        acc['lyso_x'].append(evt_lyso_x)
        acc['lyso_y'].append(evt_lyso_y)
        acc['lyso_z'].append(evt_lyso_z)
        acc['lyso_E'].append(evt_lyso_E)
        acc['lyso_t'].append(evt_lyso_t)
        acc['lyso_pdg'].append(evt_lyso_pdg)

        acc['dead_E'].append(float(evt_dead_E))

        # Positron-only ATAR energy; live_E = positron ATAR + total LYSO.
        evt_lyso_E_sum = float(sum(evt_lyso_E)) if len(evt_lyso_E) > 0 else 0.0
        acc['atar_posE'].append(float(evt_atar_posE))
        acc['live_E'].append(float(evt_atar_posE + evt_lyso_E_sum))

        acc['truth_theta'].append(float(thetaInit))
        acc['truth_phi'].append(float(phiInit))
        acc['truth_positron_energy'].append(float(positron_initial_energy))

        for p_key, start_keys, stop_keys in [
            (211,
             ('truth_pion_start_x', 'truth_pion_start_y', 'truth_pion_start_z'),
             ('truth_pion_stop_x', 'truth_pion_stop_y', 'truth_pion_stop_z')),
            (-13,
             ('truth_muon_start_x', 'truth_muon_start_y', 'truth_muon_start_z'),
             ('truth_muon_stop_x', 'truth_muon_stop_y', 'truth_muon_stop_z')),
            (-11,
             ('truth_positron_start_x', 'truth_positron_start_y', 'truth_positron_start_z'),
             ('truth_positron_stop_x', 'truth_positron_stop_y', 'truth_positron_stop_z')),
        ]:
            for idx, k in enumerate(start_keys):
                acc[k].append(endpoints[p_key]['start'][idx])
            for idx, k in enumerate(stop_keys):
                acc[k].append(endpoints[p_key]['stop'][idx])

        # Flush shard to disk when accumulator reaches shard_size.
        if len(acc['event_id']) >= shard_size:
            _flush()

    # Flush any remaining events.
    _flush()
    if writer is not None:
        writer.close()
    print(f"Done! Wrote {n_written} events to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PIONEER ROOT sim files to flattened Parquet datasets.")
    parser.add_argument("--input", type=str, required=True, help="Input ROOT file path or directory (glob supported)")
    parser.add_argument("--output", type=str, required=True, help="Output Parquet file path")
    parser.add_argument("--max_events", type=int, default=None, help="Max total events to process across all files")
    parser.add_argument("--max_files", type=int, default=None, help="Max number of ROOT files to process in batch mode")
    parser.add_argument("--shuffle_files", action='store_true', help="Randomize ROOT file ordering before chaining")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --shuffle_files (reproducible)")
    parser.add_argument("--shard_size", type=int, default=100000,
                        help="Flush to disk every N events to bound RAM (default 100k)")
    args = parser.parse_args()

    # 1. Resolve Input Files
    input_path = args.input
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, "*.root")

    file_list = sorted(glob.glob(input_path))
    if len(file_list) == 0:
        print(f"Error: No matching ROOT files found for pattern: {input_path}")
        sys.exit(1)

    if args.shuffle_files:
        rng = random.Random(args.seed)
        rng.shuffle(file_list)
        print(f"Shuffled {len(file_list)} files (seed={args.seed})")

    if args.max_files is not None:
        print(f"Capping input at {args.max_files} files (Found {len(file_list)})")
        file_list = file_list[:args.max_files]

    # 2. Load Layout Dependencies from the FIRST file
    print(f"Extracting headers from {file_list[0]}...")
    layout_file = ROOT.TFile(file_list[0])
    geo_header = layout_file.Get("GeoHeader")

    if not geo_header:
        print("Warning: GeoHeader missing from first file. Metadata might be incomplete.")

    # 3. Process
    process_root_file(file_list, geo_header, args.output, max_events=args.max_events, shard_size=args.shard_size)
