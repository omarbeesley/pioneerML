"""
Validate ROOT-to-Parquet conversion by replaying the root_to_parquet logic
with energy conservation checks. Only prints problems.

For single-particle Geant4, total deposited energy (ATAR + LYSO + dead)
must be <= positron kinetic energy + 1.022 MeV (annihilation).

Checks:
  1. Energy conservation: live_E + dead_E vs positron initial energy
  2. Volume ID consistency: GetVolume() vs GetCaloID() for LYSO crystals
  3. Pass 3 vs pass 4 accounting: tracked LYSO vs residual LYSO per crystal
  4. Smearing-induced residual bias: sum of positive residuals

Usage:
    python validate_root_to_parquet.py \
        --input /path/to/root/files/ \
        --max_events 100000
"""

import ROOT
import sys
import os
import numpy as np
import glob
import argparse
from tqdm import tqdm
from collections import defaultdict

# --- Constants (copied from root_to_parquet.py) ---
kPidif = 0x0000000010

PION     = 0b000001
MUON     = 0b000010
POSITRON = 0b000100
ELECTRON = 0b001000
GAMMA    = 0b010000
OTHER    = 0b100000

def pdg_to_mask(pdg_id):
    if pdg_id == 211: return PION
    elif pdg_id == -13: return MUON
    elif pdg_id == -11: return POSITRON
    elif pdg_id == 11:  return ELECTRON
    elif pdg_id == 22:  return GAMMA
    else:               return OTHER

GAIN_PARAMS = {
    "gain": -1.0,
    "k": 0.5,
    "alpha": 1.0,
    "a": 2.937,
    "b": -0.239,
    "z_offset": 0.065,
    "z_scale": 0.055,
    "min_s_angle": np.tan(np.radians(9.0)) * 1000.0
}

def smear_atar(energy, energy_resolution=0.15):
    mask = energy > 0
    smeared_energy = np.zeros_like(energy, dtype=np.float64)
    valid_energies = energy[mask]
    stdv = valid_energies * energy_resolution
    noise = np.random.randn(len(valid_energies)) * stdv
    smeared_energy[mask] = valid_energies + noise
    return smeared_energy

def merge_hit(hit_list, new_hit, merge_window):
    for hit in hit_list:
        if abs(hit['t'] - new_hit['t']) < merge_window:
            e_tot = hit['E'] + new_hit['E']
            if e_tot > 0:
                f_old, f_new = hit['E'] / e_tot, new_hit['E'] / e_tot
                hit['x'] = hit['x'] * f_old + new_hit['x'] * f_new
                hit['y'] = hit['y'] * f_old + new_hit['y'] * f_new
                hit['z'] = hit['z'] * f_old + new_hit['z'] * f_new
                hit['truth_t'] = hit['truth_t'] * f_old + new_hit['truth_t'] * f_new
                hit['t'] = hit['t'] * f_old + new_hit['t'] * f_new
            hit['E'] = e_tot
            hit['pdg_mask'] |= new_hit['pdg_mask']
            return
    hit_list.append(new_hit)


def validate_file(file_list, geoheader, max_events=None):
    """Replay root_to_parquet logic with validation checks."""

    chain = ROOT.TChain("sim")
    if isinstance(file_list, str):
        chain.Add(file_list)
    else:
        for f in file_list:
            chain.Add(f)
    entries = chain.GetEntries()
    if max_events is not None:
        entries = min(entries, max_events)

    print(f"Validating {entries} events...")

    # Counters
    n_processed = 0
    n_skipped = 0
    n_energy_violation = 0
    n_id_mismatch = 0
    n_large_residual = 0
    n_residual_bias = 0

    # Check volume ID mapping once
    id_mapping_checked = False
    vol_to_calo_map = {}

    LYSO_PRE_SMEAR_E = 0.05
    LYSO_TRIG_E = 0.20

    for i, entry in tqdm(enumerate(chain), total=entries):
        if i >= entries:
            break

        # 1. Same skims as root_to_parquet
        eventType = int(entry.info.GetType())
        if eventType & kPidif:
            n_skipped += 1
            continue

        triggered = 0
        for upstream in entry.upstream:
            if upstream.GetVID() == 99999:
                if upstream.GetEdep() > 0.5:
                    triggered = 1
                    break
        if not triggered:
            n_skipped += 1
            continue

        # 2. Extract positron energy
        positron_initial_energy = 0.0
        for decay in entry.decay:
            nD = decay.GetNDaughters()
            if nD == 3:
                mom = decay.GetDaughterMomAt(0)
                positron_initial_energy = np.sqrt(mom.Mag2())
                break
            elif nD == 2:
                if decay.GetDaughterPDGIDAt(0) == -13:
                    continue
                mom = decay.GetDaughterMomAt(0)
                positron_initial_energy = np.sqrt(mom.Mag2())
                break

        if positron_initial_energy <= 0:
            n_skipped += 1
            continue

        # 3. Build ancestry
        track_parent = {}
        track_pdg = {}
        for t in entry.track:
            tid = t.GetTrackID()
            track_parent[tid] = t.GetParentID()
            track_pdg[tid] = t.GetPDGID()

        positron_tid = None
        for tid, pdg_i in track_pdg.items():
            if pdg_i == -11 and (positron_tid is None or tid < positron_tid):
                positron_tid = tid

        from_positron = {}
        def _is_from_positron(tid):
            if tid in from_positron:
                return from_positron[tid]
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

        # 4. Process tracks (same as pass 3 in root_to_parquet)
        event_hits_atar = {}
        event_hits_lyso = {}
        evt_dead_E = 0.0
        evt_atar_posE = 0.0

        # Track truth energy per LYSO crystal (unsmeared, pre-threshold)
        truth_E_per_crystal = defaultdict(float)
        # Track pass-3 smeared energy per crystal (post-threshold)
        tracked_smeared_per_crystal = defaultdict(float)

        for track in entry.track:
            pdg = track.GetPDGID()
            tid = track.GetTrackID()
            track_from_positron = _is_from_positron(tid)

            post_x = np.frombuffer(track.GetPostX().data(), dtype=np.float32, count=track.GetPostX().size())
            post_y = np.frombuffer(track.GetPostY().data(), dtype=np.float32, count=track.GetPostY().size())
            post_z = np.frombuffer(track.GetPostZ().data(), dtype=np.float32, count=track.GetPostZ().size())
            post_t = np.frombuffer(track.GetPostTime().data(), dtype=np.float32, count=track.GetPostTime().size())
            edep_vec = np.frombuffer(track.GetEdep().data(), dtype=np.float32, count=track.GetEdep().size())
            volumes = np.frombuffer(track.GetVolume().data(), dtype=np.int32, count=track.GetVolume().size())

            unique_vols, inverse = np.unique(volumes, return_inverse=True)

            is_atar_lookup = np.array([
                geoheader.GetDetectorType(int(v)) == ROOT.PIDetectorType.kAtar
                for v in unique_vols], dtype=bool)
            is_lyso_lookup = np.array([
                geoheader.GetDetectorType(int(v)) == ROOT.PIDetectorType.kCalo
                for v in unique_vols], dtype=bool)

            mask_energy = (edep_vec > 1e-4) & (post_t >= 0) & (post_t <= 10000)
            inv_valid = inverse[mask_energy]
            edep_valid = edep_vec[mask_energy]

            if len(edep_valid) == 0:
                continue

            sum_e = np.bincount(inv_valid, weights=edep_valid, minlength=len(unique_vols))
            safe_div = np.where(sum_e == 0, 1.0, sum_e)

            is_dead_lookup = ~(is_atar_lookup | is_lyso_lookup)
            if is_dead_lookup.any() and track_from_positron:
                evt_dead_E += float(sum_e[is_dead_lookup].sum())

            avg_x = np.bincount(inv_valid, weights=post_x[mask_energy] * edep_valid, minlength=len(unique_vols)) / safe_div
            avg_y = np.bincount(inv_valid, weights=post_y[mask_energy] * edep_valid, minlength=len(unique_vols)) / safe_div
            avg_z = np.bincount(inv_valid, weights=post_z[mask_energy] * edep_valid, minlength=len(unique_vols)) / safe_div
            avg_t = np.bincount(inv_valid, weights=post_t[mask_energy] * edep_valid, minlength=len(unique_vols)) / safe_div
            truth_t = avg_t.copy()

            valid_indices = np.where(sum_e > 1e-4)[0]
            atar_indices = valid_indices[is_atar_lookup[valid_indices]]
            lyso_indices = valid_indices[is_lyso_lookup[valid_indices]]

            if pdg == -11 and len(atar_indices) > 0:
                evt_atar_posE += float(sum_e[atar_indices].sum())

            # Record truth (unsmeared) energy per LYSO crystal from this track
            for idx in lyso_indices:
                v_id = int(unique_vols[idx])
                truth_E_per_crystal[v_id] += float(sum_e[idx])

            # ATAR smearing
            if len(atar_indices) > 0:
                avg_t[atar_indices] += np.random.normal(0, 0.2, size=len(atar_indices))
                sum_e[atar_indices] = smear_atar(sum_e[atar_indices], 0.15)

            # LYSO smearing + threshold (same as root_to_parquet)
            if len(lyso_indices) > 0:
                keep = sum_e[lyso_indices] > LYSO_PRE_SMEAR_E
                lyso_indices = lyso_indices[keep]

            if len(lyso_indices) > 0:
                lyso_E_truth = sum_e[lyso_indices].copy()
                res_frac_raw = np.sqrt((6 / np.sqrt(lyso_E_truth))**2
                                       + (50 / lyso_E_truth)**2
                                       + 1.2**2) / 100.0
                res_frac = np.minimum(res_frac_raw, 0.10)
                sum_e[lyso_indices] = np.maximum(
                    sum_e[lyso_indices] * (1 + np.random.normal(0, res_frac)), 0.0)

                survive = sum_e[lyso_indices] > LYSO_TRIG_E
                lyso_indices = lyso_indices[survive]

            # Merge surviving hits (same as root_to_parquet)
            for idx in valid_indices:
                v_id = int(unique_vols[idx])
                if is_lyso_lookup[idx] and idx in lyso_indices:
                    new_hit = {
                        'x': float(avg_x[idx]), 'y': float(avg_y[idx]),
                        'z': float(avg_z[idx]), 't': float(avg_t[idx]),
                        'truth_t': float(truth_t[idx]),
                        'E': float(sum_e[idx]), 'pdg_mask': pdg_to_mask(pdg)}
                    if v_id not in event_hits_lyso:
                        event_hits_lyso[v_id] = []
                    merge_hit(event_hits_lyso[v_id], new_hit, merge_window=10.0)
                elif is_atar_lookup[idx]:
                    new_hit = {
                        'x': float(avg_x[idx]), 'y': float(avg_y[idx]),
                        'z': float(avg_z[idx]), 't': max(float(avg_t[idx]), 0.05),
                        'truth_t': max(float(truth_t[idx]), 0.05),
                        'E': float(sum_e[idx]), 'view': 0, 'pdg_mask': pdg_to_mask(pdg)}
                    if v_id not in event_hits_atar:
                        event_hits_atar[v_id] = []
                    merge_hit(event_hits_atar[v_id], new_hit, merge_window=2.0)

        # Record tracked (smeared) energy per LYSO crystal
        for c_id, hits in event_hits_lyso.items():
            tracked_smeared_per_crystal[c_id] = sum(h['E'] for h in hits)

        # 5. Pass 4: Calorimeter exclusive processing (residuals)
        tracked_per_crystal = {
            c_id: sum(h['E'] for h in hits)
            for c_id, hits in event_hits_lyso.items()
        }

        residual_total = 0.0
        residual_details = []

        for crystal in entry.calo:
            edep_total = crystal.GetTotalEnergyDeposit()
            if edep_total < 1e-4:
                continue

            c_id = int(crystal.GetCaloID())

            # === CHECK 2: Volume ID vs Calo ID consistency ===
            if not id_mapping_checked:
                # Check if any crystal's CaloID matches a volume ID from pass 3
                if c_id in event_hits_lyso:
                    vol_to_calo_map[c_id] = c_id  # same
                elif c_id not in truth_E_per_crystal:
                    # c_id from GetCaloID() not seen in pass 3's GetVolume() IDs
                    # But it might just be a crystal with no tracked hits
                    pass

            tracked_E = tracked_per_crystal.get(c_id, 0.0)
            edep_residual = edep_total - tracked_E

            if edep_residual < 1e-4:
                continue

            residual_total += edep_residual
            residual_details.append({
                'calo_id': c_id,
                'total_deposit': edep_total,
                'tracked': tracked_E,
                'residual': edep_residual,
                'in_pass3': c_id in tracked_per_crystal,
            })

            # Merge residual hit (same as root_to_parquet)
            t = max(crystal.GetTime()[0], 0.05)
            if t > 10000:
                continue
            pdg = crystal.GetPDGID()
            pdg_mask = pdg_to_mask(pdg)
            xyz = geoheader.GetCentre(c_id)
            x, y, z = xyz.X() * 1.19, xyz.Y() * 1.19, xyz.Z() * 1.19
            new_hit = {
                'x': float(x), 'y': float(y), 'z': float(z),
                't': float(t), 'truth_t': float(t),
                'E': float(edep_residual), 'pdg_mask': pdg_mask}
            if c_id not in event_hits_lyso:
                event_hits_lyso[c_id] = []
            merge_hit(event_hits_lyso[c_id], new_hit, merge_window=10.0)

        id_mapping_checked = True

        # 6. Compute final energies
        evt_lyso_E_sum = sum(
            h['E'] for hits in event_hits_lyso.values() for h in hits)
        live_E = evt_atar_posE + evt_lyso_E_sum

        # === CHECK 1: Energy conservation ===
        # Positron KE + annihilation (1.022 MeV) is the max possible deposited energy
        max_allowed = positron_initial_energy + 1.5  # generous margin
        total_deposited = live_E + evt_dead_E

        if total_deposited > max_allowed:
            n_energy_violation += 1
            excess = total_deposited - positron_initial_energy
            print(f"\n[ENERGY VIOLATION] event {i}: "
                  f"positron_E={positron_initial_energy:.2f} MeV, "
                  f"live_E={live_E:.2f}, dead_E={evt_dead_E:.2f}, "
                  f"total={total_deposited:.2f}, "
                  f"excess={excess:.2f} MeV ({100*excess/positron_initial_energy:.1f}%)")
            print(f"  atar_posE={evt_atar_posE:.3f}, "
                  f"lyso_sum={evt_lyso_E_sum:.3f}, "
                  f"residual_total={residual_total:.3f}")

            # === CHECK 2: ID mismatch details ===
            # Compare volume IDs from pass 3 with calo IDs from pass 4
            pass3_vol_ids = set(truth_E_per_crystal.keys())
            pass4_calo_ids = set()
            for crystal in entry.calo:
                if crystal.GetTotalEnergyDeposit() > 1e-4:
                    pass4_calo_ids.add(int(crystal.GetCaloID()))

            only_in_pass3 = pass3_vol_ids - pass4_calo_ids
            only_in_pass4 = pass4_calo_ids - pass3_vol_ids
            overlap = pass3_vol_ids & pass4_calo_ids

            if only_in_pass3 or only_in_pass4:
                n_id_mismatch += 1
                print(f"  [ID MISMATCH] "
                      f"pass3 vol_ids: {len(pass3_vol_ids)}, "
                      f"pass4 calo_ids: {len(pass4_calo_ids)}, "
                      f"overlap: {len(overlap)}, "
                      f"only_pass3: {len(only_in_pass3)}, "
                      f"only_pass4: {len(only_in_pass4)}")
                if only_in_pass3:
                    print(f"    vol_ids NOT in calo: {sorted(only_in_pass3)[:10]}...")
                if only_in_pass4:
                    print(f"    calo_ids NOT in vol: {sorted(only_in_pass4)[:10]}...")

            # === CHECK 3: Per-crystal breakdown ===
            if residual_details:
                # Sort by largest residual
                residual_details.sort(key=lambda d: d['residual'], reverse=True)
                print(f"  Top residual crystals:")
                for d in residual_details[:5]:
                    truth_from_tracks = truth_E_per_crystal.get(d['calo_id'], 0.0)
                    print(f"    calo_id={d['calo_id']}: "
                          f"GetTotalEdep={d['total_deposit']:.3f}, "
                          f"tracked(smeared)={d['tracked']:.3f}, "
                          f"truth(unsmeared)={truth_from_tracks:.3f}, "
                          f"residual={d['residual']:.3f}, "
                          f"in_pass3={d['in_pass3']}")

        # === CHECK 4: Residual bias (even for non-violating events) ===
        if residual_total > 5.0:  # > 5 MeV of residuals is suspicious
            n_large_residual += 1
            if n_large_residual <= 5:  # print first 5
                print(f"\n[LARGE RESIDUAL] event {i}: "
                      f"residual_total={residual_total:.2f} MeV, "
                      f"positron_E={positron_initial_energy:.2f}, "
                      f"live_E={live_E:.2f}")

        n_processed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Events processed: {n_processed}")
    print(f"Events skipped:   {n_skipped}")
    print(f"Energy violations (total > positron_E + 1.5 MeV): {n_energy_violation} "
          f"({100*n_energy_violation/max(n_processed,1):.3f}%)")
    print(f"ID mismatches (vol_id != calo_id):                {n_id_mismatch}")
    print(f"Large residuals (> 5 MeV):                        {n_large_residual} "
          f"({100*n_large_residual/max(n_processed,1):.3f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=str, required=True,
                        help='Path to ROOT file(s) — directory, glob, or single file')
    parser.add_argument('--max_events', type=int, default=None,
                        help='Max events to validate')
    args = parser.parse_args()

    # Resolve file list (same logic as root_to_parquet.py)
    input_path = args.input
    if os.path.isdir(input_path):
        file_list = sorted(glob.glob(os.path.join(input_path, '*.root')))
    elif '*' in input_path or '?' in input_path:
        file_list = sorted(glob.glob(input_path))
    else:
        file_list = [input_path]

    if not file_list:
        print(f"No ROOT files found matching: {input_path}")
        sys.exit(1)

    print(f"Found {len(file_list)} ROOT file(s)")

    # Get geometry header from first file (same key as root_to_parquet.py line 654)
    f0 = ROOT.TFile(file_list[0])
    geoheader = f0.Get("GeoHeader")
    if geoheader is None:
        print("ERROR: no GeoHeader found in ROOT file")
        sys.exit(1)

    validate_file(file_list, geoheader, max_events=args.max_events)


if __name__ == '__main__':
    main()