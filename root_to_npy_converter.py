import ROOT
import sys
import os
import numpy as np
import glob
from collections import Counter
import random
import argparse


# Example usage:
'''
python3 root_to_npy_converter.py \
  --root_pattern "/path/to/root/files/*.root" \
  --output_dir "ML/npy_data" \
  --geohelper "/path/to/geohelper.root" \
  --max_events 100000 \
  --batch_size 10000
'''

# Constants and Helper Functions from legacy code

pdg_to_idx = {pdg: i for i, pdg in enumerate([211, -13, -11, 11, 98105])}
# Bitmask constants (1 bit per particle type)
PION      = 0b00001  # 1
MUON      = 0b00010  # 2
POSITRON  = 0b00100  # 4
ELECTRON  = 0b01000  # 8
OTHER     = 0b10000  # 16

def pdg_to_mask(pdg_id):
    if pdg_id == 211: return PION
    elif pdg_id == -13: return MUON
    elif pdg_id == -11:       return POSITRON
    elif pdg_id == 11:        return ELECTRON
    else:                     return OTHER

MASK_TO_PDG = {
    0b00001: 211,    # pion
    0b00010: -13,     # muon
    0b00100: -11,    # positron
    0b01000: 11,     # electron
    0b10000: 98105    # other
    }

def decode_mask(mask):
    """
    Given a bitmask, return a list of PDG IDs it represents.
    """
    return [pdg for bit, pdg in MASK_TO_PDG.items() if mask & bit]

def smear(energy, energy_resolution):
    stdv = energy * energy_resolution
    smeared_energy = energy + np.random.normal(0, stdv)
    return smeared_energy

MERGE_WINDOW = 2
GROUP_WINDOW = 1

def group_hits_in_time(allHits, time_window_ns=GROUP_WINDOW):
    """
    Groups hits into clusters where each hit is within `time_window_ns`
    of the first hit in the group.
    """
    # Flatten all hits into a list of hit_array
    hit_list = []
    for hits in allHits.values():
        hit_list.extend(hits)

    # Sort hits by time (index 4)
    hit_list.sort(key=lambda hit: hit[4])

    groups = []
    current_group = []
    group_time = None

    for hit in hit_list:
        time = hit[4]
        if not current_group:
            current_group.append(hit)
            group_time = time
        elif abs(time - group_time) <= time_window_ns:
            current_group.append(hit)
            group_time = time
        else:
            # For the legacy code, they appended total energy here, but we are reconstructing the hit list differently.
            # The user wants specific 13-element arrays. The input 'hit' here is already the 11-element array (without angle).
            # We will handle the angle appending later.
            groups.append(np.array(current_group))
            current_group = [hit]
            group_time = time

    if current_group:
        groups.append(np.array(current_group))

    return groups

def groupID(groupList):
    """
    This function takes in a list of groups of hits, and returns the unique PDG IDs
    associated with each group, based on bitmask encoding. A PDG ID is kept only if it
    appears in at least two hits in the group; if none do, all are kept.
    """
    unique_pdg_per_group = []

    for group in groupList:
        all_decoded_pdgs = []
        for hit in group:
            mask = int(hit[5])
            all_decoded_pdgs.extend(decode_mask(mask))

        pdg_counts = Counter(all_decoded_pdgs)
        unique_pdgs = [pdg for pdg, count in pdg_counts.items() if count >= 2]

        if not unique_pdgs:
            unique_pdgs = list(pdg_counts.keys())

        unique_pdg_per_group.append(unique_pdgs)

    return unique_pdg_per_group

kPidif  = 0x0000000010

def process_and_save(root_files_pattern, output_dir, geohelper_file, max_events=None, batch_size=10000):
    """
    Processes ROOT files, extracts MIP time groups with positron angles, and saves to NPY in batches.
    """
    
    # Load Geometry Helper
    if not os.path.exists(geohelper_file):
        print(f"Error: Geohelper file not found at {geohelper_file}")
        return

    layout_file = ROOT.TFile(geohelper_file)
    geohelper = layout_file.Get("PIMCGeoHelper")
    # layout = geohelper.GetLayout("CALORIMETER", "TP2") # Not strictly needed if we use geohelper directly

    # Load Chain
    chain = ROOT.TChain("sim")
    # Check if files exist
    files = glob.glob(root_files_pattern)
    if not files:
        print(f"No files found matching {root_files_pattern}")
        return
    
    for f in files:
        chain.Add(f)

    print(f"Loaded {chain.GetEntries()} events from {len(files)} files.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eventTimeGroups = []
    eventGroupInfos = []
    
    eventNumber = 0
    processed_events = 0
    batch_index = 0
    
    # Iterate
    for i, entry in enumerate(chain):
        if max_events and i >= max_events:
            break
        
        eventType = int(entry.info.GetType())
        if eventType & kPidif:
            continue

        # Trigger check
        triggered = 0
        for upstream in entry.upstream:
            if upstream.GetUpstreamID() == 99999:
                energyDTAR = upstream.GetTotalEnergyDeposit()
                if energyDTAR > 1:
                    triggered = 1
        
        if not triggered:
            continue

        # Extract Positron Angle (Truth) using legacy logic
        thetaInit, phiInit = -1000, -1000
        
        # Determine run type from filename or assume based on structure?
        # The legacy code checks `if run == 'michel':` and `elif run == 'pie':`.
        # Since we are processing files, we need to infer this or handle both.
        # Let's infer from the number of daughters as a heuristic if run name isn't available,
        # or try to parse the filename.
        # However, the legacy code iterates `sample_chains.items()` where keys are run names.
        # Here we iterate a chain. Let's try to handle both cases based on nD.
        
        for decay in entry.decay:
            nD = decay.GetNDaughters()
            # Logic for 'michel' run (nD == 3)
            if nD == 3:
                mom = decay.GetDaughterMomAt(2)
                thetaInit = mom.Theta()
                phiInit = mom.Phi()
                break
            # Logic for 'pie' run (nD == 2)
            elif nD == 2:
                mom = decay.GetDaughterMomAt(1)
                thetaInit = mom.Theta()
                phiInit = mom.Phi()
                break
        
        # If no angle found (e.g. not a decay event we care about?), maybe skip?
        # But legacy code had specific logic for 'michel' and 'pie'. 
        # Let's assume if we found a positron in ATAR, there should be an angle.
        # If thetaInit is still -1000, we might want to skip this event for training angle.
        
        pion_in_atar = 0
        positron_in_atar = 0
        evil_event = 0
        
        allHits = {}
        merged_pixels = []

        for atarHit in entry.atar:
            pdg = atarHit.GetPDGID()
            if pdg not in [211, -11, -13, 11]:
                continue
            
            # Check for "evil" particles (legacy check)
            if pdg not in [211, -11, -13, 11]:
                 evil_event = 1

            pdg_binary = pdg_to_mask(pdg)

            true_time = atarHit.GetTime()
            hitTime = true_time + np.random.normal(0, 0.2)
            pixelID = atarHit.GetPixelID()
            edep = atarHit.GetEdep()
            energySmeared = smear(edep, 0.15)
            stripType = abs(abs(geohelper.GetPsi(pixelID)) - 0.0) > 0.01

            coord = geohelper.GetY(pixelID) if stripType else geohelper.GetX(pixelID)
            z = geohelper.GetZ(pixelID)

            truePositions = [atarHit.GetX1(), atarHit.GetY1(), atarHit.GetZ1()]

            # Base hit structure (13 elements) - User requested theta/phi on ALL hits
            # [coord, z, stripType, energySmeared, hitTime, pdg_binary, eventNumber, truePositions[0], truePositions[1], truePositions[2], true_time, theta, phi]
            hit = np.array([coord, z, stripType, energySmeared, hitTime, pdg_binary, i, truePositions[0], truePositions[1], truePositions[2], true_time, thetaInit, phiInit])

            if pdg == 211:
                pion_in_atar = 1
            if pdg == -11:
                positron_in_atar = 1

            if pixelID in allHits:
                merged = False
                for existing_hit in allHits[pixelID]:
                    if abs(existing_hit[4] - hitTime) < MERGE_WINDOW:
                        existing_hit[3] += energySmeared
                        existing_hit[5] = int(existing_hit[5]) | int(pdg_binary)
                        if pixelID not in merged_pixels:
                            merged_pixels.append(pixelID)
                        merged = True
                        break
                if not merged:
                    allHits[pixelID].append(hit)
            else:
                allHits[pixelID] = [hit]

        if len(allHits) < 10:
            evil_event = 1

        # Filter events
        if ((not pion_in_atar) or evil_event or (not positron_in_atar)):
            continue
            
        # Calculate event-level pion stop
        eventPionStopX, eventPionStopY, eventPionStopZ = 0.0, 0.0, 0.0
        eventMaxPionTime = -1.0
        
        for pixelHits in allHits.values():
            for hit in pixelHits:
                mask = int(hit[5])
                pdgs = decode_mask(mask)
                if 211 in pdgs:
                     # hit[10] is true_time
                     if hit[10] > eventMaxPionTime:
                         eventMaxPionTime = hit[10]
                         eventPionStopX = hit[7]
                         eventPionStopY = hit[8]
                         eventPionStopZ = hit[9]

        # Process groups to extract group-level info and hit data
        groups = group_hits_in_time(allHits)
        if groups:
            for group in groups:
                # Initialize group stats
                pionInGroup = 0
                muonInGroup = 0
                MIPinGroup = 0
                
                totalPionEnergy = 0.0
                totalMuonEnergy = 0.0
                totalMIPEnergy = 0.0
                
                # Use event-level pion stop
                pionStopX, pionStopY, pionStopZ = eventPionStopX, eventPionStopY, eventPionStopZ
                
                group_hits_data = []
                
                for hit in group:
                    # hit structure from earlier loop:
                    # [coord, z, stripType, energySmeared, hitTime, pdg_binary, eventNumber, 
                    #  truePositions[0], truePositions[1], truePositions[2], true_time, thetaInit, phiInit]
                    
                    # We need to recover the original PDG to check type correctly, or use the binary mask
                    # The binary mask is at index 5.
                    mask = int(hit[5])
                    pdgs = decode_mask(mask)
                    
                    energy = hit[3]
                    
                    # Check particle types
                    is_pion = False
                    is_muon = False
                    is_mip = False 
                    
                    if 211 in pdgs:
                        pionInGroup = 1
                        totalPionEnergy += energy
                        is_pion = True
                        # Pion stop is now calculated globally
                            
                    if -13 in pdgs:
                        muonInGroup = 1
                        totalMuonEnergy += energy
                        is_muon = True
                        
                    # MIP definition: Positrons (-11) and Electrons (11)
                    if -11 in pdgs or 11 in pdgs:
                        MIPinGroup = 1
                        totalMIPEnergy += energy
                        is_mip = True
                        
                    # Hit data for saving: [coord, z, stripType, energySmeared, pdg_binary]
                    # hit[0], hit[1], hit[2], hit[3], hit[5]
                    group_hits_data.append([hit[0], hit[1], hit[2], hit[3], hit[5]])

                # Group info: [pionInGroup, muonInGroup, MIPinGroup, pionStopX, pionStopY, pionStopZ, 
                #              totalPionEnergy, totalMuonEnergy, totalMIPEnergy, theta, phi, eventID]
                # theta is hit[11], phi is hit[12], eventID is hit[6]
                # Take from first hit (they are same for group)
                first_hit = group[0]
                theta = first_hit[11]
                phi = first_hit[12]
                eventID = first_hit[6]
                
                # Calculate Start/End Points
                # Convert group to numpy array for vectorization
                # hit structure: [coord, z, stripType, energySmeared, hitTime, pdg_binary, eventNumber, 
                #                 truePositions[0], truePositions[1], truePositions[2], true_time, ...]
                # We need indices: 5 (mask), 7,8,9 (pos), 10 (time)
                
                group_arr = np.array(group)
                masks = group_arr[:, 5].astype(int)
                times = group_arr[:, 10].astype(float)
                pos = group_arr[:, 7:10].astype(float)
                
                # Filter out electrons (mask & 8)
                # bit 3 is 8. (mask & 8) == 0 means NOT electron
                not_electron = (masks & 8) == 0
                
                if np.any(not_electron):
                    use_times = times[not_electron]
                    use_pos = pos[not_electron]
                else:
                    # Fallback to all if only electrons
                    use_times = times
                    use_pos = pos
                
                # Find start (min time) and end (max time)
                start_idx = np.argmin(use_times)
                end_idx = np.argmax(use_times)
                
                startX, startY, startZ = use_pos[start_idx]
                endX, endY, endZ = use_pos[end_idx]

                group_info = [
                    pionInGroup, muonInGroup, MIPinGroup,
                    pionStopX, pionStopY, pionStopZ,
                    totalPionEnergy, totalMuonEnergy, totalMIPEnergy,
                    theta, phi, eventID,
                    startX, startY, startZ,
                    endX, endY, endZ
                ]
                
                eventTimeGroups.append(group_hits_data)
                eventGroupInfos.append(group_info)
            
        processed_events += 1
        if processed_events % 1000 == 0:
            print(f"Processed {processed_events} events...")

        # Chunked Saving
        if len(eventTimeGroups) >= batch_size:
            hits_file = os.path.join(output_dir, f"hits_batch_{batch_index}.npy")
            info_file = os.path.join(output_dir, f"group_info_batch_{batch_index}.npy")
            
            np.save(hits_file, np.array(eventTimeGroups, dtype=object))
            np.save(info_file, np.array(eventGroupInfos, dtype=np.float32))
            
            print(f"Saved batch {batch_index} with {len(eventTimeGroups)} groups to {output_dir}")
            eventTimeGroups = [] # Clear memory
            eventGroupInfos = []
            batch_index += 1

    # Save remaining events
    if eventTimeGroups:
        hits_file = os.path.join(output_dir, f"hits_batch_{batch_index}.npy")
        info_file = os.path.join(output_dir, f"group_info_batch_{batch_index}.npy")
        
        np.save(hits_file, np.array(eventTimeGroups, dtype=object))
        np.save(info_file, np.array(eventGroupInfos, dtype=np.float32))
        
        print(f"Saved final batch {batch_index} with {len(eventTimeGroups)} groups to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ROOT files to NPY with positron angle.")
    parser.add_argument("--root_pattern", type=str, required=True, help="Glob pattern for ROOT files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for NPY files")
    parser.add_argument("--geohelper", type=str, required=True, help="Path to ROOT file containing PIMCGeoHelper")
    parser.add_argument("--max_events", type=int, default=None, help="Maximum events to process")
    parser.add_argument("--batch_size", type=int, default=10000, help="Number of events per NPY file")

    args = parser.parse_args()

    process_and_save(args.root_pattern, args.output_dir, args.geohelper, args.max_events, args.batch_size)



