import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from typing import List, Dict, Any, Optional

# Constants matching models.py
MODALITY_ATAR_XZ = 0
MODALITY_ATAR_YZ = 1
MODALITY_LYSO = 2

# Normalization constants (borrowed from calorimeter_clustering.py and upstream pipelines)
NORM_POS_LYSO = 100.0  # mm
NORM_E_LYSO = 70.0     # MeV
NORM_T_LYSO = 500.0     # ns

NORM_POS_ATAR = 10.0  # mm
NORM_E_ATAR = 1.0      # ATAR Energy norm (usually dE/dx or MeV)
NORM_T_ATAR = 500.0     # ATAR Time norm

def build_purity_data(
    atar_records: List[Any], 
    lyso_hits: np.ndarray, 
    lyso_targets: Optional[np.ndarray] = None,
    lyso_fracs: Optional[np.ndarray] = None,
    lyso_mask: Optional[np.ndarray] = None
) -> Data:
    """
    Merges ATAR GraphRecords (for a single event) and LYSO structured hits
    into a single PyG Data object formatted for PURITY.
    
    Args:
        atar_records: List of `GraphRecord` objects for this single event (Time-Sliced).
        lyso_hits: Numpy array shape [N_lyso, 5] -> [x, y, z, E, t]
        ...LYSO Targets for Object Condensation
    """
    # 1. Process ATAR Records (Time Slices)
    atar_hit_list = []
    
    # Track node-level and slice-level targets for ATAR
    atar_node_pdgs = []
    atar_slice_pdgs = []
    atar_pion_stops = []
    atar_angles = []
    slice_indices = []
    
    t_slice = 0 # Logical time slice index for AttentionalAggregation
    
    if atar_records:
        for record in atar_records:
            n_hits = len(record.coord)
            if n_hits == 0:
                continue
                
            # Parse features from record
            c = record.coord
            z = record.z
            e = record.energy
            v = record.view
            
            # Map hits to geometric axes [x, y, z, E, t, modality]
            for i in range(n_hits):
                is_yz = (v[i] == 1.0)
                # If YZ view, X=0, Y=coord. If XZ view, X=coord, Y=0.
                x_val = 0.0 if is_yz else (c[i] / NORM_POS_ATAR)
                y_val = (c[i] / NORM_POS_ATAR) if is_yz else 0.0
                z_val = z[i] / NORM_POS_ATAR
                e_val = e[i] / NORM_E_ATAR
                t_val = 0.0 # Standard ATAR records aggregate over time, explicit hit dt is usually 0
                mod_val = MODALITY_ATAR_YZ if is_yz else MODALITY_ATAR_XZ
                
                atar_hit_list.append([x_val, y_val, z_val, e_val, t_val, mod_val])
                slice_indices.append(t_slice)
                
            # Store Node Targets
            atar_node_pdgs.extend(record.hit_pdgs.tolist())
            
            # Group PDG Target heuristics (Taking the max logic from EventMixer)
            group_pdg = max(record.hit_pdgs) if len(record.hit_pdgs) > 0 else 0
            atar_slice_pdgs.append(group_pdg)
            
            # Store Kinematics
            ps = getattr(record, 'true_pion_stop', None)
            atar_pion_stops.append(ps if ps is not None else [float('nan')]*3)
            
            ang = getattr(record, 'true_angle_vector', None)
            atar_angles.append(ang if ang is not None else [float('nan')]*3)
            
            t_slice += 1

    # 2. Process LYSO Calorimeter Hits
    lyso_hit_list = []
    lyso_slice_indices = []
    
    if lyso_hits is not None and len(lyso_hits) > 0:
        for i in range(len(lyso_hits)):
            h = lyso_hits[i]
            x_val = h[0] / NORM_POS_LYSO
            y_val = h[1] / NORM_POS_LYSO
            z_val = h[2] / NORM_POS_LYSO
            e_val = h[3] / NORM_E_LYSO
            t_val = h[4] / NORM_T_LYSO
            mod_val = MODALITY_LYSO
            
            lyso_hit_list.append([x_val, y_val, z_val, e_val, t_val, mod_val])
            # Calorimeter clusters acts as a single contiguous slice internally 
            # (or we could map them to matching ATAR slices if time is synced)
            lyso_slice_indices.append(t_slice) 
            
    # 3. Assemble Unified Matrices
    unified_hits = atar_hit_list + lyso_hit_list
    all_slice_indices = slice_indices + lyso_slice_indices
    
    if len(unified_hits) == 0:
        return Data() 
        
    x_tensor = torch.tensor(unified_hits, dtype=torch.float)
    
    # Inject Slice ID as the 7th column `x[:, 6]` for PURITY Grouping logic (models.py:277)
    slice_tensor = torch.tensor(all_slice_indices, dtype=torch.float).unsqueeze(1)
    x_tensor = torch.cat([x_tensor, slice_tensor], dim=1) # Shape: [N_total, 7]
    
    data = Data(x=x_tensor)
    
    # 4. Attach Formatted Target Vectors
    if len(atar_node_pdgs) > 0:
        data.atar_node_pdg_target = torch.tensor(atar_node_pdgs, dtype=torch.long)
        data.atar_slice_pdg_target = torch.tensor(atar_slice_pdgs, dtype=torch.long)
        
        pstops = torch.tensor(atar_pion_stops, dtype=torch.float)
        pstops /= NORM_POS_ATAR  # Normalize ground truth spatial coordinates!
        data.atar_pion_stop_target = pstops
        
        data.atar_angle_target = torch.tensor(atar_angles, dtype=torch.float) # Unit vectors, no norm needed
        
    if lyso_targets is not None:
        pay = torch.tensor(lyso_targets[:, 0:4], dtype=torch.float)
        pay[:, 0:3] /= NORM_POS_LYSO
        pay[:, 3] /= NORM_E_LYSO
        data.lyso_payload_target = pay
        
        if lyso_fracs is not None:
            data.lyso_fracs_target = torch.tensor(lyso_fracs, dtype=torch.float)
        if lyso_mask is not None:
            data.lyso_mask_target = torch.tensor(lyso_mask, dtype=torch.float)
        
    return data

class PURITYDataset(Dataset):
    """
    Reads mixed Parquet events built from PileupMixer and automatically formats them 
    into PyG Data objects suitable for the PURITY Transformer.
    """
    def __init__(self, parquet_path, max_hits=300):
        super().__init__(root=None, transform=None, pre_transform=None)
        
        print(f"Loading merged parquet dataset from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        
        # Pre-filter events with more than max_hits (OOM constraint)
        atar_len = self.df['atar_x'].apply(lambda x: len(x) if x is not None else 0)
        lyso_len = self.df['lyso_x'].apply(lambda x: len(x) if x is not None else 0)
        total_hits = atar_len + lyso_len
        
        initial_len = len(self.df)
        self.df = self.df[(total_hits > 0) & (total_hits <= max_hits)].reset_index(drop=True)
        final_len = len(self.df)
        print(f"Dropped {initial_len - final_len} events exceeding {max_hits} hits or containing 0 hits. Active dataset size: {final_len}")

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        
        # Parse arrays
        atar_x = row['atar_x']
        n_atar = len(atar_x) if atar_x is not None else 0
        lyso_x = row['lyso_x']
        n_lyso = len(lyso_x) if lyso_x is not None else 0
        
        hit_list = []
        slice_indices = []
        
        atar_pdg = row.get('atar_pdg', np.zeros(n_atar))
        atar_slice = row.get('atar_slice', np.zeros(n_atar))
        
        # Process ATAR hits
        for i in range(n_atar):
            v_val = row['atar_view'][i]
            is_yz = (v_val == 1.0)
            is_xz = (v_val == 0.0)
            x_val = 0.0 if is_yz else (atar_x[i] / NORM_POS_ATAR)
            y_val = (row['atar_y'][i] / NORM_POS_ATAR) if is_yz else 0.0
            z_val = row['atar_z'][i] / NORM_POS_ATAR
            e_val = row['atar_E'][i] / NORM_E_ATAR
            t_val = row['atar_t'][i] / NORM_T_ATAR
            
            # Binary one-hot flags
            is_lyso = 0.0
            
            hit_list.append([x_val, y_val, z_val, e_val, t_val, float(is_xz), float(is_yz), is_lyso])
            slice_indices.append(atar_slice[i] if atar_slice is not None else 0)
            
        lyso_slice = row.get('lyso_slice', np.zeros(n_lyso))
        # Process LYSO hits
        for i in range(n_lyso):
            x_val = lyso_x[i] / NORM_POS_LYSO
            y_val = row['lyso_y'][i] / NORM_POS_LYSO
            z_val = row['lyso_z'][i] / NORM_POS_LYSO
            e_val = row['lyso_E'][i] / NORM_E_LYSO
            t_val = row['lyso_t'][i] / NORM_T_LYSO
            
            # Binary one-hot flags for LYSO
            is_xz = 0.0
            is_yz = 0.0
            is_lyso = 1.0
            
            hit_list.append([x_val, y_val, z_val, e_val, t_val, is_xz, is_yz, is_lyso])
            slice_indices.append(lyso_slice[i] if lyso_slice is not None else 0)
            
        if len(hit_list) == 0:
            return Data()
            
        x_tensor = torch.tensor(hit_list, dtype=torch.float)
        slice_tensor = torch.tensor(slice_indices, dtype=torch.float).unsqueeze(1)
        x_tensor = torch.cat([x_tensor, slice_tensor], dim=1)
        
        data = Data(x=x_tensor)
        
        # Format Targets
        if n_atar > 0:
            atar_pdg_tensor = torch.tensor(atar_pdg, dtype=torch.long)
            atar_origin = row.get('atar_origin', np.zeros(n_atar))
            atar_truth_t = np.array(row['atar_truth_t']) if 'atar_truth_t' in row else np.zeros(n_atar)
            
            # Use 3 bits for unified classification: [Pion (0), Muon (1), MIP (2)]
            # MIP bit (Index 2) is triggered by Positron (Bit 2), Electron (Bit 3), or Gamma (Bit 4)
            multi_hot = torch.zeros((n_atar, 3), dtype=torch.float)
            multi_hot[:, 0] = ((atar_pdg_tensor & (1 << 0)) > 0).float() # Pion
            multi_hot[:, 1] = ((atar_pdg_tensor & (1 << 1)) > 0).float() # Muon
            
            mip_mask = (atar_pdg_tensor & (1 << 2)) | (atar_pdg_tensor & (1 << 3)) | (atar_pdg_tensor & (1 << 4))
            multi_hot[:, 2] = (mip_mask > 0).float()
            
            data.atar_node_pdg_target = multi_hot
            
            # Monte Carlo Event ID per ATAR hit: 0 = triggering event, 1+ = pileup, -1 = radioactivity.
            # Used to build ground-truth edge labels: an edge (u,v) is "True" iff
            # atar_true_event_id[u] == atar_true_event_id[v].
            data.atar_true_event_id = torch.tensor(
                np.array(atar_origin, dtype=np.int64), dtype=torch.long
            )
            
            # Pre-extract arrays once for speed
            atar_x_raw = np.array(row['atar_x']) if 'atar_x' in row else np.zeros(n_atar)
            atar_y_raw = np.array(row['atar_y']) if 'atar_y' in row else np.zeros(n_atar)
            atar_z_raw = np.array(row['atar_z']) if 'atar_z' in row else np.zeros(n_atar)
            
            unique_atar_slices, slice_inverse = np.unique(atar_slice, return_inverse=True)
            slice_targets = []
            multi_event_targets = []
            s_starts = []
            s_stops = []
            
            # Bitmasks for categorization
            # PION (1<<0), MUON (1<<1), POSI (1<<2)
            is_trigger_all = (atar_origin == 0)
            
            for s_idx in range(len(unique_atar_slices)):
                # 1. Mask to this slice
                s_mask_local = (slice_inverse == s_idx)
                s_origins = atar_origin[s_mask_local]
                
                # REQ 1: Multi-event flag (Fast check: is every element equal to the first?)
                is_multi = 0.0
                if len(s_origins) > 1:
                    if np.any(s_origins != s_origins[0]):
                        is_multi = 1.0
                multi_event_targets.append(is_multi)
                
                # 2. Identify refined hits for slice-level targets (EXACT logic)
                s_trigger_mask = is_trigger_all[s_mask_local]
                
                if s_trigger_mask.any():
                    # Refine by PDG priority within the trigger
                    s_pdg_slice = atar_pdg[s_mask_local]
                    
                    s_pion = (s_pdg_slice & (1 << 0)) > 0
                    s_muon = (s_pdg_slice & (1 << 1)) > 0
                    s_posi = (s_pdg_slice & (1 << 2)) > 0
                    
                    if (s_trigger_mask & s_pion).any():
                        slice_ref_mask = s_trigger_mask & s_pion
                    elif (s_trigger_mask & s_muon).any():
                        slice_ref_mask = s_trigger_mask & s_muon
                    elif (s_trigger_mask & s_posi).any():
                        slice_ref_mask = s_trigger_mask & s_posi
                    else:
                        slice_ref_mask = s_trigger_mask
                else:
                    # Background-only slice: use all hits
                    slice_ref_mask = np.ones(len(s_origins), dtype=bool)
                
                # REQ 2: Priority-based PDG
                # multi_hot is a Tensor, we need to index it carefully
                # s_mask_local is boolean mask for n_atar hits
                # slice_ref_mask is boolean mask for current slice hits
                # Combined mask for n_atar hits:
                combined_mask = np.zeros(n_atar, dtype=bool)
                combined_mask[s_mask_local] = slice_ref_mask
                
                # Prepare the 3-bit final slice target: [Pion, Muon, MIP]
                # Since multi_hot is now pre-collapsed to 3 bits, we use it directly
                s_hits_ref_pdg = multi_hot[combined_mask]
                if s_hits_ref_pdg.size(0) > 0:
                    s_pdg_out = (s_hits_ref_pdg.sum(dim=0) > 0).float()
                else:
                    s_pdg_out = torch.zeros(3, dtype=torch.float)
                    
                slice_targets.append(s_pdg_out)
                
                # REQ 3: Priority-based Endpoints
                s_t = atar_truth_t[combined_mask]
                s_x = atar_x_raw[combined_mask]
                s_y = atar_y_raw[combined_mask]
                s_z = atar_z_raw[combined_mask]
                
                if len(s_t) > 0:
                    # REQ: Define by Time, Order by Z
                    t_min_idx = np.argmin(s_t)
                    t_max_idx = np.argmax(s_t)
                    
                    if s_z[t_min_idx] < s_z[t_max_idx]:
                        low_idx, high_idx = t_min_idx, t_max_idx
                    else:
                        low_idx, high_idx = t_max_idx, t_min_idx
                        
                    s_starts.append([s_x[low_idx] / NORM_POS_ATAR, s_y[low_idx] / NORM_POS_ATAR, s_z[low_idx] / NORM_POS_ATAR])
                    s_stops.append([s_x[high_idx] / NORM_POS_ATAR, s_y[high_idx] / NORM_POS_ATAR, s_z[high_idx] / NORM_POS_ATAR])
                else:
                    s_starts.append([0.0, 0.0, 0.0])
                    s_stops.append([0.0, 0.0, 0.0])
                
            data.atar_slice_pdg_target = torch.stack(slice_targets) if len(slice_targets) > 0 else torch.zeros((0, 3), dtype=torch.float)
            data.atar_slice_multi_target = torch.tensor(multi_event_targets, dtype=torch.float)
            data.atar_slice_start_target = torch.tensor(s_starts, dtype=torch.float) if len(s_starts) > 0 else torch.zeros((0, 3), dtype=torch.float)
            data.atar_slice_stop_target = torch.tensor(s_stops, dtype=torch.float) if len(s_stops) > 0 else torch.zeros((0, 3), dtype=torch.float)
            
            num_slices = len(unique_atar_slices)
            
            # Positron Vector Angle (Derived from truth_theta/phi for the triggering event)
            theta = row.get('truth_theta', 0.0)
            phi = row.get('truth_phi', 0.0)
            vx = np.sin(theta) * np.cos(phi)
            vy = np.sin(theta) * np.sin(phi)
            vz = np.cos(theta)
            data.atar_angle_target = torch.tensor([vx, vy, vz], dtype=torch.float).unsqueeze(0).repeat(num_slices, 1)
        else:
            data.atar_node_pdg_target = torch.zeros((0, 3), dtype=torch.float)
            data.atar_slice_pdg_target = torch.zeros((0, 3), dtype=torch.float)
            data.atar_slice_multi_target = torch.zeros(0, dtype=torch.float)
            data.atar_slice_start_target = torch.zeros((0, 3), dtype=torch.float)
            data.atar_true_event_id = torch.zeros(0, dtype=torch.long)  # Always present
            data.atar_slice_stop_target = torch.zeros((0, 3), dtype=torch.float)
            data.atar_angle_target = torch.zeros((0, 3), dtype=torch.float)
        
        # Pion Stop Targets (Global per event)
        pion_x = row.get('truth_pion_stop_x', 0.0)
        pion_y = row.get('truth_pion_stop_y', 0.0)
        pion_z = row.get('truth_pion_stop_z', 0.0)
        pstops = torch.tensor([pion_x, pion_y, pion_z], dtype=torch.float) / NORM_POS_ATAR
        data.atar_pion_stop_target = pstops.unsqueeze(0).repeat(num_slices if n_atar > 0 else 0, 1)
            
        data.positron_initial_energy_target = torch.tensor([row.get('truth_positron_energy', 0.0)], dtype=torch.float)
        
        # Object Condensation LYSO Targets
        if n_lyso > 0 and 'lyso_origin' in row:
            origins = np.array(row['lyso_origin'])
            unique_objs = np.unique(origins)
            unique_objs = unique_objs[unique_objs >= 0] # Exclude radioactivity (origin -1)
            
            MAX_OBJS = 20
            n_objs_true = min(len(unique_objs), MAX_OBJS)
            
            fracs = np.zeros((n_lyso, MAX_OBJS), dtype=np.float32)
            payloads = np.zeros((MAX_OBJS, 4), dtype=np.float32) # X, Y, Z, E
            mask = np.zeros(MAX_OBJS, dtype=np.float32)
            
            lyso_x_arr = np.array(row['lyso_x'])
            lyso_y_arr = np.array(row['lyso_y'])
            lyso_z_arr = np.array(row['lyso_z'])
            lyso_e_arr = np.array(row['lyso_E'])
            
            for obj_idx in range(n_objs_true):
                obj_id = unique_objs[obj_idx]
                obj_mask = (origins == obj_id)
                fracs[obj_mask, obj_idx] = 1.0  # Hard assignment for truth masking
                mask[obj_idx] = 1.0
                
                e_sum = np.sum(lyso_e_arr[obj_mask])
                if e_sum > 0:
                    cx = np.sum(lyso_x_arr[obj_mask] * lyso_e_arr[obj_mask]) / e_sum
                    cy = np.sum(lyso_y_arr[obj_mask] * lyso_e_arr[obj_mask]) / e_sum
                    cz = np.sum(lyso_z_arr[obj_mask] * lyso_e_arr[obj_mask]) / e_sum
                    payloads[obj_idx] = [cx / NORM_POS_LYSO, cy / NORM_POS_LYSO, cz / NORM_POS_LYSO, e_sum / NORM_E_LYSO]
                    
            data.lyso_fracs_target = torch.tensor(fracs)
            data.lyso_payload_target = torch.tensor(payloads)
            data.lyso_mask_target = torch.tensor(mask)
        else:
            MAX_OBJS = 20
            data.lyso_fracs_target = torch.zeros((0, MAX_OBJS), dtype=torch.float32)
            data.lyso_payload_target = torch.zeros((MAX_OBJS, 4), dtype=torch.float32)
            data.lyso_mask_target = torch.zeros(MAX_OBJS, dtype=torch.float32)
            
        # --- Unified Hit-Level Targets ---
        atar_is_trigger = (atar_origin == 0).astype(float) if n_atar > 0 else np.zeros(0)
        lyso_origin = np.array(row['lyso_origin']) if n_lyso > 0 and 'lyso_origin' in row else np.zeros(n_lyso)
        lyso_is_trigger = (lyso_origin == 0).astype(float) if n_lyso > 0 else np.zeros(0)
        
        is_trigger_all = np.concatenate([atar_is_trigger, lyso_is_trigger])
        data.is_trigger_target = torch.tensor(is_trigger_all, dtype=torch.float)
        #data.is_trigger_target = torch.tensor(atar_is_trigger, dtype=torch.float)
            
        return data
