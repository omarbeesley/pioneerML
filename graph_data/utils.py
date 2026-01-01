"""
Common utilities for building fully connected graph representations of
preprocessed time-group data and feeding them into torch-geometric models.

Provides:
  * fully_connected_edge_index
  * build_edge_attr
  * GraphGroupDataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, List, Dict, Any
import math

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse



PION_MASK = 0b00001
MUON_MASK = 0b00010
POSITRON_MASK = 0b00100
ELECTRON_MASK = 0b01000
OTHER_MASK = 0b10000

BIT_TO_CLASS = {
    PION_MASK: 0,
    MUON_MASK: 1,
    POSITRON_MASK: 2,  # positron collapses to mip label
    ELECTRON_MASK: 2,  # electron collapses to mip label
    # OTHER_MASK hits are ignored for supervision
}

CLASS_NAMES = {0: 'pion', 1: 'muon', 2: 'mip'}
NUM_GROUP_CLASSES = len(set(BIT_TO_CLASS.values()))
def fully_connected_edge_index(num_nodes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a directed fully-connected edge index without self loops."""
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes - 1)
    dst = torch.cat([
        torch.cat([torch.arange(0, i, device=device), torch.arange(i + 1, num_nodes, device=device)])
        for i in range(num_nodes)
    ])
    return torch.stack([src, dst], dim=0)


def build_edge_attr(node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute edge attributes [dx, dz, dE, same_view] for provided edges."""
    if edge_index.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float, device=node_features.device)

    src, dst = edge_index
    coord = node_features[:, 0]
    z_pos = node_features[:, 1]
    energy = node_features[:, 2]
    view_flag = node_features[:, 3]

    dx = (coord[dst] - coord[src]).unsqueeze(1)
    dz = (z_pos[dst] - z_pos[src]).unsqueeze(1)
    dE = (energy[dst] - energy[src]).unsqueeze(1)
    same_view = (view_flag[dst] == view_flag[src]).float().unsqueeze(1)

    return torch.cat([dx, dz, dE, same_view], dim=1)


@dataclass
class GraphRecord:
    coord: Iterable[float]
    z: Iterable[float]
    energy: Iterable[float]
    view: Iterable[float]
    labels: Optional[Sequence[int]] = None
    event_id: Optional[int] = None
    group_id: Optional[int] = None
    hit_labels: Optional[Sequence[Sequence[int]]] = None
    group_probs: Optional[Sequence[float]] = None
    hit_pdgs: Optional[Sequence[int]] = None
    class_energies: Optional[Sequence[float]] = None
    true_start: Optional[Sequence[float]] = None
    true_end: Optional[Sequence[float]] = None
    true_pion_stop: Optional[Sequence[float]] = None
    true_angle_vector: Optional[Sequence[float]] = None
    pred_pion_stop: Optional[Sequence[float]] = None
    pred_endpoints: Optional[Sequence[Sequence[Sequence[float]]]] = None   # [Start/End, XYZ, Quantiles]
    matched_pion_index: Optional[int] = None
    pion_stop_for_angle: Optional[Sequence[float]] = None
    true_arc_length: Optional[float] = None

    def __getitem__(self, key):
        """Allow subscript access for backward compatibility."""
        if not isinstance(key, str):
            raise TypeError(f"GraphRecord key must be a string, got {type(key)}")
        return getattr(self, key)


class GraphGroupDataset(Dataset):
    """Dataset that emits standardized graph Data objects for time-group records."""

    def __init__(self, records: Sequence[Dict[str, Any] | GraphRecord], *, num_classes: Optional[int] = None):
        self.items: List[GraphRecord] = [self._coerce(item) for item in records]
        if num_classes is None:
            max_label = -1
            for item in self.items:
                if item.labels:
                    max_label = max(max_label, max(item.labels))
            num_classes = max_label + 1 if max_label >= 0 else 0
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Data:
        item = self.items[index]

        coord = np.asarray(item.coord, dtype=np.float32)
        z_pos = np.asarray(item.z, dtype=np.float32)
        energy = np.asarray(item.energy, dtype=np.float32)
        view = np.asarray(item.view, dtype=np.float32)

        if not (coord.shape == z_pos.shape == energy.shape == view.shape):
            raise ValueError("All per-hit arrays must share the same shape.")

        num_hits = coord.shape[0]
        node_features = torch.tensor(
            np.stack([coord, z_pos, energy, view], axis=1), dtype=torch.float
        )

        edge_index = fully_connected_edge_index(num_hits, device=node_features.device)
        edge_attr = build_edge_attr(node_features, edge_index)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        # Add global group energy feature (shape [1, 1] for proper batching)
        data.u = torch.tensor([[energy.sum()]], dtype=torch.float)

        if item.labels is not None and self.num_classes:
            label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
            for lbl in item.labels:
                if 0 <= lbl < self.num_classes:
                    label_tensor[lbl] = 1.0
            data.y_group = label_tensor.unsqueeze(0)
            data.y = label_tensor
            
        if item.hit_pdgs is not None:
            data.y_node = torch.tensor(item.hit_pdgs, dtype=torch.long)
            
        if item.class_energies is not None:
            data.y_energy = torch.tensor(item.class_energies, dtype=torch.float).unsqueeze(0) # [1, num_classes]

        if item.hit_labels is not None:
            # Multi-label targets for splitter [N, 3]
            data.y = torch.tensor(item.hit_labels, dtype=torch.float)

        if item.event_id is not None:
            data.event_id = torch.tensor(int(item.event_id), dtype=torch.long)
        if item.group_id is not None:
            data.group_id = torch.tensor(int(item.group_id), dtype=torch.long)
            
        if item.true_start is not None and item.true_end is not None:
            # shape: [2, 3]
            start = torch.tensor(item.true_start, dtype=torch.float)
            end = torch.tensor(item.true_end, dtype=torch.float)
            data.y_pos = torch.stack([start, end], dim=0).unsqueeze(0)
            data.group_id = torch.tensor(int(item.group_id), dtype=torch.long)

        if item.true_pion_stop is not None:
            # shape: [1, 3]
            data.y_pion_stop = torch.tensor(item.true_pion_stop, dtype=torch.float).unsqueeze(0)

        if item.true_angle_vector is not None:
            # shape: [1, 3]
            data.y_angle_vector = torch.tensor(item.true_angle_vector, dtype=torch.float).unsqueeze(0)

        if item.pred_pion_stop is not None:
            # shape: [1, 3]
            data.pred_pion_stop = torch.tensor(item.pred_pion_stop, dtype=torch.float).unsqueeze(0)

        if item.group_probs is not None:
            data.group_probs = torch.tensor(item.group_probs, dtype=torch.float).unsqueeze(0)

        if item.true_arc_length is not None:
            data.y_arc = torch.tensor([item.true_arc_length], dtype=torch.float).unsqueeze(0)

        return data

    @staticmethod
    def _coerce(raw: Dict[str, Any] | GraphRecord) -> GraphRecord:
        # Fast path for same-class instance
        if isinstance(raw, GraphRecord):
            return raw
        
        # Duck typing for stale instances (from previous reloads)
        if hasattr(raw, 'coord'):
            return raw

        return GraphRecord(
            coord=raw["coord"],
            z=raw["z"],
            energy=raw["energy"],
            view=raw["view"],
            labels=raw.get("labels"),
            event_id=raw.get("event_id"),
            group_id=raw.get("group_id"),
            hit_pdgs=raw.get("hit_pdgs"),
            class_energies=raw.get("class_energies"),
            hit_labels=raw.get("hit_labels"),
            true_pion_stop=raw.get("true_pion_stop"),
            true_angle_vector=raw.get("true_angle_vector"),
            pred_pion_stop=raw.get("pred_pion_stop"),
            matched_pion_index=raw.get("matched_pion_index"),
            pion_stop_for_angle=raw.get("pion_stop_for_angle"),
            group_probs=raw.get("group_probs"),
            true_arc_length=raw.get("true_arc_length"),

        )

def build_event_graph(container, device, radius_z: float = 0.5):
    """
    Converts EventContainer to graph inputs for EventBuilder.
    Now builds inter-group connections purely based on Z-distance.
    """
    # 1. Collect all hits and their metadata
    node_features_list = []
    group_indices_list = []
    group_origins = []
    
    for g_idx, record in enumerate(container.records):
        # Base Data
        coords = record.coord
        zs = record.z
        energies = record.energy
        views = record.view
        num_hits = len(coords)
        
        # --- Feature Engineering ---
        # Upstream: Broadcast group-level probs [3] -> [num_hits, 3]
        if record.group_probs is not None:
             probs = torch.tensor(record.group_probs, dtype=torch.float32).repeat(num_hits, 1)
        else:
             probs = torch.zeros(num_hits, 3)
             
        # Upstream: Broadcast predicted endpoints [18] -> [num_hits, 18]
        if record.pred_endpoints is not None:
            eps = torch.tensor(record.pred_endpoints, dtype=torch.float32).view(-1).repeat(num_hits, 1)
        else:
            eps = torch.zeros(num_hits, 18)
            
        base = torch.tensor(np.stack([coords, zs, energies, views], axis=1), dtype=torch.float32)
        
        # Concatenate Features: [4] + [3] + [18] = 25
        feats = torch.cat([base, probs, eps], dim=1)
        node_features_list.append(feats)
        
        # Track Group IDs and Origins
        group_indices_list.append(torch.full((num_hits,), g_idx, dtype=torch.long))
        group_origins.append(container.origins[g_idx])
        
    # Stack all nodes into single tensors
    if not node_features_list:
        return None

    x = torch.cat(node_features_list, dim=0).to(device) # [TotalHits, 25]
    group_indices = torch.cat(group_indices_list, dim=0).to(device) # [TotalHits]
    num_groups = len(container.records)
    
    # 2. Build Targets [N, N]
    origins = torch.tensor(group_origins, device=device)
    affinity_targets = (origins.unsqueeze(1) == origins.unsqueeze(0)).float()
    
    # 3. Build Edges (Vectorized & Z-Only)
    
    # Extract columns for efficient masking
    z_col = x[:, 1]
    
    # Create broadcasted matrices for comparison [N, N]
    g_i = group_indices.unsqueeze(1)
    g_j = group_indices.unsqueeze(0)
    
    # A. Intra-Group Edges (Fully Connected)
    intra_mask = (g_i == g_j)
    
    # B. Inter-Group Edges (Pure Z-Radius)
    # Logic: Connect if Z-distance < radius_z AND they are in different groups.
    # We ignore X/Y distance here; the GNN will see the X/Y diff in the edge attributes.
    dist_z = torch.abs(z_col.unsqueeze(1) - z_col.unsqueeze(0))
    
    inter_mask = (dist_z < radius_z) & (g_i != g_j)
    
    # C. Combine All Edges
    final_adj = intra_mask | inter_mask
    edge_index, _ = dense_to_sparse(final_adj)
    
    # 4. Compute Edge Attributes
    src, dst = edge_index
    u, v = x[src], x[dst]
    
    diffs = u[:, :3] - v[:, :3] # coord, z, energy diffs
    is_same_view = (u[:, 3] == v[:, 3]).float().unsqueeze(1)
    is_same_group = (group_indices[src] == group_indices[dst]).float().unsqueeze(1)
    
    # Attr: [dx, dz, dE, same_view, same_group]
    edge_attr = torch.cat([diffs, is_same_view, is_same_group], dim=1)
        
    return x, edge_index, edge_attr, group_indices, num_groups, affinity_targets








