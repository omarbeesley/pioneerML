"""Reusable GNN model definitions built around the standardized graph features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, JumpingKnowledge, AttentionalAggregation

class FullGraphTransformerBlock(nn.Module):
    def __init__(self, hidden, heads=4, edge_dim=4, dropout=0.1):
        super().__init__()

        # Pre-norm is more stable for transformers
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = TransformerConv(
            hidden, hidden // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout,
            concat=True, beta=True
        )

        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden)
        )

    def forward(self, x, edge_index, edge_attr):
        # Multi-head graph attention + residual
        h = self.attn(self.ln1(x), edge_index, edge_attr)
        x = x + h           # Residual

        # Feed-forward network + residual
        h2 = self.ffn(self.ln2(x))
        x = x + h2          # Residual

        return x


class QuantileOutputHead(nn.Module):
    def __init__(self, input_dim, num_points=2, coords=3, quantiles=[0.16, 0.50, 0.84]):
        super().__init__()
        self.quantiles = sorted(quantiles)
        self.mid_index = self.quantiles.index(0.50) 
        self.num_points = num_points
        self.coords = coords
        self.num_quantiles = len(quantiles)
        
        # Project from embedding to (Points * Coords * Quantiles)
        # e.g. 2 * 3 * 3 = 18 outputs
        self.projection = nn.Linear(input_dim, num_points * coords * self.num_quantiles)

    def forward(self, x):
        """
        Input: [batch, hidden_dim]
        Output: [batch, num_points, coords, num_quantiles]
        """
        batch_size = x.shape[0]
        
        # 1. Project
        raw = self.projection(x)
        
        # 2. Reshape to [batch, num_points, coords, num_quantiles]
        raw = raw.view(batch_size, self.num_points, self.coords, self.num_quantiles)
        
        # 3. Enforce Monotonicity
        # Median is the anchor
        median = raw[..., self.mid_index]
        
        # Upper quantiles (Median + positive)
        upper_offsets = torch.nn.functional.softplus(raw[..., self.mid_index+1:])
        upper_vals = median.unsqueeze(-1) + torch.cumsum(upper_offsets, dim=-1)
        
        # Lower quantiles (Median - positive)
        lower_offsets = torch.nn.functional.softplus(raw[..., :self.mid_index])
        lower_offsets_flipped = torch.flip(lower_offsets, dims=[-1]) 
        lower_vals = median.unsqueeze(-1) - torch.cumsum(lower_offsets_flipped, dim=-1)
        lower_vals = torch.flip(lower_vals, dims=[-1])
        
        return torch.cat([lower_vals, median.unsqueeze(-1), upper_vals], dim=-1)


VIEW_X_VAL = 0  # Set to 1 if your data uses 1 for X
VIEW_Y_VAL = 1  # Set to 2 if your data uses 2 for Y

class ViewAwareEncoder(nn.Module):
    def __init__(self, prob_dim, hidden_dim):
        super().__init__()
        self.prob_dim = prob_dim
        self.feature_proj = nn.Linear(3 + prob_dim, hidden_dim)
        self.view_embedding = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.view_embedding.weight, std=0.02)
        
    def forward(self, x, probs=None):
        # 1. Physics Features
        phys_feats = x[:, :3]
        
        # 2. STRICT View Indexing (No guessing!)
        raw_view = x[:, 3].long()
        
        # Create a clean 0/1 index for the embedding layer
        # We start with zeros (default to View 0)
        embedding_idx = torch.zeros_like(raw_view)
        
        # If the raw value matches our Y-Constant, set index to 1
        embedding_idx[raw_view == VIEW_Y_VAL] = 1
        # (Anything else stays 0, which corresponds to X)
        
        if probs is None:
            probs = torch.zeros(x.size(0), self.prob_dim, device=x.device)

        features = torch.cat([phys_feats, probs], dim=1)
        hit_embed = self.feature_proj(features)
        
        return hit_embed + self.view_embedding(embedding_idx)


class GroupClassifierStereo(nn.Module):
    def __init__(self, in_dim=4, edge_dim=4, hidden=200, heads=4,
                 num_blocks=2, dropout=0.1, num_classes=3):
        super().__init__()

        self.input_embed = ViewAwareEncoder(prob_dim=0, hidden_dim=hidden)

        self.blocks = nn.ModuleList([
            FullGraphTransformerBlock(
                hidden, heads=heads, edge_dim=edge_dim, dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden * num_blocks

        # Split Pooling
        self.pool_x = AttentionalAggregation(nn.Sequential(
            nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1)
        ))
        self.pool_y = AttentionalAggregation(nn.Sequential(
            nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1)
        ))

        # --- FUSED HEAD UPDATE ---
        # Input: Pool_X (jk) + Pool_Y (jk) + Global_U (1) + Valid_X (1) + Valid_Y (1)
        # We add +2 for the valid bits
        concat_dim = (jk_dim * 2) + 1 + 2

        self.head = nn.Sequential(
            nn.Linear(concat_dim, jk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(jk_dim, jk_dim // 2),
            nn.ReLU(),
            nn.Linear(jk_dim // 2, num_classes)
        )

    def forward(self, data):
        # 1. Encode
        x = self.input_embed(data.x)
        
        # 2. GNN Backbone
        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)

        # 3. STRICT VIEW MASKING (No dynamic guessing!)
        raw_view = data.x[:, 3].long()
        mask_x = (raw_view == VIEW_X_VAL)
        mask_y = (raw_view == VIEW_Y_VAL)

        # 4. Pooling & Valid Bits
        def pool_and_count(mask, pool_layer):
            if mask.any():
                pooled = pool_layer(x_cat[mask], data.batch[mask], dim_size=data.num_graphs)
                
                # Count hits per graph to determine validity
                counts = torch.zeros(data.num_graphs, device=x.device)
                counts.index_add_(0, data.batch, mask.float())
                has_hits = (counts > 0).float().unsqueeze(1)
                return pooled, has_hits
            else:
                return (torch.zeros(data.num_graphs, x_cat.size(1), device=x.device),
                        torch.zeros(data.num_graphs, 1, device=x.device))

        pool_x, has_x = pool_and_count(mask_x, self.pool_x)
        pool_y, has_y = pool_and_count(mask_y, self.pool_y)

        # 5. Fusion & Prediction
        # Concatenate: [Pool_X, Pool_Y, Global, Has_X, Has_Y]
        # This allows the classifier to distinguish "Zero Energy" from "Missing View"
        out = torch.cat([pool_x, pool_y, data.u, has_x, has_y], dim=1)
        
        return self.head(out)


class GroupClassifier(nn.Module):
    def __init__(self, in_dim=4, edge_dim=4, hidden=200, heads=4,
                 num_blocks=2, dropout=0.1, num_classes=3):
        super().__init__()

        self.input_embed = nn.Linear(in_dim, hidden)

        self.blocks = nn.ModuleList([
            FullGraphTransformerBlock(
                hidden, heads=heads, edge_dim=edge_dim, dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        self.jk = JumpingKnowledge(mode="cat")
        concat_dim = hidden * num_blocks

        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Linear(concat_dim // 2, 1)
        ))

        self.head = nn.Sequential(
            nn.Linear(concat_dim + 1, concat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim // 2, num_classes)
        )

    def forward(self, data):
        x = self.input_embed(data.x)
        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        out = torch.cat([pooled, data.u], dim=1)
        return self.head(out)



class GroupAffinityModel(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=128,
                 heads=4, num_layers=3, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList([
            FullGraphTransformerBlock(
                hidden_channels, heads=heads, edge_dim=4, dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden_channels * num_layers

        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(jk_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        ))

        self.head = nn.Sequential(
            nn.Linear(jk_dim + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        xs = []
        for block in self.layers:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        out = torch.cat([pooled, data.u], dim=1)
        return self.head(out)


class OrthogonalEndpointRegressor(nn.Module):
    def __init__(self, in_channels=4, prob_dimension=3, hidden=160, heads=4, 
                 layers=2, dropout=0.1, quantiles=[0.16, 0.50, 0.84]):
        super().__init__()

        self.hit_encoder = ViewAwareEncoder(prob_dim=prob_dimension, hidden_dim=hidden)
        
        self.blocks = nn.ModuleList([
            FullGraphTransformerBlock(hidden, heads=heads, edge_dim=4, dropout=dropout)
            for _ in range(layers)
        ])
        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden * layers

        self.pool_x = AttentionalAggregation(nn.Sequential(
            nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1)
        ))
        self.pool_y = AttentionalAggregation(nn.Sequential(
            nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1)
        ))

        self.head_x = QuantileOutputHead(input_dim=jk_dim + 1, num_points=2, coords=1, quantiles=quantiles)
        self.head_y = QuantileOutputHead(input_dim=jk_dim + 1, num_points=2, coords=1, quantiles=quantiles)
        self.head_z = QuantileOutputHead(input_dim=jk_dim + 1, num_points=2, coords=1, quantiles=quantiles)

    def forward(self, data):
        # ... Encoding & Backbone (Same as before) ...
        if hasattr(data, 'group_probs') and data.group_probs is not None:
            probs = data.group_probs[data.batch]
        else:
            probs = None
        x = self.hit_encoder(data.x, probs)
        
        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)

        # 1. STRICT VIEW MASKING (Hard-coded)
        raw_view = data.x[:, 3].long()
        mask_x = (raw_view == VIEW_X_VAL)
        mask_y = (raw_view == VIEW_Y_VAL)

        # 2. Pooling & PER-GRAPH Valid Bits
        # We define a helper to handle the pooling and counting correctly
        def pool_and_count(mask, pool_layer):
            if mask.any():
                # Pool only the hits belonging to this view
                pooled = pool_layer(x_cat[mask], data.batch[mask], dim_size=data.num_graphs)
                
                # COUNT hits per graph to determine validity
                # Create a zero vector [Num_Graphs]
                counts = torch.zeros(data.num_graphs, device=x.device)
                # Add 1.0 for every hit to its corresponding graph index
                counts.index_add_(0, data.batch, mask.float())
                
                # Valid = 1.0 if this specific graph had > 0 hits
                has_hits = (counts > 0).float().unsqueeze(1)
                return pooled, has_hits
            else:
                # If the entire batch is empty for this view
                return (torch.zeros(data.num_graphs, x_cat.size(1), device=x.device),
                        torch.zeros(data.num_graphs, 1, device=x.device))

        # Apply the helper
        pool_x, has_x = pool_and_count(mask_x, self.pool_x)
        pool_y, has_y = pool_and_count(mask_y, self.pool_y)

        # 3. Prediction
        out_x = self.head_x(torch.cat([pool_x, data.u], dim=1)) 
        out_y = self.head_y(torch.cat([pool_y, data.u], dim=1)) 
        
        # --- FIX: STEREO MEAN AGGREGATION ---
        # 1. Sum the valid feature vectors
        #    If X is valid, we add Pool_X. If invalid (Valid_X=0), we add 0.
        sum_feat = (pool_x * has_x) + (pool_y * has_y)
        
        # 2. Count how many views contributed (0, 1, or 2)
        valid_count = has_x + has_y
        
        # 3. Safe Division (Avoid divide-by-zero for empty events)
        #    If count is 0, we divide by 1.0 (result is still 0.0)
        valid_count = valid_count.clamp(min=1.0) 
        
        # 4. Compute the actual Mean Feature
        #    If Both: (X+Y)/2
        #    If X Only: (X+0)/1 = X  <-- RESTORES FULL SIGNAL STRENGTH
        stereo_feat = sum_feat / valid_count
        
        # 5. Feed the CLEAN mean to the Z-head
        #    Note: Z-head input_dim must be reduced to (jk_dim + 1) in __init__
        out_z = self.head_z(torch.cat([stereo_feat, data.u], dim=1)) 

        return torch.cat([out_x, out_y, out_z], dim=2)



class GroupSplitter(nn.Module):
    def __init__(self, in_channels=4, prob_dimension=3, hidden=128, heads=4,
                 layers=3, dropout=0.1, num_classes=3):
        super().__init__()

        self.input_proj = nn.Linear(in_channels+prob_dimension, hidden)

        self.blocks = nn.ModuleList([
            FullGraphTransformerBlock(
                hidden, heads=heads, edge_dim=4, dropout=dropout
            )
            for _ in range(layers)
        ])

        self.node_head = nn.Linear(hidden + 1, num_classes)

        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        ))

        self.energy_head = nn.Linear(hidden + 1, num_classes)

    def forward(self, data):
        if hasattr(data, 'group_probs') and data.group_probs is not None:
            # data.group_probs is [B, 3], data.x is [N, 4]
            # Broadcast probabilities to each node in the group
            probs_expanded = data.group_probs[data.batch] # [N, 3]
            x = self.input_proj(torch.cat([data.x, probs_expanded], dim=1))
        else:
            # Fallback if no probs (though dims might mismatch if layers expect 7)
            # Assuming input_proj handles the 4-dim case or this branch isn't taken when configured for 7
            # For now, if we expect 7 features but get 4, this will fail.
            # If input_proj expects 7, we MUST provide 7.
            if self.input_proj.in_features > data.x.shape[1]:
                 # Pad with zeros if probs are missing but expected
                 padding = torch.zeros(data.x.shape[0], self.input_proj.in_features - data.x.shape[1], device=data.x.device)
                 x = self.input_proj(torch.cat([data.x, padding], dim=1))
            else:
                 x = self.input_proj(data.x)
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
        
        # Broadcast global energy to each node and concatenate
        # Node prediction (PDG)
        u_expanded = data.u[data.batch]
        node_out = torch.cat([x, u_expanded], dim=1)
        node_logits = self.node_head(node_out)

        # Graph prediction (Total Energy per Class)
        pooled = self.pool(x, data.batch)
        graph_out = torch.cat([pooled, data.u], dim=1)
        energy_preds = self.energy_head(graph_out)

        return node_logits, energy_preds



class PionStopRegressor(nn.Module):
    def __init__(self, in_channels=4, hidden=128, heads=4,
                 layers=3, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden)

        self.blocks = nn.ModuleList([
            FullGraphTransformerBlock(
                hidden, heads=heads, edge_dim=4, dropout=dropout
            )
            for _ in range(layers)
        ])

        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden * layers

        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(jk_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        ))

        self.head = nn.Sequential(
            nn.Linear(jk_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        out = torch.cat([pooled, data.u], dim=1)
        return self.head(out)


class PositronAngleModel(nn.Module):
    def __init__(self, in_channels=4, hidden=128, heads=4,
                 layers=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden)

        # Full transformer-style blocks
        self.blocks = nn.ModuleList([
            FullGraphTransformerBlock(
                hidden, heads=heads, edge_dim=4, dropout=dropout
            )
            for _ in range(layers)
        ])

        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden * layers

        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(jk_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        ))

        self.head = nn.Sequential(
            nn.Linear(jk_dim + 3, hidden), #Adds pion stop position
            nn.ReLU(),
            nn.Linear(hidden, 3)   # predicts unit vector corresponding to theta/phi
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        xs = []

        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)

        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        out = torch.cat([pooled, data.pred_pion_stop], dim=1)
        return self.head(out)


from torch_geometric.utils import scatter

class EventBuilder(nn.Module):
    def __init__(self, in_channels=25, hidden=128, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(in_channels, hidden)
        
        # Backbone
        self.blocks = nn.ModuleList([
            FullGraphTransformerBlock(hidden, heads=heads, edge_dim=5, dropout=dropout)
            for _ in range(layers)
        ])
        
        # --- POOLING STRATEGY ---
        # We will produce two vectors per group:
        # 1. Global Max (to capture Bragg peaks/hotspots regardless of view)
        # 2. Stereo Mean (The "Smart" Average from your Regressor)
        # Input to MLP is hidden * 2
        
        # Interaction Head
        self.affinity_mlp = nn.Sequential(
            nn.Linear(hidden * 4, hidden), # Input: [Group_i, Group_j]
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, group_indices, batch_indices_per_group):
        """
        Args:
            x: [NumHits, 25] (Assumes col 3 is View ID: 0 for X, 1 for Y)
            group_indices: [NumHits]
            batch_indices_per_group: [NumGroups]
        """
        # 1. Embed & Graph Process
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
            
        num_groups = batch_indices_per_group.size(0)
        
        # --- 2. STEREO-AWARE POOLING (The Upgrade) ---
        
        # A. Global Max Pooling 
        # (We keep this simple: just finding the hottest hit in the group)
        g_max = scatter(h, group_indices, dim=0, dim_size=num_groups, reduce='max')
        
        # B. Smart Stereo Mean Pooling (Mimicking OrthogonalEndpointRegressor)
        # Identify views (Assumes Column 3 is View ID)
        raw_view = x[:, 3].long()
        mask_x = (raw_view == 0) # Adjust constant if needed
        mask_y = (raw_view == 1)
        
        # Helper to pool specific hits
        def pool_view_specific(mask):
            if not mask.any():
                return torch.zeros(num_groups, h.size(1), device=h.device), torch.zeros(num_groups, 1, device=h.device)
            
            # Pool hits for this view
            pooled = scatter(h[mask], group_indices[mask], dim=0, dim_size=num_groups, reduce='mean')
            
            # Determine validity (Did this group have ANY hits in this view?)
            # We sum 1.0 for every hit
            counts = scatter(torch.ones_like(group_indices[mask], dtype=torch.float), 
                             group_indices[mask], dim=0, dim_size=num_groups, reduce='sum')
            has_hits = (counts > 0).float().unsqueeze(1)
            
            return pooled, has_hits

        # Get separate representations
        pool_x, has_x = pool_view_specific(mask_x)
        pool_y, has_y = pool_view_specific(mask_y)
        
        # Combine Logic (The "Regressor" Logic)
        # If X is valid, add X. If Y is valid, add Y.
        sum_feat = (pool_x * has_x) + (pool_y * has_y)
        
        # Valid Count is 0, 1, or 2
        valid_count = (has_x + has_y).clamp(min=1.0)
        
        # The Result:
        # If Both: (Mean_X + Mean_Y) / 2
        # If X Only: (Mean_X + 0) / 1  <-- Preserves signal, doesn't dilute
        g_stereo_mean = sum_feat / valid_count
        
        # --- 3. Interaction Matrix ---
        
        # Concatenate: [Stereo_Mean, Global_Max]
        group_embs = torch.cat([g_stereo_mean, g_max], dim=1)
        
        # Expand for All-to-All
        N = num_groups
        left = group_embs.unsqueeze(1).expand(N, N, -1)
        right = group_embs.unsqueeze(0).expand(N, N, -1)
        
        pair_features = torch.cat([left, right], dim=-1)
        
        scores = self.affinity_mlp(pair_features).squeeze(-1)
        
        # Symmetry
        scores = (scores + scores.t()) / 2.0
        
        # Masking
        batch_ids = batch_indices_per_group.unsqueeze(1)
        event_mask = (batch_ids == batch_ids.T)
        
        probs = self.output_act(scores) * event_mask.float()
        
        return probs
