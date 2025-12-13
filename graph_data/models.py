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



class EndpointRegressor(nn.Module):
    def __init__(self, in_channels=4, hidden=160, heads=4,
                 layers=2, dropout=0.1):
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
            nn.Linear(jk_dim, jk_dim // 2),
            nn.ReLU(),
            nn.Linear(jk_dim // 2, 1)
        ))

        self.head = nn.Sequential(
            nn.Linear(jk_dim + 1, jk_dim),
            nn.ReLU(),
            nn.Linear(jk_dim, 6)
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)

        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        out = torch.cat([pooled, data.u], dim=1)
        return self.head(out).view(-1, 2, 3)



class GroupSplitter(nn.Module):
    def __init__(self, in_channels=4, hidden=128, heads=4,
                 layers=3, dropout=0.1, num_classes=3):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden)

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

