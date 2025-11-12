"""Reusable GNN model definitions built around the standardized graph features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, JumpingKnowledge, AttentionalAggregation


class GroupClassifier(nn.Module):
    def __init__(self, in_dim=5, edge_dim=4, hidden=200, heads=4, num_blocks=2, dropout=0.1, num_classes=3):
        super().__init__()
        self.input_embed = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim, dropout=dropout, concat=True, beta=True)
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
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim // 2, num_classes)
        )

    def forward(self, data):
        x = self.input_embed(data.x)
        xs = []
        for conv in self.blocks:
            x = conv(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        return self.head(pooled)


class GroupAffinityModel(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=128, heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList([
            TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=4, dropout=dropout, concat=True, beta=True)
            for _ in range(num_layers)
        ])
        self.jk = JumpingKnowledge(mode="cat")
        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(hidden_channels * num_layers, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        ))
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * num_layers, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        xs = []
        for conv in self.layers:
            x = conv(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        return self.head(pooled)


class EndpointRegressor(nn.Module):
    def __init__(self, in_channels=5, hidden=160, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.blocks = nn.ModuleList([
            TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=4, dropout=dropout, concat=True, beta=True)
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
            nn.Linear(jk_dim, jk_dim),
            nn.ReLU(),
            nn.Linear(jk_dim, 6)
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        xs = []
        for conv in self.blocks:
            x = conv(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        return self.head(pooled).view(-1, 2, 3)


class GroupSplitter(nn.Module):
    def __init__(self, in_channels=5, hidden=128, heads=4, layers=3, dropout=0.1, num_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.blocks = nn.ModuleList([
            TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=4, dropout=dropout, concat=True, beta=True)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = self.input_proj(data.x)
        for conv, norm in zip(self.blocks, self.norms):
            x = conv(x, data.edge_index, data.edge_attr)
            x = norm(F.relu(x))
        return self.head(x)


class PionStopRegressor(nn.Module):
    def __init__(self, in_channels=5, hidden=128, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.blocks = nn.ModuleList([
            TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=4, dropout=dropout, concat=True, beta=True)
            for _ in range(layers)
        ])
        self.jk = JumpingKnowledge(mode="cat")
        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(hidden * layers, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        ))
        self.head = nn.Sequential(
            nn.Linear(hidden * layers, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        xs = []
        for conv in self.blocks:
            x = conv(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        return self.head(pooled)


class PositronAngleModel(nn.Module):
    def __init__(self, in_channels=5, hidden=128, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.blocks = nn.ModuleList([
            TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=4, dropout=dropout, concat=True, beta=True)
            for _ in range(layers)
        ])
        self.jk = JumpingKnowledge(mode="cat")
        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(hidden * layers, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        ))
        self.head = nn.Sequential(
            nn.Linear(hidden * layers, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        xs = []
        for conv in self.blocks:
            x = conv(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        return self.head(pooled)
