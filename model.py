"""Network of the paper: PointNet++-type classifier of one tree segment (point cloud + 6 segment descriptors).

Input of one batch (torch_geometric style object):
    x      [N, 42]  per-point features: 40 Sentinel-2 bands, LiDAR intensity, neighbor-context flag (last column)
    pos    [N, 3]   normalized point coordinates
    batch  [N]      index of the tree segment of every point
    extra  [B, 6]   segment descriptors
Output: class logits [B, 9]. Softmax and decision weights are applied outside (see class_weights.py).
"""

import torch
import torch.nn as nn
from torch_cluster import fps, radius
from torch_geometric.nn import MLP, global_max_pool, global_mean_pool
from torch_geometric.nn.conv import PointTransformerConv

N_IN = 42  # 40 bands + intensity + neighbor flag
N_CLASSES = 9


class GatedMLP(nn.Module):
    """Per-point MLP with a sigmoid gate: out = proj(extract(x) * gate(x))."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=128, dropout=0.25):
        super().__init__()
        self.extract = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.proj(self.extract(x) * self.gate(x))


class SetAbstraction(nn.Module):
    """Farthest point sampling (ratio), radius grouping (r), point-transformer convolution, subtraction of the sampled point's own feature."""

    def __init__(self, ratio, r):
        super().__init__()
        self.ratio, self.r = ratio, r
        self.conv = PointTransformerConv(in_channels=[128, 128], out_channels=128, pos_nn=MLP([3, 32, 64, 128]),
                                         attn_nn=MLP([128, 64, 64, 128]), aggr="add")

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)  # fixed start at inference: reproducible predictions
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        x_dst = x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), torch.stack([col, row], dim=0)) - x_dst
        return x, pos[idx], batch[idx]


class GlobalSetAbstraction(nn.Module):
    """All remaining points -> one vector per segment, by max and mean pooling (2 x 512)."""

    def __init__(self):
        super().__init__()
        self.nn = MLP([128 + 3, 256, 512])

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        return torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=-1)


class TreeSpeciesNet(nn.Module):
    def __init__(self, num_classes=N_CLASSES, in_dim=N_IN, embedding_dim=128):
        super().__init__()
        self.mlp_ = GatedMLP(in_dim)
        self.sa1_module = SetAbstraction(ratio=0.7, r=0.3)  # keeps 70 % of the points, radius 0.3 (normalized units)
        self.sa2_module = SetAbstraction(ratio=0.5, r=0.5)  # keeps 50 % of them, radius 0.5
        self.sa3_module = GlobalSetAbstraction()
        self.mlpextra = MLP([6, 32, 64])  # segment descriptors
        self.mlp = MLP([1024 + 64, 512, 256, embedding_dim], dropout=[0.5, 0.25, 0.0])
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x = self.mlp_(x)
        x, pos, batch = self.sa1_module(x, pos, batch)
        x, pos, batch = self.sa2_module(x, pos, batch)
        g = self.sa3_module(x, pos, batch)  # [B, 1024]
        return self.classifier(self.mlp(torch.cat([g, self.mlpextra(data.extra)], dim=1)))
