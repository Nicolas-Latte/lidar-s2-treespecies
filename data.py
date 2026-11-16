"""Input of the network: one tree segment -> normalized sample, training-time augmentation, batches.

A segment is a point cloud: positions (X, Y, Z in m) and 42 raw features per point,
    columns 0-39  Sentinel-2 bands (4 seasonal composites x 10 bands),
    column  40    LiDAR intensity,
    column  41    neighbor-context flag: 0 = point of the target tree, 1 = top point of a neighboring tree (context only).
The 6 segment descriptors are computed here from the target points. Segments are stored one per file: <dir>/<id_seg>.npz with arrays pos [N,3], x [N,42].
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

N_BANDS = 40
EXTRA_DIVS = torch.tensor([30.0, 30.0, 250.0, 15.0, 15.0, 1.0])  # scale of the segment descriptors


# ------------------------------------------------------------------ normalization (no randomness)
def load_stats(csv):
    """feature_stats.csv (variable, mean, sd) of the 41 continuous features -> (mean [42], sd [42]); the flag is left unchanged."""
    s = pd.read_csv(csv)
    mean = torch.tensor(np.append(s["mean"].to_numpy(), 0.0), dtype=torch.float32)
    sd = torch.tensor(np.append(s["sd"].to_numpy(), 1.0), dtype=torch.float32)
    return mean, sd


def crown_stats(pos):
    """Segment descriptors of the target points: 5th and 95th percentile of Z, point count, width in X and Y, width / height."""
    z5, z95 = np.percentile(pos[:, 2], [5, 95])
    wx, wy = (np.percentile(pos[:, i], 95) - np.percentile(pos[:, i], 5) for i in (0, 1))
    return np.array([z5, z95, float(len(pos)), wx, wy, (wx + wy) / 2 / ((z95 - z5) + 0.1)])


def make_sample(pos, x, stats, label=0):
    """Raw segment (pos [N,3], x [N,42]) -> normalized Data: pos centered on the target tree and scaled by the 95th percentile of |pos|,
    features z-scored, segment descriptors divided by EXTRA_DIVS."""
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(x)[:, -1] == 0
    extra = torch.tensor(crown_stats(pos[target] if target.any() else pos), dtype=torch.float32) / EXTRA_DIVS
    pos, x = torch.as_tensor(pos, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32)
    t = torch.as_tensor(target)
    center = pos[t].median(dim=0).values if t.any() else pos.median(dim=0).values
    pos = pos - center
    pos = pos / (torch.quantile(pos.abs(), 0.95) + 1e-6)
    x = (x - stats[0]) / stats[1]
    return Data(pos=pos, x=x, extra=extra.unsqueeze(0), y=torch.tensor(label, dtype=torch.long))


# ------------------------------------------------------------------ augmentation (training only)
_ROTATE_JITTER_SHEAR = T.Compose([T.RandomRotate(degrees=[-180, 180], axis=2), T.RandomJitter(translate=0.01), T.RandomShear(shear=0.1)])


def _sphere(d, radius_range):
    """Random sphere on a random target point: returns (indices of the target points, mask of those inside the sphere)."""
    idx = (d.x[:, -1] == 0).nonzero(as_tuple=False).squeeze(-1)
    center = d.pos[idx[torch.randint(0, len(idx), (1,))]].squeeze(0)
    inside = (d.pos[idx] - center).norm(dim=1) <= torch.empty(1).uniform_(*radius_range).item()
    return idx, inside


def augment_position(d):
    d = _ROTATE_JITTER_SHEAR(d)
    d.pos = d.pos + torch.empty_like(d.pos).uniform_(-0.1, 0.1)  # per-point shift
    d.pos = d.pos * torch.empty(1).uniform_(0.85, 1.15).item()  # isotropic scale
    keep = torch.rand(len(d.pos)) > 0.1  # 10 % of the points dropped
    d.pos, d.x = d.pos[keep], d.x[keep]
    n_target = int((d.x[:, -1] == 0).sum())
    if torch.rand(1).item() < 0.3 and n_target >= 10:  # occlusion: remove the target points inside a sphere
        idx, inside = _sphere(d, (0.2, 0.5))
        if (~inside).sum() >= max(10, 0.2 * n_target):
            keep = torch.ones(len(d.pos), dtype=torch.bool)
            keep[idx[inside]] = False
            d.pos, d.x = d.pos[keep], d.x[keep]
    dup = (torch.rand(len(d.pos)) < 0.1).nonzero(as_tuple=False).squeeze(-1)  # 10 % of the points duplicated with jitter
    if len(dup):
        d.pos = torch.cat([d.pos, d.pos[dup] + torch.randn(len(dup), 3) * 0.05])
        d.x = torch.cat([d.x, d.x[dup] + torch.randn(len(dup), d.x.shape[1]) * 0.05])  # as in the trained model, the flag is jittered too
    if torch.rand(1).item() < 0.3 and (d.x[:, -1] == 0).any():  # locally denser LiDAR: duplicate the target points inside a sphere
        idx, inside = _sphere(d, (0.15, 0.4))
        sel = idx[inside]
        if len(sel):
            dx = d.x[sel].clone()
            dx[:, :-1] += torch.randn_like(dx[:, :-1]) * 0.02
            d.pos = torch.cat([d.pos, d.pos[sel] + torch.randn(len(sel), 3) * 0.02])
            d.x = torch.cat([d.x, dx])
    nb, tg = d.x[:, -1] == 1, d.x[:, -1] == 0  # neighbor points: noise, scaling and rotation relative to the target tree
    if nb.any() and tg.sum() >= 2:
        ext = d.pos[tg].std() if d.pos[tg].std() >= 1e-6 else torch.tensor(0.1)
        ctr = d.pos[tg].mean(dim=0)
        d.pos[nb, :2] += torch.randn(int(nb.sum()), 2) * ext * 0.15
        d.pos[nb, 2] += torch.randn(int(nb.sum())) * ext * 0.08
        if torch.rand(1) < 0.4:
            d.pos[nb] = ctr + (d.pos[nb] - ctr) * (0.8 + torch.rand(1) * 0.4)
        if torch.rand(1) < 0.3:
            th = torch.rand(1) * 2 * math.pi
            c, s = torch.cos(th).squeeze(), torch.sin(th).squeeze()
            r = d.pos[nb, :2] - ctr[:2]
            d.pos[nb, :2] = ctr[:2] + torch.stack([r[:, 0] * c - r[:, 1] * s, r[:, 0] * s + r[:, 1] * c], dim=1)
    return d


def augment_features(d):
    d.x[:, :-1] = d.x[:, :-1] * torch.empty_like(d.x[:, :-1]).uniform_(0.95, 1.05) + torch.empty_like(d.x[:, :-1]).uniform_(-0.05, 0.05)
    if torch.rand(1) < 0.5:  # random drop of 0-10 % of the band columns (haze, missing date)
        d.x[:, :N_BANDS] *= (torch.rand(N_BANDS) > torch.empty(1).uniform_(0.0, 0.10).item()).float()
    nb = d.x[:, -1] == 1
    if nb.any() and torch.rand(1) < 0.5:
        d.x[nb, :-1] += torch.randn(int(nb.sum()), d.x.shape[1] - 1) * 0.1
    return d


def augment(d):
    d = augment_features(augment_position(d))
    d.extra = d.extra * torch.empty_like(d.extra).uniform_(0.9, 1.1)
    return d


# ------------------------------------------------------------------ batches
class SegmentBatches(Dataset):
    """Every item is a whole batch (use DataLoader(batch_size=None)).
    table: DataFrame with id_seg, class_id, split ('train' / 'valid'), weight (sampling weight of the segment).
    train: `steps` batches per epoch drawn with probability proportional to weight, without replacement within a batch, augmented.
    valid: the validation segments in order, not augmented."""

    def __init__(self, table, seg_dir, stats, mode, batch_size=128, steps=100):
        self.t = table[table["split"] == mode].reset_index(drop=True)
        self.dir, self.stats, self.mode, self.bs, self.steps = Path(seg_dir), stats, mode, batch_size, steps
        self.p = self.t["weight"].to_numpy(dtype=float) / self.t["weight"].sum()

    def __len__(self):
        return self.steps if self.mode == "train" else math.ceil(len(self.t) / self.bs)

    def __getitem__(self, i):
        rows = self.t.iloc[np.random.choice(len(self.t), self.bs, replace=False, p=self.p)] if self.mode == "train" else self.t.iloc[i * self.bs:(i + 1) * self.bs]
        out = []
        for id_seg, c in zip(rows["id_seg"], rows["class_id"]):
            z = np.load(self.dir / f"{id_seg}.npz")
            d = make_sample(z["pos"], z["x"], self.stats, int(c))
            out.append(augment(d) if self.mode == "train" else d)
        return Batch.from_data_list(out)
