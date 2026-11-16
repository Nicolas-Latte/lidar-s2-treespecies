"""Species of the segments of a point table: softmax probabilities, standard argmax and decision weights d_c (the species of the map).

    python predict.py --points points.parquet --weights run/weights.pt --class-weights run/class_weights.csv --stats data/feature_stats.csv --out species.csv

points: one row per LiDAR point (first returns), columns id_seg, X, Y, Z, the 40 Sentinel-2 band columns and Intensity (names as in
feature_stats.csv), and plus (0 = target tree, 1 = top point of a neighboring tree). Output: one row per segment with prob_<class_id>
columns, class_argmax, class_map (argmax of p * decision weight, file class_weights.csv) and prob_map (probability of that class).
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from data import load_stats, make_sample
from model import TreeSpeciesNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--class-weights", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TreeSpeciesNet().to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device))
    model.eval()
    stats = load_stats(a.stats)
    cols = pd.read_csv(a.stats)["variable"].tolist() + ["plus"]  # 40 bands, Intensity, plus: the 42 inputs, in this order
    cw = pd.read_csv(a.class_weights).sort_values("class_id")["class_weight"].to_numpy()

    pts = pd.read_parquet(a.points) if a.points.endswith(".parquet") else pd.read_csv(a.points)
    groups = list(pts.groupby("id_seg"))
    ids, probs = [], []
    with torch.no_grad():
        for i in range(0, len(groups), a.batch):
            chunk = groups[i:i + a.batch]
            batch = Batch.from_data_list([make_sample(g[["X", "Y", "Z"]].to_numpy(), g[cols].to_numpy(), stats) for _, g in chunk]).to(device)
            probs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
            ids += [k for k, _ in chunk]
    probs = np.concatenate(probs)
    cls = (probs * cw).argmax(axis=1)
    out = pd.DataFrame(probs, columns=[f"prob_{c}" for c in range(probs.shape[1])])
    out.insert(0, "id_seg", ids)
    out["class_argmax"], out["class_map"], out["prob_map"] = probs.argmax(axis=1), cls, probs[np.arange(len(cls)), cls]
    out.to_csv(a.out, index=False)
    print(f"{len(out)} segments -> {a.out}")


if __name__ == "__main__":
    main()
