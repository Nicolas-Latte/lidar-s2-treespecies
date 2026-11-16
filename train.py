"""Training (Section 3.7): 375 epochs of 100 weighted batches of 128 segments, cross-entropy, Adam, cyclic learning rate, validation every
fifth epoch. At every validation the standard and the regional macro F1 of Section 3.5 are logged; the checkpoint with the highest regional macro F1
(decision weights fitted on the validation set, segments weighted by r_c / q_c) is kept.

    python train.py --table segments.csv --seg-dir segments/ --stats feature_stats.csv --prior prior.csv --out run/

segments.csv: id_seg, class_id (0-8), split ('train' / 'valid'), weight (sampling weight); one <seg-dir>/<id_seg>.npz per segment (see data.py).
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

from class_weights import evaluate, load_prior
from data import SegmentBatches, load_stats
from model import TreeSpeciesNet


def worker_init(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, y = [], []
    for batch in loader:
        probs.append(torch.softmax(model(batch.to(device)), dim=1).cpu().numpy())
        y.append(batch.y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--prior", required=True, help="csv class_id, share: class shares r_c of the region (RFI canopy stems)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=375)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=231)
    a = ap.parse_args()

    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, stats, prior = pd.read_csv(a.table), load_stats(a.stats), load_prior(a.prior)
    kw = dict(batch_size=None, shuffle=False, num_workers=a.workers, worker_init_fn=worker_init, persistent_workers=a.workers > 0)
    train = DataLoader(SegmentBatches(table, a.seg_dir, stats, "train"), **kw)
    valid = DataLoader(SegmentBatches(table, a.seg_dir, stats, "valid"), **kw)

    model = TreeSpeciesNet().to(device)
    opt = Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
    # 75-epoch pyramids (37 up, 38 down) between 1e-5 and 1e-4, each lower than the previous one: 5 pyramids in 375 epochs
    sched = CyclicLR(opt, base_lr=1e-5, max_lr=1e-4, step_size_up=37, step_size_down=38, mode="triangular2", cycle_momentum=False)

    best, log = -1.0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        loss = 0.0
        for batch in train:
            batch = batch.to(device)
            opt.zero_grad()
            l = torch.nn.functional.cross_entropy(model(batch), batch.y)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            loss += l.item() / len(train)
        sched.step()
        row = {"epoch": epoch, "train_loss": loss}
        if epoch in (1, a.epochs) or epoch % 5 == 0:
            probs, y = predict_probs(model, valid, device)
            scores, weights = evaluate(y, probs, prior)
            row.update(scores)
            if scores["regional_f1"] > best:  # checkpoint criterion: regional macro F1 (decision weights)
                best = scores["regional_f1"]
                torch.save(model.state_dict(), os.path.join(a.out, "weights.pt"))
                pd.DataFrame({"class_id": range(len(weights)), "class_weight": weights}).to_csv(os.path.join(a.out, "class_weights.csv"), index=False)
        log.append(row)
        pd.DataFrame(log).to_csv(os.path.join(a.out, "training_log.csv"), index=False)
        print({k: round(v, 4) for k, v in row.items()})


if __name__ == "__main__":
    main()
