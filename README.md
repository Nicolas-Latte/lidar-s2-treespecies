# Tree species from airborne LiDAR point clouds and Sentinel-2 (code of the paper)

Code of the network, the augmentation, the training and the prediction described in *Regional-scale individual tree species mapping using deep learning, airborne LiDAR, Sentinel-2 and heterogeneous forest inventory data* (Latte, Delplanque, Lhoest, Dethier, Lisein and Lejeune).
Not included: the R scripts for tree segmentation and for the preparation of the reference data, the reference data, and the trained weights. `train.py` reproduces the training procedure on your own reference segments.

| File | Content |
|---|---|
| `model.py` | the network (gated MLP, two point-transformer set-abstraction modules, global pooling, segment-descriptor branch, classifier) |
| `data.py` | normalization of a segment, augmentation, batches with weighted sampling |
| `class_weights.py` | standard and regional macro F1 and the fit of the decision weights d_c (the file `class_weights.csv` holds the d_c) |
| `train.py` | training and validation, 375 epochs |
| `predict.py` | species of the segments of a point table, with the weights of a training run |
| `data/` | `feature_stats.csv` (mean and standard deviation of the features), `class_weights.csv` (decision weights d_c of the paper), `prior.csv` (class shares r_c of Wallonia) |

## Input
One tree segment = its LiDAR points (first returns) with, for every point, X, Y, Z (m) and 42 features: 40 Sentinel-2 bands (four seasonal composites of ten bands), LiDAR intensity, and a flag (`plus`): 0 = point of the target tree, 1 = top point of a neighboring tree, added as context. Names and order of the 41 continuous features: `data/feature_stats.csv` (with the mean and standard deviation used to z-score them). The six segment descriptors are computed from the target points in `data.py`.

## Classes
0 Oaks, 1 Beech, 2 Birches, 3 Poplars, 4 Other broadleaves, 5 Spruce, 6 Douglas-fir, 7 Pines, 8 Larches.

## Training
```
python train.py --table segments.csv --seg-dir segments/ --stats data/feature_stats.csv --prior data/prior.csv --out run/
```
`segments.csv`: `id_seg, class_id, split (train / valid), weight`; one `segments/<id_seg>.npz` per segment (`pos` [N,3], `x` [N,42], raw values). At every validation (every fifth epoch) the log gives the standard macro F1 (`standard_f1`: argmax, every validation segment counted once) and the regional macro F1 (`regional_f1`: argmax of the probabilities times the decision weights d_c, validation segments weighted by r_c / q_c; `regional_f1_argmax` is the same with the plain argmax); `run/weights.pt` and `run/class_weights.csv` are those of the epoch with the best regional macro F1. `data/prior.csv` is the share r_c of each class among the canopy stems of the Regional Forest Inventory (replace it for another region); q_c is the share of each class among the validation segments.

## Prediction
```
python predict.py --points points.parquet --weights run/weights.pt --class-weights run/class_weights.csv --stats data/feature_stats.csv --out species.csv
```
`class_map` is the species of the map: argmax of the softmax probability times the decision weight d_c of the class; `class_argmax` is the standard argmax.

## Map
The wall-to-wall species map of Wallonia can be downloaded at https://dox.uliege.be/s/YioDKTHnToGtcAD in two versions: with the decision weights (`pred_wal`) and with the standard argmax (`pred_argmax`), each with its QGIS style file (`.qml`).

## Requirements
Python 3.11, PyTorch 2.3.1 (CUDA 12.1), PyTorch Geometric 2.8.0 with `torch-cluster`, numpy, pandas, pyarrow. See `requirements.txt`.

## Licence and citation
MIT licence (see `LICENSE`). Citation: [to be added after publication]

## AI assistance
The code was written and reviewed with AI assistance (Claude Code, Anthropic); see the declaration in the paper.
