"""Standard and regional macro F1, and the decision weights (Section 3.5 of the paper).

Shares: r_c = share of class c in the region (``prior``, RFI canopy stems), q_c = share of class c among the validation segments.
standard macro F1: argmax of the softmax probabilities p, every validation segment counted once.
regional macro F1: a segment is assigned to argmax_c(p_c * d_c), with one decision weight d_c per class (called ``a`` below, ``class_weights.csv``
                   in the files), fitted so that the weighted number of segments predicted in every class equals the weighted number of reference
                   segments; precision and recall count each validation segment with the segment weight r_c / q_c of its reference class.
The macro F1 is always the unweighted mean of the nine per-class F1: the weights act on the segments, not on the classes.
"""

import numpy as np
import pandas as pd


def load_prior(csv, n_classes=9):
    """Share r_c of every class among the canopy stems of the Regional Forest Inventory (csv: class_id, share) -> vector summing to 1."""
    p = pd.read_csv(csv).sort_values("class_id")["share"].to_numpy(dtype=float)
    assert len(p) == n_classes
    return p / p.sum()


def segment_weights(y, prior):
    """Weight of every validation segment: r_c / q_c of its reference class (rescaled to a mean of 1)."""
    share = np.bincount(y, minlength=len(prior)) / len(y)
    w = np.where(share > 0, prior / np.where(share > 0, share, 1.0), 0.0)[y]
    return w * len(y) / w.sum()


def fit_decision_weights(y, probs, w=None, n_iter=500, damping=0.5, tol=1e-3):
    """Decision weights d_c (geometric mean 1) by damped iterative proportional adjustment of log d_c, starting from 1."""
    n, k = probs.shape
    w = np.ones(n) if w is None else w
    true_c = np.bincount(y, weights=w, minlength=k)
    floor = 1e-9 * true_c.sum()
    logp, b = np.log(np.clip(probs, 1e-12, 1.0)), np.zeros(k)
    active = true_c > 0
    best_b, best_err = b.copy(), np.inf
    for _ in range(n_iter):
        pred_c = np.bincount((logp + b).argmax(axis=1), weights=w, minlength=k)
        err = np.max(np.abs(pred_c[active] - true_c[active]) / true_c[active])
        if err < best_err:
            best_err, best_b = err, b.copy()
        if err < tol:
            break
        step = damping * (np.log(np.maximum(true_c, floor)) - np.log(np.maximum(pred_c, floor)))
        step[~active] = 0.0
        b = b + np.clip(step, -1.0, 1.0)
        b = b - b[active].mean()
    return np.exp(best_b - best_b[active].mean())


def decide(probs, d=None):
    """Predicted class: argmax_c(p_c * d_c); the standard argmax without decision weights."""
    return probs.argmax(axis=1) if d is None else (probs * d).argmax(axis=1)


def macro_f1(y, pred, w=None, k=9):
    cm = np.zeros((k, k))
    np.add.at(cm, (y, pred), 1.0 if w is None else w)
    tp, p, r = np.diag(cm), cm.sum(axis=0), cm.sum(axis=1)
    prec, rec = np.divide(tp, p, out=np.zeros(k), where=p > 0), np.divide(tp, r, out=np.zeros(k), where=r > 0)
    return float(np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0).mean())


def evaluate(y, probs, prior):
    """The scores logged at every validation: standard macro F1, regional macro F1 of the standard argmax (reference), regional macro F1 with the
    decision weights (the checkpoint criterion). Returns (scores, decision weights)."""
    w = segment_weights(y, prior)
    d = fit_decision_weights(y, probs, w)
    arg = decide(probs)
    return {"standard_f1": macro_f1(y, arg), "regional_f1_argmax": macro_f1(y, arg, w), "regional_f1": macro_f1(y, decide(probs, d), w)}, d
