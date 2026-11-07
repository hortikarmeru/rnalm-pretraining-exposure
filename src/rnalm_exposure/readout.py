"""Zero-shot base-pair readout from attention maps, following Wang et al. (2026) and their released code.

Scores: symmetrize each head's map (A + A^T), apply APC on the full map including the BOS/EOS
tokens, drop those tokens, then take the sigmoid. A pair (i, j) with j > i is predicted when its
score is >= the head's threshold. Heads and thresholds are chosen by F1 pooled over a selection set
(TP, FP and FN summed over all its sequences).

F1 and MCC use Wang's formulas, including their eps = 1e-6 terms, so the numbers can be compared
with theirs directly.
"""
import numpy as np
import torch

EPS = 1e-6


def contact_scores(attentions, n_prefix=1, n_suffix=1):
    """attentions: [layers, heads, T, T] for one sequence. Returns [layers, heads, L, L] float32 scores,
    L = T - n_prefix - n_suffix (the special tokens are dropped after APC, as in Wang's code)."""
    x = attentions.float()
    x = x + x.transpose(-1, -2)
    a1 = x.sum(-1, keepdim=True)
    a2 = x.sum(-2, keepdim=True)
    x = x - a1 * a2 / x.sum((-1, -2), keepdim=True)
    x = x[..., n_prefix:x.shape[-2] - n_suffix, n_prefix:x.shape[-1] - n_suffix]
    return torch.sigmoid(x)


def thresholds(grid):
    """[start, stop, step] -> the threshold values, as np.arange(0, 1 + step, step) in Wang's code."""
    start, stop, step = grid
    return np.arange(start, stop + step, step)


def upper(scores, targets):
    """Flatten the j > i entries. scores: [..., L, L] tensor; targets: [L, L] 0/1 array."""
    L = targets.shape[0]
    i, j = torch.triu_indices(L, L, offset=1, device=scores.device)
    return scores[..., i, j], torch.as_tensor(targets, device=scores.device)[i, j].bool()


def threshold_counts(scores, targets, grid):
    """Predicted-positive counts at every threshold, for one sequence and every head at once.

    scores: [heads, n] (upper-triangle scores), targets: [n] bool, grid: float64 tensor of thresholds.
    Returns (tp, pp): [heads, len(grid)] int64, tp = true pairs with score >= t, pp = all pairs with
    score >= t. Same comparison as Wang's torch.greater_equal(pred, T) with T in float64."""
    k = len(grid)
    # idx = number of thresholds <= score, so score >= grid[m] exactly when idx > m
    idx = torch.searchsorted(grid, scores.double().contiguous(), right=True)
    rows = torch.arange(scores.shape[0], device=scores.device)[:, None] * (k + 1)
    hist_all = torch.bincount((idx + rows).ravel(), minlength=scores.shape[0] * (k + 1)).view(-1, k + 1)
    hist_pos = torch.bincount((idx + rows)[:, targets].ravel(), minlength=scores.shape[0] * (k + 1)).view(-1, k + 1)
    # count of idx >= m + 1 for m = 0..k-1: reverse cumulative sum, dropping bin 0
    pp = hist_all.flip(-1).cumsum(-1).flip(-1)[:, 1:]
    tp = hist_pos.flip(-1).cumsum(-1).flip(-1)[:, 1:]
    return tp, pp


def f1_mcc(tp, fp, fn, tn):
    """Wang's formulas (resm_tools.py), elementwise on arrays or floats."""
    prec = tp / (tp + fp + EPS)
    sens = tp / (tp + fn + EPS)
    f1 = 2 * (prec * sens) / (prec + sens + EPS)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + EPS)
    return f1, mcc


def select_head(tp, pp, n_true, n_total, grid):
    """Choose layer, head and threshold by pooled F1.

    tp, pp: [layers, heads, len(grid)] counts summed over the selection set; n_true, n_total: summed
    numbers of true pairs and of all j > i pairs. Ties go to the first maximum in (layer, head,
    threshold) order, as with np.where / np.nanargmax in Wang's code."""
    tp, pp = np.asarray(tp, dtype=np.float64), np.asarray(pp, dtype=np.float64)
    fp, fn = pp - tp, n_true - tp
    f1, _ = f1_mcc(tp, fp, fn, n_total - tp - fp - fn)
    best_t = np.nanargmax(f1, axis=-1)
    max_f1 = np.take_along_axis(f1, best_t[..., None], -1)[..., 0]
    layer, head = (int(v[0]) for v in np.where(max_f1 == max_f1.max()))
    return {"layer": layer, "head": head, "threshold": float(grid[best_t[layer, head]]),
            "pooled_f1": float(max_f1[layer, head]), "max_f1": max_f1, "thresholds": grid[best_t]}


def score_pairs(scores, targets, threshold):
    """F1 and MCC of one sequence for one head's [L, L] scores at a fixed threshold."""
    s, t = upper(scores, targets)
    pred = s.double() >= threshold
    tp = int((pred & t).sum())
    fp = int((pred & ~t).sum())
    fn = int((~pred & t).sum())
    tn = int(t.numel()) - tp - fp - fn
    f1, mcc = f1_mcc(float(tp), float(fp), float(fn), float(tn))
    return {"f1": float(f1), "mcc": float(mcc), "tp": tp, "fp": fp, "fn": fn}
