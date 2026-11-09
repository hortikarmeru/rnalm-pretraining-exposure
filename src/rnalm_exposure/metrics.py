"""Base-pair F1 and MCC with Wang et al. (2026)'s formulas, including their eps = 1e-6 terms
(resm_tools.py). Shared by the attention readout and the ViennaRNA baseline."""
import numpy as np

EPS = 1e-6


def f1_mcc(tp, fp, fn, tn):
    """Wang's formulas, elementwise on arrays or floats."""
    prec = tp / (tp + fp + EPS)
    sens = tp / (tp + fn + EPS)
    f1 = 2 * (prec * sens) / (prec + sens + EPS)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + EPS)
    return f1, mcc


def score_pair_sets(predicted, reference, length):
    """F1 and MCC of predicted vs reference pairs (sets of (i, j), i < j) over all j > i pairs."""
    tp = len(predicted & reference)
    fp, fn = len(predicted) - tp, len(reference) - tp
    tn = length * (length - 1) // 2 - tp - fp - fn
    f1, mcc = f1_mcc(float(tp), float(fp), float(fn), float(tn))
    return {"f1": float(f1), "mcc": float(mcc), "tp": tp, "fp": fp, "fn": fn}
