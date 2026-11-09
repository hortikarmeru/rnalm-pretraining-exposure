import numpy as np
import torch

from rnalm_exposure import readout
from rnalm_exposure.readout import (contact_scores, f1_mcc, one_pair_per_base, score_pairs, select_head,
                                    threshold_counts, thresholds, upper)


def random_case(rng, layers=2, heads=3, L=12):
    attn = torch.tensor(rng.random((layers, heads, L + 2, L + 2)), dtype=torch.float32)
    attn = attn / attn.sum(-1, keepdim=True)   # rows sum to 1, like softmax output
    targets = np.zeros((L, L), dtype=np.int64)
    for i, j in [(0, 11), (1, 10), (2, 9), (3, 4)]:
        targets[i, j] = targets[j, i] = 1
    return attn, targets


def test_scores_match_apc_on_full_map_then_drop_special_tokens():
    attn, _ = random_case(np.random.default_rng(0))
    x = attn.double() + attn.double().transpose(-1, -2)
    x = x - x.sum(-1, keepdim=True) * x.sum(-2, keepdim=True) / x.sum((-1, -2), keepdim=True)
    expected = torch.sigmoid(x)[..., 1:-1, 1:-1]
    assert torch.allclose(contact_scores(attn).double(), expected, atol=1e-6)


def test_threshold_counts_match_a_loop_over_thresholds():
    rng = np.random.default_rng(1)
    grid = torch.tensor(thresholds([0.0, 1.0, 0.001]), dtype=torch.float64)
    attn, targets = random_case(rng)
    s, t = upper(contact_scores(attn).flatten(0, 1), targets)
    s[0, :5] = torch.tensor([0.5, 0.25, 0.0, 1.0, 0.501], dtype=torch.float32)   # scores exactly on the grid
    tp, pp = threshold_counts(s, t, grid)
    for h in range(s.shape[0]):
        for m in range(0, len(grid), 37):
            pred = s[h].double() >= grid[m]
            assert pp[h, m] == pred.sum() and tp[h, m] == (pred & t).sum()


def test_select_head_takes_the_first_maximum():
    grid = np.array([0.0, 0.5, 1.0])
    tp = np.array([[[4, 4, 0], [4, 4, 0]]])       # both heads reach F1 1.0 at threshold 0.5
    pp = np.array([[[10, 4, 0], [10, 4, 0]]])
    best = select_head(tp, pp, n_true=4, n_total=10, grid=grid)
    assert (best["layer"], best["head"], best["threshold"]) == (0, 0, 0.5)
    assert abs(best["pooled_f1"] - 1.0) < 1e-5


def test_score_pairs_counts_only_j_greater_than_i():
    targets = np.zeros((4, 4), dtype=np.int64)
    targets[0, 3] = targets[3, 0] = 1
    scores = torch.zeros((4, 4))
    scores[0, 3] = scores[3, 0] = 0.9
    scores[1, 2] = 0.9                             # false positive above the diagonal
    scores[2, 1] = 0.9                             # below the diagonal: ignored
    r = score_pairs(scores, targets, 0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)
    assert abs(r["f1"] - f1_mcc(1.0, 1.0, 0.0, 4.0)[0]) < 1e-12


def test_one_pair_per_base_matches_a_greedy_loop(monkeypatch):
    monkeypatch.setattr(readout, "HEAD_CHUNK", 2)   # several chunks
    rng = np.random.default_rng(3)
    scores = torch.tensor(rng.random((5, 15, 15)), dtype=torch.float32)
    out = one_pair_per_base(scores)
    for h in range(5):
        used, kept = set(), set()
        for _, i, j in sorted((-float(scores[h, i, j]), i, j) for i in range(15) for j in range(i + 1, 15)):   # ties: lower index first, as torch
            if i not in used and j not in used:
                used |= {i, j}
                kept.add((i, j))
        got = {(i, j) for i in range(15) for j in range(i + 1, 15) if out[h, i, j] >= 0}
        assert got == kept
        assert all(out[h, i, j] == scores[h, i, j] for i, j in kept)
