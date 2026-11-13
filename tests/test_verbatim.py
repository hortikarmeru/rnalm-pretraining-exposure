import numpy as np
import pytest

from rnalm_exposure.verbatim import fill_spans, min_k, place_spans, self_repeat, tiled_masks

LETTERS = [4, 5, 6, 7]   # token ids of A, C, G, U
MASK = 9


def test_place_spans_is_deterministic_and_keeps_flanks():
    a = place_spans("set:x", 100, [5, 10, 20, 50], 2, 10, 2026)
    assert a == place_spans("set:x", 100, [5, 10, 20, 50], 2, 10, 2026)
    assert [(k, j) for k, j, _ in a] == [(k, j) for k in (5, 10, 20, 50) for j in (0, 1)]
    for k, _, start in a:
        assert 10 <= start and start + k <= 90
    assert a != place_spans("set:y", 100, [5, 10, 20, 50], 2, 10, 2026)


def test_place_spans_one_start_per_part():
    for seq_id in ("a", "b", "c", "d"):
        (_, _, first), (_, _, second) = place_spans(seq_id, 60, [10], 2, 10, 0)   # starts 10..40: parts 10..24 and 25..40
        assert 10 <= first < 25 <= second <= 40


def test_place_spans_skips_lengths_that_do_not_fit():
    assert {k for k, _, _ in place_spans("s", 69, [5, 50], 2, 10, 0)} == {5}    # 50 + 20 leaves no start
    assert {k for k, _, _ in place_spans("s", 70, [5, 50], 2, 10, 0)} == {5}    # one start, two wanted
    assert {k for k, _, _ in place_spans("s", 71, [5, 50], 2, 10, 0)} == {5, 50}
    assert [s for _, _, s in place_spans("s", 71, [50], 2, 10, 0)] == [10, 11]


def test_self_repeat():
    assert self_repeat("ACGUACGUGG", 0, 4)
    assert self_repeat("ACGUACGUGG", 4, 4)
    assert not self_repeat("ACGUACGUGG", 6, 4)
    assert self_repeat("AAAAAC", 0, 4)   # overlapping copy


def tokens(seq):
    return np.array([0] + [LETTERS["ACGU".index(c)] for c in seq] + [2])


def test_fill_commits_most_confident_first():
    """A model that is sure of position 4 from the start and of an earlier span position only once
    its right neighbour is filled: the fill recovers the span, the first pass alone does not."""
    truth = "ACGUACGUAC"
    ids = tokens(truth)
    calls = []

    def predict(x):
        calls.append(x.copy())
        out = np.full((len(x), len(truth), 4), np.log(0.25))
        for r in range(len(x)):
            for p in range(len(truth)):
                if x[r, p + 1] == MASK and (p == 4 or (p < 4 and x[r, p + 2] != MASK)):
                    out[r, p] = np.log(0.01)
                    out[r, p, "ACGU".index(truth[p])] = np.log(0.97)
        return out

    filled, one_pass = fill_spans(predict, ids, [(2, 3)], LETTERS, MASK)
    assert "".join("ACGU"[i] for i in filled[0]) == truth[2:5]
    assert list(one_pass[0]) == [0, 0, 0]          # ties at 0.25 give A; only the last letter (A) is right
    assert [int((c == MASK).sum()) for c in calls] == [3, 2, 1]
    assert calls[1][0, 5] == ids[5] and calls[1][0, 3] == MASK   # position 4 went first
    assert (ids == tokens(truth)).all()            # the input is not changed


def test_fill_rows_are_independent_and_finish_separately():
    truth = "ACGUACGUACGU"
    ids = tokens(truth)
    sizes = []

    def predict(x):
        sizes.append(len(x))
        assert all((row == MASK).sum() > 0 for row in x)
        out = np.full((len(x), len(truth), 4), np.log(0.1))
        for p, c in enumerate(truth):
            out[:, p, "ACGU".index(c)] = np.log(0.7)
        return out

    filled, one_pass = fill_spans(predict, ids, [(1, 2), (5, 4)], LETTERS, MASK)
    assert sizes == [2, 2, 1, 1]
    assert "".join("ACGU"[i] for i in filled[0]) == truth[1:3]
    assert "".join("ACGU"[i] for i in filled[1]) == truth[5:9]
    assert list(one_pass[1]) == list(filled[1])


def test_fill_writes_token_ids_and_breaks_ties_leftmost():
    ids = tokens("AAAAAA")
    seen = []

    def predict(x):
        seen.append(x.copy())
        out = np.full((len(x), 6, 4), np.log(0.1))
        out[:, :, 3] = np.log(0.7)   # always U, equally sure everywhere
        return out

    filled, _ = fill_spans(predict, ids, [(1, 3)], LETTERS, MASK)
    assert list(filled[0]) == [3, 3, 3]
    assert list(seen[1][0, 2:5]) == [LETTERS[3], MASK, MASK]
    assert list(seen[2][0, 2:5]) == [LETTERS[3], LETTERS[3], MASK]


def test_tiled_masks_cover_every_position_once():
    tiles = tiled_masks("set:x", 53, 7, 2026)
    assert len(tiles) == 7
    assert sorted(np.concatenate(tiles)) == list(range(53))
    assert {len(t) for t in tiles} == {7, 8}
    assert all((t == tiled_masks("set:x", 53, 7, 2026)[i]).all() for i, t in enumerate(tiles))


def test_min_k():
    assert min_k([-1.0, -5.0, -2.0, -3.0, -4.0], 0.4) == pytest.approx(-4.5)
    assert min_k([-1.0, -5.0, -2.0], 0.2) == pytest.approx(-5.0)    # 0.6 of a value rounds up to one
    assert min_k([-1.0, -3.0], 1.0) == pytest.approx(-2.0)
