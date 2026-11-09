import pytest

from rnalm_exposure.structures import parse_bpseq, parse_bprna_st, parse_ct
from rnalm_exposure.test_sets import filter_sets

CT = """6 ENERGY = -1.0 toy
1 G 0 2 6 1
2 g 1 3 5 2
3 A 2 4 0 3
4 A 3 5 0 4
5 C 4 6 2 5
6 C 5 0 1 6
"""

BPSEQ = """# toy
1 G 6
2 G 5
3 A 0
4 A 0
5 C 2
6 C 1
"""

# helix (0-8, 1-7), a [] pseudoknot 2-9 and a letter pair (Aa) 3-10
ST = """#Name: toy
#Length: 11
GGGGAAAACCC
(([A...))]a
EEEEEEEEEEE
"""


def test_ct():
    assert parse_ct(CT) == ("GgAACC", {(0, 5), (1, 4)})


def test_bpseq():
    assert parse_bpseq(BPSEQ) == ("GGAACC", {(0, 5), (1, 4)})


def test_bprna_st_pseudoknots():
    seq, pairs = parse_bprna_st(ST)
    assert seq == "GGGGAAAACCC"
    assert pairs == {(0, 8), (1, 7), (2, 9), (3, 10)}


def test_ct_rejects_wrong_length():
    with pytest.raises(ValueError):
        parse_ct("7 toy\n" + CT.split("\n", 1)[1])


def rec(dataset, name, seq, pairs=()):
    return {"seq_id": f"{dataset}:{name}", "dataset": dataset, "sequence": seq, "pairs": set(pairs)}


def test_filter_sets():
    sets = {
        "a": [rec("a", "1", "ACGU", [(0, 3)]), rec("a", "2", "ACGU", [(0, 3)]),   # same structure: keep 1
              rec("a", "3", "GGCC", [(0, 3)]), rec("a", "4", "GGCC", [(1, 2)]),   # conflicting: drop both
              rec("a", "5", "ACGN"), rec("a", "6", "ACGUA")],                     # non-ACGU, too long
        "b": [rec("b", "1", "ACGU", [(1, 2)]), rec("b", "2", "UUUU")],           # b:1 is a's sequence
    }
    kept, dropped = filter_sets(sets, max_length=4, priority=["a", "b"])
    assert [r["seq_id"] for r in kept] == ["a:1", "b:2"]
    reasons = {r["seq_id"]: (reason, detail) for r, reason, detail in dropped}
    assert reasons == {
        "a:2": ("duplicate_within_set", "a:1"),
        "a:3": ("duplicate_within_set_conflicting", None),
        "a:4": ("duplicate_within_set_conflicting", None),
        "a:5": ("non_acgu", "N"),
        "a:6": ("over_length_cutoff", 5),
        "b:1": ("duplicate_across_sets", "a:1"),
    }
