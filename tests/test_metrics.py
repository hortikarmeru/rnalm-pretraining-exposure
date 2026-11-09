from rnalm_exposure.metrics import f1_mcc, score_pair_sets
from rnalm_exposure.structures import dot_bracket_pairs


def test_score_pair_sets():
    ref = dot_bracket_pairs("((..))")        # (0, 5), (1, 4)
    pred = dot_bracket_pairs("(....)")       # (0, 5)
    r = score_pair_sets(pred, ref, 6)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 1)
    assert r["f1"] == f1_mcc(1.0, 0.0, 1.0, 15.0 - 2)[0]
    assert abs(score_pair_sets(ref, ref, 6)["f1"] - 1) < 1e-5
