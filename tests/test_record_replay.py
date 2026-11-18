import importlib.util

import numpy as np

from rnalm_exposure.config import repo_path
from rnalm_exposure.pair_balance import scores


def script_26():
    spec = importlib.util.spec_from_file_location("injection_sets", repo_path("scripts/26_injection_sets.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def toy_file(path, rng):
    """200 distinct sequences; the first 50 are written four times, one record is over the length limit.
    (The sampler draws tokens_per_seed / 100 records, so sequences average over 100 tokens.)"""
    seqs = ["".join(rng.choice(list("ACGT"), int(n))) for n in rng.integers(120, 200, 200)]
    records = [(f"s{i}", s) for i, s in enumerate(seqs)] + [(f"s{i}", s.lower()) for i, s in enumerate(seqs[:50])] * 3
    order = rng.permutation(len(records))
    with open(path, "w") as f:
        for k in order:
            f.write(f">{records[k][0]} some description\n{records[k][1]}\n")
        f.write(">long\n" + "A" * 300 + "\n")
    return {f"s{i}": s.replace("T", "U") for i, s in enumerate(seqs)}


def test_record_replay_budget_held_out_and_repeats(tmp_path):
    rng = np.random.default_rng(0)
    seqs = toy_file(tmp_path / "corpus.fasta", rng)
    draws, held, counts = script_26().record_replay(tmp_path / "corpus.fasta", 200, 20, 40000, [0, 1], 7)
    assert counts["replay_records"] == 351 and counts["longer_than_max_length"] == 1
    assert len(held) == 20 and len(set(held.values())) == 20
    assert all(seqs[name] == seq for name, seq in held.items())
    # every copy of a held-out sequence is set aside
    assert counts["records_held_out"] == sum(4 if int(name[1:]) < 50 else 1 for name in held)
    for s in (0, 1):
        tokens = [len(seq) + 2 for _, seq in draws[s]]
        assert sum(tokens) >= 40000 > sum(tokens[:-1])   # stops at the record that reaches the budget
        assert all(seqs[name] == seq for name, seq in draws[s])   # normalized: uppercase, T -> U
        assert not {name for name, _ in draws[s]} & set(held)
        assert "long" not in {name for name, _ in draws[s]}
    assert draws[0] != draws[1]
    # records are drawn uniformly, so a sequence written four times is drawn about four times as often
    n = {name: 0 for name in seqs if name not in held}
    for name, _ in draws[0] + draws[1]:
        n[name] += 1
    repeated = np.mean([v for name, v in n.items() if int(name[1:]) < 50])
    single = np.mean([v for name, v in n.items() if int(name[1:]) >= 50])
    assert 3.3 < repeated / single < 4.7


def test_record_replay_is_reproducible(tmp_path):
    toy_file(tmp_path / "corpus.fasta", np.random.default_rng(1))
    a = script_26().record_replay(tmp_path / "corpus.fasta", 200, 10, 5000, [0], 3)
    b = script_26().record_replay(tmp_path / "corpus.fasta", 200, 10, 5000, [0], 3)
    assert a == b


def test_scores_reads_the_requested_readout(tmp_path):
    (tmp_path / "28_scores_m_released.tsv").write_text(
        "seq_id\trole\treadout\tf1\tmcc\ttp\tfp\tfn\n"
        "a\teval\tprimary\t0.1\t0\t0\t0\t0\n"
        "a\teval\tapc_off\t0.4\t0\t0\t0\t0\n"
        "b\tsentinel_test\tprimary\t0.2\t0\t0\t0\t0\n")
    assert scores(tmp_path, "m", "released", "eval") == {"a": 0.1}
    assert scores(tmp_path, "m", "released", "eval", readout="apc_off") == {"a": 0.4}
    assert scores(tmp_path, "m", "released", "sentinel") == {"b": 0.2}
