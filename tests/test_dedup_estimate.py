import gzip
import importlib.util
import random

from rnalm_exposure.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("est", REPO_ROOT / "scripts" / "03_estimate_dedup_size.py")
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)


def brute_force_clusters(seqs):
    """Representatives under cd-hit-est -c 1.0 -r 1: longest first, ties by file order; a sequence
    joins a cluster if it occurs (either strand) inside an existing representative."""
    order = sorted(range(len(seqs)), key=lambda i: (-len(seqs[i]), i))
    reps = []
    for i in order:
        s = seqs[i]
        if not any(s in r or est.reverse_complement(s) in r for r in reps):
            reps.append(s)
    return len(reps)


def test_matches_brute_force_when_everything_is_sampled(tmp_path):
    rng = random.Random(1)
    base = ["".join(rng.choice("ACGT") for _ in range(rng.randint(20, 60))) for _ in range(40)]
    seqs = list(base)
    for s in base[:25]:  # substrings, exact copies, reverse complements, lowercase/T variants
        a = rng.randint(0, 5)
        seqs += [s[a:a + 18], s, s.translate(str.maketrans("ACGT", "TGCA"))[::-1], s.lower()]
    path = tmp_path / "x.fasta.gz"
    with gzip.open(path, "wt") as f:
        for i, s in enumerate(seqs):
            f.write(f">r{i}\n{s}\n")
    normalized = [s.upper().replace("T", "U").encode() for s in seqs]
    expected = brute_force_clusters(normalized)
    result = est.estimate(("t", [path], {"seed": 0, "anchor_length": 8, "sample_size": len(seqs)}))
    assert result["estimated_clusters"] == expected
