import random

import numpy as np

from rnalm_exposure.containment import LETTER_CODES, representatives

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def cdhit_rules(seqs, min_length):
    """cd-hit-est -c 1.0 -r 1 by brute force: drop sequences of <= min_length, letters other than
    ACGTU become N, longest first with ties in file order, and a sequence joins a cluster if it
    occurs (either strand) inside an existing representative."""
    norm = [s.encode().translate(LETTER_CODES).translate(bytes.maketrans(b"\0\1\2\3\4", b"ACGTN")).decode()
            for s in seqs]
    order = sorted((i for i in range(len(seqs)) if len(seqs[i]) > min_length), key=lambda i: (-len(seqs[i]), i))
    reps = []
    for i in order:
        s = norm[i]
        if not any(s in norm[r] or s.translate(COMPLEMENT)[::-1] in norm[r] for r in reps):
            reps.append(i)
    return sorted(reps)


def run(seqs, **kw):
    codes = np.frombuffer(b"".join(s.encode().translate(LETTER_CODES) for s in seqs), np.uint8)
    lengths = np.array([len(s) for s in seqs], np.int64)
    reps, _ = representatives(codes, lengths, log=lambda *_: None, **kw)
    return list(np.flatnonzero(reps))


def random_seqs(rng, n):
    base = []
    for _ in range(n):
        s = "".join(rng.choice("ACGT") for _ in range(rng.randint(4, 90)))
        if rng.random() < 0.15:   # ambiguity letters, RNA letters, lowercase
            s = "".join(rng.choice("NRYacgtU") if rng.random() < 0.05 else c for c in s)
        base.append(s)
    seqs = list(base)
    for s in base[: n // 2]:
        a = rng.randint(0, max(0, len(s) - 5))
        b = rng.randint(a + 1, len(s))
        rc = s.upper().replace("U", "T").translate(str.maketrans("ACGTNRY", "TGCANYR"))[::-1]
        seqs += [s[a:b], rc[a:b], s, rc, s + rc]   # pieces, exact copies, reverse complement, palindrome
    rng.shuffle(seqs)
    return seqs


def test_matches_cdhit_rules():
    rng = random.Random(0)
    for _ in range(20):
        seqs = random_seqs(rng, 60)
        assert run(seqs) == cdhit_rules(seqs, 10)


def test_shorter_anchors_and_length_cutoff():
    rng = random.Random(1)
    for _ in range(10):
        seqs = random_seqs(rng, 60)
        assert run(seqs, min_length=4, short_anchor=5, long_anchor=12) == cdhit_rules(seqs, 4)


def test_many_near_identical_sequences():
    rng = random.Random(2)
    core = "".join(rng.choice("ACGT") for _ in range(200))
    seqs = []
    for _ in range(300):   # one family: point mutations and random start/end, like 16S rRNA
        s = list(core)
        for _ in range(rng.randint(0, 2)):
            s[rng.randrange(200)] = rng.choice("ACGT")
        a, b = rng.randint(0, 60), rng.randint(140, 200)
        seqs.append("".join(s)[a:b])
    assert run(seqs) == cdhit_rules(seqs, 10)
