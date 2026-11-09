"""Read secondary structures from the test-set file formats.

Each parser takes the file's text and returns (sequence, pairs): the sequence as written in the
file and the base pairs as a set of 0-indexed (i, j) tuples with i < j.
"""
OPENERS, CLOSERS = "([{<", ")]}>"


def _check(n, pairs):
    bases = [b for p in pairs for b in p]
    if len(bases) != len(set(bases)):
        raise ValueError("a base is in more than one pair")
    if any(not 0 <= i < j < n for i, j in pairs):
        raise ValueError("pair outside the sequence")
    return pairs


def parse_ct(text):
    """Connectivity table: header line starting with the length, then one line per base:
    index, base, previous, next, partner (0 = unpaired), natural index."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    n = int(lines[0][0])
    seq, pairs = [], set()
    for k, fields in enumerate(lines[1:n + 1]):
        i, j = int(fields[0]), int(fields[4])
        if i != k + 1:
            raise ValueError(f"ct line {k + 2}: index {i}, expected {k + 1}")
        seq.append(fields[1])
        if j > i:
            pairs.add((i - 1, j - 1))
    if len(seq) != n:
        raise ValueError(f"ct header says {n} bases, file has {len(seq)}")
    return "".join(seq), _check(n, pairs)


def parse_bpseq(text):
    """BPSEQ: one line per base: index, base, partner (0 = unpaired). Lines that don't start with an
    index (comments, headers) are skipped."""
    seq, pairs = [], set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        i, j = int(fields[0]), int(fields[2])
        if i != len(seq) + 1:
            raise ValueError(f"bpseq index {i}, expected {len(seq) + 1}")
        seq.append(fields[1])
        if j > i:
            pairs.add((i - 1, j - 1))
    return "".join(seq), _check(len(seq), pairs)


def parse_bprna_st(text):
    """bpRNA structure-type file: '#' header lines, then the sequence and its dot-bracket line.
    Pseudoknots use [] {} <> and then letter pairs (A opens, a closes)."""
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    seq, dot_bracket = lines[0], lines[1]
    if len(seq) != len(dot_bracket):
        raise ValueError("sequence and dot-bracket lengths differ")
    return seq, dot_bracket_pairs(dot_bracket)


def dot_bracket_pairs(dot_bracket):
    """Pairs of a dot-bracket string; () [] {} <> and letter pairs (A opens, a closes)."""
    stacks, pairs = {}, set()
    for i, c in enumerate(dot_bracket):
        if c in OPENERS or c.isupper():
            stacks.setdefault(c, []).append(i)
        elif c in CLOSERS or c.islower():
            opener = OPENERS[CLOSERS.index(c)] if c in CLOSERS else c.upper()
            if not stacks.get(opener):
                raise ValueError(f"unmatched {c!r} at {i}")
            pairs.add((stacks[opener].pop(), i))
    if any(stacks.values()):
        raise ValueError("unmatched opening bracket")
    return _check(len(dot_bracket), pairs)


PARSERS = {"ct": parse_ct, "bpseq": parse_bpseq, "bprna_st": parse_bprna_st}
