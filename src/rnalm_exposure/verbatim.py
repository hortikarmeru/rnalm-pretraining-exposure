"""Span placement, greedy span filling and per-position scores for the verbatim recovery check
(scripts 34-36; config checks.verbatim_recovery)."""
import math
import zlib

import numpy as np


def place_spans(seq_id, length, span_lengths, n, min_flank, seed):
    """[(span_length, span number, start)], 0-indexed starts, for one sequence. A span keeps at least
    min_flank letters on each side. The allowed starts are cut into n equal parts with one start drawn
    in each (seeded by seed, sequence id, span length and span number), so the spans of one length are
    spread along the sequence. A span length with fewer than n allowed starts gets no spans."""
    key = zlib.crc32(seq_id.encode())
    out = []
    for k in span_lengths:
        lo, n_starts = min_flank, length - k - 2 * min_flank + 1
        if n_starts < n:
            continue
        for j in range(n):
            a, b = lo + n_starts * j // n, lo + n_starts * (j + 1) // n
            rng = np.random.default_rng([seed, 34, key, k, j])   # 34: a stream separate from script 31's masks
            out.append((k, j, int(rng.integers(a, b))))
    return out


def self_repeat(sequence, start, k):
    """Whether the span's letters also occur somewhere else in the sequence (overlaps count)."""
    span = sequence[start:start + k]
    return sequence.find(span) != start or sequence.find(span, start + 1) != -1


def fill_spans(predict, ids, spans, letters, mask_id):
    """Greedy fill of masked spans. ids: token ids of one sequence with one special token at each end
    (integer array). spans: [(start, k)] in sequence coordinates; each span gets its own copy of the
    sequence with only that span masked. predict(x) takes [rows, tokens] ids and returns
    [rows, tokens - 2, 4] log-probabilities of the four letters at every sequence position.

    Every pass, each unfinished row commits one letter: the masked position whose top letter has the
    highest probability (the leftmost on ties) is set to that letter, and the row is predicted again.
    Returns (filled, one_pass): per span, the letter indices (0-3) from the fill and from the first
    pass alone (every position's top letter with the whole span masked)."""
    letters = np.asarray(letters)
    x = np.tile(np.asarray(ids), (len(spans), 1))
    for b, (start, k) in enumerate(spans):
        x[b, start + 1:start + 1 + k] = mask_id
    filled = [np.full(k, -1) for _, k in spans]
    one_pass = [None] * len(spans)
    active = list(range(len(spans)))
    while active:
        logp = predict(x[active])
        for row, b in enumerate(active):
            start, k = spans[b]
            lp = logp[row, start:start + k]
            if one_pass[b] is None:
                one_pass[b] = lp.argmax(-1)
            top = np.where(filled[b] < 0, lp.max(-1), -np.inf)
            j = int(top.argmax())
            filled[b][j] = int(lp[j].argmax())
            x[b, start + 1 + j] = letters[filled[b][j]]
        active = [b for b in active if (filled[b] < 0).any()]
    return filled, one_pass


def tiled_masks(seq_id, length, n, seed):
    """n disjoint sets of positions that together cover the sequence once (a random split, seeded by
    seed and sequence id), each sorted."""
    rng = np.random.default_rng([seed, 35, zlib.crc32(seq_id.encode())])   # 35: separate from script 31's masks
    order = rng.permutation(length)
    return [np.sort(order[t::n]) for t in range(n)]


def min_k(logp, fraction):
    """Mean of the lowest fraction of the values (at least one value; Min-K% Prob, Shi et al. 2024)."""
    logp = np.sort(np.asarray(logp, dtype=float))
    return float(logp[:max(1, math.ceil(fraction * len(logp)))].mean())
