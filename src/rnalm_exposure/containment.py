"""100%-identity deduplication by exact containment, following cd-hit-est -c 1.0 -r 1.

A sequence is dropped when it occurs, on either strand, inside a sequence that cd-hit-est would
process before it. Containment is transitive, so checking against every earlier sequence gives
the same result as checking against cluster representatives only. This is checked directly
instead of aligning against every representative, which takes weeks on RNAcentral.

cd-hit-est itself differs slightly (script 04: 16 of 300,000 random RNAcentral 15.0 records,
none of 30,000 16S rRNA records). It also merges a sequence that matches a longer one only with
a gap, because its identity is matches over the shorter sequence's length, and its word filter
misses some exact containments of short sequences.

Rules copied from cd-hit-est (cdhit-common.c++):
  - sequences of <= min_length letters (its -l, default 10) are discarded before clustering;
  - A, C, G and T/U are bases 0-3; every other letter becomes N (4), and N matches N;
  - processing order is length descending, ties in file order (stable counting sort).

Search: each sequence and its reverse complement get an anchor, their first long_anchor
letters (short_anchor for shorter sequences), packed in base 5 so N is a letter like the others.
One pass over every sequence looks up each of its windows in the anchor tables; where an anchor
occurs, a rolling hash compares the whole candidate stretch, and a letter-by-letter comparison
confirms a match.
"""
from array import array

import numba
import numpy as np
from numba import njit, prange

from rnalm_exposure.fasta import iter_gz_fasta

_code = np.full(256, 4, np.uint8)
for _ch, _v in zip(b"ACGTUacgtu", (0, 1, 2, 3, 3, 0, 1, 2, 3, 3)):
    _code[_ch] = _v
LETTER_CODES = bytes(_code)   # bytes.translate table: letter -> 0..4

# Two hash moduli below 2^31, so a product of two residues fits in int64; together ~62 bits.
M1, M2 = 2147483647, 2147483629
B1, B2 = 1000003, 999983


def load_codes(paths, codes_path):
    """Write every sequence of the FASTA files, coded 0-4, to codes_path; return the lengths."""
    lengths = array("q")
    with open(codes_path, "wb") as out:
        for path in paths:
            for _, seq in iter_gz_fasta(path):
                out.write(seq.translate(LETTER_CODES))
                lengths.append(len(seq))
    return np.frombuffer(lengths, dtype=np.int64).copy()


@njit(cache=True)
def _powers(n):
    p1 = np.ones(n + 1, np.int64)
    p2 = np.ones(n + 1, np.int64)
    for i in range(n):
        p1[i + 1] = p1[i] * B1 % M1
        p2[i + 1] = p2[i] * B2 % M2
    return p1, p2


@njit(inline="always")
def _letter(codes, start, length, strand, i):
    """Letter i of a sequence (strand 0) or of its reverse complement (strand 1)."""
    if strand == 0:
        return codes[start + i]
    c = codes[start + length - 1 - i]
    return c if c == 4 else 3 - c


@njit(parallel=True, cache=True)
def _anchors(codes, offsets, lengths, keep, short_anchor, long_anchor):
    """One entry per sequence and strand: anchor key (first short_anchor or long_anchor letters
    in base 5) and the hash of the whole strand."""
    n = len(lengths)
    key = np.zeros(2 * n, np.uint64)
    h = np.zeros(2 * n, np.int64)
    for j in prange(2 * n):
        q, strand = j // 2, j % 2
        if not keep[q]:
            continue
        start, length = offsets[q], lengths[q]
        a = long_anchor if length >= long_anchor else short_anchor
        h1, h2 = 0, 0
        k = np.uint64(0)
        for i in range(length):
            c = _letter(codes, start, length, strand, i)
            h1 = (h1 * B1 + c + 1) % M1
            h2 = (h2 * B2 + c + 1) % M2
            if i < a:
                k = k * np.uint64(5) + np.uint64(c)
        key[j] = k
        h[j] = h1 * 2147483648 + h2
    return key, h


def _groups(key, length, h):
    """Sort entries by (key, length, hash). A group is a run sharing (key, length): one substring
    hash of the scanned sequence checks the whole group."""
    order = np.lexsort((h, length, key))
    key, length, h = key[order], length[order], h[order]
    new = np.ones(len(key), bool)
    new[1:] = (key[1:] != key[:-1]) | (length[1:] != length[:-1])
    g_start = np.flatnonzero(new)
    g_end = np.append(g_start[1:], len(key))
    g_key = key[g_start]
    k_new = np.ones(len(g_key), bool)
    k_new[1:] = g_key[1:] != g_key[:-1]
    k_start = np.flatnonzero(k_new)
    k_end = np.append(k_start[1:], len(g_key))
    return order, h, g_start, g_end, length[g_start], g_key[k_start], k_start, k_end


@njit(cache=True)
def _slot(k, bits):
    return (k * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(64 - bits)


@njit(cache=True)
def _hash_table(keys, k_start, k_end, bits):
    """Open-addressing table: anchor key -> range of its groups."""
    size = 1 << bits
    t_key = np.zeros(size, np.uint64)
    t_start = np.full(size, -1, np.int64)
    t_end = np.zeros(size, np.int64)
    mask = np.uint64(size - 1)
    for i in range(len(keys)):
        s = _slot(keys[i], bits)
        while t_start[s] >= 0:
            s = (s + np.uint64(1)) & mask
        t_key[s], t_start[s], t_end[s] = keys[i], k_start[i], k_end[i]
    return t_key, t_start, t_end


@njit(cache=True)
def _check(codes, t, t_start, t_len, s, gs, ge, g_len, g_es, g_ee, e_h, e_q, e_strand,
           H1, H2, P1, P2, rank, offsets, absorbed):
    """An anchor of groups gs..ge starts at position s of the scanned sequence t: mark every
    query of those groups that occurs in t at s and that cd-hit-est would process after t."""
    for g in range(gs, ge):
        L = g_len[g]
        if s + L > t_len:
            continue
        x1 = (H1[s + L] - H1[s] * P1[L] % M1) % M1
        x2 = (H2[s + L] - H2[s] * P2[L] % M2) % M2
        x = x1 * 2147483648 + x2
        lo, hi = g_es[g], g_ee[g]
        while lo < hi:   # first entry with hash >= x
            mid = (lo + hi) // 2
            if e_h[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        while lo < g_ee[g] and e_h[lo] == x:
            q = e_q[lo]
            if q != t and absorbed[q] == 0 and (L < t_len or rank[t] < rank[q]):
                same = True
                for i in range(L):
                    if codes[t_start + s + i] != _letter(codes, offsets[q], L, e_strand[lo], i):
                        same = False
                        break
                if same:
                    absorbed[q] = 1
            lo += 1


@njit(cache=True)
def _lookup(t_key, t_start, t_end, bits, k):
    """Range of groups for anchor key k (empty range if absent)."""
    mask = np.uint64((1 << bits) - 1)
    s = _slot(k, bits)
    while t_start[s] >= 0:
        if t_key[s] == k:
            return t_start[s], t_end[s]
        s = (s + np.uint64(1)) & mask
    return 0, 0


@njit(parallel=True, cache=True)
def _scan(codes, offsets, lengths, keep, rank, order, n_threads, short_anchor, long_anchor,
          s_t_key, s_t_start, s_t_end, s_bits, s_g_len, s_g_es, s_g_ee, s_e_h, s_e_q, s_e_strand,
          l_t_key, l_t_start, l_t_end, l_bits, l_g_len, l_g_es, l_g_ee, l_e_h, l_e_q, l_e_strand,
          absorbed):
    max_len = lengths.max()
    P1, P2 = _powers(max_len)
    lead_s = np.uint64(5 ** (short_anchor - 1))
    lead_l = np.uint64(5 ** (long_anchor - 1))
    for b in prange(n_threads):
        H1 = np.zeros(max_len + 1, np.int64)
        H2 = np.zeros(max_len + 1, np.int64)
        for idx in range(b, len(order), n_threads):
            t = order[idx]
            if not keep[t]:
                continue
            t_start, t_len = offsets[t], lengths[t]
            for p in range(t_len):
                c = codes[t_start + p]
                H1[p + 1] = (H1[p] * B1 + c + 1) % M1
                H2[p + 1] = (H2[p] * B2 + c + 1) % M2
            ks, kl = np.uint64(0), np.uint64(0)
            for p in range(t_len):
                c = np.uint64(codes[t_start + p])
                # rolling base-5 codes of the windows ending at p (drop the letter leaving)
                if p >= short_anchor:
                    ks -= np.uint64(codes[t_start + p - short_anchor]) * lead_s
                if p >= long_anchor:
                    kl -= np.uint64(codes[t_start + p - long_anchor]) * lead_l
                ks = ks * np.uint64(5) + c
                kl = kl * np.uint64(5) + c
                if p >= short_anchor - 1:
                    gs, ge = _lookup(s_t_key, s_t_start, s_t_end, s_bits, ks)
                    if ge > gs:
                        _check(codes, t, t_start, t_len, p - short_anchor + 1, gs, ge, s_g_len,
                               s_g_es, s_g_ee, s_e_h, s_e_q, s_e_strand, H1, H2, P1, P2, rank,
                               offsets, absorbed)
                if p >= long_anchor - 1:
                    gs, ge = _lookup(l_t_key, l_t_start, l_t_end, l_bits, kl)
                    if ge > gs:
                        _check(codes, t, t_start, t_len, p - long_anchor + 1, gs, ge, l_g_len,
                               l_g_es, l_g_ee, l_e_h, l_e_q, l_e_strand, H1, H2, P1, P2, rank,
                               offsets, absorbed)


def representatives(codes, lengths, min_length=10, short_anchor=11, long_anchor=27, threads=None,
                    log=print):
    """Boolean mask over sequences (file order): True for representatives, i.e. sequences not
    contained in any sequence processed before them.
    long_anchor <= 27 so a base-5 anchor fits in 64 bits; short_anchor <= min_length + 1 so every
    kept sequence has an anchor."""
    assert short_anchor <= min_length + 1 and short_anchor < long_anchor <= 27
    if threads:
        numba.set_num_threads(threads)
    n = len(lengths)
    offsets = np.zeros(n, np.int64)
    np.cumsum(lengths[:-1], out=offsets[1:])
    keep = lengths > min_length
    order = np.argsort(-lengths, kind="stable")   # cd-hit-est processing order
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)

    key, h = _anchors(codes, offsets, lengths, keep, short_anchor, long_anchor)
    query = np.repeat(np.arange(n, dtype=np.int64), 2)
    strand = np.tile(np.array([0, 1], np.int8), n)
    q_len = lengths[query]
    q_keep = keep[query]

    tables = []
    for sel in (np.flatnonzero(q_keep & (q_len < long_anchor)), np.flatnonzero(q_keep & (q_len >= long_anchor))):
        o, e_h, g_s, g_e, g_len, keys, k_s, k_e = _groups(key[sel], q_len[sel], h[sel])
        bits = max(4, int(np.ceil(np.log2(2 * len(keys) + 1))))
        t_key, t_start, t_end = _hash_table(keys, k_s, k_e, bits)
        tables.append((t_key, t_start, t_end, bits, g_len, g_s, g_e, e_h, query[sel[o]], strand[sel[o]]))
        log(f"{len(sel)} strands, {len(g_s)} groups, {len(keys)} distinct anchors")

    absorbed = np.zeros(n, np.uint8)
    _scan(codes, offsets, lengths, keep, rank, order, numba.get_num_threads(), short_anchor,
          long_anchor, *tables[0], *tables[1], absorbed)
    return keep & (absorbed == 0), {"discarded_short": int((~keep).sum())}
