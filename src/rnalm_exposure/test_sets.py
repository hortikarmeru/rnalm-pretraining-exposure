"""Filter and deduplicate the test sets (rules in config test_sets).

Each record is a dict with seq_id, dataset, sequence (normalized: uppercase, T -> U) and pairs.
Steps, in order: drop sequences with letters other than ACGU, drop sequences over the length
cutoff, drop sequences with no reference pairs, resolve exact duplicates within each set, then across sets. Every dropped record is
returned with its reason.
"""
from collections import defaultdict

ACGU = set("ACGU")


def normalize(seq):
    return seq.upper().replace("T", "U")


def within_set(records):
    """First copy by seq_id if every copy has the same pairs; otherwise drop all copies."""
    groups = defaultdict(list)
    for r in sorted(records, key=lambda r: r["seq_id"]):
        groups[r["sequence"]].append(r)
    kept, dropped = [], []
    for copies in groups.values():
        if all(c["pairs"] == copies[0]["pairs"] for c in copies):
            kept.append(copies[0])
            dropped += [(c, "duplicate_within_set", copies[0]["seq_id"]) for c in copies[1:]]
        else:
            dropped += [(c, "duplicate_within_set_conflicting", None) for c in copies]
    return kept, dropped


def filter_sets(records_by_set, max_length, priority):
    """records_by_set: {dataset: [record, ...]}. Returns (kept records in priority order, dropped
    as (record, reason, detail) tuples); detail names the copy that was kept, when there is one."""
    kept_by_set, dropped = {}, []
    for name, records in records_by_set.items():
        keep = []
        for r in records:
            if not set(r["sequence"]) <= ACGU:
                dropped.append((r, "non_acgu", "".join(sorted(set(r["sequence"]) - ACGU))))
            elif len(r["sequence"]) > max_length:
                dropped.append((r, "over_length_cutoff", len(r["sequence"])))
            elif not r["pairs"]:   # F1 would be 0 whatever the model predicts
                dropped.append((r, "no_reference_pairs", None))
            else:
                keep.append(r)
        kept_by_set[name], d = within_set(keep)
        dropped += d
    owner, kept = {}, []
    for name in priority:   # a sequence stays in the first set that still has it
        for r in kept_by_set[name]:
            if r["sequence"] in owner:
                dropped.append((r, "duplicate_across_sets", owner[r["sequence"]]))
            else:
                owner[r["sequence"]] = r["seq_id"]
                kept.append(r)
    return kept, dropped
