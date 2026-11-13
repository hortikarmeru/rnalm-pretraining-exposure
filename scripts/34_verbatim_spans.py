"""Masked spans for the verbatim recovery check (config checks.verbatim_recovery) on the matched seen
and unseen sequences of script 30.

The masked-letter check (scripts 31-32) hides scattered single letters. This check hides one
contiguous stretch and asks whether a model writes all of it back, the masked-model version of the
extraction test of Carlini et al. (2023). Every sequence gets spans_per_length spans of each length
in span_lengths. A span keeps at least min_flank letters on each side (the ends have context on one
side only), and the starts of one length are drawn one per equal part of the allowed range, seeded by
(random_seed, sequence id, span length, span number), so every model gets the same spans. A sequence
too short for a span length gets no spans of that length; a pair counts at a span length only when
both of its sequences have spans there.

self_repeat marks spans whose letters also occur elsewhere in the same sequence: such a span can be
copied from its context.

Usage: python scripts/34_verbatim_spans.py   (environment-extra.yml)

Writes results/34_verbatim_spans.tsv (seq_id, span_length, span, start (0-indexed), self_repeat) and
results/34_verbatim_spans.json (spans, sequences and pairs per span length) plus its .meta.json.
"""
import csv
import json

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.verbatim import place_spans, self_repeat


def main():
    config = load_config()
    paths, cfg = config["paths"], config["checks"]["verbatim_recovery"]
    results = repo_path(paths["results_dir"])
    with open(results / "30_recovery_sequences.tsv") as f:
        seqs = list(csv.DictReader(f, delimiter="\t"))

    rows, has = [], {k: set() for k in cfg["span_lengths"]}
    for r in seqs:
        for k, j, start in place_spans(r["seq_id"], len(r["sequence"]), cfg["span_lengths"], cfg["spans_per_length"],
                                       cfg["min_flank"], config["random_seed"]):
            rows.append({"seq_id": r["seq_id"], "span_length": k, "span": j, "start": start,
                         "self_repeat": int(self_repeat(r["sequence"], start, k))})
            has[k].add(r["seq_id"])
    out = results / "34_verbatim_spans.tsv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    by_pair = {}
    for r in seqs:
        by_pair.setdefault(r["pair"], []).append(r["seq_id"])
    summary = {"sequences": len(seqs), "pairs": len(by_pair), "spans": len(rows),
               "model_passes_per_model": sum(r["span_length"] for r in rows),   # one pass per masked letter
               "span_lengths": {str(k): {"spans": sum(r["span_length"] == k for r in rows),
                                         "self_repeats": sum(r["self_repeat"] for r in rows if r["span_length"] == k),
                                         "sequences": len(has[k]),
                                         "pairs": sum(all(q in has[k] for q in qs) for qs in by_pair.values())}
                                for k in cfg["span_lengths"]},
               "spans_file": display_path(out)}
    print(json.dumps(summary, indent=2))
    js = results / "34_verbatim_spans.json"
    js.write_text(json.dumps(summary, indent=2))
    write_meta(js, config, packages=["numpy"])


if __name__ == "__main__":
    main()
