"""Compare deduplicated RNA-FM corpus candidates with the corpus statistics RNA-FM's authors published.

Usage: python scripts/06_corpus_fingerprints.py [candidate name ...]   (default: every candidate)

Reads each candidate's deduplicated FASTA from script 05 and computes the statistics listed in
corpora.rna_fm.reference_fingerprint (Shen et al. 2024, RhoFold+, Supplementary Information):
number of sequences, length minimum, quartiles, maximum and mean, the number of sequences shorter
than 1,023 nt, and counts of rare letters. A release that matches on all of them is very likely
the one RNA-FM was trained on.

Writes results/06_corpus_fingerprints.json plus its .meta.json.
"""
import json
import sys

import numpy as np

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.meta import write_meta


def fingerprint(path, rare_letters):
    lengths = []
    letter_counts = np.zeros(256, dtype=np.int64)
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                continue
            seq = line.rstrip(b"\n").upper()   # script 05 writes each sequence on one line
            lengths.append(len(seq))
            letter_counts += np.bincount(np.frombuffer(seq, dtype=np.uint8), minlength=256)
    lengths = np.array(lengths)
    q1, median, q3 = np.percentile(lengths, [25, 50, 75])
    return {
        "sequences": int(lengths.size),
        "total_letters": int(lengths.sum()),
        "min_length": int(lengths.min()),
        "q1": float(q1), "median": float(median), "q3": float(q3),
        "max_length": int(lengths.max()),
        "mean_length": round(float(lengths.mean()), 1),
        "under_1023": int((lengths < 1023).sum()),
        "letters": {ch: int(letter_counts[ord(ch)]) for ch in rare_letters},
        "all_letters": {chr(i): int(n) for i, n in enumerate(letter_counts) if n},
    }


def main():
    config = load_config()
    paths = config["paths"]
    reference = config["corpora"]["rna_fm"]["reference_fingerprint"]
    candidates = [c["name"] for c in config["corpora"]["rna_fm"]["candidates"]]
    names = sys.argv[1:] or candidates
    unknown = [n for n in names if n not in candidates]
    if unknown:
        sys.exit(f"unknown candidates: {unknown}; choose from {candidates}")

    result = {"reference": reference, "candidates": {}}
    for name in names:
        prefix = repo_path(paths["rnacentral_dedup"].format(processed_dir=paths["processed_dir"], name=name))
        fasta = prefix.parent / f"{prefix.name}.fasta"
        fp = fingerprint(fasta, list(reference["letters"]))
        fp["file"] = display_path(fasta)
        result["candidates"][name] = fp
        print(json.dumps({k: v for k, v in fp.items() if k != "all_letters"}), flush=True)

    out = repo_path(paths["results_dir"]) / "06_corpus_fingerprints.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    write_meta(out, config, packages=["numpy"])


if __name__ == "__main__":
    main()
