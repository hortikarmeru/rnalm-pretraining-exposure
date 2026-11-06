"""Estimate how many sequences cd-hit-est at 100% identity would keep, by random sampling.

Usage: python scripts/03_estimate_dedup_size.py [candidate name ...]   (default: every candidate)

cd-hit-est -c 1.0 -r 1 keeps a sequence only if no longer sequence (or its reverse complement)
contains it exactly; identical copies collapse to one. So the number of clusters is
records x (1 - g), where g is the fraction of records absorbed by another record. This script
estimates g from a random sample of records instead of clustering everything:
  1. draw corpora.dedup_estimate.sample_size records uniformly at random (one pass)
  2. stream the whole candidate once, finding every place a sampled sequence (or its reverse
     complement) occurs inside another record, via an Aho-Corasick automaton built on the first
     anchor_length letters of each sampled sequence, then checking the full match
  3. a sampled record is absorbed if it occurs in a longer record, or in a record of equal
     length that comes earlier in the file (identical copies keep exactly one)
Sequences are compared after uppercase + T->U. The 95% interval is the binomial interval on g.
Writes results/03_dedup_estimates.json plus its .meta.json.
"""
import json
import math
import random
import sys
from multiprocessing import Pool

import ahocorasick

from rnalm_exposure.config import candidate_files, display_path, load_config, repo_path
from rnalm_exposure.fasta import iter_gz_fasta, normalize
from rnalm_exposure.meta import write_meta

COMPLEMENT = bytes.maketrans(b"ACGU", b"UGCA")


def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1]


def stream(files):
    for path in files:
        for _, seq in iter_gz_fasta(path):
            yield normalize(seq)


def estimate(job):
    name, files, settings = job
    rng = random.Random(settings["seed"])
    k, n = settings["anchor_length"], settings["sample_size"]

    # Pass 1: reservoir sample of record indices and sequences.
    sample, records = [], 0
    for i, seq in enumerate(stream(files)):
        records += 1
        if len(sample) < n:
            sample.append((i, seq))
        else:
            j = rng.randrange(records)
            if j < n:
                sample[j] = (i, seq)
    print(f"{name}: {records} records, sampled {len(sample)}", flush=True)

    # Each sampled sequence is searched as itself and as its reverse complement.
    automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_STRING)
    for q, (idx, seq) in enumerate(sample):
        for strand in (seq, reverse_complement(seq)):
            anchor = strand[:k].decode()
            hits = automaton.get(anchor, []) if anchor in automaton else []
            hits.append((q, strand))
            automaton.add_word(anchor, hits)
    automaton.make_automaton()

    # Pass 2: find containers.
    absorbed = [False] * len(sample)
    for i, seq in enumerate(stream(files)):
        text = seq.decode()
        for end, hits in automaton.iter(text):
            for q, strand in hits:
                if absorbed[q]:
                    continue
                q_idx, q_seq = sample[q]
                if i == q_idx or not (len(seq) > len(q_seq) or (len(seq) == len(q_seq) and i < q_idx)):
                    continue
                start = end - min(k, len(strand)) + 1
                if seq[start:start + len(strand)] == strand:
                    absorbed[q] = True

    g = sum(absorbed) / len(sample)
    half = 1.96 * math.sqrt(g * (1 - g) / len(sample))
    result = {"name": name, "files": [display_path(f) for f in files], "records": records,
              "sample_size": len(sample), "absorbed_in_sample": sum(absorbed),
              "absorbed_fraction": round(g, 5),
              "estimated_clusters": round(records * (1 - g)),
              "ci95_clusters": [round(records * (1 - g - half)), round(records * (1 - g + half))]}
    print(json.dumps(result), flush=True)
    return result


def main():
    config = load_config()
    settings = config["corpora"]["dedup_estimate"]
    candidates = {c["name"]: c for c in config["corpora"]["rna_fm"]["candidates"]}
    names = sys.argv[1:] or list(candidates)
    jobs = [(nm, candidate_files(config, candidates[nm]), settings) for nm in names]
    with Pool(len(jobs)) as pool:
        results = pool.map(estimate, jobs)
    out = repo_path(config["paths"]["results_dir"]) / "03_dedup_estimates.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    write_meta(out, config, packages=["pyahocorasick"])


if __name__ == "__main__":
    main()
