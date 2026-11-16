"""Training structures for the supervised probe (config training_experiment.probe).

Usage: python scripts/37_probe_sets.py   (env rnalm-exposure; after script 26)

The pool is every test sequence with an Rfam family, minus: the experiment's families and every family
in the same Rfam clan as one of them, the sentinel tests (results/26_sentinels.tsv), sequences outside
probe.length, and any sequence with an MMseqs2 hit of >= exclude_min_seq_id identity (no coverage
requirement) to an eval sequence. From the families left, n_train_families are drawn with the seed,
and max_per_family sequences from each. None of them is a homolog of an injected family, so the
injected sets add nothing to them (script 26 checks that only the set's own families gain homologs).

Writes results/37_probe_sequences.tsv (seq_id, family, length, pairs) and results/37_probe_sets.json
with its .meta.json.
"""
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.mmseqs import STRAND, mmseqs
from rnalm_exposure.pair_balance import read_tsv
from rnalm_exposure.queries import write_fasta


def main():
    config = load_config()
    paths, exp, hs = config["paths"], config["training_experiment"], config["homolog_search"]
    cfg = exp["probe"]
    results = repo_path(paths["results_dir"])

    tests = {}
    with open(repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))) as f:
        for line in f:
            r = json.loads(line)
            tests[r["seq_id"]] = r
    fam_rows = {r["seq_id"]: r for r in read_tsv(results / "15_families.tsv")}
    family_of = {q: fam_rows[q]["family"] for q in tests if fam_rows[q]["family"]}
    accession = {r["family_name"]: r["family"] for r in fam_rows.values() if r["family"]}

    experiment = {r["family"] for r in read_tsv(results / "26_assignment.tsv")}
    clans = []
    for line in open(repo_path(paths["raw_dir"]) / config["family_assignment"]["rfam_clanin"]):
        clans.append({accession[m] for m in line.split()[1:] if m in accession})
    excluded = set(experiment)
    for members in clans:
        if members & experiment:
            excluded |= members
    sentinels = {r["seq_id"] for r in read_tsv(results / "26_sentinels.tsv")}
    eval_ids = [r["seq_id"] for r in read_tsv(results / "26_eval_sequences.tsv")]
    lo, hi = cfg["length"]
    pool = sorted(q for q, fam in family_of.items()
                  if fam not in excluded and q not in sentinels and lo <= len(tests[q]["sequence"]) <= hi)
    families_before = len({family_of[q] for q in pool})

    # >= exclude_min_seq_id identity to any eval sequence, over any aligned length
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_fasta({q: tests[q]["sequence"] for q in pool}, tmp / "pool.fasta")
        write_fasta({q: tests[q]["sequence"] for q in eval_ids}, tmp / "eval.fasta")
        mmseqs("easy-search", tmp / "pool.fasta", tmp / "eval.fasta", tmp / "hits.tsv", tmp / "work",
               "--search-type", hs["search_type"], "-k", hs["kmer_length"], "-e", hs["evalue"],
               "--min-seq-id", cfg["exclude_min_seq_id"], "-c", 0, "--strand", STRAND[hs["strand"]],
               "--threads", hs["threads"], "--format-output", "query,target,fident")
        similar = {line.split("\t")[0] for line in open(tmp / "hits.tsv")}
    pool = [q for q in pool if q not in similar]

    by_family = defaultdict(list)
    for q in pool:
        by_family[family_of[q]].append(q)
    rng = np.random.default_rng(cfg["seed"])
    families = sorted(by_family)
    chosen = sorted(rng.choice(families, min(cfg["n_train_families"], len(families)), replace=False).tolist())
    train = sorted(q for fam in chosen
                   for q in rng.choice(by_family[fam], min(cfg["max_per_family"], len(by_family[fam])), replace=False).tolist())

    out = results / "37_probe_sequences.tsv"
    with open(out, "w") as f:
        f.write("seq_id\tfamily\tlength\tpairs\n")
        for q in train:
            f.write(f"{q}\t{family_of[q]}\t{len(tests[q]['sequence'])}\t{len(tests[q]['pairs'])}\n")
    summary = {"excluded_families": len(excluded), "families_before_identity_filter": families_before,
               "sequences_similar_to_eval": len(similar), "families_available": len(families),
               "families_chosen": len(chosen), "sequences": len(train),
               "true_pairs": int(sum(len(tests[q]["pairs"]) for q in train)),
               "mmseqs": subprocess.run(["mmseqs", "version"], capture_output=True, text=True).stdout.strip()}
    path = results / "37_probe_sets.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    write_meta(path, config, packages=["numpy"])


if __name__ == "__main__":
    main()
