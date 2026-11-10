"""Same-family check of exposure (config exposure_checks.log_same_family_count): for each test
sequence with an Rfam family (script 15, Rfam 15.0), the number of corpus sequences that RNAcentral's
own Rfam scan of that corpus's release assigns to the same family. Families that never occur in the
release's annotations (absent from its Rfam version) get no value; sequences without a family too.

Usage: python scripts/22_same_family_counts.py <corpus> [<corpus> ...]   (exposure_checks.rfam_annotations)

Only corpus sequences with RNAcentral IDs can be counted: RiNALMo's Rfam 14.9 and RefSeq sequences and
AIDO.RNA's inactive sequences have no annotations (RNAcentral annotates active sequences only).

Writes results/22_same_family_<corpus>.tsv (seq_id, family, same_family, log_same_family) and
results/22_same_family_<corpus>.json (counts) plus its .meta.json.
"""
import csv
import gzip
import json
import math
import sys
from collections import Counter

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.queries import query_sequences


def corpus_ids(path):
    """RNAcentral IDs (URS...) of a corpus FASTA; headers start with the ID."""
    ids = set()
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                i = line[1:].split(None, 1)[0].decode()
                if i.startswith("URS"):
                    ids.add(i.split("_", 1)[0])   # species-specific IDs end in _<taxid>
    return ids


def main():
    config = load_config()
    paths, hs = config["paths"], config["homolog_search"]
    with open(repo_path(paths["results_dir"]) / "15_families.tsv") as f:
        family = {r["seq_id"]: r["family"] for r in csv.DictReader(f, delimiter="\t")}
    queries = query_sequences(config)
    for name in sys.argv[1:]:
        ids = corpus_ids(repo_path(paths["corpus_fasta"].format(processed_dir=paths["processed_dir"], name=name)))
        per_family, annotated = Counter(), 0
        seen = set()   # one family per sequence and family, however many hits it has
        with gzip.open(repo_path(paths["raw_dir"]) / config["exposure_checks"]["rfam_annotations"][name], "rt") as f:
            for line in f:
                urs, fam = line.split("\t", 2)[:2]
                if urs in ids and (urs, fam) not in seen:
                    seen.add((urs, fam))
                    per_family[fam] += 1
        annotated = len({u for u, _ in seen})
        del seen
        out_tsv = repo_path(paths["results_dir"]) / f"22_same_family_{name}.tsv"
        with open(out_tsv, "w") as f:
            f.write("seq_id\tfamily\tsame_family\tlog_same_family\n")
            for q in queries:
                fam = family[q]
                if fam and fam in per_family:
                    n = per_family[fam]
                    f.write(f"{q}\t{fam}\t{n}\t{math.log(hs['log_offset'] + n):.6f}\n")
                else:   # no family, or a family absent from this release's Rfam version
                    f.write(f"{q}\t{fam}\t\t\n")
        result = {"corpus": name, "corpus_rnacentral_ids": len(ids), "annotated_corpus_sequences": annotated,
                  "families_in_release": len(per_family),
                  "queries_with_value": sum(bool(family[q]) and family[q] in per_family for q in queries),
                  "queries_with_family_absent_from_release": sum(bool(family[q]) and family[q] not in per_family for q in queries)}
        print(json.dumps(result), flush=True)
        out = repo_path(paths["results_dir"]) / f"22_same_family_{name}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        write_meta(out, config)


if __name__ == "__main__":
    main()
