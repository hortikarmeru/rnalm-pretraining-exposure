"""Count records and exact-duplicate sequences in each candidate RNAcentral release.

Usage: python scripts/01_count_rnacentral_releases.py

Reads corpora.rna_fm.candidate_releases and paths.rnacentral_fasta from configs/config.yaml and
writes results/01_rnacentral_release_counts.json (one entry per release) plus its .meta.json.

Per release:
  records            number of '>' headers
  unique_raw         distinct sequences as written in the file
  unique_normalized  distinct sequences after uppercase + T->U (RNA-FM's preprocessing)

cd-hit-est at 100% (RNA-FM's actual dedup) also merges sequences contained in longer
identical ones, so it can only go lower than unique_normalized.
"""
import json
import sys
from multiprocessing import Pool

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.fasta import count_unique_sequences
from rnalm_exposure.meta import write_meta


def count_release(job):
    release, path = job
    counts = count_unique_sequences(path)
    print(f"{release}: {counts['records']} records", flush=True)
    return {"release": release, "file": display_path(path), **counts}


def main():
    config = load_config()
    paths = config["paths"]
    releases = config["corpora"]["rna_fm"]["candidate_releases"]
    jobs = [(r, repo_path(paths["rnacentral_fasta"].format(raw_dir=paths["raw_dir"], release=r)))
            for r in releases]
    for _, path in jobs:
        if not path.exists():
            sys.exit(f"missing {path}")

    # One process per release: each holds two digest sets, roughly 5 GB of RAM per release.
    with Pool(len(jobs)) as pool:
        results = pool.map(count_release, jobs)

    out = repo_path(paths["results_dir"]) / "01_rnacentral_release_counts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    write_meta(out, config, packages=["pyyaml"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
