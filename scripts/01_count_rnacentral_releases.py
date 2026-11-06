"""Count records and exact-duplicate sequences in each RNA-FM corpus candidate.

Usage: python scripts/01_count_rnacentral_releases.py

A candidate is one RNAcentral release plus the files taken from it
(corpora.rna_fm.candidates in configs/config.yaml). Writes
results/01_rnacentral_release_counts.json (one entry per candidate) plus its .meta.json.

Per candidate:
  records            number of '>' headers across its files
  unique_raw         distinct sequences as written
  unique_normalized  distinct sequences after uppercase + T->U (RNA-FM's preprocessing)

cd-hit-est at 100% (RNA-FM's actual dedup) also merges sequences contained in longer
identical ones, so it can only go lower than unique_normalized.
"""
import json
import sys
from multiprocessing import Pool

from rnalm_exposure.config import candidate_files, display_path, load_config, repo_path
from rnalm_exposure.fasta import count_unique_sequences
from rnalm_exposure.meta import write_meta


def count_candidate(job):
    candidate, files = job
    counts = count_unique_sequences(files)
    print(f"{candidate['name']}: {counts['records']} records", flush=True)
    return {"name": candidate["name"], "release": candidate["release"],
            "files": [display_path(f) for f in files], **counts}


def main():
    config = load_config()
    candidates = config["corpora"]["rna_fm"]["candidates"]
    jobs = [(c, candidate_files(config, c)) for c in candidates]
    missing = [f for _, files in jobs for f in files if not f.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(map(str, missing)))

    # One process per candidate: each holds two digest sets, roughly 5 GB of RAM per 30M sequences.
    with Pool(len(jobs)) as pool:
        results = pool.map(count_candidate, jobs)

    out = repo_path(config["paths"]["results_dir"]) / "01_rnacentral_release_counts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    write_meta(out, config, packages=["pyyaml"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
