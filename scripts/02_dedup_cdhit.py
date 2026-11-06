"""Deduplicate RNA-FM corpus candidates with cd-hit-est at 100% identity, as in RNA-FM's methods.

Usage: python scripts/02_dedup_cdhit.py [candidate name ...]   (default: every candidate)

For each candidate (corpora.rna_fm.candidates), its files are decompressed into one FASTA,
clustered with the settings in corpora.dedup, and the representatives are written to
paths.rnacentral_dedup + .fasta (cluster membership in .fasta.clstr). The number of clusters is
the deduplicated corpus size to compare with RNA-FM's ~23.7M.

Writes results/02_cdhit_<name>.json plus its .meta.json per candidate.
"""
import json
import subprocess
import sys
import time

from rnalm_exposure.config import candidate_files, display_path, load_config, repo_path
from rnalm_exposure.meta import write_meta


def count_records(path):
    n = 0
    with open(path, "rb") as f:
        for line in f:
            n += line.startswith(b">")
    return n


def main():
    config = load_config()
    dedup = config["corpora"]["dedup"]
    paths = config["paths"]
    candidates = {c["name"]: c for c in config["corpora"]["rna_fm"]["candidates"]}
    names = sys.argv[1:] or list(candidates)
    unknown = [n for n in names if n not in candidates]
    if unknown:
        sys.exit(f"unknown candidates: {unknown}; choose from {list(candidates)}")

    for name in names:
        files = candidate_files(config, candidates[name])
        prefix = repo_path(paths["rnacentral_dedup"].format(processed_dir=paths["processed_dir"], name=name))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        merged = prefix.with_suffix(".input.fasta")
        output = prefix.with_suffix(".fasta")

        # cd-hit-est takes one input file, so the candidate's files are concatenated first.
        with open(merged, "wb") as f:
            subprocess.run(["gzip", "-dc", *map(str, files)], stdout=f, check=True)

        cmd = ["cd-hit-est", "-i", str(merged), "-o", str(output),
               "-c", str(dedup["identity"]), "-r", "1" if dedup["both_strands"] else "0",
               "-M", str(dedup["memory_mb"]), "-T", str(dedup["threads"]), *dedup["other_options"]]
        print(" ".join(cmd), flush=True)
        start = time.time()
        subprocess.run(cmd, check=True)
        hours = (time.time() - start) / 3600
        merged.unlink()

        result = {"name": name, "files": [display_path(f) for f in files],
                  "input_records": None, "clusters": count_records(output),
                  "output": display_path(output), "hours": round(hours, 2), "command": cmd}
        counts = repo_path(paths["results_dir"]) / "01_rnacentral_release_counts.json"
        if counts.exists():
            match = [r for r in json.load(open(counts)) if r["name"] == name]
            result["input_records"] = match[0]["records"] if match else None
        out = repo_path(paths["results_dir"]) / f"02_cdhit_{name}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        write_meta(out, config, extra={"cd_hit_version": subprocess.run(
            ["cd-hit-est", "-h"], capture_output=True, text=True).stdout.splitlines()[0:1]})
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
