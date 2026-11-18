"""Deduplicate RNA-FM corpus candidates at 100% identity with the exact containment method.

Usage: python scripts/05_dedup_containment.py [candidate name ...]   (default: every candidate)
  A release or file another corpus is built from (corpora.<corpus>.dedup_release) is deduplicated the same
  way when its name is given.

Keeps what cd-hit-est -c 1.0 -r 1 keeps, apart from rare differences measured by script 04 (see
rnalm_exposure.containment), in hours instead of weeks. For each candidate, its files are read
once into memory (one byte per letter), the representatives are found, and a second pass writes
them with their original headers to paths.rnacentral_dedup + .fasta. The number of
representatives is the deduplicated corpus size to compare with RNA-FM's 23,735,169.

Writes results/05_dedup_<name>.json plus its .meta.json per candidate.
"""
import json
import sys
import time

import numpy as np

from rnalm_exposure.config import candidate_files, display_path, load_config, repo_path
from rnalm_exposure.containment import load_codes, representatives
from rnalm_exposure.fasta import iter_gz_fasta
from rnalm_exposure.meta import write_meta


def main():
    config = load_config()
    settings = config["corpora"]["dedup_containment"]
    paths = config["paths"]
    candidates = {c["name"]: c for c in config["corpora"]["rna_fm"]["candidates"]}
    names = sys.argv[1:] or list(candidates)
    candidates.update({c["dedup_release"]["name"]: c["dedup_release"] for c in config["corpora"].values()
                       if c and "dedup_release" in c})
    unknown = [n for n in names if n not in candidates]
    if unknown:
        sys.exit(f"unknown candidates: {unknown}; choose from {list(candidates)}")

    for name in names:
        files = candidate_files(config, candidates[name])
        prefix = repo_path(paths["rnacentral_dedup"].format(processed_dir=paths["processed_dir"], name=name))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        codes_path = prefix.parent / f"{prefix.name}.codes"
        output = prefix.parent / f"{prefix.name}.fasta"
        minutes = {}

        start = time.time()
        lengths = load_codes(files, codes_path)
        codes = np.fromfile(codes_path, np.uint8)
        codes_path.unlink()
        minutes["read"] = (time.time() - start) / 60
        print(f"{name}: {len(lengths)} records, {codes.size} letters ({minutes['read']:.1f} min)", flush=True)

        start = time.time()
        keep, info = representatives(codes, lengths, settings["min_length"], settings["short_anchor"],
                                     settings["long_anchor"], settings["threads"])
        minutes["dedup"] = (time.time() - start) / 60
        del codes
        print(f"{name}: {int(keep.sum())} representatives ({minutes['dedup']:.1f} min)", flush=True)

        start = time.time()
        i = 0
        with open(output, "wb") as f:
            for path in files:
                for header, seq in iter_gz_fasta(path):
                    if keep[i]:
                        f.write(b">" + header + b"\n" + seq + b"\n")
                    i += 1
        assert i == len(keep)
        minutes["write"] = (time.time() - start) / 60

        result = {"name": name, "files": [display_path(f) for f in files], "input_records": len(lengths),
                  "input_letters": int(lengths.sum()), "discarded_short": info["discarded_short"],
                  "clusters": int(keep.sum()), "output": display_path(output),
                  "minutes": {k: round(v, 1) for k, v in minutes.items()}}
        out = repo_path(paths["results_dir"]) / f"05_dedup_{name}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        write_meta(out, config, packages=["numpy", "numba"])
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
