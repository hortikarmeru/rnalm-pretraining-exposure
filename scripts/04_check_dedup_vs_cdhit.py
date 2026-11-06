"""Check that the exact containment dedup keeps the same representatives as cd-hit-est.

Usage: python scripts/04_check_dedup_vs_cdhit.py

Draws two subsets of corpora.dedup_check.candidate in one pass: a uniform sample of records, and
a sample of records containing dense_motif (16S rRNA), where near-identical sequences make
containment common. Each subset is clustered with cd-hit-est (corpora.dedup settings) and with
rnalm_exposure.containment, and the two sets of representative IDs are compared.

Writes results/04_dedup_check.json plus its .meta.json.
"""
import json
import random
import subprocess
import time

import numpy as np

from rnalm_exposure.config import candidate_files, load_config, repo_path
from rnalm_exposure.containment import LETTER_CODES, representatives
from rnalm_exposure.fasta import iter_gz_fasta
from rnalm_exposure.meta import write_meta

COMPLEMENT = bytes.maketrans(b"ACGT", b"TGCA")


def sample_subsets(files, check):
    """Reservoir samples, one pass: uniform records, and records containing the motif."""
    rng = random.Random(check["seed"])
    motif = check["dense_motif"].encode()
    motifs = (motif, motif.translate(COMPLEMENT)[::-1])
    subsets = {"random": [], "dense": []}
    sizes = {"random": check["random_size"], "dense": check["dense_size"]}
    seen = {"random": 0, "dense": 0}
    for path in files:
        for record in iter_gz_fasta(path):
            names = ["random"]
            if any(m in record[1].upper().replace(b"U", b"T") for m in motifs):
                names.append("dense")
            for name in names:
                seen[name] += 1
                if len(subsets[name]) < sizes[name]:
                    subsets[name].append(record)
                else:
                    j = rng.randrange(seen[name])
                    if j < sizes[name]:
                        subsets[name][j] = record
    return subsets, seen


def record_id(header):
    return header.split()[0].decode()


def run_cdhit(records, workdir, name, dedup):
    fasta, output = workdir / f"{name}.fasta", workdir / f"{name}.cdhit.fasta"
    with open(fasta, "wb") as f:
        for header, seq in records:
            f.write(b">" + header + b"\n" + seq + b"\n")
    cmd = ["cd-hit-est", "-i", str(fasta), "-o", str(output), "-c", str(dedup["identity"]),
           "-r", "1" if dedup["both_strands"] else "0", "-M", str(dedup["memory_mb"]),
           "-T", str(dedup["threads"]), *dedup["other_options"]]
    start = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    with open(output, "rb") as f:
        reps = {record_id(line[1:]) for line in f if line.startswith(b">")}
    return reps, time.time() - start


def run_containment(records, settings):
    codes = np.frombuffer(b"".join(seq.translate(LETTER_CODES) for _, seq in records), np.uint8)
    lengths = np.array([len(seq) for _, seq in records], np.int64)
    start = time.time()
    keep, _ = representatives(codes, lengths, settings["min_length"], settings["short_anchor"],
                              settings["long_anchor"], settings["threads"])
    return {record_id(records[i][0]) for i in np.flatnonzero(keep)}, time.time() - start


def main():
    config = load_config()
    corpora = config["corpora"]
    check = corpora["dedup_check"]
    candidate = next(c for c in corpora["rna_fm"]["candidates"] if c["name"] == check["candidate"])
    workdir = repo_path(config["paths"]["processed_dir"]) / "dedup_check"
    workdir.mkdir(parents=True, exist_ok=True)

    subsets, seen = sample_subsets(candidate_files(config, candidate), check)
    print(f"records seen: {seen}", flush=True)
    result = {"candidate": check["candidate"], "records_matching_dense_motif": seen["dense"]}
    for name, records in subsets.items():
        lengths = {record_id(h): len(s) for h, s in records}
        cdhit, cdhit_s = run_cdhit(records, workdir, name, corpora["dedup"])
        ours, ours_s = run_containment(records, corpora["dedup_containment"])
        only_cdhit, only_ours = sorted(cdhit - ours), sorted(ours - cdhit)
        result[name] = {
            "records": len(records), "cdhit_representatives": len(cdhit),
            "containment_representatives": len(ours),
            "kept_only_by_cdhit": len(only_cdhit), "kept_only_by_containment": len(only_ours),
            "examples_only_cdhit": [(i, lengths[i]) for i in only_cdhit[:20]],
            "examples_only_containment": [(i, lengths[i]) for i in only_ours[:20]],
            "cdhit_seconds": round(cdhit_s), "containment_seconds": round(ours_s, 1),
        }
        print(json.dumps(result[name]), flush=True)

    out = repo_path(config["paths"]["results_dir"]) / "04_dedup_check.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    write_meta(out, config, packages=["numpy", "numba"])


if __name__ == "__main__":
    main()
