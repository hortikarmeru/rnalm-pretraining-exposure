"""Count records and exact-duplicate sequences in each candidate RNAcentral release (C1 evidence).

Usage: python scripts/00_count_rnacentral_release.py

Reads corpora.rna_fm.candidate_releases and paths.rnacentral_fasta from config.yaml and writes
results/00_rnacentral_release_counts.json (one entry per release) plus its .meta.json.

Per release:
  records            number of '>' headers
  unique_raw         distinct sequences as written in the file
  unique_normalized  distinct sequences after uppercase + T->U (RNA-FM's preprocessing)

Sequences are stored as 16-byte MD5 digests so ~36M of them fit in a few GB.
cd-hit-est at 100% (RNA-FM's actual dedup) also merges sequences contained in
longer identical ones, so it can only go lower than unique_normalized.
"""
import hashlib
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, repo_path  # noqa: E402
from src.meta import write_meta  # noqa: E402


def count_release(args):
    release, path = args
    raw_seen, norm_seen = set(), set()
    records = 0
    chunks = []

    def flush():
        if not chunks:
            return
        seq = b"".join(chunks)
        raw_seen.add(hashlib.md5(seq).digest())
        norm_seen.add(hashlib.md5(seq.upper().replace(b"T", b"U")).digest())
        chunks.clear()

    # gzip -dc is several times faster than Python's gzip module on multi-GB files.
    proc = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE, bufsize=1 << 20)
    for line in proc.stdout:
        if line.startswith(b">"):
            flush()
            records += 1
        else:
            chunks.append(line.rstrip())
    flush()
    if proc.wait() != 0:
        raise RuntimeError(f"gzip failed on {path} (incomplete download?)")
    print(f"{release}: {records} records", flush=True)
    return {
        "release": release,
        "file": str(path),
        "records": records,
        "unique_raw": len(raw_seen),
        "unique_normalized": len(norm_seen),
    }


def main():
    config = load_config()
    raw_dir = config["paths"]["raw_dir"]
    releases = config["corpora"]["rna_fm"]["candidate_releases"]
    jobs = [(r, repo_path(config["paths"]["rnacentral_fasta"].format(raw_dir=raw_dir, release=r)))
            for r in releases]
    for _, path in jobs:
        if not path.exists():
            sys.exit(f"missing {path}")

    # One process per release: each holds two digest sets (a few GB), so this needs
    # roughly 5 GB of RAM per release.
    with Pool(len(jobs)) as pool:
        results = pool.map(count_release, jobs)

    out = repo_path(config["paths"]["results_dir"]) / "00_rnacentral_release_counts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    write_meta(out, config, packages=["pyyaml"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
