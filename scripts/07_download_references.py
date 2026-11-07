"""Download the test sets, validation sets and reference files listed in data_manifests/downloads.tsv.

Usage: python scripts/07_download_references.py [name ...]   (default: every row)

Each row gives a URL, the path under raw_dir to save it to, and, where known, the size and MD5.
Files already present at the listed size are not downloaded again. Rows without a listed size
or MD5 (files generated on request, e.g. Europe PMC supplement bundles) are accepted as
downloaded and their size and MD5 are recorded, so they can be copied into the manifest.

Writes <raw_dir>/download_check_references.json (size and MD5 per file) plus its .meta.json.
Exits with an error if any file does not match the manifest.
"""
import csv
import hashlib
import json
import subprocess
import sys

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.meta import write_meta


def md5sum(path, block=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def main():
    config = load_config()
    paths = config["paths"]
    with open(repo_path(paths["manifests"]["downloads"])) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    names = sys.argv[1:]
    unknown = [n for n in names if n not in {r["name"] for r in rows}]
    if unknown:
        sys.exit(f"unknown names: {unknown}")
    rows = [r for r in rows if not names or r["name"] in names]

    report, ok = [], True
    for r in rows:
        path = repo_path(paths["raw_dir"]) / r["dest"]
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = int(r["bytes"]) if r["bytes"] else None
        if not (path.exists() and (expected is None or path.stat().st_size == expected)):
            # curl follows redirects (Dropbox, Zenodo, GitHub releases); --fail turns HTTP errors into exit codes
            subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10", "-sS",
                            "-o", str(path), r["url"]], check=True)
        size, digest = path.stat().st_size, md5sum(path)
        match = (expected is None or size == expected) and (not r["md5"] or digest == r["md5"])
        ok &= match
        report.append({"name": r["name"], "file": display_path(path), "bytes": size, "md5": digest,
                       "matches_manifest": match})
        print(f"{r['name']}: {size} bytes, md5 {digest}, {'OK' if match else 'MISMATCH'}", flush=True)

    out = repo_path(paths["raw_dir"]) / "download_check_references.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    write_meta(out, config)
    if not ok:
        sys.exit("some files do not match data_manifests/downloads.tsv")


if __name__ == "__main__":
    main()
