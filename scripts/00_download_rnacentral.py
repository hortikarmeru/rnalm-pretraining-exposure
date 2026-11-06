"""Download the RNAcentral releases listed in data_manifests/rnacentral.tsv and verify them.

Usage: python scripts/00_download_rnacentral.py

Uses aria2c with the options in download.aria2c_options. EBI limits the speed of each
connection, so several connections per file and several files at once are much faster than
a single wget. Files already present at the expected size are not downloaded again.

Writes <raw_dir>/rnacentral/download_check.json (size and MD5 per file) plus its .meta.json.
Exits with an error if any file does not match the manifest.
"""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile

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
    with open(repo_path(paths["manifests"]["rnacentral"])) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["path"] = repo_path(paths["rnacentral_fasta"].format(raw_dir=paths["raw_dir"], release=row["release"]))

    todo = [r for r in rows if not (r["path"].exists() and r["path"].stat().st_size == int(r["bytes"]))]
    if todo:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for r in todo:
                r["path"].parent.mkdir(parents=True, exist_ok=True)
                f.write(f"{r['url']}\n  dir={r['path'].parent}\n  out={r['path'].name}\n")
        subprocess.run(["aria2c", *config["download"]["aria2c_options"], "-i", f.name], check=True)

    report, ok = [], True
    for r in rows:
        size = r["path"].stat().st_size
        digest = md5sum(r["path"])
        match = size == int(r["bytes"]) and (not r["md5"] or digest == r["md5"])
        ok &= match
        report.append({"release": r["release"], "file": display_path(r["path"]), "bytes": size,
                       "md5": digest, "matches_manifest": match})
        print(f"{r['release']}: {size} bytes, md5 {digest}, {'OK' if match else 'MISMATCH'}", flush=True)

    out = repo_path(paths["raw_dir"]) / "rnacentral" / "download_check.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    write_meta(out, config)
    if not ok:
        sys.exit("some files do not match data_manifests/rnacentral.tsv")


if __name__ == "__main__":
    main()
