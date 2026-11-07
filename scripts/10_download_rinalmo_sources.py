"""Download the Rfam and RefSeq parts of the approximated RiNALMo corpus.

Usage: python scripts/10_download_rinalmo_sources.py [rfam] [refseq]   (default: both)

rfam:   every family FASTA in Rfam 14.9's fasta_files/ (one file per family), with curl.
refseq: the NR_ and XR_ records of RefSeq release 216, at the exact accession.version listed in
        the release catalog (script 07 downloads it). NCBI keeps no FASTA of old releases, so the
        sequences are fetched from E-utilities efetch in batches. Batches already on disk are
        skipped, so the download can be stopped and restarted. Accessions efetch does not return
        (e.g. removed records) are listed in the check file rather than stopping the run.

Writes <raw_dir>/download_check_rinalmo_sources.json (Rfam files with size and MD5; RefSeq counts
of requested, returned and missing accessions) plus its .meta.json.
"""
import gzip
import hashlib
import http.client
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.meta import write_meta


def md5sum(path, block=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def download_rfam(cfg, raw_dir):
    out_dir = raw_dir / cfg["rfam_dest"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(cfg["rfam_fasta_url"]) as r:
        names = sorted(set(re.findall(r'href="(RF\d{5}\.fa\.gz)"', r.read().decode())))
    files = []
    for name in names:
        path = out_dir / name
        if not path.exists():
            part = path.with_name(name + ".part")
            subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10", "-sS",
                            "-o", str(part), cfg["rfam_fasta_url"] + name], check=True)
            part.rename(path)
        files.append({"file": display_path(path), "bytes": path.stat().st_size, "md5": md5sum(path)})
    print(f"rfam: {len(files)} family files", flush=True)
    return {"url": cfg["rfam_fasta_url"], "n_files": len(files), "files": files}


def catalog_accessions(catalog, prefixes):
    """accession.version of every catalog record with one of the prefixes (third column)."""
    with gzip.open(catalog, "rt") as f:
        return [line.split("\t")[2] for line in f if line.split("\t")[2].startswith(tuple(prefixes))]


def efetch(ids, ef):
    data = urllib.parse.urlencode({"db": "nuccore", "rettype": "fasta", "retmode": "text",
                                   "tool": ef["tool"], "id": ",".join(ids)}).encode()
    for attempt in range(ef["retries"]):
        try:
            with urllib.request.urlopen(urllib.request.Request(ef["url"], data=data), timeout=300) as r:
                text = r.read().decode()
            if not text.startswith(">"):   # NCBI sometimes answers 200 with an error message instead of FASTA
                raise OSError(f"not FASTA: {text[:200]!r}")
            return text
        except (OSError, http.client.HTTPException) as e:   # HTTP 429/5xx, timeouts, cut-off replies, error pages: wait and retry
            print(f"efetch retry {attempt + 1}: {e!r}", flush=True)
            time.sleep(ef["retry_wait_s"] * (attempt + 1))
    raise RuntimeError(f"efetch failed {ef['retries']} times for batch starting {ids[0]}")


def download_refseq(cfg, raw_dir):
    ef = cfg["refseq_efetch"]
    ids = catalog_accessions(raw_dir / cfg["refseq_catalog"], cfg["refseq_prefixes"])
    out_dir = raw_dir / cfg["refseq_dest"]
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [ids[i:i + ef["batch_size"]] for i in range(0, len(ids), ef["batch_size"])]
    missing, returned = [], 0
    for b, batch in enumerate(batches):
        path = out_dir / f"batch_{b:05d}.fa"
        if not path.exists():
            start = time.time()
            text = efetch(batch, ef)
            part = path.with_name(path.name + ".part")
            part.write_text(text)
            part.rename(path)
            time.sleep(max(0.0, ef["min_interval_s"] - (time.time() - start)))
        got = set(re.findall(r"^>(\S+)", path.read_text(), flags=re.M))
        returned += len(got & set(batch))
        missing += [a for a in batch if a not in got]
        if b % 100 == 0:
            print(f"refseq: batch {b + 1}/{len(batches)}, {returned} returned, {len(missing)} missing", flush=True)
    print(f"refseq: {len(ids)} requested, {returned} returned, {len(missing)} missing", flush=True)
    return {"requested": len(ids), "returned": returned, "missing": missing, "batches": len(batches),
            "dir": display_path(out_dir)}


def main():
    config = load_config()
    cfg = config["corpora"]["rinalmo"]
    raw_dir = repo_path(config["paths"]["raw_dir"])
    parts = sys.argv[1:] or ["rfam", "refseq"]
    report = {}
    if "rfam" in parts:
        report["rfam"] = download_rfam(cfg, raw_dir)
    if "refseq" in parts:
        report["refseq"] = download_refseq(cfg, raw_dir)
    out = raw_dir / "download_check_rinalmo_sources.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    write_meta(out, config)


if __name__ == "__main__":
    main()
