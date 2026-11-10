"""Build the post-snapshot placebo corpus (config corpora.post_snapshot): sequences of a release
published after every model's training data, minus every sequence (exact, after uppercase + T->U)
found in the releases the models were trained on, deduplicated at 100% by exact containment on both
strands (corpora.dedup_containment, as for the RNA-FM corpus). Homologs in this corpus are ones no
model can have seen, so their count should not predict F1 once real exposure is accounted for.

Usage: python scripts/20_build_placebo.py

Writes <processed_dir>/corpora/post_snapshot.fasta (normalized, one line per sequence; script 13
searches it like any corpus) and results/20_placebo_corpus.json (counts) plus its .meta.json.
"""
import hashlib
import json
import time

import numpy as np

from rnalm_exposure.config import display_path, load_config, repo_path, rnacentral_file
from rnalm_exposure.containment import load_codes, representatives
from rnalm_exposure.fasta import iter_gz_fasta, normalize
from rnalm_exposure.meta import write_meta


def main():
    config = load_config()
    paths, cfg, dedup = config["paths"], config["corpora"]["post_snapshot"], config["corpora"]["dedup_containment"]
    if cfg["exclusion_match"] != "exact" or cfg["dedup"] != "containment":
        raise ValueError("only exact exclusion followed by containment dedup is implemented")
    out_path = repo_path(paths["corpus_fasta"].format(processed_dir=paths["processed_dir"], name="post_snapshot"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start, counts = time.time(), {}

    excluded = set()   # 16-byte MD5 digests of normalized sequences
    for entry in cfg["exclude_releases"]:
        for f in entry["files"]:
            n = 0
            for _, seq in iter_gz_fasta(rnacentral_file(config, entry["release"], f)):
                excluded.add(hashlib.md5(normalize(seq)).digest())
                n += 1
            counts[f"excluded_{entry['release']}_{f}"] = n
            print(f"read {entry['release']}/{f}: {n} records, {len(excluded)} distinct so far", flush=True)
    counts["excluded_distinct"] = len(excluded)

    # exact pass: new sequences only, one copy each, written normalized
    exact_path = out_path.with_suffix(".exact.fasta")
    seen, records, kept = set(), 0, 0
    with open(exact_path, "wb") as out:
        for f in cfg["files"]:
            for header, seq in iter_gz_fasta(rnacentral_file(config, cfg["release"], f)):
                records += 1
                seq = normalize(seq)
                d = hashlib.md5(seq).digest()
                if d in excluded or d in seen:
                    continue
                seen.add(d)
                kept += 1
                out.write(b">" + header + b"\n" + seq + b"\n")
    del excluded, seen
    counts.update(release_records=records, new_distinct=kept)
    print(f"{cfg['release']}: {records} records, {kept} not in any training release", flush=True)

    # 100% containment dedup of the new sequences (both strands, as cd-hit-est -c 1.0 -r 1)
    codes_path = out_path.with_suffix(".codes")
    lengths = load_codes([exact_path], codes_path)
    codes = np.fromfile(codes_path, np.uint8)
    codes_path.unlink()
    keep, info = representatives(codes, lengths, dedup["min_length"], dedup["short_anchor"],
                                 dedup["long_anchor"], dedup["threads"])
    del codes
    i = 0
    with open(out_path, "wb") as out:
        for header, seq in iter_gz_fasta(exact_path):
            if keep[i]:
                out.write(b">" + header + b"\n" + seq + b"\n")
            i += 1
    exact_path.unlink()
    counts.update(discarded_short=info["discarded_short"], sequences=int(keep.sum()))

    result = {"release": cfg["release"], "excluded_releases": cfg["exclude_releases"], **counts,
              "output": display_path(out_path), "minutes": round((time.time() - start) / 60, 1)}
    out = repo_path(paths["results_dir"]) / "20_placebo_corpus.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    write_meta(out, config, packages=["numpy", "numba"])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
