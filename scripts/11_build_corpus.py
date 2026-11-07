"""Build a pretraining corpus that is deduplicated exactly (not by containment): merge its sources,
apply its length filter, drop repeated sequences.

Usage: python scripts/11_build_corpus.py <corpus> [<corpus> ...]   (corpora.<corpus>.build_sources,
e.g. rinalmo, aido_rna)

Sequences are normalized (uppercase, T -> U) before the length check and the duplicate check, and
the first copy of each sequence is kept, with its header, in source order. This is what seqkit
rmdup -s does after normalization; RiNALMo and AIDO.RNA both report exact dedup with SeqKit.
Source paths are globs under raw_dir, expanded in sorted order.

Writes <processed_dir>/corpora/<corpus>.fasta (one line per sequence) and
results/11_corpus_<corpus>.json (counts per source) plus its .meta.json.
"""
import hashlib
import json
import sys
import time

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.fasta import iter_gz_fasta, normalize
from rnalm_exposure.meta import write_meta


def build(cfg, raw_dir, out_path):
    min_len, max_len = cfg.get("min_length") or 0, cfg.get("max_length") or float("inf")
    seen = set()   # 16-byte MD5 digests: ~42M sequences fit in a few GB
    sources = []
    with open(out_path, "wb") as out:
        for pattern in cfg["build_sources"]:
            files = sorted(raw_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"no files match {raw_dir / pattern}")
            counts = {"pattern": pattern, "files": len(files), "records": 0, "outside_length": 0,
                      "duplicates": 0, "kept": 0}
            for path in files:
                for header, seq in iter_gz_fasta(path):
                    counts["records"] += 1
                    seq = normalize(seq)
                    if not min_len <= len(seq) <= max_len:
                        counts["outside_length"] += 1
                        continue
                    digest = hashlib.md5(seq).digest()
                    if digest in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(digest)
                    counts["kept"] += 1
                    out.write(b">" + header + b"\n" + seq + b"\n")
            print(json.dumps(counts), flush=True)
            sources.append(counts)
    return {"min_length": cfg.get("min_length"), "max_length": cfg.get("max_length"), "sources": sources,
            "records": sum(s["records"] for s in sources), "sequences": len(seen)}


def main():
    config = load_config()
    paths = config["paths"]
    raw_dir = repo_path(paths["raw_dir"])
    for name in sys.argv[1:]:
        out_path = repo_path(paths["corpus_fasta"].format(processed_dir=paths["processed_dir"], name=name))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        result = {"corpus": name, **build(config["corpora"][name], raw_dir, out_path),
                  "output": display_path(out_path), "minutes": round((time.time() - start) / 60, 1)}
        print(f"{name}: {result['sequences']} sequences from {result['records']} records", flush=True)
        out = repo_path(paths["results_dir"]) / f"11_corpus_{name}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        write_meta(out, config)


if __name__ == "__main__":
    main()
