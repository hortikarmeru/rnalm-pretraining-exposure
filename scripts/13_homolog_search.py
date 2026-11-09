"""Count each query sequence's homologs in a pretraining corpus with MMseqs2 (settings in config
homolog_search): corpus sequences >= 80% identical over >= 80% of the query's length, forward
strand only. The queries are the test sequences (script 12) and Wang et al. 2026's selection and
reproduction sets, whose exposure sets the low-exposure split used to choose heads.

Usage: python scripts/13_homolog_search.py <corpus> [<corpus> ...]   (paths.corpus_fasta, script 11)

MMseqs2's prefilter keeps at most --max-seqs candidates per query. Queries that reach the cap are
searched again on their own with a 10x higher cap until none reach it.

Writes <exposure_dir>/<corpus>/hits.tsv (query, target, identity, query coverage),
results/13_exposure_<corpus>.tsv (seq_id, homologs, log_homologs) and
results/13_exposure_<corpus>.json (search settings and caps used) plus its .meta.json.
"""
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.test_sets import normalize
from rnalm_exposure.wang import load_set

STRAND = {"forward": 1, "reverse": 0, "both": 2}


def mmseqs(*args):
    subprocess.run(["mmseqs", *map(str, args)], check=True, stdout=subprocess.DEVNULL)


def write_queries(config, path):
    paths, val = config["paths"], config["validation"]
    queries = {}
    with open(repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))) as f:
        for line in f:
            r = json.loads(line)
            queries[r["seq_id"]] = r["sequence"]
    zip_path = repo_path(val["wang2026_zip"].format(raw_dir=paths["raw_dir"]))
    for role in ("selection_set", "reproduction_set"):
        name = val[role]
        for r in load_set(zip_path, val["wang2026_files"][name], name):
            queries[r["seq_id"]] = normalize(r["sequence"])
    write_fasta(queries, path)
    return queries


def write_fasta(seqs, path):
    with open(path, "w") as f:
        for seq_id, seq in seqs.items():
            f.write(f">{seq_id}\n{seq}\n")


def search(qdb, tdb, work, hs, max_seqs):
    """One MMseqs2 search; returns (result db, prefilter candidates per query id)."""
    res, tmp = work / f"res_{max_seqs}", work / f"tmp_{max_seqs}"
    mmseqs("search", qdb, tdb, res, tmp, "--search-type", hs["search_type"], "-k", hs["kmer_length"],
           "-s", hs["sensitivity"], "-e", hs["evalue"], "--max-seqs", max_seqs,
           "--min-seq-id", hs["min_identity"], "-c", hs["min_coverage"], "--cov-mode", hs["coverage_mode"],
           "--seq-id-mode", hs["seq_id_mode"], "--strand", STRAND[hs["strand"]], "--threads", hs["threads"],
           "--remove-tmp-files", 0)
    (pref,) = [p.with_suffix("") for p in tmp.glob("*/search/pref_0.dbtype")]   # prefilter output, kept to check the cap
    stats = work / f"pref_count_{max_seqs}"
    mmseqs("result2stats", qdb, tdb, pref, stats, "--stat", "linecount", "--threads", hs["threads"])
    mmseqs("createtsv", qdb, stats, f"{stats}.tsv")
    counts = {}
    for line in open(f"{stats}.tsv"):
        q, n = line.split("\t")[:2]
        counts[q] = int(n)
    subprocess.run(["rm", "-rf", str(tmp)], check=True)
    return res, counts


def run(name, config, queries, queries_fasta):
    paths, hs = config["paths"], config["homolog_search"]
    work = repo_path(paths["exposure_dir"].format(processed_dir=paths["processed_dir"])) / name
    work.mkdir(parents=True, exist_ok=True)
    qdb, tdb = work / "queries", work / "corpus"
    mmseqs("createdb", queries_fasta, qdb, "--dbtype", 2)
    if not Path(f"{tdb}.dbtype").exists():
        corpus = repo_path(paths["corpus_fasta"].format(processed_dir=paths["processed_dir"], name=name))
        mmseqs("createdb", corpus, tdb, "--dbtype", 2)

    hits_path = work / "hits.tsv"
    hits = defaultdict(set)
    rounds, todo_db, max_seqs = [], qdb, hs["max_target_seqs"]
    with open(hits_path, "w") as out:
        while True:
            res, pref = search(todo_db, tdb, work, hs, max_seqs)
            capped = {q for q, n in pref.items() if n >= max_seqs}
            tsv = work / f"res_{max_seqs}.tsv"
            mmseqs("convertalis", todo_db, tdb, res, tsv, "--format-output", "query,target,fident,qcov",
                   "--threads", hs["threads"])
            for line in open(tsv):
                q = line.split("\t", 1)[0]
                if q in capped:   # searched again with a higher cap
                    continue
                out.write(line)
                hits[q].add(line.split("\t", 2)[1])
            rounds.append({"max_seqs": max_seqs, "queries": len(pref), "capped": sorted(capped)})
            print(name, json.dumps(rounds[-1]), flush=True)
            if not capped:
                break
            max_seqs *= 10
            todo_db = work / f"queries_capped_{max_seqs}"
            write_fasta({q: queries[q] for q in sorted(capped)}, f"{todo_db}.fasta")
            mmseqs("createdb", f"{todo_db}.fasta", todo_db, "--dbtype", 2)

    out_tsv = repo_path(paths["results_dir"]) / f"13_exposure_{name}.tsv"
    with open(out_tsv, "w") as f:
        f.write("seq_id\thomologs\tlog_homologs\n")
        for q in queries:
            n = len(hits.get(q, ()))
            f.write(f"{q}\t{n}\t{math.log(hs['log_offset'] + n):.6f}\n")
    return {"corpus": name, "queries": len(queries), "with_homologs": sum(q in hits for q in queries),
            "rounds": rounds, "hits": display_path(hits_path), "per_sequence": display_path(out_tsv)}


def main():
    config = load_config()
    paths, hs = config["paths"], config["homolog_search"]
    if hs["log_base"] != "e" or hs["count"] != "unique_targets":
        raise ValueError("only natural log of unique-target counts is implemented")
    exposure_dir = repo_path(paths["exposure_dir"].format(processed_dir=paths["processed_dir"]))
    exposure_dir.mkdir(parents=True, exist_ok=True)
    queries_fasta = exposure_dir / "queries.fasta"
    queries = write_queries(config, queries_fasta)
    for name in sys.argv[1:]:
        result = run(name, config, queries, queries_fasta)
        out = repo_path(paths["results_dir"]) / f"13_exposure_{name}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        write_meta(out, config, extra={"mmseqs": subprocess.run(["mmseqs", "version"], capture_output=True,
                                                                  text=True).stdout.strip()})


if __name__ == "__main__":
    main()
