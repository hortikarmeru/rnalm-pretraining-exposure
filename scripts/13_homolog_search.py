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
from pathlib import Path

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.mmseqs import mmseqs, search
from rnalm_exposure.queries import query_sequences, write_fasta


def count_targets(tsv, skip):
    """Distinct targets per query in a convertalis table, whose rows come grouped by query."""
    counts, current, targets = {}, None, set()
    for line in open(tsv):
        q, t = line.split("\t", 2)[:2]
        if q != current:
            if current is not None:
                counts[current] = len(targets)
            if q in counts:
                raise ValueError(f"{tsv}: rows for {q} are not contiguous")
            current, targets = q, set()
        targets.add(t)
    if current is not None:
        counts[current] = len(targets)
    return {q: n for q, n in counts.items() if q not in skip}


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
    homologs = {}
    rounds, todo_db, max_seqs = [], qdb, hs["max_target_seqs"]
    with open(hits_path, "w") as out:
        while True:
            res, pref = search(todo_db, tdb, work, hs, max_seqs)
            capped = {q for q, n in pref.items() if n >= max_seqs}
            tsv = work / f"res_{max_seqs}.tsv"
            mmseqs("convertalis", todo_db, tdb, res, tsv, "--format-output", "query,target,fident,qcov",
                   "--threads", hs["threads"])
            homologs.update(count_targets(tsv, skip=capped))   # capped queries are searched again
            for line in open(tsv):
                if line.split("\t", 1)[0] not in capped:
                    out.write(line)
            tsv.unlink()
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
            n = homologs.get(q, 0)
            f.write(f"{q}\t{n}\t{math.log(hs['log_offset'] + n):.6f}\n")
    return {"corpus": name, "queries": len(queries), "with_homologs": sum(homologs.get(q, 0) > 0 for q in queries),
            "rounds": rounds, "hits": display_path(hits_path), "per_sequence": display_path(out_tsv)}


def main():
    config = load_config()
    paths, hs = config["paths"], config["homolog_search"]
    if hs["log_base"] != "e" or hs["count"] != "unique_targets":
        raise ValueError("only natural log of unique-target counts is implemented")
    exposure_dir = repo_path(paths["exposure_dir"].format(processed_dir=paths["processed_dir"]))
    exposure_dir.mkdir(parents=True, exist_ok=True)
    queries_fasta = exposure_dir / "queries.fasta"
    queries = query_sequences(config)
    write_fasta(queries, queries_fasta)
    for name in sys.argv[1:]:
        result = run(name, config, queries, queries_fasta)
        out = repo_path(paths["results_dir"]) / f"13_exposure_{name}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        write_meta(out, config, extra={"mmseqs": subprocess.run(["mmseqs", "version"], capture_output=True,
                                                                  text=True).stdout.strip()})


if __name__ == "__main__":
    main()
