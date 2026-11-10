"""Cluster check of exposure for RiNALMo (config exposure_checks.log_cluster_count): RiNALMo samples
one sequence per cluster of its corpus in each epoch, so the number of distinct clusters among a test
sequence's homologs is closer to what the model saw than the raw homolog count. The corpus is
clustered as in corpora.rinalmo.clusters (MMseqs2 linclust), and each query's homologs from
script 13 (hits.tsv) are mapped to their clusters.

Usage: python scripts/21_cluster_counts.py   (after script 13 for rinalmo)

Writes results/21_cluster_counts_rinalmo.tsv (seq_id, clusters, log_clusters = ln(1 + clusters)) and
results/21_cluster_counts_rinalmo.json (cluster totals) plus its .meta.json.
"""
import json
import math
import subprocess

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.mmseqs import mmseqs
from rnalm_exposure.queries import query_sequences


def main():
    config = load_config()
    paths, hs = config["paths"], config["homolog_search"]
    cl = config["corpora"]["rinalmo"]["clusters"]
    if cl["tool"] != "mmseqs2 easy-linclust":
        raise ValueError("only MMseqs2 linclust is implemented")
    work = repo_path(paths["exposure_dir"].format(processed_dir=paths["processed_dir"])) / "rinalmo"
    clu, tmp, tsv = work / "clusters", work / "tmp_linclust", work / "clusters.tsv"
    if not tsv.exists():
        # the corpus database script 13 built; easy-linclust's clustering on that database
        mmseqs("linclust", work / "corpus", clu, tmp, "--min-seq-id", cl["min_seq_id"], "-c", cl["coverage"],
               "--threads", hs["threads"])
        mmseqs("createtsv", work / "corpus", work / "corpus", clu, tsv, "--threads", hs["threads"])
        subprocess.run(["rm", "-rf", str(tmp)], check=True)

    rep = {}   # member id -> cluster representative id
    for line in open(tsv):
        r, m = line.rstrip("\n").split("\t")
        rep[m] = r
    n_clusters = len(set(rep.values()))
    print(f"{len(rep)} sequences in {n_clusters} clusters", flush=True)

    clusters = {}   # query -> set of cluster representatives among its homologs
    for line in open(work / "hits.tsv"):
        q, t = line.split("\t", 2)[:2]
        clusters.setdefault(q, set()).add(rep[t])

    queries = query_sequences(config)
    out_tsv = repo_path(paths["results_dir"]) / "21_cluster_counts_rinalmo.tsv"
    with open(out_tsv, "w") as f:
        f.write("seq_id\tclusters\tlog_clusters\n")
        for q in queries:
            n = len(clusters.get(q, ()))
            f.write(f"{q}\t{n}\t{math.log(hs['log_offset'] + n):.6f}\n")
    out = repo_path(paths["results_dir"]) / "21_cluster_counts_rinalmo.json"
    with open(out, "w") as f:
        json.dump({"corpus_sequences": len(rep), "clusters": n_clusters, "min_seq_id": cl["min_seq_id"],
                   "coverage": cl["coverage"], "queries": len(queries),
                   "with_homologs": sum(q in clusters for q in queries)}, f, indent=2)
    write_meta(out, config, extra={"mmseqs": subprocess.run(["mmseqs", "version"], capture_output=True,
                                                              text=True).stdout.strip()})


if __name__ == "__main__":
    main()
