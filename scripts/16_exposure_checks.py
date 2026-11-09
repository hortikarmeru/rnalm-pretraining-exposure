"""Check measures of exposure (config exposure_checks) for the same queries as script 13:
exact_match = the query occurs letter for letter inside a corpus sequence (forward strand), and
max_identity = the best identity among MMseqs2 hits covering >= 80% of the query, with no identity
floor (empty if no hit at all).

Usage: python scripts/16_exposure_checks.py <corpus> [<corpus> ...]   (after script 13 for that corpus)

exact_match runs GNU grep -F over the corpus sequence lines. grep -o reports non-overlapping
matches only, so one pass can miss a query that overlaps another query's match; found queries are
removed and grep runs again on the rest until a pass finds none, which proves the rest absent.

max_identity: script 13's hits.tsv holds every hit at >= 80% identity, so a query with homologs takes
its best identity from there. Queries without homologs are searched again with --min-seq-id 0.

Writes results/16_exposure_checks_<corpus>.tsv (seq_id, exact_match, max_identity) and
results/16_exposure_checks_<corpus>.json (counts, grep passes) plus its .meta.json.
"""
import json
import subprocess
import sys

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.mmseqs import mmseqs, search
from rnalm_exposure.queries import query_sequences, write_fasta


def contained(queries, corpus, work):
    """Queries found inside some corpus sequence line, and the number of grep passes."""
    found, passes = set(), 0
    while True:
        rest = sorted(set(queries.values()) - found)
        if not rest:
            return set(queries), passes
        patterns = work / "patterns.txt"
        patterns.write_text("".join(s + "\n" for s in rest))
        passes += 1
        run = subprocess.run(["bash", "-o", "pipefail", "-c",   # -a: read the corpus as text even if a stray byte looks binary
                              f"LC_ALL=C grep -a -v '^>' {corpus} | LC_ALL=C grep -a -F -o -f {patterns} | LC_ALL=C sort -u"],
                             capture_output=True, text=True)
        if run.returncode not in (0, 1):   # grep exits 1 when nothing matches
            raise RuntimeError(run.stderr)
        out = run.stdout.split()
        new = set(out) - found
        print(f"grep pass {passes}: {len(rest)} patterns, {len(new)} found", flush=True)
        if not new:
            return {q for q, s in queries.items() if s in found}, passes
        found |= new


def best_identity(tsv):
    best = {}
    for line in open(tsv):
        q, _, fident = line.split("\t")[:3]
        best[q] = max(best.get(q, 0.0), float(fident))
    return best


def search_without_floor(seqs, tdb, work, hs):
    """Best identity per query among hits covering >= 80% of it, searched with --min-seq-id 0."""
    write_fasta(seqs, work / "no_homologs.fasta")
    qdb = work / "no_homologs"
    mmseqs("createdb", work / "no_homologs.fasta", qdb, "--dbtype", 2)
    res, pref = search(qdb, tdb, work, {**hs, "min_identity": 0}, hs["max_target_seqs"])
    capped = sorted(q for q, n in pref.items() if n >= hs["max_target_seqs"])
    if capped:
        raise RuntimeError(f"prefilter cap reached for {capped}")
    mmseqs("convertalis", qdb, tdb, res, work / "no_homologs.tsv", "--format-output", "query,target,fident,qcov",
           "--threads", hs["threads"])
    return best_identity(work / "no_homologs.tsv")


def run(name, config, queries):
    paths, hs = config["paths"], config["homolog_search"]
    exposure_dir = repo_path(paths["exposure_dir"].format(processed_dir=paths["processed_dir"]))
    work = exposure_dir / name / "checks"
    subprocess.run(["rm", "-rf", str(work)], check=True)   # mmseqs refuses to overwrite a result db from an earlier run
    work.mkdir(parents=True)

    identity = best_identity(exposure_dir / name / "hits.tsv")
    if not identity.keys() <= queries.keys():
        raise ValueError(f"{name}/hits.tsv has queries outside the current query list; rerun script 13")
    no_homologs = {q: s for q, s in queries.items() if q not in identity}
    if no_homologs:
        identity.update(search_without_floor(no_homologs, exposure_dir / name / "corpus", work, hs))

    corpus = repo_path(paths["corpus_fasta"].format(processed_dir=paths["processed_dir"], name=name))
    exact, passes = contained(queries, corpus, work)

    with open(repo_path(paths["results_dir"]) / f"16_exposure_checks_{name}.tsv", "w") as f:
        f.write("seq_id\texact_match\tmax_identity\n")
        for q in queries:
            f.write(f"{q}\t{int(q in exact)}\t{identity.get(q, '')}\n")
    return {"corpus": name, "queries": len(queries), "exact_match": len(exact), "grep_passes": passes,
            "searched_without_identity_floor": len(no_homologs),
            "with_any_hit": sum(q in identity for q in queries),
            # MMseqs2 splits targets > 10,000 nt into non-overlapping pieces and masks low-complexity
            # regions, so it can miss an exact match that grep finds
            "exact_match_without_full_identity_hit": sum(identity.get(q, 0) < 1 for q in exact)}


def main():
    config = load_config()
    if config["exposure_checks"]["exact_match"] != "containment" or \
            config["exposure_checks"]["max_identity"] != "coverage_0.8" or config["homolog_search"]["min_coverage"] != 0.8:
        raise ValueError("only containment and best identity at 80% query coverage are implemented")
    queries = query_sequences(config)
    for name in sys.argv[1:]:
        result = run(name, config, queries)
        print(json.dumps(result), flush=True)
        out = repo_path(config["paths"]["results_dir"]) / f"16_exposure_checks_{name}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        write_meta(out, config, extra={"mmseqs": subprocess.run(["mmseqs", "version"], capture_output=True,
                                                                  text=True).stdout.strip(),
                                       "grep": subprocess.run(["grep", "--version"], capture_output=True,
                                                              text=True).stdout.splitlines()[0]})


if __name__ == "__main__":
    main()
