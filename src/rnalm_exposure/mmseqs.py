"""Run MMseqs2 nucleotide searches with the config homolog_search settings."""
import subprocess

STRAND = {"forward": 1, "reverse": 0, "both": 2}


def mmseqs(*args):
    subprocess.run(["mmseqs", *map(str, args)], check=True, stdout=subprocess.DEVNULL)


def search(qdb, tdb, work, hs, max_seqs):
    """One MMseqs2 search; returns (result db, prefilter candidates per query id)."""
    res, tmp = work / f"res_{max_seqs}", work / f"tmp_{max_seqs}"
    mmseqs("search", qdb, tdb, res, tmp, "--search-type", hs["search_type"], "-k", hs["kmer_length"],
           "-s", hs["sensitivity"], "-e", hs["evalue"], "--max-seqs", max_seqs,
           "--min-seq-id", hs["min_identity"], "-c", hs["min_coverage"], "--cov-mode", hs["coverage_mode"],
           "--seq-id-mode", hs["seq_id_mode"], "--strand", STRAND[hs["strand"]], "--threads", hs["threads"],
           "--remove-tmp-files", 0)
    pref = tmp / "latest" / "search" / "pref_0"   # prefilter output, kept to check the cap ("latest" links to the run's folder)
    stats = work / f"pref_count_{max_seqs}"
    mmseqs("result2stats", qdb, tdb, pref, stats, "--stat", "linecount", "--threads", hs["threads"])
    mmseqs("createtsv", qdb, stats, f"{stats}.tsv")
    counts = {}
    for line in open(f"{stats}.tsv"):
        q, n = line.split("\t")[:2]
        counts[q] = int(n)
    subprocess.run(["rm", "-rf", str(tmp)], check=True)
    return res, counts
