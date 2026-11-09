"""Assign each query sequence an Rfam family with Infernal's cmscan, following RNA3DB's rule
(settings in config family_assignment; RNA3DB's code: rna3db/tabular.py, scripts/get_nohits.py):
default cmscan on both strands, then a --max rescan of the sequences with no hit at all or, if
shorter than rescan_below_length, no hit at E <= cmscan_evalue. Each sequence takes its lowest
E-value hit at E <= cmscan_evalue (first in table order on ties); sequences without one get no
family and later form their own group. A gathering-threshold scan with clan competition is run as
a check. Queries are the same as script 13's (test sequences plus Wang et al. 2026's sets).

Usage: python scripts/15_assign_families.py   (Infernal's cmscan of config infernal_version on PATH)

Writes <processed_dir>/family/*.tbl (cmscan tables), results/15_families.tsv (seq_id, family,
family_name, evalue, strand, scan, check_family) and results/15_families.json (counts per set)
plus its .meta.json.
"""
import json
import os
import subprocess
from collections import Counter

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.queries import query_sequences, write_fasta


def infernal_version():
    return subprocess.run(["cmscan", "-h"], capture_output=True, text=True).stdout.splitlines()[1].split()[2]


def cmscan(cm, fasta, tbl, options):
    """Run cmscan unless tbl is already complete (Infernal ends finished tables with '# [ok]')."""
    if tbl.exists() and tbl.read_text().rstrip().endswith("# [ok]"):
        return
    subprocess.run(["cmscan", "--cpu", str(os.cpu_count()), "-o", os.devnull, "--tblout", str(tbl),
                    *options, str(cm), str(fasta)], check=True)


def read_tbl(tbl, fmt2=False):
    """Hits of a cmscan table in file order: query, accession, name, strand, E-value (and the
    overlap mark of --fmt 2)."""
    hits = []
    for line in open(tbl):
        if line.startswith("#"):
            continue
        f = line.split()
        if fmt2:   # idx, name, accession, query, query acc, clan, mdl, mdl from/to, seq from/to, strand, ..., E-value, inc, olp
            hits.append({"query": f[3], "accession": f[2], "name": f[1], "strand": f[11], "evalue": float(f[17]),
                         "olp": f[19]})
        else:      # name, accession, query, query acc, mdl, mdl from/to, seq from/to, strand, ..., E-value, inc
            hits.append({"query": f[2], "accession": f[1], "name": f[0], "strand": f[9], "evalue": float(f[15])})
    return hits


def top_hits(hits, cutoff):
    """Lowest E-value hit per query among hits with E <= cutoff; the first one on ties."""
    best = {}
    for h in sorted((h for h in hits if h["evalue"] <= cutoff), key=lambda h: h["evalue"]):
        best.setdefault(h["query"], h)
    return best


def main():
    config = load_config()
    paths, cfg = config["paths"], config["family_assignment"]
    if infernal_version() != cfg["infernal_version"]:
        raise RuntimeError(f"cmscan on PATH is Infernal {infernal_version()}, config asks for {cfg['infernal_version']}")
    raw_dir = repo_path(paths["raw_dir"])
    work = repo_path(paths["processed_dir"]) / "family"
    work.mkdir(parents=True, exist_ok=True)
    cm = work / "Rfam.cm"
    if not cm.with_suffix(".cm.i1m").exists():
        subprocess.run(f"gzip -dc {raw_dir / cfg['rfam_cm']} > {cm}", shell=True, check=True)
        subprocess.run(["cmpress", "-F", str(cm)], check=True, stdout=subprocess.DEVNULL)

    queries = query_sequences(config)
    write_fasta(queries, work / "queries.fasta")
    cmscan(cm, work / "queries.fasta", work / "main.tbl", cfg["cmscan_options"])
    main_hits = read_tbl(work / "main.tbl")

    any_hit = {h["query"] for h in main_hits}
    good_hit = set(top_hits(main_hits, cfg["cmscan_evalue"]))
    rescan = {q: s for q, s in queries.items()
              if q not in any_hit or (q not in good_hit and len(s) < cfg["rescan_below_length"])}
    write_fasta(rescan, work / "rescan.fasta")
    rescan_hits = []
    if rescan:
        cmscan(cm, work / "rescan.fasta", work / "rescan.tbl", cfg["cmscan_options"] + cfg["rescan_options"])
        rescan_hits = read_tbl(work / "rescan.tbl")
    for h in main_hits:
        h["scan"] = "main"
    for h in rescan_hits:
        h["scan"] = "rescan"
    best = top_hits(main_hits + rescan_hits, cfg["cmscan_evalue"])

    check_options = [a for o in cfg["check_options"] for a in ([o, str(raw_dir / cfg["rfam_clanin"])] if o == "--clanin" else [o])]
    cmscan(cm, work / "queries.fasta", work / "check.tbl", check_options)
    # --fmt 2 marks hits overlapped by a better hit of the same clan with "="; the rest compete by E-value
    check = top_hits([h for h in read_tbl(work / "check.tbl", fmt2=True) if h["olp"] != "="], float("inf"))

    out_tsv = repo_path(paths["results_dir"]) / "15_families.tsv"
    per_set = {}
    with open(out_tsv, "w") as f:
        f.write("seq_id\tfamily\tfamily_name\tevalue\tstrand\tscan\tcheck_family\n")
        for q in queries:
            h, c = best.get(q), check.get(q)
            family, check_family = (h["accession"] if h else ""), (c["accession"] if c else "")
            f.write(f"{q}\t{family}\t{h['name'] if h else ''}\t{h['evalue'] if h else ''}\t"
                    f"{h['strand'] if h else ''}\t{h['scan'] if h else ''}\t{check_family}\n")
            s = per_set.setdefault(q.split(":", 1)[0], Counter())
            s["sequences"] += 1
            s["with_family"] += bool(h)
            s["minus_strand"] += bool(h) and h["strand"] == "-"
            s["from_rescan"] += bool(h) and h["scan"] == "rescan"
            s["check_agrees"] += family == check_family
    families = {}
    for q, h in best.items():
        families.setdefault(q.split(":", 1)[0], set()).add(h["accession"])
    summary = {name: {**c, "families": len(families.get(name, ()))} for name, c in per_set.items()}
    print(json.dumps(summary, indent=2))
    out = repo_path(paths["results_dir"]) / "15_families.json"
    with open(out, "w") as f:
        json.dump({"queries": len(queries), "rescanned": len(rescan), "per_set": summary}, f, indent=2)
    write_meta(out, config, extra={"infernal": infernal_version()})


if __name__ == "__main__":
    main()
