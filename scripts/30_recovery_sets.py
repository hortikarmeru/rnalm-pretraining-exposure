"""Matched seen and unseen sequences for the masked-letter recovery check (config checks.masked_recovery).

Seen: random sequences of the first seen corpus (RNA-FM's) whose exact sequence is also in every other
seen corpus (RiNALMo's, AIDO.RNA's). Unseen: random sequences of the post-snapshot corpus (already
free of exact matches to every model's data, script 20). Both sides get their Rfam family from one
RNAcentral annotation release (the lowest E-value hit). Within a family, each unseen sequence is
paired with the closest unused seen sequence within the length and GC tolerances (closest in
|log length ratio| / tolerance + |GC difference| / tolerance). Families are taken in random order.

The paired sequences are then searched against every seen corpus with 80/80 settings (homolog_search
otherwise): an unseen sequence covered end to end at 100% identity by a corpus sequence is contained
in it, so its pair is dropped. The same hits give each sequence's homolog count per corpus (a seen
sequence's own copy is not counted), to compare the two sides.

Usage: python scripts/30_recovery_sets.py   (environment.yml: MMseqs2; the corpora's search databases from script 13)

Writes results/30_recovery_sequences.tsv (one row per sequence: pair, family, side, length, GC,
homologs and best identity per corpus, sequence) and results/30_recovery_sets.json (counts per step,
the two sides compared) plus its .meta.json.
"""
import csv
import gzip
import hashlib
import json
import math
import subprocess
from collections import defaultdict

import numpy as np

from rnalm_exposure.config import display_path, load_config, repo_path
from rnalm_exposure.fasta import iter_gz_fasta
from rnalm_exposure.meta import write_meta
from rnalm_exposure.mmseqs import mmseqs, search
from rnalm_exposure.queries import write_fasta


def corpus_fasta(paths, name):
    return repo_path(paths["corpus_fasta"].format(processed_dir=paths["processed_dir"], name=name))


def sample(path, k, lo, hi, alphabet, rng):
    """Uniform sample (reservoir) of k {id: sequence} among sequences of lo..hi nt using only alphabet."""
    ids, seqs, n = [], [], 0
    for header, seq in iter_gz_fasta(path):
        if not lo <= len(seq) <= hi or seq.translate(None, alphabet):
            continue
        n += 1
        if len(ids) < k:
            ids.append(header.split()[0].decode())
            seqs.append(seq.decode())
        else:
            j = rng.integers(n)
            if j < k:
                ids[j], seqs[j] = header.split()[0].decode(), seq.decode()
    return dict(zip(ids, seqs)), n


def in_corpus(seqs, path):
    """Ids of seqs whose exact sequence occurs in the corpus at path."""
    digest = {hashlib.md5(s.encode()).digest(): q for q, s in seqs.items()}
    found = set()
    for _, seq in iter_gz_fasta(path):
        q = digest.get(hashlib.md5(seq).digest())
        if q is not None:
            found.add(q)
    return found


def families(path, ids):
    """{id: Rfam accession} by lowest E-value annotation (first in file order on ties)."""
    best = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = line.split("\t")
            if r[0] in ids:
                e = float(r[3])
                if r[0] not in best or e < best[r[0]][1]:
                    best[r[0]] = (r[1], e)
    return {q: fam for q, (fam, _) in best.items()}


def gc(s):
    return (s.count("G") + s.count("C")) / len(s)


def match(fam_u, fam_s, seqs, cfg, rng):
    """{family: [(unseen, seen)]} in random family order, up to max_pairs_per_family * oversample each,
    until max_pairs * oversample pairs."""
    want, cap = cfg["max_pairs"] * cfg["oversample"], cfg["max_pairs_per_family"] * cfg["oversample"]
    shared = sorted(set(fam_u) & set(fam_s))
    out, total = {}, 0
    for f in rng.permutation(shared):
        us = list(rng.permutation(sorted(fam_u[f])))
        ss = sorted(fam_s[f])
        s_len = np.log([len(seqs[s]) for s in ss])
        s_gc = np.array([gc(seqs[s]) for s in ss])
        free = np.ones(len(ss), bool)
        pairs = []
        for u in us:
            dl = np.abs(s_len - math.log(len(seqs[u])))
            dg = np.abs(s_gc - gc(seqs[u]))
            ok = free & (dl <= math.log1p(cfg["length_tolerance"])) & (dg <= cfg["gc_tolerance"])
            if ok.any():
                score = np.where(ok, dl / math.log1p(cfg["length_tolerance"]) + dg / cfg["gc_tolerance"], np.inf)
                j = int(np.argmin(score))
                free[j] = False
                pairs.append((u, ss[j]))
                if len(pairs) == cap:
                    break
        if pairs:
            out[str(f)] = pairs
            total += min(len(pairs), cfg["max_pairs_per_family"])
            if total >= want:
                break
    return out


def hits(seqs, corpus, work, config, cfg):
    """{id: (homologs, best identity, contained)} for seqs against one corpus's search database."""
    paths, hs = config["paths"], config["homolog_search"]
    hs = {**hs, **{k: cfg["search"][k] for k in ("min_identity", "min_coverage")}}
    cw = work / corpus
    cw.mkdir(parents=True, exist_ok=True)
    qdb = work / "queries"
    tdb = repo_path(paths["exposure_dir"].format(processed_dir=paths["processed_dir"])) / corpus / "corpus"
    res, pref = search(qdb, tdb, cw, hs, cfg["search"]["max_seqs"])
    tsv = cw / "hits.tsv"
    mmseqs("convertalis", qdb, tdb, res, tsv, "--format-output", "query,target,fident,qcov,tcov", "--threads", hs["threads"])
    out = {q: [0, 0.0, False] for q in seqs}
    for line in open(tsv):
        q, _, fid, qcov, tcov = line.rstrip("\n").split("\t")
        fid, qcov, tcov = float(fid), float(qcov), float(tcov)
        if fid >= 1.0 and qcov >= 1.0:
            out[q][2] = True
            if tcov >= 1.0:   # the sequence's own copy
                continue
        out[q][0] += 1
        out[q][1] = max(out[q][1], fid)
    capped = {q for q, n in pref.items() if n >= cfg["search"]["max_seqs"]}
    return out, capped


def main():
    config = load_config()
    paths, cfg = config["paths"], config["checks"]["masked_recovery"]
    rng = np.random.default_rng(config["random_seed"])
    work = repo_path(cfg["work_dir"].format(processed_dir=paths["processed_dir"]))
    work.mkdir(parents=True, exist_ok=True)
    lo, hi = cfg["length"]
    alphabet = cfg["alphabet"].encode()
    first, *others = cfg["seen_corpora"]
    counts = {}

    seen, counts["seen_eligible"] = sample(corpus_fasta(paths, first), cfg["seen_sample"], lo, hi, alphabet, rng)
    for c in others:
        keep = in_corpus(seen, corpus_fasta(paths, c))
        seen = {q: s for q, s in seen.items() if q in keep}
        counts[f"seen_sampled_also_in_{c}"] = len(seen)
    unseen, counts["unseen_eligible"] = sample(corpus_fasta(paths, cfg["unseen_corpus"]), cfg["unseen_sample"], lo, hi, alphabet, rng)
    counts["unseen_sampled"] = len(unseen)
    print(json.dumps(counts), flush=True)

    fam = families(repo_path(paths["raw_dir"]) / cfg["family_annotations"], set(seen) | set(unseen))
    fam_s, fam_u = defaultdict(list), defaultdict(list)
    for q in seen:
        if q in fam:
            fam_s[fam[q]].append(q)
    for q in unseen:
        if q in fam:
            fam_u[fam[q]].append(q)
    counts.update({"seen_with_family": sum(map(len, fam_s.values())), "unseen_with_family": sum(map(len, fam_u.values())),
                   "families_both_sides": len(set(fam_s) & set(fam_u))})
    seqs = {**seen, **unseen}
    side = {**{q: "seen" for q in seen}, **{q: "unseen" for q in unseen}}
    matched = match(fam_u, fam_s, seqs, cfg, rng)
    counts["prematched_pairs"] = sum(map(len, matched.values()))
    counts["prematched_families"] = len(matched)
    print(json.dumps(counts), flush=True)

    query = {q: seqs[q] for pairs in matched.values() for p in pairs for q in p}
    write_fasta(query, work / "queries.fasta")
    mmseqs("createdb", work / "queries.fasta", work / "queries", "--dbtype", 2)
    stats, capped = {}, {}
    for c in cfg["seen_corpora"]:
        stats[c], capped[c] = hits(query, c, work, config, cfg)
        print(c, "searched", flush=True)

    rows, n_contained = [], 0
    for f, pairs in matched.items():   # random family order from match()
        kept = []
        for u, s in pairs:
            if any(stats[c][u][2] for c in cfg["seen_corpora"]):
                n_contained += 1
                continue
            if not all(stats[c][s][2] for c in cfg["seen_corpora"]):   # cannot happen unless the search missed the copy
                raise RuntimeError(f"seen sequence {s} not found in every seen corpus by the search")
            kept.append((u, s))
        for u, s in kept[:cfg["max_pairs_per_family"]]:
            if len(rows) // 2 >= cfg["max_pairs"]:
                break
            pair = f"p{len(rows) // 2}"
            for q in (s, u):
                r = {"seq_id": q, "pair": pair, "family": f, "side": side[q], "length": len(seqs[q]),
                     "gc": round(gc(seqs[q]), 4)}
                for c in cfg["seen_corpora"]:
                    n, ident, _ = stats[c][q]
                    r[f"homologs_{c}"], r[f"capped_{c}"], r[f"max_identity_{c}"] = n, int(q in capped[c]), ident
                r["sequence"] = seqs[q]
                rows.append(r)
    counts["unseen_contained_dropped"] = n_contained
    counts["pairs"] = len(rows) // 2
    counts["families"] = len({r["family"] for r in rows})

    results = repo_path(paths["results_dir"])
    out = results / "30_recovery_sequences.tsv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    def side_summary(name):
        rs = [r for r in rows if r["side"] == name]
        d = {"median_length": float(np.median([r["length"] for r in rs])), "median_gc": float(np.median([r["gc"] for r in rs]))}
        for c in cfg["seen_corpora"]:
            d[f"median_log_homologs_{c}"] = float(np.median([math.log1p(r[f"homologs_{c}"]) for r in rs]))
            d[f"capped_{c}"] = sum(r[f"capped_{c}"] for r in rs)
            d[f"median_max_identity_{c}"] = float(np.median([r[f"max_identity_{c}"] for r in rs]))
        return d

    summary = {"counts": counts, "seen": side_summary("seen"), "unseen": side_summary("unseen"),
               "sequences": display_path(out)}
    print(json.dumps(summary, indent=2))
    js = results / "30_recovery_sets.json"
    js.write_text(json.dumps(summary, indent=2))
    version = subprocess.run(["mmseqs", "version"], capture_output=True, text=True).stdout.strip()
    write_meta(js, config, packages=["numpy"], extra={"mmseqs": version})


if __name__ == "__main__":
    main()
