"""Injection sets for the continued-training experiment (config training_experiment): which test
families get extra homologs, which sequences are evaluated, what is injected, and the replay data.

Usage: python scripts/26_injection_sets.py   (after script 13 on the injection corpus and script 21)

Steps:
  1. Candidates are injection-corpus sequences (post_snapshot: added to RNAcentral after every
     model's snapshot, absent from the RiNALMo corpus) that are 80/80 homologs of test sequences
     (script 13's hits). A candidate is kept only if every query that hits it is a test sequence of one
     Rfam family outside exclude_families, so injecting it adds exposure to that family only and never
     to a selection-pool sequence.
  2. Dropped: candidates equal to, contained in, or containing a test or selection sequence, and
     candidates >= near_duplicate identity and mutual coverage to a sequence of the replay corpus
     (otherwise the model has, in effect, already seen them).
  3. Eligible sequences have >= min_candidates surviving candidates and <= max_corpus_homologs corpus
     homologs; eligible families have >= min_eligible_per_family of them. If fewer than
     fallback.min_families qualify, the fallback thresholds are used. Up to max_eval_per_family
     eligible sequences per family are evaluated.
  4. Families are paired within exposure strata (median corpus homologs <= strata_split or above) by
     their baseline F1 (RiNALMo micro, frozen head), and a coin flip puts one of each pair in set A,
     the other in set B; an unpaired family is flipped alone. Each model trains on one set, so every
     family is treated in one run and a control in the other.
  5. Per evaluated sequence, up to max_injected_per_sequence of its candidates are drawn; the union
     per set is that set's injection FASTA. The injected sets are then searched with script 13's exact
     settings, every query against each set, which checks that only the set's own families gain
     homologs and gives each sequence's added exposure.
  6. Replay data, sampled as RiNALMo samples its corpus (a uniform cluster of the script 21 linclust
     clustering, then a uniform member), per seed until replay.tokens_per_seed tokens; held_out
     clusters are kept out of replay and give the held-out MLM-loss sequences. Sentinels: the
     selection pool plus n_sentinel_tests test sequences from families outside the experiment.

Writes <work_dir>/ (candidates, injected_<A|B>.fasta, replay_seed<n>.fasta, held_out.fasta) and
results/26_assignment.tsv (family, stratum, pair, set, baseline F1), results/26_eval_sequences.tsv
(seq_id, family, set, corpus homologs, injected homologs, added log exposure),
results/26_sentinels.tsv and results/26_injection_sets.json with its .meta.json.
"""
import csv
import json
import math
import subprocess
from collections import defaultdict

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.mmseqs import STRAND, mmseqs, search
from rnalm_exposure.queries import query_sequences, write_fasta


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = []
        else:
            seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def fetch(ids, db, work, name, threads):
    """Sequences of the given accessions from an MMseqs2 database (by its lookup file)."""
    ids_path = work / f"{name}.ids"
    ids_path.write_text("".join(f"{i}\n" for i in ids))
    sub = work / f"{name}_db"
    mmseqs("createsubdb", ids_path, db, sub, "--id-mode", 1)   # also subsets the header database
    mmseqs("convert2fasta", sub, work / f"{name}.fasta")
    subprocess.run(f"rm -f {sub} {sub}.* {sub}_h {sub}_h.*", shell=True, check=True)
    return read_fasta(work / f"{name}.fasta")


def owned_candidates(hits_path, family_of, excluded):
    """{candidate: family} for candidates every hitting query of which is a test sequence of one
    allowed family; and {query: [candidates]}."""
    owner = {}
    for line in open(hits_path):
        q, t = line.split("\t", 2)[:2]
        f = family_of.get(q) or "BAD"   # selection-pool queries and test sequences without a family
        if f in excluded:
            f = "BAD"
        prev = owner.get(t)
        owner[t] = f if prev in (None, f) else "BAD"
    owner = {t: f for t, f in owner.items() if f != "BAD"}
    by_query = defaultdict(list)
    for line in open(hits_path):
        q, t = line.split("\t", 2)[:2]
        if t in owner:
            by_query[q].append(t)
    return owner, by_query


def near_duplicates(cand_fasta, replay_db, work, cfg, threads):
    qdb, res, tmp = work / "cand_db", work / "near_dup", work / "tmp_near_dup"
    mmseqs("createdb", cand_fasta, qdb, "--dbtype", 2)
    mmseqs("search", qdb, replay_db, res, tmp, "--search-type", 3, "--min-seq-id", cfg["min_seq_id"],
           "-c", cfg["coverage"], "--cov-mode", cfg["coverage_mode"], "--strand", STRAND[cfg["strand"]],
           "-e", 10, "--max-seqs", 100, "--threads", threads)
    mmseqs("convertalis", qdb, replay_db, res, work / "near_dup.tsv", "--format-output", "query,target,fident",
           "--threads", threads)
    subprocess.run(["rm", "-rf", str(tmp)], check=True)
    return {line.split("\t", 1)[0] for line in open(work / "near_dup.tsv")}


def assign(families, base_f1, median_homologs, split, rng):
    """[(family, stratum, pair, set)]: neighbours by baseline F1 within a stratum form a pair, and a
    coin flip decides which of the two goes to set A."""
    rows = []
    for stratum in ("low", "mid"):
        fams = sorted((f for f in families if (median_homologs[f] <= split) == (stratum == "low")),
                      key=lambda f: (base_f1[f], f))
        for k in range(0, len(fams), 2):
            pair = fams[k:k + 2]
            flip = rng.random() < 0.5
            sets = ["A", "B"] if flip else ["B", "A"]
            for f, s in zip(pair, sets):
                rows.append({"family": f, "stratum": stratum, "pair": f"{stratum}{k // 2}", "set": s})
    return rows


def sample_replay(clusters_tsv, n_held_out, tokens_per_seed, seeds, rng_seed):
    """Member accessions per seed (RiNALMo's sampling: uniform cluster, then uniform member) and the
    held-out clusters' representatives. Lengths are only known after fetching, so more members are
    drawn than the budget needs (at >= 150 tokens each) and the caller stops at the budget."""
    import pandas as pd
    t = pd.read_csv(clusters_tsv, sep="\t", header=None, names=["rep", "member"], dtype=str)
    rep_codes, reps = pd.factorize(t["rep"], sort=False)
    order = np.argsort(rep_codes, kind="stable")
    members = t["member"].to_numpy()[order]
    starts = np.searchsorted(rep_codes[order], np.arange(len(reps)))
    sizes = np.diff(np.append(starts, len(members)))
    rng = np.random.default_rng(rng_seed)
    held = rng.choice(len(reps), n_held_out, replace=False)
    keep = np.ones(len(reps), bool)
    keep[held] = False
    allowed = np.flatnonzero(keep)
    draws = {}
    for s in seeds:
        r = np.random.default_rng([rng_seed, s])
        c = r.choice(allowed, size=int(tokens_per_seed / 150), replace=True)   # over-drawn; trimmed by length later
        draws[s] = members[starts[c] + (r.random(len(c)) * sizes[c]).astype(np.int64)]
    return draws, [str(reps[h]) for h in held]


def main():
    config = load_config()
    paths, cfg, hs = config["paths"], config["training_experiment"], config["homolog_search"]
    results = repo_path(paths["results_dir"])
    exposure_dir = repo_path(paths["exposure_dir"].format(processed_dir=paths["processed_dir"]))
    work = repo_path(cfg["work_dir"].format(processed_dir=paths["processed_dir"]))
    work.mkdir(parents=True, exist_ok=True)
    threads = hs["threads"]
    seed = cfg["seed"]

    queries = query_sequences(config)
    fam_rows = {r["seq_id"]: r for r in read_tsv(results / "15_families.tsv")}
    test_ids = [q for q in queries if q in fam_rows]
    family_of = {q: fam_rows[q]["family"] for q in test_ids if fam_rows[q]["family"]}
    names = {r["family"]: r["family_name"] for r in fam_rows.values() if r["family"]}
    corpus_hom = {r["seq_id"]: int(r["homologs"]) for r in read_tsv(results / f"13_exposure_{cfg['replay_corpus']}.tsv")}
    base = {r["seq_id"]: float(r["f1"]) for r in read_tsv(results / f"18_test_scores_{cfg['baseline_model']}.tsv")
            if r["readout"] == "primary"}

    # 1. candidates owned by one family
    hits = exposure_dir / cfg["injection_corpus"] / "hits.tsv"
    owner, by_query = owned_candidates(hits, family_of, set(cfg["exclude_families"]))
    low = {q for q in test_ids if q in family_of and corpus_hom[q] <= cfg["max_corpus_homologs"]}
    wanted = sorted({t for q in low for t in by_query.get(q, [])})
    print(f"owned candidates {len(owner)}, hitting low-exposure test sequences {len(wanted)}", flush=True)
    cand = fetch(wanted, exposure_dir / cfg["injection_corpus"] / "corpus", work, "candidates", threads)

    # 2. no copies of test or selection sequences, no near-duplicates of the replay corpus
    joined = "|".join(queries.values())
    hitters = defaultdict(list)
    for q in low:
        for t in by_query.get(q, []):
            hitters[t].append(q)
    copies = {t for t, s in cand.items() if s in joined or any(queries[q] in s for q in hitters[t])}
    near = near_duplicates(work / "candidates.fasta", exposure_dir / cfg["replay_corpus"] / "corpus", work,
                           cfg["near_duplicate"], threads)
    survive = set(cand) - copies - near
    print(f"candidates {len(cand)}: copies {len(copies)}, near-duplicates {len(near)}, kept {len(survive)}", flush=True)

    # 3. eligibility, with the preset fallback
    def eligible(min_cand, min_seq):
        seqs = defaultdict(list)
        for q in low:
            if sum(t in survive for t in by_query.get(q, [])) >= min_cand:
                seqs[family_of[q]].append(q)
        return {f: sorted(s) for f, s in seqs.items() if len(s) >= min_seq}
    used = "main"
    fams = eligible(cfg["min_candidates"], cfg["min_eligible_per_family"])
    if len(fams) < cfg["fallback"]["min_families"]:
        used = "fallback"
        fams = eligible(cfg["fallback"]["min_candidates"], cfg["fallback"]["min_eligible_per_family"])
    rng = np.random.default_rng(seed)
    eval_ids = {f: sorted(rng.choice(s, min(len(s), cfg["max_eval_per_family"]), replace=False).tolist())
                for f, s in sorted(fams.items())}

    # 4. paired randomization
    base_f1 = {f: float(np.mean([base[q] for q in s])) for f, s in eval_ids.items()}
    med_hom = {f: float(np.median([corpus_hom[q] for q in s])) for f, s in eval_ids.items()}
    assignment = assign(eval_ids, base_f1, med_hom, cfg["strata_split"], rng)
    set_of = {a["family"]: a["set"] for a in assignment}

    # 5. injection sets, then the check search with script 13's settings
    injected = {"A": set(), "B": set()}
    for f, s in eval_ids.items():
        for q in s:
            pool = sorted(t for t in by_query[q] if t in survive)
            k = min(len(pool), cfg["max_injected_per_sequence"])
            injected[set_of[f]].update(rng.choice(pool, k, replace=False).tolist())
    check = {}
    for s in ("A", "B"):
        write_fasta({t: cand[t] for t in sorted(injected[s])}, work / f"injected_{s}.fasta")
        sw = work / f"check_{s}"
        sw.mkdir(exist_ok=True)
        mmseqs("createdb", work / f"injected_{s}.fasta", sw / "target", "--dbtype", 2)
        res, _ = search(exposure_dir / cfg["replay_corpus"] / "queries", sw / "target", sw, hs, hs["max_target_seqs"])
        mmseqs("convertalis", exposure_dir / cfg["replay_corpus"] / "queries", sw / "target", res, sw / "hits.tsv",
               "--format-output", "query,target,fident,qcov", "--threads", threads)
        counts = defaultdict(set)
        for line in open(sw / "hits.tsv"):
            q, t = line.split("\t", 2)[:2]
            counts[q].add(t)
        bad = [q for q in counts if family_of.get(q) is None or set_of.get(family_of[q]) != s]
        if bad:
            raise RuntimeError(f"set {s}: {len(bad)} queries outside its families gain homologs, e.g. {bad[:5]}")
        check[s] = {q: len(v) for q, v in counts.items()}

    with open(results / "26_assignment.tsv", "w") as f:
        f.write("family\tfamily_name\tstratum\tpair\tset\tn_eval\tmedian_corpus_homologs\tbaseline_f1\n")
        for a in assignment:
            fam = a["family"]
            f.write(f"{fam}\t{names[fam]}\t{a['stratum']}\t{a['pair']}\t{a['set']}\t{len(eval_ids[fam])}\t"
                    f"{med_hom[fam]:g}\t{base_f1[fam]:.6f}\n")
    with open(results / "26_eval_sequences.tsv", "w") as f:
        f.write("seq_id\tfamily\tset\tcorpus_homologs\tinjected_homologs\tdelta_log_exposure\n")
        for fam, s in eval_ids.items():
            for q in s:
                h, n = corpus_hom[q], check[set_of[fam]].get(q, 0)
                f.write(f"{q}\t{fam}\t{set_of[fam]}\t{h}\t{n}\t{math.log1p(h + n) - math.log1p(h):.6f}\n")

    # 6. sentinels and replay
    experiment = set(eval_ids)
    outside = sorted(q for q in test_ids if family_of.get(q) not in experiment)
    sentinel_tests = sorted(rng.choice(outside, cfg["n_sentinel_tests"], replace=False).tolist())
    with open(results / "26_sentinels.tsv", "w") as f:
        f.write("seq_id\tfamily\n")
        for q in sentinel_tests:
            f.write(f"{q}\t{family_of.get(q, '')}\n")

    rp = cfg["replay"]
    draws, held = sample_replay(exposure_dir / cfg["replay_corpus"] / "clusters.tsv", rp["held_out"],
                                rp["tokens_per_seed"], rp["seeds"], seed)
    replay_db = exposure_dir / cfg["replay_corpus"] / "corpus"
    fetch(held, replay_db, work, "held_out", threads)
    replay_summary = {}
    for s, ids in draws.items():
        # fetch the distinct members, then write them in draw order (repeats kept) up to the token budget
        seqs = fetch(sorted(set(ids)), replay_db, work, f"replay_pool_{s}", threads)
        tokens, n = 0, 0
        with open(work / f"replay_seed{s}.fasta", "w") as f:
            for i in ids:
                if tokens >= rp["tokens_per_seed"]:
                    break
                f.write(f">{i}\n{seqs[i]}\n")
                tokens += min(len(seqs[i]), rp["crop"]) + 2
                n += 1
        if tokens < rp["tokens_per_seed"]:
            raise RuntimeError(f"replay seed {s}: only {tokens} tokens drawn")
        (work / f"replay_pool_{s}.fasta").unlink()
        replay_summary[s] = {"sequences": n, "tokens": tokens}

    summary = {
        "thresholds_used": used, "owned_candidates": len(owner), "candidates_low_exposure": len(cand),
        "dropped_copies": len(copies), "dropped_near_duplicates": len(near), "surviving": len(survive),
        "families": len(eval_ids), "eval_sequences": sum(len(s) for s in eval_ids.values()),
        "injected": {s: {"sequences": len(v), "nt": sum(len(cand[t]) for t in v)} for s, v in injected.items()},
        "sentinel_tests": len(sentinel_tests), "held_out": len(held), "replay": replay_summary,
    }
    out = results / "26_injection_sets.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    write_meta(out, config, packages=["numpy", "pandas"], extra={
        "mmseqs": subprocess.run(["mmseqs", "version"], capture_output=True, text=True).stdout.strip()})


if __name__ == "__main__":
    main()
