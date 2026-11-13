"""Memorization gap in verbatim span recovery (scripts 30, 34, 35; config checks.verbatim_recovery).

Per sequence and span length: the share of its spans recovered exactly (every letter right). Per
matched pair and span length: gap = share on the seen sequence - share on the unseen one; positive
values mean the model writes back more of the sequences it was trained on. A pair counts at a span
length only when both of its sequences have spans there.

Headline (primary_metric, pooled): the pair's gap averaged over the span lengths, each span length
weighing the same. One number per model, so no span length has to be singled out, and a gap at any
length shows up in it. Every span length is also tested on its own, with Holm-adjusted p values
across the span lengths. Estimate, 95% interval and p come from script 32's test: mean over families
of the family's mean gap, bootstrap over families, sign flips of whole families.

Also reported, pooled and per span length: the same gap for the first pass alone (exact_one_pass),
and for the share of span letters right (correct, correct_one_pass). Per side and span length: the
recovery rates themselves; a model with near-zero rates on both sides cannot show a gap.

Checks on the primary metric, as in scripts 32 and 33:
  - adjusted for the difference in log(1 + homologs), on pairs with balanced homolog counts, and by
    the unseen sequence's best identity to any seen corpus (settings of checks.masked_recovery);
  - minus the DNA model's gap on the same pair (it saw neither side);
  - the size contrasts (per pair, gap of the larger model - gap of the smaller);
  - without spans whose letters also occur elsewhere in the same sequence (script 34's self_repeat);
  - by the seen sequence's cluster size in the RiNALMo corpus (script 33's bins).

Per-position scores (script 35): per pair, seen - unseen of the mean log-probability over all
positions and of the mean over the lowest min_k_fraction (Min-K% Prob), with the same checks, and the
AUROC of each score for telling seen from unseen sequences.

Usage: python scripts/36_verbatim_analysis.py   (after scripts 21 and 35; environment-extra.yml)

Writes results/36_verbatim_analysis.json plus its .meta.json.
"""
import importlib.util
import json
import math

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta

SECONDARY = ["exact", "exact_one_pass", "correct", "correct_one_pass"]
SCORES = ["mean_logp", "min_k_logp"]


def load_script(number_and_name):
    spec = importlib.util.spec_from_file_location(number_and_name, repo_path(f"scripts/{number_and_name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_sets(by_pair, cfg):
    """Per pair: family, homolog difference (unseen minus seen), and the pair sets of the checks, as
    script 32 defines them (cfg: checks.masked_recovery)."""
    corpora = cfg["seen_corpora"]
    family = {p: d["seen"]["family"] for p, d in by_pair.items()}

    def log_h(r, c):
        return math.log1p(int(r[f"homologs_{c}"]))

    homolog_diff = {p: float(np.mean([log_h(d["unseen"], c) - log_h(d["seen"], c) for c in corpora])) for p, d in by_pair.items()}
    balanced = {p for p, d in by_pair.items()
                if not any(int(d[s][f"capped_{c}"]) for s in d for c in corpora)
                and all(abs(log_h(d["unseen"], c) - log_h(d["seen"], c)) <= cfg["balanced_log_homolog_difference"] for c in corpora)}
    identity = {p: max(float(d["unseen"][f"max_identity_{c}"]) for c in corpora) for p, d in by_pair.items()}
    lo_bin, hi_bin = cfg["identity_bins"]
    bins = {f"below_{lo_bin}": {p for p, v in identity.items() if v < lo_bin},
            f"{lo_bin}_to_{hi_bin}": {p for p, v in identity.items() if lo_bin <= v < hi_bin},
            f"{hi_bin}_and_above": {p for p, v in identity.items() if v >= hi_bin}}
    return family, homolog_diff, balanced, bins


def shares(rows, metric, skip=frozenset()):
    """{span length: {seq_id: share}}: spans recovered (exact metrics) or span letters right (correct
    metrics), over the sequence's spans of that length; spans in skip are left out."""
    total = {}
    for r in rows:
        k = int(r["span_length"])
        if (r["seq_id"], k, r["span"]) in skip:
            continue
        t = total.setdefault(k, {}).setdefault(r["seq_id"], [0, 0])
        t[0] += int(r[metric])
        t[1] += k if metric.startswith("correct") else 1
    return {k: {q: a / b for q, (a, b) in d.items()} for k, d in total.items()}


def pair_gaps(share, by_pair):
    """{span length: {pair: seen - unseen}} for pairs with both sides, and under "pooled" the pair's
    mean over its span lengths."""
    out = {k: {p: d[s["seen"]["seq_id"]] - d[s["unseen"]["seq_id"]] for p, s in by_pair.items()
               if s["seen"]["seq_id"] in d and s["unseen"]["seq_id"] in d} for k, d in sorted(share.items())}
    pooled = {}
    for g in out.values():
        for p, v in g.items():
            pooled.setdefault(p, []).append(v)
    return {"pooled": {p: float(np.mean(v)) for p, v in pooled.items()}, **{f"span_{k}": g for k, g in out.items()}}


def holm(p):
    """Holm-adjusted p values, {name: p} -> {name: adjusted p}."""
    names = sorted(p, key=p.get)
    out, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - i) * p[name]))
        out[name] = running
    return out


def auroc(pos, neg):
    """Probability that a random pos value is above a random neg value (ties count half)."""
    values = np.concatenate([pos, neg])
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():   # average ranks of tied values
        sums = np.bincount(inverse, weights=ranks)
        ranks = (sums / counts)[inverse]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    config = load_config()
    paths, cfg, mr = config["paths"], config["checks"]["verbatim_recovery"], config["checks"]["masked_recovery"]
    results = repo_path(paths["results_dir"])
    seed, primary = config["random_seed"], cfg["primary_metric"]
    if cfg["pooling"] != "span_lengths_equal":   # the only pooling pair_gaps implements
        raise ValueError(f"unknown checks.verbatim_recovery.pooling {cfg['pooling']}")
    s32, s33 = load_script("32_recovery_analysis"), load_script("33_recovery_by_cluster_size")
    by_pair = {}
    for r in s32.read_tsv(results / "30_recovery_sequences.tsv"):
        by_pair.setdefault(r["pair"], {})[r["side"]] = r
    family, homolog_diff, balanced, bins = pair_sets(by_pair, mr)
    side = {d[s]["seq_id"]: s for d in by_pair.values() for s in d}
    repeats = {(r["seq_id"], int(r["span_length"]), r["span"])
               for r in s32.read_tsv(results / "34_verbatim_spans.tsv") if int(r["self_repeat"])}

    clusters = repo_path(paths["exposure_dir"].format(processed_dir=paths["processed_dir"])) / "rinalmo" / "clusters.tsv"
    seen = {d["seen"]["seq_id"]: p for p, d in by_pair.items()}
    size = {seen[q]: n for q, n in s33.cluster_sizes(clusters, set(seen)).items()}
    edges = mr["cluster_size_bins"]
    cluster_bins = {(f"{lo}_to_{hi}" if hi else f"{lo}_and_above"): {p for p, n in size.items() if n >= lo and (hi is None or n <= hi)}
                    for lo, hi in zip([1] + edges, [e - 1 for e in edges] + [None])}

    def test(v):
        return s32.test(v, family, mr, seed)

    def checks(v):   # as script 32's report
        out = {"all": test(v),
               "adjusted_for_homologs": s32.test(v, family, mr, seed, homolog_diff),
               "balanced_homologs": test({p: x for p, x in v.items() if p in balanced})}
        out.update({f"unseen_identity_{b}": test({p: x for p, x in v.items() if p in ps}) for b, ps in bins.items()})
        return out

    def minus(a, b):   # per pair, where both have a value
        return {p: a[p] - b[p] for p in a if p in b}

    out = {"primary_metric": primary, "pairs": len(by_pair), "families": len(set(family.values())),
           "balanced_pairs": len(balanced), "identity_bins": {b: len(ps) for b, ps in bins.items()},
           "pairs_with_cluster": len(size), "cluster_size_bins": {b: len(ps) for b, ps in cluster_bins.items()},
           "self_repeat_spans": len(repeats), "models": {}, "minus_dna_control": {}, "size_contrasts": {}}
    gaps = {}
    for key in mr["models"]:
        rows = s32.read_tsv(results / f"35_verbatim_{key}.tsv")
        gaps[key] = g = {m: pair_gaps(shares(rows, m), by_pair) for m in SECONDARY}
        rates = {}
        for r in rows:
            t = rates.setdefault(side[r["seq_id"]], {}).setdefault(f"span_{r['span_length']}", {"spans": 0, **dict.fromkeys(SECONDARY, 0)})
            t["spans"] += 1
            for m in SECONDARY:
                t[m] += int(r[m])
        for by_k in rates.values():
            for name, t in by_k.items():
                k = int(name.split("_")[1])
                t.update({f"{m}_rate": t[m] / (t["spans"] * (k if m.startswith("correct") else 1)) for m in SECONDARY})
        res = {"rates": rates, primary: {name: checks(v) for name, v in g[primary].items()}}
        res[f"{primary}_holm_p"] = holm({name: t["all"]["p_sign_flip"] for name, t in res[primary].items()
                                         if name != "pooled" and "p_sign_flip" in t["all"]})
        for m in SECONDARY:
            if m != primary:
                res[m] = {name: test(v) for name, v in g[m].items()}
        res[f"{primary}_without_self_repeats"] = {name: test(v) for name, v in pair_gaps(shares(rows, primary, repeats), by_pair).items()}
        res[f"{primary}_by_cluster_size"] = {b: test({p: v for p, v in g[primary]["pooled"].items() if p in ps}) for b, ps in cluster_bins.items()}

        score = {r["seq_id"]: r for r in s32.read_tsv(results / f"35_verbatim_scores_{key}.tsv")}
        for m in SCORES:
            v = {p: float(score[d["seen"]["seq_id"]][m]) - float(score[d["unseen"]["seq_id"]][m]) for p, d in by_pair.items()}
            g[m] = {"all_positions": v}
            res[m] = checks(v)
            res[m]["auroc"] = auroc(np.array([float(score[d["seen"]["seq_id"]][m]) for d in by_pair.values()]),
                                    np.array([float(score[d["unseen"]["seq_id"]][m]) for d in by_pair.values()]))
        out["models"][key] = res
        print(key, "pooled", {k: round(v, 4) if isinstance(v, float) else v for k, v in res[primary]["pooled"]["all"].items()}, flush=True)

    for key, g in gaps.items():
        if key != s32.CONTROL:
            out["minus_dna_control"][key] = {m: {name: checks(minus(v, gaps[s32.CONTROL][m][name])) for name, v in g[m].items()}
                                             for m in [primary] + SCORES}
    for big, small in s32.CONTRASTS:
        out["size_contrasts"][f"{big}_minus_{small}"] = {
            m: {name: checks(minus(v, gaps[small][m][name])) for name, v in gaps[big][m].items()} for m in [primary] + SCORES}
    path = results / "36_verbatim_analysis.json"
    path.write_text(json.dumps(out, indent=2))
    write_meta(path, config, packages=["numpy"])


if __name__ == "__main__":
    main()
