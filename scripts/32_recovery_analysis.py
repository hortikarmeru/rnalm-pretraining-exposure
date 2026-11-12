"""Memorization gap in masked-letter recovery (scripts 30-31; config checks.masked_recovery).

Per matched pair: gap = recovery accuracy on the seen sequence - on the unseen one (the share of
masked letters predicted correctly, per sequence), and loss gap = mean cross-entropy per masked letter
on the unseen sequence - on the seen one; positive values mean the model does better on sequences it
was trained on. The estimate is the mean over families of the family's mean gap (families weigh
equally). 95% interval: bootstrap over families (percentile). p: sign flips of single pairs (two-sided,
p = (k + 1) / (n + 1)): with no memorization, which member of a matched pair was seen does not matter.

Also, with the same machinery, the size contrasts (per pair, gap of the larger model - gap of the
smaller: RiNALMo mega - micro, giga - micro, AIDO.RNA 1.6B - 650M), the DNA model as a control that
saw neither side, and every estimate again on the pairs whose two sides have similar homolog counts
in every seen corpus (balanced_log_homolog_difference; pairs with a capped count left out).

Usage: python scripts/32_recovery_analysis.py   (environment-extra.yml)

Writes results/32_recovery_analysis.json plus its .meta.json.
"""
import csv
import json
import math

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta

CONTRASTS = [("rinalmo_mega", "rinalmo_micro"), ("rinalmo_giga", "rinalmo_micro"), ("aido_rna_1.6b", "aido_rna_650m")]


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test(values, family, cfg, rng_seed):
    """values: {pair: gap}. Mean over families of family means, family bootstrap, pair sign flips."""
    pairs = sorted(values)
    fams = sorted({family[p] for p in pairs})
    index = {f: k for k, f in enumerate(fams)}
    fam_idx = np.array([index[family[p]] for p in pairs])
    v = np.array([values[p] for p in pairs])
    size = np.bincount(fam_idx, minlength=len(fams))

    weight = np.zeros((len(v), len(fams)))   # pair values @ weight = family means
    weight[np.arange(len(v)), fam_idx] = 1 / size[fam_idx]

    def stat(x):   # mean of family means, for one or many rows of pair values
        return (x @ weight).mean(axis=-1)

    est = float(stat(v))
    rng = np.random.default_rng(rng_seed)
    fam_means = np.bincount(fam_idx, weights=v, minlength=len(fams)) / size
    boot = fam_means[rng.integers(0, len(fams), size=(cfg["bootstrap"], len(fams)))].mean(axis=1)
    flips = rng.choice([-1.0, 1.0], size=(cfg["sign_flip_permutations"], len(v)))
    null = stat(flips * v)
    k = int(np.sum(np.abs(null) >= abs(est) - 1e-12))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi), "pairs": len(pairs), "families": len(fams),
            "p_sign_flip": (k + 1) / (len(null) + 1)}


def main():
    config = load_config()
    paths, cfg = config["paths"], config["checks"]["masked_recovery"]
    results = repo_path(paths["results_dir"])
    seqs = read_tsv(results / "30_recovery_sequences.tsv")
    by_pair = {}
    for r in seqs:
        by_pair.setdefault(r["pair"], {})[r["side"]] = r
    family = {p: d["seen"]["family"] for p, d in by_pair.items()}

    def similar(d):
        return all(not int(d[s][f"capped_{c}"]) for s in d for c in cfg["seen_corpora"]) and all(
            abs(math.log1p(int(d["seen"][f"homologs_{c}"])) - math.log1p(int(d["unseen"][f"homologs_{c}"])))
            <= cfg["balanced_log_homolog_difference"] for c in cfg["seen_corpora"])

    balanced = {p for p, d in by_pair.items() if similar(d)}

    gaps = {}
    for key in cfg["models"]:
        rec = {r["seq_id"]: r for r in read_tsv(results / f"31_recovery_{key}.tsv")}
        acc = {q: int(r["correct"]) / int(r["masked"]) for q, r in rec.items()}
        loss = {q: float(r["loss_sum"]) / int(r["masked"]) for q, r in rec.items()}
        gaps[key] = {"accuracy": {p: acc[d["seen"]["seq_id"]] - acc[d["unseen"]["seq_id"]] for p, d in by_pair.items()},
                     "loss": {p: loss[d["unseen"]["seq_id"]] - loss[d["seen"]["seq_id"]] for p, d in by_pair.items()}}

    seed = config["random_seed"]
    out = {"pairs": len(by_pair), "balanced_pairs": len(balanced), "models": {}, "size_contrasts": {}}
    for key, g in gaps.items():
        out["models"][key] = {metric: {"all": test(v, family, cfg, seed),
                                       "balanced": test({p: x for p, x in v.items() if p in balanced}, family, cfg, seed)}
                              for metric, v in g.items()}
    for big, small in CONTRASTS:
        out["size_contrasts"][f"{big}_minus_{small}"] = {
            metric: {"all": test({p: gaps[big][metric][p] - gaps[small][metric][p] for p in by_pair}, family, cfg, seed),
                     "balanced": test({p: gaps[big][metric][p] - gaps[small][metric][p] for p in balanced}, family, cfg, seed)}
            for metric in ("accuracy", "loss")}
    path = results / "32_recovery_analysis.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    write_meta(path, config, packages=["numpy"])


if __name__ == "__main__":
    main()
