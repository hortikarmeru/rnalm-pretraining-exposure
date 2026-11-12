"""Memorization gap in masked-letter recovery (scripts 30-31; config checks.masked_recovery).

Per matched pair: loss gap = mean cross-entropy per masked letter on the unseen sequence - on the seen
one (primary_metric), and accuracy gap = share of masked letters recovered on the seen sequence - on
the unseen one; positive values mean the model does better on sequences it was trained on. The
estimate is the mean over families of the family's mean gap (families weigh equally). 95% interval:
bootstrap over families (percentile). p: sign flips of whole families (two-sided, p = (k + 1) / (n + 1)):
with no difference between seen and unseen sequences, swapping the labels in a family only flips the
sign of its pair gaps. Pairs within a family are not independent, so families are the unit throughout.

A gap can also come from seen and unseen sequences differing in other ways (source databases, how
typical they are). Checks, each with the same machinery:
  - minus the DNA model's gap on the same pair (it saw neither side, so it carries only such differences);
  - adjusted for the difference in log(1 + homologs) (mean over the seen corpora; unseen minus seen):
    the intercept of a family-weighted regression of the gap on it, i.e. the gap at equal homolog counts;
  - on pairs whose homolog counts are similar in every seen corpus (balanced_log_homolog_difference;
    pairs with a capped count left out, which removes most large families);
  - by the unseen sequence's best identity to any seen corpus (identity_bins): a near-copy of a seen
    sequence is not new to the model in practice.
And the size contrasts (per pair, gap of the larger model - gap of the smaller).

Usage: python scripts/32_recovery_analysis.py   (environment-extra.yml)

Writes results/32_recovery_analysis.json plus its .meta.json.
"""
import csv
import json
import math

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta

CONTRASTS = [("rinalmo_mega", "rinalmo_micro"), ("rinalmo_giga", "rinalmo_micro"), ("rinalmo_giga", "rinalmo_mega"),
             ("aido_rna_1.6b", "aido_rna_650m")]
CONTROL = "aido_dna_300m"


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def intercept(w, y, x):
    """Weighted mean of y (x None) or intercept of the weighted regression of y on x. w: [..., pairs]
    weights; y and x: [..., pairs] (broadcast)."""
    s, sy = w.sum(-1), (w * y).sum(-1)
    if x is None:
        return sy / s
    sx, sxx, sxy = (w * x).sum(-1), (w * x * x).sum(-1), (w * x * y).sum(-1)
    return (sxx * sy - sx * sxy) / (s * sxx - sx ** 2)


def test(values, family, cfg, seed, covariate=None):
    """values: {pair: gap}. Family-weighted estimate, family bootstrap, family sign flips."""
    pairs = sorted(values)
    fams = sorted({family[p] for p in pairs})
    if len(fams) < 2:
        return {"pairs": len(pairs), "families": len(fams)}
    index = {f: k for k, f in enumerate(fams)}
    fi = np.array([index[family[p]] for p in pairs])
    size = np.bincount(fi, minlength=len(fams))
    y = np.array([values[p] for p in pairs])
    x = None if covariate is None else np.array([covariate[p] for p in pairs])
    w = 1 / size[fi]   # every family sums to 1
    est = float(intercept(w, y, x))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(fams), size=(cfg["bootstrap"], len(fams)))
    times = np.stack([np.bincount(d, minlength=len(fams)) for d in draws])   # how often each family was drawn
    boot = intercept(times[:, fi] * w, y, x)
    flips = rng.choice([-1.0, 1.0], size=(cfg["sign_flip_permutations"], len(fams)))[:, fi]
    null = intercept(w, flips * y, None if x is None else flips * x)   # swapping labels negates gap and covariate
    k = int(np.sum(np.abs(null) >= abs(est) - 1e-12))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi), "pairs": len(pairs), "families": len(fams),
            "p_sign_flip": (k + 1) / (len(null) + 1)}


def main():
    config = load_config()
    paths, cfg = config["paths"], config["checks"]["masked_recovery"]
    results = repo_path(paths["results_dir"])
    seed = config["random_seed"]
    corpora = cfg["seen_corpora"]
    by_pair = {}
    for r in read_tsv(results / "30_recovery_sequences.tsv"):
        by_pair.setdefault(r["pair"], {})[r["side"]] = r
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

    gaps = {}
    for key in cfg["models"]:
        rec = {r["seq_id"]: r for r in read_tsv(results / f"31_recovery_{key}.tsv")}
        acc = {q: int(r["correct"]) / int(r["masked"]) for q, r in rec.items()}
        loss = {q: float(r["loss_sum"]) / int(r["masked"]) for q, r in rec.items()}
        gaps[key] = {"loss": {p: loss[d["unseen"]["seq_id"]] - loss[d["seen"]["seq_id"]] for p, d in by_pair.items()},
                     "accuracy": {p: acc[d["seen"]["seq_id"]] - acc[d["unseen"]["seq_id"]] for p, d in by_pair.items()}}

    def report(v):   # one gap (or contrast) per pair: estimate and its checks
        out = {"all": test(v, family, cfg, seed),
               "adjusted_for_homologs": test(v, family, cfg, seed, homolog_diff),
               "balanced_homologs": test({p: x for p, x in v.items() if p in balanced}, family, cfg, seed)}
        out.update({f"unseen_identity_{b}": test({p: x for p, x in v.items() if p in ps}, family, cfg, seed)
                    for b, ps in bins.items()})
        return out

    out = {"primary_metric": cfg["primary_metric"], "pairs": len(by_pair), "families": len(set(family.values())),
           "balanced_pairs": len(balanced), "identity_bins": {b: len(ps) for b, ps in bins.items()},
           "median_log_homolog_difference_unseen_minus_seen": float(np.median(list(homolog_diff.values()))),
           "models": {}, "minus_dna_control": {}, "size_contrasts": {}}
    for key, g in gaps.items():
        out["models"][key] = {m: report(v) for m, v in g.items()}
        if key != CONTROL:
            out["minus_dna_control"][key] = {m: report({p: v[p] - gaps[CONTROL][m][p] for p in v}) for m, v in g.items()}
    for big, small in CONTRASTS:
        out["size_contrasts"][f"{big}_minus_{small}"] = {
            m: report({p: gaps[big][m][p] - gaps[small][m][p] for p in by_pair}) for m in ("loss", "accuracy")}
    path = results / "32_recovery_analysis.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("pairs", "families", "balanced_pairs", "identity_bins")}))
    for key in cfg["models"]:
        a = out["models"][key][cfg["primary_metric"]]["all"]
        print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in a.items()}, flush=True)
    write_meta(path, config, packages=["numpy"])


if __name__ == "__main__":
    main()
