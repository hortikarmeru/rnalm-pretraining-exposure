"""Pair-balanced estimates for the training experiment (scripts 29, 39, 40): families were randomized
by clan within matched pairs (script 26), so uncertainty comes from the pairs."""
import csv

import numpy as np


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def scores(results, key, tag, role, column="f1", kind="scores", readout="primary"):
    rows = read_tsv(results / f"28_{kind}_{key}_{tag}.tsv")
    if kind == "scores":
        return {r["seq_id"]: float(r[column]) for r in rows if r["readout"] == readout and r["role"].startswith(role)}
    return {r["seq_id"]: float(r["loss"]) for r in rows if r["role"] == role}


def family_differences(eval_rows, own_other):
    """{family: D_f} from {seq_id: [own - other per seed]}."""
    fams = {}
    for r in eval_rows:
        fams.setdefault(r["family"], []).append(np.mean(own_other[r["seq_id"]]))
    return {f: float(np.mean(v)) for f, v in fams.items()}


def pair_test(d, pair_of, set_of_family, cfg):
    """Pair-balanced mean of D_f (A side and B side weighted equally within each pair), bootstrap over
    pairs, sign-flip permutation over pairs. Pairs with families on one side only are left out."""
    sides = {}
    for f, v in d.items():
        sides.setdefault(pair_of[f], {}).setdefault(set_of_family[f], []).append(v)
    pairs = sorted(p for p, s in sides.items() if len(s) == 2)
    one_sided = {f: v for f, v in d.items() if pair_of[f] not in pairs}
    vals = np.array([(np.mean(sides[p]["A"]) + np.mean(sides[p]["B"])) / 2 for p in pairs])
    est = float(vals.mean())
    rng = np.random.default_rng(cfg["seed"])
    boot = vals[rng.integers(0, len(vals), size=(cfg["bootstrap"], len(vals)))].mean(axis=1)
    # swapping A/B within a pair negates its value, so the null flips the sign of whole pair values
    null = rng.choice([-1, 1], size=(cfg["sign_flip_permutations"], len(vals))) @ vals / len(vals)
    k = int(np.sum(np.abs(null) >= abs(est) - 1e-12))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi), "pairs": len(pairs),
            "families": len(d) - len(one_sided), "p_sign_flip": (k + 1) / (len(null) + 1),
            "one_sided_pair_families": one_sided}
