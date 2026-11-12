"""Analysis of the training experiment (config training_experiment.analysis), from script 28's scores.

Usage: python scripts/29_training_analysis.py <model key> [--simulate]   (environment-extra.yml)
  --simulate: every model is the released one (script 28 "released") plus a known effect (+0.02 F1
  and -0.05 loss on its own set's families) and noise, to check the analysis recovers it before
  real results exist.

Primary estimand: per family f, D_f = mean over its eval sequences and over seeds of
F1(model trained on f's set) - F1(model trained on the other set), primary readout, last checkpoint;
the estimate is the unweighted mean of D_f over families. Families were randomized by clan within
matched pairs (script 26), so uncertainty comes from the pairs: a bootstrap resamples pairs
(95% percentile interval), and the permutation test flips the A/B labels of whole pairs (two-sided,
p = (k + 1) / (n + 1)). Verdict: positive if the interval is above 0; null at this dose if it lies
inside +/- null_band; inconclusive otherwise; uninterpretable if the manipulation check fails.

Also: the manipulation check (masked-token loss on eval sequences, own-set minus other-set model;
should be negative), the same estimate at each checkpoint, a generic-effect check (sentinel mean F1,
model B minus model A: injecting different amounts of new data should not move unrelated families;
also each injected model vs the replay-only one), and the estimate per stratum.

Writes results/29_training_analysis_<model>[_simulated].json with its .meta.json.
"""
import csv
import json
import sys

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def scores(results, key, tag, role, column="f1", kind="scores"):
    rows = read_tsv(results / f"28_{kind}_{key}_{tag}.tsv")
    if kind == "scores":
        return {r["seq_id"]: float(r[column]) for r in rows if r["readout"] == "primary" and r["role"].startswith(role)}
    return {r["seq_id"]: float(r["loss"]) for r in rows if r["role"] == role}


def family_differences(eval_rows, own_other):
    """{family: D_f} from {seq_id: [own - other per seed]}."""
    fams = {}
    for r in eval_rows:
        fams.setdefault(r["family"], []).append(np.mean(own_other[r["seq_id"]]))
    return {f: float(np.mean(v)) for f, v in fams.items()}


def pair_test(d, pair_of, cfg):
    """Mean of D_f, bootstrap over pairs, sign-flip permutation over pairs."""
    pairs = sorted({pair_of[f] for f in d})   # only pairs with families in d (by-stratum subsets)
    by_pair = {p: [d[f] for f in d if pair_of[f] == p] for p in pairs}
    est = float(np.mean(list(d.values())))
    rng = np.random.default_rng(cfg["seed"])
    boot = []
    for _ in range(cfg["bootstrap"]):
        pick = rng.choice(len(pairs), len(pairs), replace=True)
        vals = [x for k in pick for x in by_pair[pairs[k]]]
        boot.append(np.mean(vals))
    sums = np.array([sum(by_pair[p]) for p in pairs])
    n = sum(len(v) for v in by_pair.values())
    flips = rng.choice([-1, 1], size=(cfg["sign_flip_permutations"], len(pairs)))
    null = flips @ sums / n
    k = int(np.sum(np.abs(null) >= abs(est) - 1e-12))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi), "families": len(d), "pairs": len(pairs),
            "p_sign_flip": (k + 1) / (len(null) + 1)}


def main():
    config = load_config()
    exp, results = config["training_experiment"], repo_path(config["paths"]["results_dir"])
    cfg = exp["analysis"]
    key = sys.argv[1]
    simulate = "--simulate" in sys.argv
    eval_rows = read_tsv(results / "26_eval_sequences.tsv")
    assign = {r["family"]: r for r in read_tsv(results / "26_assignment.tsv")}
    pair_of = {f: r["pair"] for f, r in assign.items()}
    set_of = {r["seq_id"]: r["set"] for r in eval_rows}
    seeds = exp["training"]["seeds"]
    epochs = sorted(exp["training"]["checkpoints_at_epochs"])
    last = epochs[-1]
    rng = np.random.default_rng(cfg["seed"])

    def f1(arm, seed, epoch, role="eval", kind="scores"):
        s = scores(results, key, "released" if simulate else f"{arm}_s{seed}_e{epoch}", role, kind=kind)
        if simulate:   # known effect on the model's own families, plus noise per sequence
            shift, sd = (0.02, 0.05) if kind == "scores" else (-0.05, 0.02)
            s = {q: v + (shift if set_of.get(q) == arm else 0.0) + rng.normal(0, sd) for q, v in s.items()}
        return s

    def estimate(epoch, keep=lambda r: True):
        own_other = {}
        for sd in seeds:
            a, b = f1("A", sd, epoch), f1("B", sd, epoch)
            for r in eval_rows:
                q = r["seq_id"]
                own, other = (a[q], b[q]) if r["set"] == "A" else (b[q], a[q])
                own_other.setdefault(q, []).append(own - other)
        rows = [r for r in eval_rows if keep(r)]
        return pair_test(family_differences(rows, own_other), pair_of, cfg)

    out = {"model": key, "simulated": simulate, "primary": estimate(last)}
    p = out["primary"]
    if p["ci_low"] > 0:
        verdict = "positive"
    elif -cfg["null_band"] <= p["ci_low"] and p["ci_high"] <= cfg["null_band"]:
        verdict = "null at this dose"
    else:
        verdict = "inconclusive"
    # manipulation check: own-set model's masked-token loss on eval sequences minus the other's
    loss_diff = {}
    for sd in seeds:
        la, lb = f1("A", sd, last, "eval", "mlm"), f1("B", sd, last, "eval", "mlm")
        for r in eval_rows:
            q = r["seq_id"]
            loss_diff.setdefault(q, []).append((la[q] - lb[q]) if r["set"] == "A" else (lb[q] - la[q]))
    manip = pair_test(family_differences(eval_rows, loss_diff), pair_of, cfg)
    out["manipulation_check_loss"] = manip
    if manip["ci_high"] >= 0:
        verdict = "uninterpretable (no loss gain on injected families)"
    out["verdict"] = verdict
    out["by_checkpoint"] = {str(e): estimate(e) for e in epochs}
    out["by_stratum"] = {st: estimate(last, lambda r, st=st: assign[r["family"]]["stratum"] == st) for st in ("low", "mid")}
    generic = {}
    for sd in seeds:
        sa, sb = f1("A", sd, last, "sentinel"), f1("B", sd, last, "sentinel")
        generic[f"B_minus_A_s{sd}"] = float(np.mean([sb[q] - sa[q] for q in sa]))
    if simulate or (results / f"28_scores_{key}_0_s0_e{last}.tsv").exists():
        s0 = f1("0", 0, last, "sentinel")
        for arm in ("A", "B"):
            s = f1(arm, 0, last, "sentinel")
            generic[f"{arm}_minus_replay_only_s0"] = float(np.mean([s[q] - s0[q] for q in s0]))
    out["generic_effect_sentinel_f1"] = generic

    name = f"29_training_analysis_{key}{'_simulated' if simulate else ''}.json"
    path = results / name
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    write_meta(path, config, packages=["numpy"])


if __name__ == "__main__":
    main()
