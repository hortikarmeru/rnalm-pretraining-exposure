"""Training-experiment analysis of the supervised probe (script 38), with script 29's estimand.

Usage: python scripts/39_probe_analysis.py <model key>   (environment-extra.yml)

For each probe readout (probe_refit, probe_released) and each of probe.checkpoints_at_epochs:
D_f = mean over a family's eval sequences and over seeds of F1(own-set model) - F1(other-set model),
pair-balanced, pair bootstrap and sign-flip over pairs (pair_balance.pair_test, as in script 29).
Also: sentinel mean F1 of model B minus model A per seed, each injected model vs the replay-only
one (seed 0), and the released model's probe F1 next to its frozen-head F1 (script 28).

Writes results/39_probe_analysis_<model>.json with its .meta.json.
"""
import json
import sys

import numpy as np

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.pair_balance import family_differences, pair_test, read_tsv

READOUTS = ("probe_refit", "probe_released")


def probe_f1(results, key, tag, readout, role):
    return {r["seq_id"]: float(r["f1"]) for r in read_tsv(results / f"38_probe_scores_{key}_{tag}.tsv")
            if r["readout"] == readout and r["role"] == role}


def main():
    config = load_config()
    exp, results = config["training_experiment"], repo_path(config["paths"]["results_dir"])
    cfg = exp["analysis"]
    key = sys.argv[1]
    eval_rows = read_tsv(results / "26_eval_sequences.tsv")
    assign = {r["family"]: r for r in read_tsv(results / "26_assignment.tsv")}
    pair_of = {f: r["pair"] for f, r in assign.items()}
    set_of_family = {f: r["set"] for f, r in assign.items()}
    seeds = exp["training"]["seeds"]

    out = {"model": key, "estimates": {}, "generic_effect_sentinel_f1": {}}
    for readout in READOUTS:
        for e in exp["probe"]["checkpoints_at_epochs"]:
            own_other = {}
            for sd in seeds:
                a = probe_f1(results, key, f"A_s{sd}_e{e}", readout, "eval")
                b = probe_f1(results, key, f"B_s{sd}_e{e}", readout, "eval")
                for r in eval_rows:
                    q = r["seq_id"]
                    own_other.setdefault(q, []).append(a[q] - b[q] if r["set"] == "A" else b[q] - a[q])
            out["estimates"][f"{readout}_e{e}"] = pair_test(family_differences(eval_rows, own_other),
                                                            pair_of, set_of_family, cfg)
            generic = {}
            for sd in seeds:
                sa = probe_f1(results, key, f"A_s{sd}_e{e}", readout, "sentinel_test")
                sb = probe_f1(results, key, f"B_s{sd}_e{e}", readout, "sentinel_test")
                generic[f"B_minus_A_s{sd}"] = float(np.mean([sb[q] - sa[q] for q in sa]))
            s0 = probe_f1(results, key, f"0_s0_e{e}", readout, "sentinel_test")
            for arm in ("A", "B"):
                s = probe_f1(results, key, f"{arm}_s0_e{e}", readout, "sentinel_test")
                generic[f"{arm}_minus_replay_only_s0"] = float(np.mean([s[q] - s0[q] for q in s0]))
            out["generic_effect_sentinel_f1"][f"{readout}_e{e}"] = generic

    frozen = {r["seq_id"]: float(r["f1"]) for r in read_tsv(results / f"28_scores_{key}_released.tsv")
              if r["readout"] == "primary"}
    out["released"] = {}
    for role in ("eval", "sentinel_test"):
        probe = probe_f1(results, key, "released", "probe_refit", role)
        out["released"][role] = {"probe_mean_f1": float(np.mean(list(probe.values()))),
                                 "frozen_head_mean_f1": float(np.mean([frozen[q] for q in probe])), "n": len(probe)}

    path = results / f"39_probe_analysis_{key}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    write_meta(path, config, packages=["numpy"])


if __name__ == "__main__":
    main()
