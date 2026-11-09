"""Choose and freeze each model's attention head on low-exposure validation sequences (config
validation and readout).

Usage: python scripts/17_select_heads.py <implementation> <model key> [<model key> ...] [--overwrite]
  implementation: a key of inference.implementations, run in its own env (as script 09).
  A head already in paths.frozen_heads is never changed unless --overwrite is given.

Selection pool: Wang et al. 2026's structure sets in validation.head_selection_sets, minus sequences
identical to a test sequence or from the same PDB chain as an Archive-NoFam entry. For each model,
the low-exposure half is the pool sequences with log_homologs (script 13, the model's own corpus) at
or below that model's median over the pool. The layer, head and threshold with the best pooled F1
on the low-exposure half are frozen, once for the primary readout and once for each readout variant
(one_pair_per_base: greedy maximum-weight matching, pairs taken in descending score order and
skipped if they reuse a base; ties broken by index order; apc_off: the same scores without APC). Scores
and labels follow script 09: residues with 3D structure only, contact matrices as labels.

As a stability check, the same choice is also made on the high-exposure half, on the whole pool, and
on the low-exposure half of the first set alone (its own median); these are reported, not used. For
every subset the frozen head's pooled F1 and the five best heads are reported too.

Writes results/17_head_selection_<implementation>.json (subsets, choices, pooled F1, md5 of the
input files; models from earlier runs are kept) with its .meta.json, and adds the models' frozen
heads to paths.frozen_heads.
"""
import csv
import hashlib
import json
import sys

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.metrics import f1_mcc
from rnalm_exposure.models import attention_maps, load
from rnalm_exposure.readout import contact_scores, one_pair_per_base, select_head, threshold_counts, thresholds, upper
from rnalm_exposure.test_sets import normalize
from rnalm_exposure.wang import load_set

def chain_key(seq_id):
    pdb, chain = seq_id.split(":", 1)[1].split("_", 1)
    return pdb.lower(), chain


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def selection_pool(config):
    """Records of the head-selection sets after removing test-set overlap, and the removed ids."""
    paths, val = config["paths"], config["validation"]
    if val["exclude_test_overlap"] != ["exact_sequence", "pdb_chain"]:
        raise ValueError("only exact_sequence and pdb_chain overlap removal is implemented")
    with open(repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))) as f:
        tests = [json.loads(line) for line in f]
    test_seqs = {r["sequence"] for r in tests}
    # Archive-NoFam ids are <pdb>_<chain> with a lowercase PDB code; Wang's are uppercase. Chain IDs
    # are case-sensitive, so only the PDB code is compared without case.
    nofam_chains = {chain_key(r["seq_id"]) for r in tests if r["dataset"] == "archive_nofam"}
    zip_path = repo_path(val["wang2026_zip"].format(raw_dir=paths["raw_dir"]))
    pool, removed = [], {}
    for name in val["head_selection_sets"]:
        for r in load_set(zip_path, val["wang2026_files"][name], name):
            r["set"] = name
            if normalize(r["sequence"]) in test_seqs:
                removed[r["seq_id"]] = "exact_sequence"
            elif chain_key(r["seq_id"]) in nofam_chains:
                removed[r["seq_id"]] = "pdb_chain"
            else:
                pool.append(r)
    return pool, removed


def read_exposure(config, corpus):
    with open(repo_path(config["paths"]["results_dir"]) / f"13_exposure_{corpus}.tsv") as f:
        return {r["seq_id"]: float(r["log_homologs"]) for r in csv.DictReader(f, delimiter="\t")}


def subsets(pool, log_homologs, first_set):
    """Sequence ids of the low/high-exposure halves (median over the pool), the whole pool, and the
    first set's own low-exposure half."""
    median = float(np.median([log_homologs[r["seq_id"]] for r in pool]))
    first = [r for r in pool if r["set"] == first_set]
    first_median = float(np.median([log_homologs[r["seq_id"]] for r in first]))
    ids = {
        "low_exposure": {r["seq_id"] for r in pool if log_homologs[r["seq_id"]] <= median},
        "high_exposure": {r["seq_id"] for r in pool if log_homologs[r["seq_id"]] > median},
        "pool": {r["seq_id"] for r in pool},
        f"low_exposure_{first_set}_only": {r["seq_id"] for r in first if log_homologs[r["seq_id"]] <= first_median},
    }
    return ids, median


def select(implementation, model_id, pool, groups, readouts, grid, device, autocast):
    model, tokenizer = load(implementation, model_id, device)
    grid_t = torch.tensor(grid, dtype=torch.float64, device=device)
    counts = {(g, v): [None, None, 0, 0] for g in groups for v in readouts}   # tp, pp, n_true, n_total
    for record in pool:
        attentions = attention_maps(model, tokenizer, record["sequence"], device, autocast)
        keep = torch.tensor(record["keep"], device=device)
        flat = {apc: contact_scores(attentions, apc=apc)[..., keep, :][..., keep].flatten(0, 1) for apc in (True, False)}
        layers, heads = attentions.shape[:2]
        for v in readouts:
            s = one_pair_per_base(flat[True]) if v == "one_pair_per_base" else flat[v != "apc_off"]
            s, t = upper(s, record["contacts"])
            tp_i, pp_i = threshold_counts(s, t, grid_t)
            for g, members in groups.items():
                if record["seq_id"] in members:
                    c = counts[(g, v)]
                    c[0] = tp_i if c[0] is None else c[0] + tp_i
                    c[1] = pp_i if c[1] is None else c[1] + pp_i
                    c[2] += int(t.sum())
                    c[3] += t.numel()
    del model
    torch.cuda.empty_cache()
    chosen = {}
    for v in readouts:   # the frozen choice (low-exposure half) first, then the other subsets
        frozen = None
        for g in ["low_exposure", *(g for g in groups if g != "low_exposure")]:
            tp, pp, n_true, n_total = counts[(g, v)]
            tp, pp = tp.view(layers, heads, -1).cpu().numpy(), pp.view(layers, heads, -1).cpu().numpy()
            best = select_head(tp, pp, n_true, n_total, grid)
            frozen = frozen or best
            m = int(np.searchsorted(grid, frozen["threshold"]))   # the frozen threshold's grid index
            tp_f, pp_f = float(tp[frozen["layer"], frozen["head"], m]), float(pp[frozen["layer"], frozen["head"], m])
            f1, _ = f1_mcc(tp_f, pp_f - tp_f, n_true - tp_f, n_total - pp_f - (n_true - tp_f))
            top = np.argsort(-best["max_f1"], axis=None, kind="stable")[:5]
            chosen.setdefault(v, {})[g] = {
                **{k: best[k] for k in ("layer", "head", "threshold", "pooled_f1")},
                "frozen_head_pooled_f1": float(f1),
                "top5": [{"layer": int(l), "head": int(h), "threshold": float(best["thresholds"][l, h]),
                          "pooled_f1": float(best["max_f1"][l, h])}
                         for l, h in zip(*np.unravel_index(top, best["max_f1"].shape))],
            }
    return chosen


def main():
    config = load_config()
    paths, val = config["paths"], config["validation"]
    if val["low_exposure"] != "at_or_below_median" or config["readout"]["selection_metric"] != "pooled_f1":
        raise ValueError("only the median split and pooled F1 are implemented")
    if config["readout"]["variants"] != {"one_pair_per_base": "greedy_by_score", "apc_off": "no_apc"}:
        raise ValueError("only the greedy one-pair-per-base and APC-off variants are implemented")
    overwrite = "--overwrite" in sys.argv
    implementation, keys = sys.argv[1], [a for a in sys.argv[2:] if a != "--overwrite"]
    models = config["inference"]["implementations"][implementation]
    readouts = ["primary", *config["readout"]["variants"]]
    grid = thresholds(config["readout"]["threshold_grid"])
    device = config["inference"]["device"]
    autocast = config["inference"]["dtype"] == "float16_autocast"

    pool, removed = selection_pool(config)
    print(f"selection pool: {len(pool)} structures, {len(removed)} removed", flush=True)
    out = repo_path(paths["results_dir"]) / f"17_head_selection_{implementation}.json"
    earlier = json.loads(out.read_text())["models"] if out.exists() else {}
    test_sets = repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))
    results = {"implementation": implementation, "sets": val["head_selection_sets"], "n_pool": len(pool),
               "removed": removed, "test_sets_md5": md5(test_sets), "models": earlier}
    frozen_path = repo_path(paths["frozen_heads"])
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    for key in keys:
        corpus = val["exposure_corpus"][key]
        log_homologs = read_exposure(config, corpus)
        groups, median = subsets(pool, log_homologs, val["head_selection_sets"][0])
        chosen = select(implementation, models[key], pool, groups, readouts, grid, device, autocast)
        results["models"][key] = {"corpus": corpus, "median_log_homologs": median,
                                  "subsets": {g: sorted(m) for g, m in groups.items()}, "chosen": chosen}
        results["models"][key]["exposure_md5"] = md5(repo_path(paths["results_dir"]) / f"13_exposure_{corpus}.tsv")
        new = {v: {k: chosen[v]["low_exposure"][k] for k in ("layer", "head", "threshold")} for v in readouts}
        changed = [v for v in readouts if v in frozen.get(key, {}) and frozen[key][v] != new[v]]
        if changed and not overwrite:
            raise RuntimeError(f"{key}: frozen {changed} would change ({frozen[key]} -> {new}); pass --overwrite")
        frozen[key] = {**frozen.get(key, {}), **new}
        print(key, json.dumps({v: chosen[v] for v in readouts}), flush=True)

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    frozen_path.write_text(json.dumps(dict(sorted(frozen.items())), indent=2) + "\n")
    write_meta(out, config, packages=["torch", "transformers", "multimolecule", "modelgenerator"])


if __name__ == "__main__":
    main()
