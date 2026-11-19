"""ERNIE-RNA after continued training: does the readout head move, and how much do the attention maps
follow the hand-built pairing prior? (config training_experiment.prior_checks)

Usage: python scripts/41_prior_checks.py   (GPU; env rnalm-infer; after scripts 27-29 for the model)

ERNIE-RNA adds a fixed base-pairing score (AU 2, CG 3, GU 0.8, everything else 0) to its first layer's
attention, and every layer passes its attention scores on to the next, so head h of map l is
map 0 of head h (the projected prior) plus the learned scores of layers 1..l. For the released model,
the replay-only run and both injected arms (every seed and checkpoint of script 27):

Head drift. The head is chosen again for each checkpoint exactly as script 17 chose the frozen one
(its select function: the same low-exposure selection structures, eligible layers, threshold grid and
pooled F1), for the readouts in prior_checks.readouts; frozen_heads.json is not touched. Reported per
checkpoint: the chosen map, head, threshold and pooled F1, next to the frozen head's pooled F1 on the
same structures (from script 28's rows). The eval sequences are then scored with each checkpoint's own
head (script 18's score_model).

Pairing-prior reliance, two measures per eval sequence:
  prior_r      Pearson correlation, over all pairs j > i, between the frozen primary head's symmetrized
               map (before APC and the sigmoid) and the pairing score of the two letters. 1 would mean the
               map is the prior up to scale; lower means the learned part moved it away. Also the mean
               over all heads of every map (by_layer in the JSON).
  prior_drop   F1 of the frozen head minus its F1 when the pairing score is set to 0 for every pair
               (the model then runs without the prior): how much of the readout's accuracy needs the prior.

Threshold-free readout. The frozen primary threshold is 1.0 (the float32 sigmoid of the score equals 1),
so a change in the scale of the scores changes how many pairs are predicted. f1_top_k predicts the
frozen primary head's K highest-scoring pairs (symmetrized map, no APC, no sigmoid; K = the number of
reference pairs, so precision = recall = F1): it depends on the order of the scores only.

Estimand, as script 29: per family, the mean over its eval sequences and seeds of (value under the model
trained on the family's own set) - (value under the model trained on the other set); A and B sides
averaged with equal weight within each pair, mean over pairs, bootstrap and sign-flip test over pairs
(pair_balance.pair_test). It is given at every checkpoint for F1 with the frozen head (equals script
29), F1 with each checkpoint's own head, precision and recall of the frozen head, f1_top_k, prior_r and
prior_drop. vs_replay_only gives the pair-balanced mean of own-set model minus replay-only model and
of other-set model minus replay-only model (first seed, the only replay-only run), with the
bootstrap interval; levels gives the mean over families of the family means.

Writes results/41_prior_checks.json and results/41_prior_checks.tsv (one row per checkpoint and eval
sequence) with the .meta.json.
"""
import contextlib
import csv
import importlib.util
import json

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.models import attention_maps, load
from rnalm_exposure.pair_balance import family_differences, pair_test, read_tsv
from rnalm_exposure.readout import thresholds

IMPLEMENTATION = "multimolecule"


def script(name):
    spec = importlib.util.spec_from_file_location(name[3:], repo_path(f"scripts/{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def zero_prior():
    """Inside the block ERNIE-RNA's pairing score is 0 for every pair of letters."""
    from multimolecule import ErnieRnaModel
    original = ErnieRnaModel.get_pairwise_bias
    ErnieRnaModel.get_pairwise_bias = lambda self, *a, **k: torch.zeros_like(original(self, *a, **k))
    try:
        yield
    finally:
        ErnieRnaModel.get_pairwise_bias = original


def map_measures(model_path, records, head, device, autocast):
    """Per sequence, over the pairs j > i of the symmetrized maps (before APC and the sigmoid):
    {seq_id: [maps, heads] Pearson r between each map and the pairing score}, and {seq_id: share of
    true pairs among the K highest-scoring pairs of one head, K = number of reference pairs}."""
    model, tokenizer = load(IMPLEMENTATION, model_path, device)
    out, top = {}, {}
    for r in records:
        maps = attention_maps(model, tokenizer, r["sequence"], device, autocast).float()[..., 1:-1, 1:-1]
        ids = tokenizer([r["sequence"]], return_tensors="pt")["input_ids"].to(device)
        prior = model.get_pairwise_bias(ids)[0, 1:-1, 1:-1].float()
        i, j = torch.triu_indices(len(prior), len(prior), offset=1, device=device)
        x = (maps + maps.transpose(-1, -2))[..., i, j]
        p = prior[i, j]
        true = torch.as_tensor(r["contacts"], device=device)[i, j].bool()
        k = int(true.sum())
        top[r["seq_id"]] = float(true[x[head["layer"], head["head"]].topk(k).indices].float().mean()) if k else 0.0
        x, p = x - x.mean(-1, keepdim=True), p - p.mean()
        out[r["seq_id"]] = (x @ p / (x.norm(dim=-1) * p.norm()).clamp_min(1e-12)).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out, top


def precision_recall(row):
    tp, fp, fn = int(row["tp"]), int(row["fp"]), int(row["fn"])
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def main():
    config = load_config()
    paths, exp = config["paths"], config["training_experiment"]
    cfg, results = exp["prior_checks"], repo_path(paths["results_dir"])
    key, readouts = cfg["model"], cfg["readouts"]
    work = repo_path(exp["work_dir"].format(processed_dir=paths["processed_dir"]))
    device, autocast = config["inference"]["device"], config["inference"]["dtype"] == "float16_autocast"
    grid = thresholds(config["readout"]["threshold_grid"])
    eligible = config["readout"]["eligible_layers"].get(key)
    frozen = {v: json.loads(repo_path(paths["frozen_heads"]).read_text())[key][v] for v in readouts}
    s17, s18 = script("17_select_heads"), script("18_score_test_sets")

    seeds, epochs = exp["training"]["seeds"], sorted(exp["training"]["checkpoints_at_epochs"])
    checkpoints = {"released": config["inference"]["implementations"][IMPLEMENTATION][key]}
    for arm, sd in [("0", seeds[0]), *((a, sd) for sd in seeds for a in ("A", "B"))]:
        for e in epochs:
            checkpoints[f"{arm}_s{sd}_e{e}"] = str(work / "runs" / f"{key}_{arm}_s{sd}" / f"epoch_{e}")

    eval_rows = read_tsv(results / "26_eval_sequences.tsv")
    eval_ids = {r["seq_id"] for r in eval_rows}
    tests = [r for r in s18.test_records(config) if r["seq_id"] in eval_ids]
    pool, _ = s17.selection_pool(config)
    groups, _ = s17.subsets(pool, s17.read_exposure(config, config["validation"]["exposure_corpus"][key]),
                            config["validation"]["head_selection_sets"][0])
    low = {"low_exposure": groups["low_exposure"]}
    recorded = json.loads((results / f"17_head_selection_{IMPLEMENTATION}.json").read_text())["models"][key]["chosen"]

    selection, values, by_layer = {}, {}, {}
    for tag, path in checkpoints.items():
        rows28 = read_tsv(results / f"28_scores_{key}_{tag}.tsv")
        ref = {(r["seq_id"], r["readout"]): r for r in rows28}
        counts = [{**r, **{k: int(r[k]) for k in ("tp", "fp", "fn")}} for r in rows28]
        chosen = s17.select(IMPLEMENTATION, path, pool, low, readouts, grid, device, autocast, "low_exposure", eligible)
        selection[tag] = {}
        for v in readouts:
            c = chosen[v]["low_exposure"]
            selection[tag][v] = {**{k: c[k] for k in ("layer", "head", "threshold", "pooled_f1")},
                                 "frozen_head_pooled_f1": s18.pooled_f1(counts, v, low["low_exposure"])}
        own = {v: {k: selection[tag][v][k] for k in ("layer", "head", "threshold")} for v in readouts}
        if tag == "released":   # this path must give script 17's choice and pooled F1 for the released model
            for v in readouts:
                s = selection[tag][v]
                if own[v] != frozen[v] or max(abs(s[k] - recorded[v]["low_exposure"]["pooled_f1"])
                                              for k in ("pooled_f1", "frozen_head_pooled_f1")) > 1e-9:
                    raise RuntimeError(f"released model: {v} selection {s} differs from script 17")
        own_rows, _ = s18.score_model(IMPLEMENTATION, path, own, tests, device, autocast)
        with zero_prior():
            zero_rows, _ = s18.score_model(IMPLEMENTATION, path, frozen, tests, device, autocast)
        h = frozen["primary"]
        r_all, top = map_measures(path, tests, h, device, autocast)
        by_layer[tag] = {s: np.mean([r_all[r["seq_id"]].mean(-1) for r in eval_rows if r["set"] == s], axis=0).tolist()
                         for s in ("A", "B")}
        m = values[tag] = {}
        for q in eval_ids:
            m[q] = {"prior_r": float(r_all[q][h["layer"], h["head"]]), "f1_top_k": top[q]}
        for v in readouts:
            for q in eval_ids:
                m[q][f"precision_{v}"], m[q][f"recall_{v}"] = precision_recall(ref[(q, v)])
            for r in own_rows:
                if r["readout"] == v:
                    if own[v] == frozen[v] and (r["tp"], r["fp"]) != (int(ref[(r["seq_id"], v)]["tp"]), int(ref[(r["seq_id"], v)]["fp"])):
                        raise RuntimeError(f"{tag} {r['seq_id']} {v}: frozen-head score differs from script 28")
                    m[r["seq_id"]][f"f1_frozen_{v}"] = float(ref[(r["seq_id"], v)]["f1"])
                    m[r["seq_id"]][f"f1_own_head_{v}"] = r["f1"]
            for r in zero_rows:
                if r["readout"] == v:
                    m[r["seq_id"]][f"prior_drop_{v}"] = float(ref[(r["seq_id"], v)]["f1"]) - r["f1"]
        print(tag, json.dumps(selection[tag]), flush=True)

    measures = sorted(next(iter(values["released"].values())))
    with open(results / "41_prior_checks.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["checkpoint", "seq_id", "set", *measures])
        for tag in checkpoints:
            for r in eval_rows:
                w.writerow([tag, r["seq_id"], r["set"], *(f"{values[tag][r['seq_id']][k]:.6f}" for k in measures)])

    assign = {r["family"]: r for r in read_tsv(results / "26_assignment.tsv")}
    pair_of = {f: r["pair"] for f, r in assign.items()}
    set_of_family = {f: r["set"] for f, r in assign.items()}
    other = {"A": "B", "B": "A"}

    def test(differences):   # {seq_id: [difference per seed]} -> pair-balanced estimate
        return pair_test(family_differences(eval_rows, differences), pair_of, set_of_family, exp["analysis"])

    def value(arm, sd, e, q, k):
        return values[f"{arm}_s{sd}_e{e}"][q][k]

    def family_mean(per_sequence):
        return float(np.mean(list(family_differences(eval_rows, {q: [x] for q, x in per_sequence.items()}).values())))

    out = {"model": key, "frozen_heads": frozen, "selection": selection, "own_minus_other": {}, "vs_replay_only": {},
           "levels": {}, "prior_r_by_layer": by_layer}
    for k in measures:
        out["own_minus_other"][k], out["vs_replay_only"][k] = {}, {}
        out["levels"][k] = {"released": family_mean({q: values["released"][q][k] for q in eval_ids})}
        for e in epochs:
            sets = {r["seq_id"]: r["set"] for r in eval_rows}
            out["own_minus_other"][k][str(e)] = test(
                {q: [value(s, sd, e, q, k) - value(other[s], sd, e, q, k) for sd in seeds] for q, s in sets.items()})
            vs = {}
            for name, arm_of in (("own", lambda s: s), ("other", other.get)):
                t = test({q: [value(arm_of(s), seeds[0], e, q, k) - value("0", seeds[0], e, q, k)] for q, s in sets.items()})
                vs[name] = {x: t[x] for x in ("estimate", "ci_low", "ci_high", "pairs")}
            out["vs_replay_only"][k][str(e)] = vs
            out["levels"][k][str(e)] = {
                "replay_only": family_mean({q: value("0", seeds[0], e, q, k) for q in sets}),
                "own": family_mean({q: np.mean([value(s, sd, e, q, k) for sd in seeds]) for q, s in sets.items()}),
                "other": family_mean({q: np.mean([value(other[s], sd, e, q, k) for sd in seeds]) for q, s in sets.items()})}

    path = results / "41_prior_checks.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {e: [round(t[x], 4) for x in ("estimate", "ci_low", "ci_high")] for e, t in v.items()}
                      for k, v in out["own_minus_other"].items()}, indent=1))
    write_meta(path, config, packages=["torch", "transformers", "multimolecule", "numpy"])


if __name__ == "__main__":
    main()
