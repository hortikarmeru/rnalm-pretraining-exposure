"""Supervised probe on a training-experiment checkpoint (config training_experiment.probe).

Usage: python scripts/38_probe_scores.py <model key> (released | <arm> <seed> <epoch>)   (env rnalm-infer; released first)

Features: every layer and head's contact score (readout.contact_scores: symmetrized, APC, sigmoid)
for each (i, j), j > i. An L1 logistic regression (Rao et al. 2020; C fixed, not tuned) is fitted on
the probe structures of script 37: all true pairs plus negatives_per_positive unpaired (i, j) drawn
with a seed per sequence id, so every checkpoint is fitted on the same rows. The threshold is the
readout.threshold_grid value with the highest F1 pooled over all j > i pairs of the probe structures.

Two readouts are scored on the eval sequences and sentinel tests:
  probe_refit: probe and threshold fitted on this checkpoint (a user fits a readout to the model they have);
  probe_released: the released model's probe and threshold, applied unchanged (separates a change in
  the attention maps from a change in the readout). For "released" the two are the same probe.

Writes results/38_probe_scores_<model>_<tag>.tsv (seq_id, role, readout, f1, mcc, tp, fp, fn) and
results/38_probe_<model>_<tag>.json (weights, bias, non-zero weights, threshold, pooled training F1)
with its .meta.json; tag = released or <arm>_s<seed>_e<epoch>.
"""
import csv
import importlib.util
import json
import sys
import zlib

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.models import attention_maps, load
from rnalm_exposure.pair_balance import read_tsv
from rnalm_exposure.readout import contact_scores, score_pairs, select_head, threshold_counts, thresholds, upper


def script_18():
    spec = importlib.util.spec_from_file_location("score_test_sets", repo_path("scripts/18_score_test_sets.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def layer_scores(attentions):
    """Yield (layer, [heads, L, L] contact scores), one layer at a time to bound memory."""
    for layer in range(attentions.shape[0]):
        yield layer, contact_scores(attentions[layer][None])[0]


def training_rows(attentions, record, cfg):
    """(features [rows, layers * heads], labels) for one probe structure: every true pair and
    negatives_per_positive unpaired (i, j), j > i, drawn with a seed from the sequence id."""
    L = record["contacts"].shape[0]
    i, j = np.triu_indices(L, 1)
    y = record["contacts"][i, j].astype(bool)
    neg = np.flatnonzero(~y)
    rng = np.random.default_rng([cfg["seed"], zlib.crc32(record["seq_id"].encode())])
    k = min(len(neg), cfg["negatives_per_positive"] * int(y.sum()))
    rows = np.sort(np.concatenate([np.flatnonzero(y), rng.choice(neg, k, replace=False)]))
    ri = torch.as_tensor(i[rows], device=attentions.device)
    rj = torch.as_tensor(j[rows], device=attentions.device)
    feats = [s[:, ri, rj].T.cpu().numpy() for _, s in layer_scores(attentions)]
    return np.concatenate(feats, axis=1), y[rows]


def probe_map(attentions, probe):
    """[L, L] probe probabilities: sigmoid(bias + sum over layers and heads of weight * score)."""
    w = torch.as_tensor(probe["weights"], dtype=torch.float32, device=attentions.device)
    logit = None
    for layer, s in layer_scores(attentions):
        part = torch.einsum("h,hij->ij", w[layer], s)
        logit = part if logit is None else logit + part
    return torch.sigmoid(logit + probe["bias"])


def main():
    config = load_config()
    paths, exp = config["paths"], config["training_experiment"]
    cfg = exp["probe"]
    results = repo_path(paths["results_dir"])
    work = repo_path(exp["work_dir"].format(processed_dir=paths["processed_dir"]))
    key = sys.argv[1]
    if sys.argv[2] == "released":
        tag, model_path = "released", config["inference"]["implementations"]["multimolecule"][key]
    else:
        arm, seed, epoch = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        tag = f"{arm}_s{seed}_e{epoch}"
        model_path = str(work / "runs" / f"{key}_{arm}_s{seed}" / f"epoch_{epoch}")
    released_probe = results / f"38_probe_{key}_released.json"
    if tag != "released" and not released_probe.exists():
        raise SystemExit(f"run {key} released first: {released_probe} is missing")
    device = config["inference"]["device"]
    autocast = config["inference"]["dtype"] == "float16_autocast"

    roles = {r["seq_id"]: "eval" for r in read_tsv(results / "26_eval_sequences.tsv")}
    roles.update({r["seq_id"]: "sentinel_test" for r in read_tsv(results / "26_sentinels.tsv")})
    train_ids = {r["seq_id"] for r in read_tsv(results / "37_probe_sequences.tsv")}
    if train_ids & set(roles):
        raise RuntimeError("probe structures overlap the eval or sentinel sequences")
    records = [r for r in script_18().test_records(config) if r["seq_id"] in roles or r["seq_id"] in train_ids]
    train = [r for r in records if r["seq_id"] in train_ids]
    scored = [r for r in records if r["seq_id"] in roles]
    model, tokenizer = load("multimolecule", model_path, device)

    def attn(record):
        return attention_maps(model, tokenizer, record["sequence"], device, autocast)

    # fit
    from sklearn.linear_model import LogisticRegression
    xs, ys = zip(*(training_rows(attn(r), r, cfg) for r in train))
    X, y = np.concatenate(xs), np.concatenate(ys)
    fit = LogisticRegression(penalty=cfg["penalty"], C=cfg["C"], solver=cfg["solver"], max_iter=cfg["max_iter"],
                             random_state=cfg["seed"]).fit(X, y)
    n_layers, n_heads = attn(train[0]).shape[:2]
    probe = {"weights": fit.coef_[0].reshape(n_layers, n_heads).tolist(), "bias": float(fit.intercept_[0])}

    # threshold: pooled F1 over every j > i pair of the probe structures
    grid = torch.as_tensor(thresholds(config["readout"]["threshold_grid"]), dtype=torch.float64, device=device)
    tp = pp = 0
    n_true = n_total = 0
    for r in train:
        s, t = upper(probe_map(attn(r), probe), r["contacts"])
        c_tp, c_pp = threshold_counts(s[None], t, grid)
        tp, pp = tp + c_tp[0].cpu().numpy(), pp + c_pp[0].cpu().numpy()
        n_true, n_total = n_true + int(t.sum()), n_total + int(t.numel())
    chosen = select_head(tp[None, None], pp[None, None], n_true, n_total, grid.cpu().numpy())
    probe["threshold"], probe["train_pooled_f1"] = chosen["threshold"], chosen["pooled_f1"]
    probe.update({"nonzero_weights": int(np.count_nonzero(fit.coef_)), "n_weights": int(fit.coef_.size),
                  "rows": int(len(y)), "true_pairs": int(y.sum()), "sequences": len(train),
                  "converged": bool(fit.n_iter_.max() < cfg["max_iter"])})
    if not probe["converged"]:
        raise RuntimeError(f"probe fit hit max_iter = {cfg['max_iter']} without converging")

    probes = {"probe_refit": probe}
    if tag == "released":
        probes["probe_released"] = probe
    else:
        probes["probe_released"] = json.loads(released_probe.read_text())

    rows = []
    for r in scored:
        a = None   # free the previous maps before computing the next
        a = attn(r)
        for name, p in probes.items():
            rows.append({"seq_id": r["seq_id"], "readout": name, **score_pairs(probe_map(a, p), r["contacts"], p["threshold"])})

    with open(results / f"38_probe_scores_{key}_{tag}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seq_id", "role", "readout", "f1", "mcc", "tp", "fp", "fn"])
        for r in rows:
            w.writerow([r["seq_id"], roles[r["seq_id"]], r["readout"], f"{r['f1']:.6f}", f"{r['mcc']:.6f}", r["tp"], r["fp"], r["fn"]])
    out = results / f"38_probe_{key}_{tag}.json"
    out.write_text(json.dumps(probe, indent=1))
    means = {f"{name}_{role}_mean_f1": float(np.mean([r["f1"] for r in rows if r["readout"] == name and roles[r["seq_id"]] == role]))
             for name in probes for role in ("eval", "sentinel_test")}
    print(json.dumps({k: v for k, v in probe.items() if k != "weights"} | means), flush=True)
    write_meta(out, config, packages=["torch", "transformers", "multimolecule", "scikit-learn"])


if __name__ == "__main__":
    main()
