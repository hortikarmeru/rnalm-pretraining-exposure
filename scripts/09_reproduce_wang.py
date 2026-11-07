"""Reproduce Wang et al. (2026) zero-shot secondary structure results with our readout code.

Usage: python scripts/09_reproduce_wang.py <implementation> [model key ...]
  implementation: a key of inference.implementations (multimolecule, genbio), run in its own env;
  default: every model of that implementation.

Uses Wang's own data (Zenodo 18218776), not our test sets: for each model, picks the layer, head
and threshold with the best pooled F1 on their 40-structure selection set (VL1), as they did, then
scores each of their 70 test structures (TS) at that head and threshold and compares the median and
mean F1 and median MCC with their Supplementary Table 4. This head choice is for the reproduction
only; the study's heads are chosen on low-exposure sequences and frozen separately.

As in Wang's code, scores and labels are kept only at the residues that have 3D structure (the
fourth field of each record), and the labels are their contact matrices (any j > i pair).

Writes results/09_wang_reproduction_<implementation>.json (summary plus per-structure F1/MCC) and
its .meta.json.
"""
import io
import json
import pickle
import sys
import zipfile

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.models import attention_maps, load
from rnalm_exposure.readout import contact_scores, score_pairs, select_head, threshold_counts, thresholds, upper

# Wang's files are Python pickles. This unpickler rebuilds plain lists, strings and numpy arrays and
# refuses everything else, so the file can't run code when it is loaded.
ALLOWED = {("numpy", "dtype"), ("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct"),
           ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer")}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def load_set(zip_path, member, name):
    with zipfile.ZipFile(zip_path) as z:
        records = SafeUnpickler(io.BytesIO(z.read(member))).load()
    return [{"seq_id": f"{name}:{r[0]}", "sequence": r[1], "contacts": np.asarray(r[2]), "keep": list(r[3])}
            for r in records]


def kept_scores(model, tokenizer, record, device, autocast):
    scores = contact_scores(attention_maps(model, tokenizer, record["sequence"], device, autocast))
    keep = torch.tensor(record["keep"], device=device)
    return scores[..., keep, :][..., keep]


def reproduce(implementation, model_id, sets, grid, device, autocast):
    model, tokenizer = load(implementation, model_id, device)
    grid_t = torch.tensor(grid, dtype=torch.float64, device=device)
    tp = pp = None
    n_true = n_total = 0
    for record in sets["selection"]:
        scores = kept_scores(model, tokenizer, record, device, autocast)
        layers, heads = scores.shape[:2]
        s, t = upper(scores.flatten(0, 1), record["contacts"])
        tp_i, pp_i = threshold_counts(s, t, grid_t)
        tp = tp_i if tp is None else tp + tp_i
        pp = pp_i if pp is None else pp + pp_i
        n_true += int(t.sum())
        n_total += t.numel()
    best = select_head(tp.view(layers, heads, -1).cpu().numpy(), pp.view(layers, heads, -1).cpu().numpy(),
                       n_true, n_total, grid)

    per_structure = {}
    for record in sets["reproduction"]:
        scores = kept_scores(model, tokenizer, record, device, autocast)[best["layer"], best["head"]]
        per_structure[record["seq_id"]] = score_pairs(scores, record["contacts"], best["threshold"])
    del model
    torch.cuda.empty_cache()

    f1 = np.array([r["f1"] for r in per_structure.values()])
    mcc = np.array([r["mcc"] for r in per_structure.values()])
    return {
        "model": model_id,
        "layer": best["layer"], "head": best["head"], "threshold": best["threshold"],
        "selection_pooled_f1": best["pooled_f1"],
        "median_f1": float(np.median(f1)), "mean_f1": float(f1.mean()), "sd_f1": float(f1.std(ddof=1)),
        "median_mcc": float(np.median(mcc)), "mean_mcc": float(mcc.mean()),
        "per_structure": per_structure,
    }


def main():
    config = load_config()
    paths, val = config["paths"], config["validation"]
    targets = val["reproduction_targets"]
    implementation = sys.argv[1]
    models = config["inference"]["implementations"][implementation]
    keys = sys.argv[2:] or list(models)
    zip_path = repo_path(val["wang2026_zip"].format(raw_dir=paths["raw_dir"]))
    sets = {role: load_set(zip_path, val["wang2026_files"][val[f"{role}_set"]], val[f"{role}_set"])
            for role in ("selection", "reproduction")}
    grid = thresholds(config["readout"]["threshold_grid"])
    device = config["inference"]["device"]
    autocast = config["inference"]["dtype"] == "float16_autocast"

    results = {"implementation": implementation, "selection_set": val["selection_set"], "reproduction_set": val["reproduction_set"],
               "n_selection": len(sets["selection"]), "n_reproduction": len(sets["reproduction"]), "models": {}}
    for key in keys:
        r = reproduce(implementation, models[key], sets, grid, device, autocast)
        r["target"] = targets[key]
        r["difference"] = {k: round(r[k] - v, 3) for k, v in targets[key].items()}
        results["models"][key] = r
        print(key, json.dumps({k: v for k, v in r.items() if k != "per_structure"}), flush=True)

    out = repo_path(paths["results_dir"]) / f"09_wang_reproduction_{implementation}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    write_meta(out, config, packages=["torch", "transformers", "multimolecule", "modelgenerator"])


if __name__ == "__main__":
    main()
