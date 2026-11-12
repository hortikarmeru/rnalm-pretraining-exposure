"""Head specialization by exposure (config analysis.head_specialization): in larger models, does a
different attention head carry base-pairing information for heavily exposed RNAs?

Usage: python scripts/25_head_specialization.py <implementation> <model key> [<model key> ...]
  (GPU; after scripts 13, 17 and 18; each implementation in its own env, as script 18)

A. Every head, at its own best threshold, scored by pooled F1 on the test sequences of one Rfam family
   (family), split into its most-exposed members (homologs >= top_homologs) and the rest; reported with
   the frozen primary head's F1 at its frozen threshold.
B. Every test sequence scored with a second head: the one chosen on the high-exposure half of the
   validation pool (script 17, alternative_head). Per-sequence F1 is written so it can be compared with
   the frozen head (script 18) by exposure.

Writes results/25_head_specialization_<implementation>.json (part A, per model) and
results/25_alternative_head_<model key>.tsv (part B: seq_id, f1) with the .meta.json.
"""
import csv
import json
import sys

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.metrics import f1_mcc
from rnalm_exposure.models import attention_maps, load
from rnalm_exposure.readout import contact_scores, score_pairs, select_head, threshold_counts, thresholds, upper


def contacts(record):
    L = len(record["sequence"])
    c = np.zeros((L, L), dtype=np.int64)
    for i, j in record["pairs"]:
        c[i, j] = c[j, i] = 1
    return c


def all_heads(model, tokenizer, records, grid, grid_t, device, autocast, frozen):
    """Pooled F1 of every head at its best threshold, and of the frozen head at its frozen threshold."""
    tp = pp = None
    n_true = n_total = 0
    for r in records:
        scores = contact_scores(attention_maps(model, tokenizer, r["sequence"], device, autocast))
        layers, heads = scores.shape[:2]
        s, t = upper(scores.flatten(0, 1), contacts(r))
        a, b = threshold_counts(s, t, grid_t)
        tp, pp = (a, b) if tp is None else (tp + a, pp + b)
        n_true += int(t.sum())
        n_total += t.numel()
    tp, pp = tp.view(layers, heads, -1).cpu().numpy(), pp.view(layers, heads, -1).cpu().numpy()
    best = select_head(tp, pp, n_true, n_total, grid)
    m = int(np.searchsorted(grid, frozen["threshold"]))
    tf, pf = float(tp[frozen["layer"], frozen["head"], m]), float(pp[frozen["layer"], frozen["head"], m])
    frozen_f1 = float(f1_mcc(tf, pf - tf, n_true - tf, 0.0)[0])
    return {"n": len(records), "frozen_f1": frozen_f1,
            "frozen_head_best_threshold_f1": float(best["max_f1"][frozen["layer"], frozen["head"]]),
            "best_head": {"layer": best["layer"], "head": best["head"], "threshold": best["threshold"], "f1": best["pooled_f1"]},
            "heads_above_frozen": int((best["max_f1"] > frozen_f1).sum())}


def main():
    config = load_config()
    paths, cfg = config["paths"], config["analysis"]["head_specialization"]
    implementation, keys = sys.argv[1], sys.argv[2:]
    models = config["inference"]["implementations"][implementation]
    device, autocast = config["inference"]["device"], config["inference"]["dtype"] == "float16_autocast"
    grid = thresholds(config["readout"]["threshold_grid"])
    grid_t = torch.tensor(grid, dtype=torch.float64, device=device)
    results_dir = repo_path(paths["results_dir"])
    frozen = json.loads(repo_path(paths["frozen_heads"]).read_text())
    selection = json.loads((results_dir / f"17_head_selection_{implementation}.json").read_text())["models"]
    with open(results_dir / "15_families.tsv") as f:
        family = {r["seq_id"]: r["family_name"] for r in csv.DictReader(f, delimiter="\t")}
    with open(repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))) as f:
        tests = [json.loads(line) for line in f]

    out = {}
    for key in keys:
        corpus = config["validation"]["exposure_corpus"][key]
        with open(results_dir / f"13_exposure_{corpus}.tsv") as f:
            homologs = {r["seq_id"]: int(r["homologs"]) for r in csv.DictReader(f, delimiter="\t")}
        fam = [r for r in tests if family[r["seq_id"]] == cfg["family"]]
        groups = {"most_exposed": [r for r in fam if homologs[r["seq_id"]] >= cfg["top_homologs"]],
                  "rest": [r for r in fam if homologs[r["seq_id"]] < cfg["top_homologs"]]}
        model, tokenizer = load(implementation, models[key], device)
        alt = selection[key]["chosen"]["primary"][cfg["alternative_head"]]
        out[key] = {"family": cfg["family"], "top_homologs": cfg["top_homologs"],
                    "frozen_head": frozen[key]["primary"],
                    "alternative_head": {k: alt[k] for k in ("layer", "head", "threshold")},
                    "groups": {g: all_heads(model, tokenizer, recs, grid, grid_t, device, autocast, frozen[key]["primary"])
                               for g, recs in groups.items()}}
        print(key, json.dumps(out[key]["groups"]), flush=True)
        with open(results_dir / f"25_alternative_head_{key}.tsv", "w") as f:
            f.write("seq_id\tf1\n")
            for r in tests:
                maps = contact_scores(attention_maps(model, tokenizer, r["sequence"], device, autocast)[alt["layer"], alt["head"]][None, None])
                f.write(f"{r['seq_id']}\t{score_pairs(maps[0, 0], contacts(r), alt['threshold'])['f1']:.6f}\n")
        del model
        torch.cuda.empty_cache()

    out_path = results_dir / f"25_head_specialization_{implementation}.json"
    earlier = json.loads(out_path.read_text()) if out_path.exists() else {}
    with open(out_path, "w") as f:
        json.dump({**earlier, **out}, f, indent=2)
    write_meta(out_path, config, packages=["torch", "transformers", "multimolecule", "modelgenerator"])


if __name__ == "__main__":
    main()
