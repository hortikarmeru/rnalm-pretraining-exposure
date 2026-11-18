"""Score every test sequence with each model's frozen heads (paths.frozen_heads, script 17).

Usage: python scripts/18_score_test_sets.py <implementation> <model key> [<model key> ...] (--test | --validation)
  implementation: a key of inference.implementations, run in its own env (as scripts 09 and 17).
  --test: score the test sets (one of the two flags is required, so test data is never scored by accident).
  --validation: score the head-selection pool instead of the test sets and check that the pooled F1
  of each frozen head on its selection subset equals the one script 17 recorded. This proves the
  script without touching test data.

For each sequence the attention maps are computed once; each readout (primary, one_pair_per_base,
apc_off) takes its frozen layer and head, applies its own post-processing, and predicts the pairs
with score >= its frozen threshold. Test sequences are scored over all j > i pairs against their
reference pairs; validation structures as in scripts 09 and 17 (residues with 3D structure only,
contact matrices as labels). F1 and MCC use Wang's formulas (readout.score_pairs).

The frozen heads' score maps are saved (float32, upper triangle) to
<scores_dir>/<model key>[_validation].npz, so readouts can be rescored later without the GPU.

Writes results/18_test_scores_<model key>.tsv (seq_id, readout, f1, mcc, tp, fp, fn) and
results/18_test_scores_<implementation>.json (per-model summary; models from earlier runs are kept) with its .meta.json; with
--validation, results/18_validation_check_<implementation>.json instead.
"""
import csv
import json
import sys

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.metrics import f1_mcc
from rnalm_exposure.models import SPECIAL_TOKENS, attention_maps, load
from rnalm_exposure.readout import contact_scores, one_pair_per_base, score_pairs

APC = {"primary": True, "one_pair_per_base": True, "apc_off": False}


def test_records(config):
    paths = config["paths"]
    with open(repo_path(paths["test_sets"].format(processed_dir=paths["processed_dir"]))) as f:
        for line in f:
            r = json.loads(line)
            L = len(r["sequence"])
            contacts = np.zeros((L, L), dtype=np.int64)
            for i, j in r["pairs"]:
                contacts[i, j] = contacts[j, i] = 1
            yield {"seq_id": r["seq_id"], "sequence": r["sequence"], "contacts": contacts, "keep": None}


def validation_records(config):
    import importlib.util
    spec = importlib.util.spec_from_file_location("select_heads", repo_path("scripts/17_select_heads.py"))
    select_heads = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(select_heads)
    pool, _ = select_heads.selection_pool(config)
    return pool


def score_model(implementation, model_id, heads, records, device, autocast):
    """Per-sequence scores for each readout, and each readout's map of its frozen head."""
    model, tokenizer = load(implementation, model_id, device)
    rows, maps = [], {}
    for record in records:
        attentions = attention_maps(model, tokenizer, record["sequence"], device, autocast)
        for v, h in heads.items():
            # only the frozen head's map: APC works per head, so this equals slicing the all-head scores,
            # without holding every head in float32 (AIDO.RNA 1.6B has 1,024 heads)
            m = contact_scores(attentions[h["layer"], h["head"]][None, None], *SPECIAL_TOKENS[implementation], apc=APC[v])[0, 0]
            if record["keep"] is not None:
                keep = torch.tensor(record["keep"], device=device)
                m = m[keep, :][:, keep]
            if v == "one_pair_per_base":
                m = one_pair_per_base(m[None])[0]
            r = score_pairs(m, record["contacts"], h["threshold"])
            rows.append({"seq_id": record["seq_id"], "readout": v, **r})
            i, j = np.triu_indices(m.shape[0], 1)
            maps[f"{record['seq_id']}|{v}"] = m.float().cpu().numpy()[i, j]
    del model
    torch.cuda.empty_cache()
    return rows, maps


def pooled_f1(rows, readout, members):
    tp = sum(r["tp"] for r in rows if r["readout"] == readout and r["seq_id"] in members)
    fp = sum(r["fp"] for r in rows if r["readout"] == readout and r["seq_id"] in members)
    fn = sum(r["fn"] for r in rows if r["readout"] == readout and r["seq_id"] in members)
    return float(f1_mcc(float(tp), float(fp), float(fn), 0.0)[0])


def main():
    config = load_config()
    paths = config["paths"]
    flags = {a for a in sys.argv[2:] if a.startswith("--")}
    if len(flags & {"--test", "--validation"}) != 1:
        raise SystemExit("pass exactly one of --test or --validation")
    validation = "--validation" in flags
    implementation, keys = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
    models = config["inference"]["implementations"][implementation]
    frozen = json.loads(repo_path(paths["frozen_heads"]).read_text())
    device = config["inference"]["device"]
    autocast = config["inference"]["dtype"] == "float16_autocast"
    scores_dir = repo_path(paths["scores_dir"].format(processed_dir=paths["processed_dir"]))
    scores_dir.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(paths["results_dir"])
    if validation:
        selection = json.loads((results_dir / f"17_head_selection_{implementation}.json").read_text())["models"]

    summary = {}
    for key in keys:
        records = validation_records(config) if validation else test_records(config)
        rows, maps = score_model(implementation, models[key], frozen[key], records, device, autocast)
        np.savez_compressed(scores_dir / f"{key}{'_validation' if validation else ''}.npz", **maps)
        if validation:   # pooled F1 on the selection subset must match script 17
            s = selection[key]
            selected_on = s.get("selected_on", "low_exposure")   # entries written before the DNA model was added lack it
            members = set(s["subsets"][selected_on])
            summary[key] = {}
            for v in frozen[key]:
                ours, theirs = pooled_f1(rows, v, members), s["chosen"][v][selected_on]["frozen_head_pooled_f1"]
                summary[key][v] = {"pooled_f1": ours, "script_17": theirs, "match": abs(ours - theirs) < 1e-9}
        else:
            with open(results_dir / f"18_test_scores_{key}.tsv", "w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["seq_id", "readout", "f1", "mcc", "tp", "fp", "fn"])
                for r in rows:
                    w.writerow([r["seq_id"], r["readout"], f"{r['f1']:.6f}", f"{r['mcc']:.6f}", r["tp"], r["fp"], r["fn"]])
            summary[key] = {v: {"n": sum(r["readout"] == v for r in rows),
                                "median_f1": float(np.median([r["f1"] for r in rows if r["readout"] == v]))}
                            for v in frozen[key]}
        print(key, json.dumps(summary[key]), flush=True)

    name = f"18_validation_check_{implementation}.json" if validation else f"18_test_scores_{implementation}.json"
    out = results_dir / name
    earlier = json.loads(out.read_text()) if out.exists() else {"frozen_heads": {}, "models": {}}   # keep models from earlier runs
    with open(out, "w") as f:
        json.dump({"implementation": implementation, "frozen_heads": {**earlier["frozen_heads"], **{k: frozen[k] for k in keys}},
                   "models": {**earlier["models"], **summary}}, f, indent=2)
    write_meta(out, config, packages=["torch", "transformers", "multimolecule", "modelgenerator"])
    if validation and not all(c["match"] for m in summary.values() for c in m.values()):
        raise RuntimeError("validation F1 differs from script 17")


if __name__ == "__main__":
    main()
