"""Check that each model returns full attention maps within GPU memory at the longest allowed input.

Usage: python scripts/08_check_model_memory.py [model key ...]   (default: every model in
inference.memory_check.models)

For each model, runs one random sequence of models.rna_fm.max_input_length nucleotides (seeded,
not from any test set) with all attention maps requested, under the configured dtype, and records
the peak GPU memory, the attention tensor shape and the time taken. A model that fails (for
example out of memory) is recorded with its error instead of stopping the run.

Writes results/08_model_memory.json plus its .meta.json.
"""
import json
import sys
import time

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta


def check(model_id, sequence, device, autocast):
    import multimolecule  # noqa: F401  (registers the RNA models with transformers)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    inputs = tokenizer([sequence], return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        out = model(**inputs, output_attentions=True)
    torch.cuda.synchronize()
    attentions = out.attentions
    result = {
        "model": model_id,
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokens": int(inputs["input_ids"].shape[-1]),
        "layers_returned": len(attentions) if attentions is not None else 0,
        "attention_shape": list(attentions[0].shape) if attentions else None,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "seconds": round(time.time() - start, 2),
    }
    del model, out, attentions
    torch.cuda.empty_cache()
    return result


def main():
    config = load_config()
    check_cfg = config["inference"]["memory_check"]
    models = check_cfg["models"]
    keys = sys.argv[1:] or list(models)
    length = config["models"]["rna_fm"]["max_input_length"]
    rng = np.random.default_rng(check_cfg["seed"])
    sequence = "".join(rng.choice(list("ACGU"), size=length))
    device = config["inference"]["device"]
    autocast = config["inference"]["dtype"] == "float16_autocast"

    results = {"sequence_length": length, "gpu": torch.cuda.get_device_name(0),
               "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20), "models": {}}
    for key in keys:
        try:
            results["models"][key] = check(models[key], sequence, device, autocast)
        except Exception as e:   # recorded, so one failing model doesn't hide the others
            results["models"][key] = {"model": models[key], "error": repr(e)[:500]}
            torch.cuda.empty_cache()
        print(key, json.dumps(results["models"][key]), flush=True)

    out = repo_path(config["paths"]["results_dir"]) / "08_model_memory.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    write_meta(out, config, packages=["torch", "transformers", "multimolecule"])


if __name__ == "__main__":
    main()
