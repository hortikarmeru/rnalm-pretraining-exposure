"""Check that each model returns full attention maps within GPU memory at the longest allowed input.

Usage: python scripts/08_check_model_memory.py <implementation> [model key ...]
  implementation: a key of inference.implementations (multimolecule, genbio, rnakm), run in its own env;
  default: every model of that implementation.

For each model, runs one random sequence of models.rna_fm.max_input_length nucleotides (seeded,
not from any test set) with all attention maps requested, under the configured dtype, and records
the peak GPU memory, the attention tensor shape and the time taken. A model that fails (for
example out of memory) is recorded with its error instead of stopping the run.

Writes results/08_model_memory_<implementation>.json plus its .meta.json.
"""
import json
import sys
import time

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.models import attention_maps, load


def check(implementation, model_id, sequence, device, autocast):
    model, tokenizer = load(implementation, model_id, device)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    attentions = attention_maps(model, tokenizer, sequence, device, autocast)
    torch.cuda.synchronize()
    result = {
        "model": model_id,
        "parameters": sum(p.numel() for p in model.parameters()),
        "layers": attentions.shape[0],
        "attention_shape": list(attentions.shape),   # [layers, heads, tokens, tokens]
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "seconds": round(time.time() - start, 2),
    }
    del model, attentions
    torch.cuda.empty_cache()
    return result


def main():
    config = load_config()
    check_cfg = config["inference"]["memory_check"]
    implementation = sys.argv[1]
    models = config["inference"]["implementations"][implementation]
    keys = sys.argv[2:] or list(models)
    length = config["models"]["rna_fm"]["max_input_length"]
    rng = np.random.default_rng(check_cfg["seed"])
    sequence = "".join(rng.choice(list("ACGU"), size=length))
    device = config["inference"]["device"]
    autocast = config["inference"]["dtype"] == "float16_autocast"

    results = {"implementation": implementation, "sequence_length": length, "gpu": torch.cuda.get_device_name(0),
               "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20), "models": {}}
    for key in keys:
        try:
            results["models"][key] = check(implementation, models[key], sequence, device, autocast)
        except Exception as e:   # recorded, so one failing model doesn't hide the others
            results["models"][key] = {"model": models[key], "error": repr(e)[:500]}
            torch.cuda.empty_cache()
        print(key, json.dumps(results["models"][key]), flush=True)

    out = repo_path(config["paths"]["results_dir"]) / f"08_model_memory_{implementation}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    write_meta(out, config, packages=["torch", "transformers", "multimolecule", "modelgenerator"])


if __name__ == "__main__":
    main()
