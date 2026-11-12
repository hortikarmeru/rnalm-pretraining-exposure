"""Masked-letter recovery of the released models on the matched seen and unseen sequences (script 30;
config checks.masked_recovery).

Each sequence gets n_masks masks: every position is masked with probability mask_fraction, all with
the mask token, seeded by (random_seed, sequence id, mask number), so every model sees the same masks.
The masks of one sequence go through the model as one batch (bf16 autocast). At masked positions the
model's scores for the four letters (A, C, G, U; T for the AIDO models) are renormalized among
themselves, so models with different vocabularies are compared on the same four-way choice. Per
sequence: masked positions, correct predictions (top letter), and summed cross-entropy (nats).

Usage: python scripts/31_masked_recovery.py <implementation> <model key> [<model key> ...]
  multimolecule keys in env rnalm-infer, genbio keys in env rnalm-aido.

Writes results/31_recovery_<model>.tsv (seq_id, masked, correct, loss_sum) and
results/31_recovery_<model>.json (accuracy and mean loss per side) plus its .meta.json.
"""
import csv
import json
import sys
import zlib

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.models import load_mlm


def recover(model, tokenizer, letters, mask_id, seq_id, seq, cfg, seed, device):
    ids = tokenizer([seq], return_tensors="pt")["input_ids"][0]
    if len(ids) != len(seq) + 2:
        raise ValueError(f"{seq_id}: {len(seq)} nt gave {len(ids)} tokens")
    letter_of = {t: k for k, t in enumerate(letters)}
    truth = torch.tensor([letter_of[int(t)] for t in ids[1:-1]])   # also checks every position is one of the four letters
    k = zlib.crc32(seq_id.encode())
    masks = []
    for m in range(cfg["n_masks"]):
        rng = np.random.default_rng([seed, k, m])
        masks.append(torch.from_numpy(np.flatnonzero(rng.random(len(seq)) < cfg["mask_fraction"])))
    masks = [p for p in masks if len(p)]
    x = ids.repeat(len(masks), 1)
    for b, p in enumerate(masks):
        x[b, p + 1] = mask_id
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x.to(device)).logits[:, 1:-1, letters].float()
    logp = torch.log_softmax(logits, dim=-1).cpu()
    masked = correct = 0
    loss = 0.0
    for b, p in enumerate(masks):
        lp, t = logp[b, p], truth[p]
        masked += len(p)
        correct += int((lp.argmax(-1) == t).sum())
        loss -= float(lp.gather(1, t[:, None]).sum())
    return masked, correct, loss


def main():
    config = load_config()
    paths, cfg = config["paths"], config["checks"]["masked_recovery"]
    implementation, keys = sys.argv[1], sys.argv[2:]
    device = config["inference"]["device"]
    results = repo_path(paths["results_dir"])
    with open(results / "30_recovery_sequences.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for key in keys:
        if key not in cfg["models"]:
            raise ValueError(f"{key} is not in checks.masked_recovery.models")
        model, tokenizer, letters, mask_id = load_mlm(implementation, config["inference"]["implementations"][implementation][key], device)
        out = results / f"31_recovery_{key}.tsv"
        per_side = {}
        with open(out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["seq_id", "masked", "correct", "loss_sum"])
            for n, r in enumerate(rows):
                masked, correct, loss = recover(model, tokenizer, letters, mask_id, r["seq_id"], r["sequence"], cfg,
                                                config["random_seed"], device)
                w.writerow([r["seq_id"], masked, correct, f"{loss:.6f}"])
                s = per_side.setdefault(r["side"], [0, 0, 0.0])
                s[0] += masked
                s[1] += correct
                s[2] += loss
                if n % 500 == 0:
                    print(key, n, len(rows), flush=True)
        summary = {"model": key, **{side: {"masked": m, "accuracy": c / m, "mean_loss": l / m}
                                    for side, (m, c, l) in per_side.items()}}
        print(json.dumps(summary), flush=True)
        js = results / f"31_recovery_{key}.json"
        js.write_text(json.dumps(summary, indent=2))
        packages = ["torch", "transformers", "multimolecule"] if implementation == "multimolecule" else ["torch", "transformers", "modelgenerator"]
        write_meta(js, config, packages=packages)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
