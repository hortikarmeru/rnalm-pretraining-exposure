"""Verbatim span recovery of the released models on the matched seen and unseen sequences (scripts 30
and 34; config checks.verbatim_recovery).

Span fill. Each span of script 34 is masked in its own copy of the sequence (mask token at every span
position, the rest of the sequence visible). The model then fills the span greedily: every pass, the
masked position whose top letter has the highest probability is set to that letter and the sequence
goes through the model again, until the span is full (one pass per letter). This is the masked-model
counterpart of greedy decoding in Carlini et al. (2023): a span counts as recovered only if every
letter is right. The first pass alone (every position's top letter with the whole span masked, all
letters predicted at once) is recorded too. All spans of one sequence go through the model as
one batch (bf16 autocast). As in script 31, scores are renormalized among the four letters (A, C, G,
U; T for the AIDO models), so every model makes the same four-way choice.

Per-position scores. The positions of each sequence are split at random into tiled_masks disjoint
masks (seeded by random_seed and sequence id), one copy of the sequence per mask, so every position
gets one log-probability of its true letter with about the share of letters masked that the models
were trained with. Per sequence: the mean over all positions, and the mean of the lowest
min_k_fraction of them (Min-K% Prob, Shi et al. 2024, with masked instead of left-to-right
probabilities). Masked-model scoring one position at a time (Salazar et al. 2020) would need one pass
per letter.

Usage: python scripts/35_verbatim_recovery.py <implementation> <model key> [<model key> ...]
  multimolecule keys in env rnalm-infer, genbio keys in env rnalm-aido.

Writes results/35_verbatim_<model>.tsv (one row per span: exact, correct letters, and the same for
the first pass), results/35_verbatim_scores_<model>.tsv (one row per sequence: positions, mean_logp,
min_k_logp) and results/35_verbatim_<model>.json (rates per side and span length) plus its .meta.json.
"""
import csv
import json
import sys

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta
from rnalm_exposure.models import load_mlm
from rnalm_exposure.verbatim import fill_spans, min_k, tiled_masks


def predictor(model, letters, device):
    """[rows, tokens] ids -> [rows, tokens - 2, 4] log-probabilities of the four letters."""
    def predict(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=torch.from_numpy(x).to(device)).logits[:, 1:-1, letters].float()
        return torch.log_softmax(logits, dim=-1).cpu().numpy()
    return predict


def encode(tokenizer, letters, seq_id, seq):
    """(token ids, index 0-3 of every letter)."""
    ids = tokenizer([seq], return_tensors="pt")["input_ids"][0].numpy()
    if len(ids) != len(seq) + 2:
        raise ValueError(f"{seq_id}: {len(seq)} nt gave {len(ids)} tokens")
    letter_of = {t: k for k, t in enumerate(letters)}
    return ids, np.array([letter_of[int(t)] for t in ids[1:-1]])   # also checks every position is one of the four letters


def position_scores(predict, ids, truth, masks, mask_id):
    """Log-probability of the true letter at every position, each masked in exactly one of masks."""
    x = np.tile(ids, (len(masks), 1))
    for b, p in enumerate(masks):
        x[b, p + 1] = mask_id
    logp = predict(x)
    out = np.empty(len(truth))
    for b, p in enumerate(masks):
        out[p] = logp[b, p, truth[p]]
    return out


def main():
    config = load_config()
    paths, cfg = config["paths"], config["checks"]["verbatim_recovery"]
    models = config["checks"]["masked_recovery"]["models"]
    implementation, keys = sys.argv[1], sys.argv[2:]
    device, seed = config["inference"]["device"], config["random_seed"]
    results = repo_path(paths["results_dir"])
    with open(results / "30_recovery_sequences.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    spans = {}
    with open(results / "34_verbatim_spans.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            spans.setdefault(r["seq_id"], []).append((int(r["span_length"]), int(r["span"]), int(r["start"])))
    for key in keys:
        if key not in models:
            raise ValueError(f"{key} is not in checks.masked_recovery.models")
        model, tokenizer, letters, mask_id = load_mlm(implementation, config["inference"]["implementations"][implementation][key], device)
        predict = predictor(model, letters, device)
        out, out_scores = results / f"35_verbatim_{key}.tsv", results / f"35_verbatim_scores_{key}.tsv"
        per_side, scores = {}, {}
        with open(out, "w", newline="") as f, open(out_scores, "w", newline="") as fs:
            w, ws = csv.writer(f, delimiter="\t"), csv.writer(fs, delimiter="\t")
            w.writerow(["seq_id", "span_length", "span", "start", "exact", "correct", "exact_one_pass", "correct_one_pass"])
            ws.writerow(["seq_id", "positions", "mean_logp", "min_k_logp"])
            for n, r in enumerate(rows):
                ids, truth = encode(tokenizer, letters, r["seq_id"], r["sequence"])
                mine = spans.get(r["seq_id"], [])
                filled, one_pass = fill_spans(predict, ids, [(start, k) for k, _, start in mine], letters, mask_id)
                for (k, j, start), a, b in zip(mine, filled, one_pass):
                    t = truth[start:start + k]
                    ca, cb = int((a == t).sum()), int((b == t).sum())
                    w.writerow([r["seq_id"], k, j, start, int(ca == k), ca, int(cb == k), cb])
                    s = per_side.setdefault(r["side"], {}).setdefault(k, [0, 0, 0, 0, 0])
                    for i, v in enumerate((1, ca == k, ca, cb == k, cb)):
                        s[i] += int(v)
                logp = position_scores(predict, ids, truth, tiled_masks(r["seq_id"], len(truth), cfg["tiled_masks"], seed), mask_id)
                mean, low = float(logp.mean()), min_k(logp, cfg["min_k_fraction"])
                ws.writerow([r["seq_id"], len(logp), f"{mean:.6f}", f"{low:.6f}"])
                scores.setdefault(r["side"], []).append((mean, low))
                f.flush()
                fs.flush()
                if n % 100 == 0:
                    print(key, n, len(rows), flush=True)
        summary = {"model": key}
        for side, by_k in per_side.items():
            summary[side] = {"span_lengths": {str(k): {"spans": s[0], "exact": s[1] / s[0], "letters_correct": s[2] / (k * s[0]),
                                                       "exact_one_pass": s[3] / s[0], "letters_correct_one_pass": s[4] / (k * s[0])}
                                              for k, s in sorted(by_k.items())},
                             "mean_logp": float(np.mean([m for m, _ in scores[side]])),      # mean over sequences
                             "min_k_logp": float(np.mean([l for _, l in scores[side]]))}
        print(json.dumps(summary), flush=True)
        js = results / f"35_verbatim_{key}.json"
        js.write_text(json.dumps(summary, indent=2))
        packages = ["torch", "transformers", "multimolecule"] if implementation == "multimolecule" else ["torch", "transformers", "modelgenerator"]
        write_meta(js, config, packages=packages)
        del model, predict
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
