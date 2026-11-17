"""Score a trained checkpoint (script 27) or the released model for the training experiment.

Usage: python scripts/28_score_checkpoints.py <model key> (released | initial | <arm> <seed> <epoch>)   (env rnalm-infer)
  released: the hub model; run first. initial: the released model saved through script 27's checkpoint
  path (--save-initial); its scores and losses must equal the released ones.

On the experiment's eval sequences (results/26_eval_sequences.tsv) and the sentinels (selection pool
and results/26_sentinels.tsv):
  - F1 with the model's frozen heads from frozen_heads.json, with script 18's scoring code
    (score_model; same post-processing, fp16 autocast, one sequence at a time).
  - Masked-token cross-entropy (manipulation check): eval sequences with n_masks fixed masks each
    (15%, all [MASK]; seeded by the sequence id, so every model sees the same masks), and the
    held-out corpus sequences (script 26) with one mask each, cropped to the first crop + 2 tokens.
For "released", the eval and sentinel test F1 must equal results/18_test_scores_<model>.tsv, which
checks that this path scores exactly as script 18 did.

Each checkpoint is compared with the released model: abort (exit code 2) if the held-out loss
rises by more than abort.held_out_loss_rise or the sentinel mean F1 (primary readout) drops by more
than abort.sentinel_f1_drop.

Writes results/28_scores_<model>_<tag>.tsv (seq_id, role, readout, f1, mcc, tp, fp, fn),
results/28_mlm_<model>_<tag>.tsv (seq_id, role, loss, masked) and results/28_summary_<model>_<tag>.json
with its .meta.json; tag = released or <arm>_s<seed>_e<epoch>.
"""
import csv
import importlib.util
import math
import zlib
import json
import sys

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta

N_MASKS = 8


def script_18():
    spec = importlib.util.spec_from_file_location("score_test_sets", repo_path("scripts/18_score_test_sets.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_fasta(path):
    out, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            out[name] = ""
        else:
            out[name] += line
    return out


def masked_loss(model, tokenizer, seqs, n_masks, fraction, max_tokens, seed):
    """[(id, mean cross-entropy over masked tokens, masked count)], masks fixed by (seed, sequence id, mask)."""
    rows = []
    for sid, s in seqs:
        k = zlib.crc32(sid.encode())
        ids = tokenizer([s], return_tensors="pt")["input_ids"][0][:max_tokens]
        loss_sum, count = 0.0, 0
        for m in range(n_masks):
            rng = np.random.default_rng([seed, k, m])
            pos = np.flatnonzero(rng.random(len(ids) - 2) < fraction) + 1   # never <cls> / <eos>
            if len(pos) == 0:
                continue
            x = ids.clone()
            x[pos] = tokenizer.mask_token_id
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x[None].cuda()).logits[0, pos].float()
            loss_sum += float(torch.nn.functional.cross_entropy(logits, ids[pos].cuda(), reduction="sum"))
            count += len(pos)
        rows.append((sid, loss_sum / max(count, 1), count))
    return rows


def main():
    config = load_config()
    paths, exp = config["paths"], config["training_experiment"]
    results = repo_path(paths["results_dir"])
    work = repo_path(exp["work_dir"].format(processed_dir=paths["processed_dir"]))
    key = sys.argv[1]
    if sys.argv[2] == "released":
        tag, model_path = "released", config["inference"]["implementations"]["multimolecule"][key]
    elif sys.argv[2] == "initial":
        tag, model_path = "initial", str(work / "runs" / f"{key}_initial" / "epoch_0")
    else:
        arm, seed, epoch = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        tag = f"{arm}_s{seed}_e{epoch}"
        model_path = str(work / "runs" / f"{key}_{arm}_s{seed}" / f"epoch_{epoch}")
    s18 = script_18()
    frozen = json.loads(repo_path(paths["frozen_heads"]).read_text())[key]
    device = config["inference"]["device"]
    autocast = config["inference"]["dtype"] == "float16_autocast"

    eval_rows = read_tsv(results / "26_eval_sequences.tsv")
    roles = {r["seq_id"]: "eval" for r in eval_rows}
    roles.update({r["seq_id"]: "sentinel_test" for r in read_tsv(results / "26_sentinels.tsv")})
    tests = [r for r in s18.test_records(config) if r["seq_id"] in roles]
    pool = s18.validation_records(config)
    for r in pool:
        roles[r["seq_id"]] = "sentinel_pool"
    rows, _ = s18.score_model("multimolecule", model_path, frozen, tests + pool, device, autocast)

    if tag == "released":   # this path must reproduce script 18 exactly
        ref = {(r["seq_id"], r["readout"]): r for r in read_tsv(results / f"18_test_scores_{key}.tsv")}
        matched = [r for r in rows if (r["seq_id"], r["readout"]) in ref]
        bad = [r["seq_id"] for r in matched
               if (int(ref[(r["seq_id"], r["readout"])]["tp"]), int(ref[(r["seq_id"], r["readout"])]["fp"])) != (r["tp"], r["fp"])]
        if len(matched) != 3 * len(tests) or bad:
            raise RuntimeError(f"{len(matched)} rows compared for {len(tests)} sequences; {len(bad)} differ, e.g. {bad[:3]}")

    import multimolecule
    from multimolecule import RnaTokenizer
    tokenizer = RnaTokenizer.from_pretrained(model_path)
    mlm = getattr(multimolecule, exp["mlm_class"][key]).from_pretrained(model_path).to(device).eval()
    seq_of = {r["seq_id"]: r["sequence"] for r in tests}
    max_tokens = exp["replay"]["crop"] + 2
    fraction = exp["training"]["mask"]["fraction"]
    seed = exp["seed"]
    losses = [(*r, "eval") for r in masked_loss(mlm, tokenizer, [(r["seq_id"], seq_of[r["seq_id"]]) for r in eval_rows],
                                                N_MASKS, fraction, max_tokens, seed)]
    data = work / exp["model_data"][key]["subdir"] if key in exp.get("model_data", {}) else work   # script 26 <model>
    held = sorted(read_fasta(data / "held_out.fasta").items())
    losses += [(*r, "held_out") for r in masked_loss(mlm, tokenizer, held, 1, fraction, max_tokens, seed)]

    with open(results / f"28_scores_{key}_{tag}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seq_id", "role", "readout", "f1", "mcc", "tp", "fp", "fn"])
        for r in rows:
            w.writerow([r["seq_id"], roles[r["seq_id"]], r["readout"], f"{r['f1']:.6f}", f"{r['mcc']:.6f}", r["tp"], r["fp"], r["fn"]])
    with open(results / f"28_mlm_{key}_{tag}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seq_id", "role", "loss", "masked"])
        for sid, loss, n, role in losses:
            w.writerow([sid, role, f"{loss:.6f}", n])

    def mean_f1(role_prefix, rs):
        return float(np.mean([r["f1"] for r in rs if r["readout"] == "primary" and roles[r["seq_id"]].startswith(role_prefix)]))

    held_loss = float(np.sum([l * n for _, l, n, role in losses if role == "held_out"]) /
                      np.sum([n for _, _, n, role in losses if role == "held_out"]))
    summary = {"model": key, "checkpoint": tag, "eval_mean_f1": mean_f1("eval", rows),
               "sentinel_mean_f1": mean_f1("sentinel", rows), "held_out_loss": held_loss,
               "eval_loss": float(np.mean([l for _, l, _, role in losses if role == "eval"]))}
    if tag == "released" and held_loss >= math.log(4) - 0.1:   # the MLM head must predict well above chance (ln 4)
        raise RuntimeError(f"released model's held-out loss {held_loss:.3f} is near chance: MLM head not loaded correctly")
    if tag == "initial":   # the checkpoint save/load path must change nothing
        ref = {(r["seq_id"], r["readout"]): (int(r["tp"]), int(r["fp"])) for r in read_tsv(results / f"28_scores_{key}_released.tsv")}
        ref_loss = {(r["seq_id"], r["role"]): float(r["loss"]) for r in read_tsv(results / f"28_mlm_{key}_released.tsv")}
        bad = [r["seq_id"] for r in rows if ref[(r["seq_id"], r["readout"])] != (r["tp"], r["fp"])]
        worst = max(abs(l - ref_loss[(sid, role)]) for sid, l, _, role in losses)
        summary["max_loss_difference"] = worst
        if bad or worst > 1e-4:
            raise RuntimeError(f"saved checkpoint differs from the released model: {len(bad)} F1 rows, loss {worst:.2e}")
    elif tag != "released":
        base = json.loads((results / f"28_summary_{key}_released.json").read_text())
        ab = exp["abort"]
        summary["held_out_loss_rise"] = held_loss - base["held_out_loss"]
        summary["sentinel_f1_drop"] = base["sentinel_mean_f1"] - summary["sentinel_mean_f1"]
        summary["abort"] = bool(summary["held_out_loss_rise"] > ab["held_out_loss_rise"]
                                or summary["sentinel_f1_drop"] > ab["sentinel_f1_drop"])
    out = results / f"28_summary_{key}_{tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    write_meta(out, config, packages=["torch", "transformers", "multimolecule"])
    if summary.get("abort"):
        print(f"ABORT: {key} {tag} drifted past the preset limits", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
