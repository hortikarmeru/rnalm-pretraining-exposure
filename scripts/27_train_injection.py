"""Continued masked-LM training of a released RiNALMo or RNA-FM model with one injection set (config
training_experiment.training; sets from script 26; a model in training_experiment.model_data reads its
replay and injection files from its own subdirectory).

Usage: python scripts/27_train_injection.py <model key> <arm> <seed> [--benchmark N]   (env rnalm-infer)
       python scripts/27_train_injection.py <model key> --save-initial
  arm: A or B (inject that set) or 0 (replay only). Runs of the same seed share the replay stream.
  --benchmark N: N optimizer steps, then report throughput and peak memory; nothing is saved.
  --save-initial: save the released model through the same save path as a checkpoint, to
  runs/<model>_initial/epoch_0, so script 28 can check that a saved checkpoint scores exactly like the
  released model.

The run sees tokens_per_run tokens. The injected set is shown injected_epochs times (each epoch in its
own shuffled order) and spread evenly through the replay stream: each item's place is the fraction of
its own stream before it, so injected epoch k ends k / epochs of the way through the run. Replay fills
the rest of the budget, in the order script 26 drew it. Sequences longer than crop + 2 tokens are cut
to a random window of crop + 2 tokens, as RiNALMo was trained.

Batching: the stream is cut into windows of 64 steps, and a window always ends where an injected epoch
in checkpoints_at_epochs ends (for the replay-only run, at the same fraction of the run), so a
checkpoint holds exactly the epochs before it. Inside a window, items are sorted by length and packed
into micro-batches of at most micro_batch_tokens padded tokens and max_attention_cells (batch x
length^2; training.memory_overrides lowers both per model), which are then shuffled. A step accumulates
micro-batches until tokens_per_step real tokens; a partial step carries over into the next window, and
at a checkpoint boundary a remainder under half a step joins the step before it. Masking (15%,
80/10/10, random tokens from A/C/G/U) and cropping are seeded by the micro-batch's index, so a resumed
run sees the same batches (dropout's random state is not restored).

Optimizer: AdamW (fused), linear warmup to learning_rate, then constant; gradient clipping; bf16
autocast with TF32 matmuls; SDPA attention. The loss is the cross-entropy summed over masked tokens
and divided by the step's masked-token count, so micro-batches weigh by their masked tokens.

Writes <work_dir>/runs/<model>_<arm>_s<seed>/: epoch_<k>/ (model and tokenizer, loadable by scripts
18 and 28), train_log.tsv (step, tokens, loss, lr, seconds), state.pt (for resuming; deleted at the
end), and results/27_training_<model>_<arm>_s<seed>.json with its .meta.json.
"""
import json
import math
import sys
import time

import numpy as np
import torch

from rnalm_exposure.config import load_config, repo_path
from rnalm_exposure.meta import write_meta

WINDOW_STEPS = 64


def read_fasta(path):
    seqs, cur = [], []
    for line in open(path):
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


class Tokens:
    """Token arrays of many sequences in one uint8 buffer (with <cls> and <eos>), so dataloader
    workers share it instead of copying millions of small objects."""

    def __init__(self, seqs, tokenizer):
        vocab = tokenizer.get_vocab()
        lut = np.full(256, vocab[tokenizer.unk_token], np.uint8)
        for tok, i in vocab.items():
            if len(tok) == 1:
                lut[ord(tok)] = i
        self.lengths = np.array([len(s) + 2 for s in seqs], np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        body = np.frombuffer("".join(f"\x01{s}\x02" for s in seqs).encode(), np.uint8)
        self.buf = lut[body]
        self.buf[body == 1], self.buf[body == 2] = tokenizer.cls_token_id, tokenizer.eos_token_id
        for s in seqs[:50] + seqs[-50:]:   # the tokenizer itself must agree
            if s and lut[np.frombuffer(s.encode(), np.uint8)].tolist() != tokenizer(s)["input_ids"][1:-1]:
                raise RuntimeError(f"token mismatch for {s[:40]}")

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        return self.buf[self.offsets[i]:self.offsets[i + 1]]


def stream(n_replay_len, inj_len, cfg, seed):
    """Items in training order as (source, index, position): 0 = replay, 1 = injected; position =
    fraction of its own stream before the item."""
    epochs = cfg["injected_epochs"]
    rng = np.random.default_rng([seed, 1])
    inj_order = np.concatenate([rng.permutation(len(inj_len)) for _ in range(epochs)]) if len(inj_len) else np.array([], int)
    inj_tokens = int(inj_len.sum()) * epochs
    budget = cfg["tokens_per_run"] - inj_tokens
    n_rep = int(np.searchsorted(np.cumsum(n_replay_len), budget)) + 1
    if n_rep > len(n_replay_len):
        raise RuntimeError("replay data too short for the token budget")
    rep_pos = np.concatenate([[0], np.cumsum(n_replay_len[:n_rep])[:-1]]) / budget
    items = [(p, 0, i) for i, p in enumerate(rep_pos)]
    if len(inj_order):
        inj_pos = np.concatenate([[0], np.cumsum(inj_len[inj_order])[:-1]]) / inj_tokens
        items += [(p, 1, int(i)) for p, i in zip(inj_pos, inj_order)]
    items.sort(key=lambda x: (x[0], x[1]))
    return [(s, i, p) for p, s, i in items], inj_tokens


def micro_batches(items, lengths, cfg, seed):
    """([step: [micro-batch: [item index]]], {last step before a checkpoint boundary: epoch})."""
    rng = np.random.default_rng([seed, 2])
    step_tokens, mb_tokens, cells = cfg["tokens_per_step"], cfg["micro_batch_tokens"], cfg["max_attention_cells"]
    epochs = cfg["injected_epochs"]
    bounds = sorted(cfg["checkpoints_at_epochs"])
    window = WINDOW_STEPS * step_tokens
    steps, save_at, start, group, real, since_bound = [], {}, 0, [], 0, 0
    while start < len(items):
        nxt = next((k for k in bounds if items[start][2] < k / epochs), None)
        tokens, end = 0, start
        while end < len(items) and tokens < window and (nxt is None or items[end][2] < nxt / epochs):
            tokens += lengths[end]
            end += 1
        idx = sorted(range(start, end), key=lambda k: lengths[k])
        mbs, cur = [], []
        for k in idx:   # ascending length, so the newest item is the longest
            L = lengths[k]
            if cur and ((len(cur) + 1) * L > mb_tokens or (len(cur) + 1) * L * L > cells):
                mbs.append(cur)
                cur = []
            cur.append(k)
        if cur:
            mbs.append(cur)
        rng.shuffle(mbs)
        for mb in mbs:
            group.append(mb)
            real += sum(lengths[k] for k in mb)
            if real >= step_tokens:
                steps.append(group)
                since_bound += 1
                group, real = [], 0
        at_bound = end == len(items) or (nxt is not None and items[end][2] >= nxt / epochs)
        if at_bound and group:   # flush, so the checkpoint holds exactly the items before the boundary
            if real < step_tokens / 2 and since_bound:
                steps[-1] = steps[-1] + group
            else:
                steps.append(group)
            group, real = [], 0
        if at_bound and nxt is not None and end < len(items) or end == len(items):
            save_at[len(steps) - 1] = nxt if nxt is not None else epochs
            since_bound = 0
        start = end
    return steps, save_at


class Batcher:
    def __init__(self, sources, items, mask_cfg, max_tokens, special, nucleotides, pad, mask_id, seed):
        self.sources, self.items, self.cfg, self.max_tokens = sources, items, mask_cfg, max_tokens
        self.special, self.nuc, self.pad, self.mask_id, self.seed = special, nucleotides, pad, mask_id, seed

    def __call__(self, mb_index, members):
        rng = np.random.default_rng([self.seed, 3, mb_index])
        toks = []
        for k in members:
            src, i = self.items[k][:2]
            a = self.sources[src][i]
            if len(a) > self.max_tokens:
                s = int(rng.integers(0, len(a) - self.max_tokens + 1))
                a = a[s:s + self.max_tokens]
            toks.append(a)
        L = max(len(a) for a in toks)
        ids = np.full((len(toks), L), self.pad, np.int64)
        for r, a in enumerate(toks):
            ids[r, :len(a)] = a
        attn = (ids != self.pad).astype(np.int64)
        can = ~np.isin(ids, self.special)
        chosen = can & (rng.random(ids.shape) < self.cfg["fraction"])
        labels = np.where(chosen, ids, -100)
        u = rng.random(ids.shape)
        ids = np.where(chosen & (u < self.cfg["mask_token"]), self.mask_id, ids)
        rand = chosen & (u >= self.cfg["mask_token"]) & (u < self.cfg["mask_token"] + self.cfg["random_token"])
        ids = np.where(rand, rng.choice(self.nuc, ids.shape), ids)
        return torch.from_numpy(ids), torch.from_numpy(attn), torch.from_numpy(labels)


class Steps(torch.utils.data.Dataset):
    """One item = one optimizer step's micro-batches, built in a worker."""

    def __init__(self, steps, batcher, start):
        self.steps, self.batcher, self.start = steps, batcher, start
        self.offsets = np.concatenate([[0], np.cumsum([len(g) for g in steps])])

    def __len__(self):
        return len(self.steps) - self.start

    def __getitem__(self, i):
        s = self.start + i
        return s, [self.batcher(int(self.offsets[s]) + j, mb) for j, mb in enumerate(self.steps[s])]


def main():
    config = load_config()
    paths, exp = config["paths"], config["training_experiment"]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    key = args[0]
    # per-model memory settings: they change how a step is split into forward passes, not its tokens
    cfg = {**exp["training"], **exp["training"].get("memory_overrides", {}).get(key, {})}
    work = repo_path(exp["work_dir"].format(processed_dir=paths["processed_dir"]))
    model_id = config["inference"]["implementations"]["multimolecule"][key]
    import multimolecule  # registers the models with transformers
    from multimolecule import RnaTokenizer
    tokenizer = RnaTokenizer.from_pretrained(model_id)
    model = getattr(multimolecule, exp["mlm_class"][key]).from_pretrained(
        model_id, attn_implementation="sdpa", **exp["training"].get("model_config_overrides", {}).get(key, {}))
    tied = model.lm_head.decoder.weight.data_ptr() == model.model.embeddings.word_embeddings.weight.data_ptr()
    print(f"MLM decoder tied to input embeddings: {tied}", flush=True)   # script 28's released loss is the functional check
    if "--save-initial" in sys.argv:
        out = work / "runs" / f"{key}_initial" / "epoch_0"
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"saved {out}", flush=True)
        return

    arm, seed = args[1], int(args[2])
    bench = int(sys.argv[sys.argv.index("--benchmark") + 1]) if "--benchmark" in sys.argv else 0
    if key not in exp["models"] or arm not in ("A", "B", "0") or seed not in cfg["seeds"]:
        raise SystemExit("model must be in training_experiment.models, arm A, B or 0, seed in training.seeds")
    run = work / "runs" / f"{key}_{arm}_s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    model = model.cuda()

    max_tokens = exp["replay"]["crop"] + 2
    data = work / exp["model_data"][key]["subdir"] if key in exp.get("model_data", {}) else work   # script 26 <model>
    replay = Tokens(read_fasta(data / f"replay_seed{seed}.fasta"), tokenizer)
    injected = Tokens(read_fasta(data / f"injected_{arm}.fasta"), tokenizer) if arm != "0" else Tokens([], tokenizer)
    items, inj_tokens = stream(np.minimum(replay.lengths, max_tokens), np.minimum(injected.lengths, max_tokens), cfg, seed)
    sources = {0: replay, 1: injected}
    lengths = [int(min(sources[s].lengths[i], max_tokens)) for s, i, _ in items]
    steps, save_at = micro_batches(items, lengths, cfg, seed)
    n_steps = len(steps)
    print(f"{key} {arm} s{seed}: {len(items)} items, {n_steps} steps, injected tokens {inj_tokens}, "
          f"checkpoints {save_at}", flush=True)

    vocab = tokenizer.get_vocab()
    special = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id]
    batcher = Batcher(sources, items, cfg["mask"], max_tokens, special, [vocab[c] for c in "ACGU"],
                      tokenizer.pad_token_id, tokenizer.mask_token_id, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"], fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup_steps"]))

    start, tokens_done, elapsed = 0, 0, 0.0
    state_path = run / "state.pt"
    if state_path.exists() and not bench:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, tokens_done, elapsed = st["step"] + 1, st["tokens"], st["seconds"]
        print(f"resumed at step {start}", flush=True)

    workers = cfg["dataloader_workers"]
    loader = torch.utils.data.DataLoader(Steps(steps, batcher, start), batch_size=None, num_workers=workers,
                                         **({"prefetch_factor": 4} if workers else {}))
    log_path = run / "train_log.tsv"
    if start == 0:
        log_path.write_text("step\ttokens\tloss\tlr\tseconds\n")
    else:   # drop rows logged after the saved state
        lines = log_path.read_text().splitlines()
        log_path.write_text("\n".join([lines[0]] + [ln for ln in lines[1:] if int(ln.split("\t")[0]) < start]) + "\n")
    log = open(log_path, "a")
    model.train()
    t0, last_state = time.time() - elapsed, time.time()
    torch.cuda.reset_peak_memory_stats()
    for s, mbs in loader:
        n_masked = sum(int((lab != -100).sum()) for _, _, lab in mbs)
        total = 0.0
        if n_masked:
            for ids, attn, lab in mbs:
                ids, attn, lab = ids.cuda(non_blocking=True), attn.cuda(non_blocking=True), lab.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=attn).logits
                loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), lab.view(-1),
                                                         ignore_index=-100, reduction="sum") / n_masked
                loss.backward()
                total += float(loss)
                tokens_done += int(attn.sum())
            if not math.isfinite(total):
                raise RuntimeError(f"step {s}: loss {total}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
        lr = sched.get_last_lr()[0]
        sched.step()
        if s % 20 == 0 or s == n_steps - 1:
            log.write(f"{s}\t{tokens_done}\t{total:.5f}\t{lr:.3e}\t{time.time() - t0:.1f}\n")
            log.flush()
        if bench and s - start + 1 >= bench:
            rate = tokens_done / (time.time() - t0)
            print(json.dumps({"steps": bench, "tokens_per_second": round(rate),
                              "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                              "estimated_hours": round(cfg["tokens_per_run"] / rate / 3600, 2)}), flush=True)
            return
        if s in save_at:
            out = run / f"epoch_{save_at[s]}"
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            print(f"step {s}: saved {out.name}", flush=True)
        if time.time() - last_state > 1800 or s in save_at:   # resumable state every 30 min (the PC has lost power before)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": s,
                        "tokens": tokens_done, "seconds": time.time() - t0}, state_path.with_suffix(".tmp"))
            state_path.with_suffix(".tmp").replace(state_path)
            last_state = time.time()
    log.close()
    out = repo_path(paths["results_dir"]) / f"27_training_{key}_{arm}_s{seed}.json"
    train_log = np.genfromtxt(log_path, names=True, delimiter="\t")
    out.write_text(json.dumps({
        "model": key, "arm": arm, "seed": seed, "steps": n_steps, "items": len(items), "tokens": tokens_done,
        "injected_sequences": len(injected), "injected_tokens": inj_tokens,
        "checkpoint_steps": {str(v): k for k, v in save_at.items()},
        "final_loss_last_100_steps": float(np.mean(train_log["loss"][-5:])),
        "hours": round((time.time() - t0) / 3600, 2)}, indent=2))
    write_meta(out, config, packages=["torch", "transformers", "multimolecule"])
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
