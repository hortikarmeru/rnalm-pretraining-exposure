"""Continued masked-LM training of a released RiNALMo model with one injection set (config
training_experiment.training; sets from script 26).

Usage: python scripts/27_train_injection.py <model key> <arm> <seed> [--benchmark N]   (env rnalm-infer)
  arm: A or B (inject that set) or 0 (replay only). Runs of the same seed share the replay stream.
  --benchmark N: N optimizer steps, then report throughput and peak memory; nothing is saved.

The run sees tokens_per_run tokens. The injected set is shown injected_epochs times (each epoch in its
own shuffled order) and spread evenly through the replay stream, so injected epoch k ends k / epochs
of the way through the run; the model is saved there for each k in checkpoints_at_epochs. Replay
fills the rest of the budget, in the order script 26 drew it. Sequences longer than crop + 2 tokens
are cut to a random window of crop + 2 tokens, as RiNALMo was trained.

Batching: the stream is cut into windows of 64 steps; inside a window, items are sorted by length and
packed into micro-batches of at most micro_batch_tokens padded tokens and max_attention_cells
(batch x length^2), which are then shuffled. A step accumulates micro-batches until tokens_per_step
real tokens. Masking (15%, 80/10/10, random tokens from A/C/G/U) and cropping are seeded by the
micro-batch's index, so a run resumed from its last state continues exactly.

Optimizer: AdamW (fused), linear warmup to learning_rate, then constant; gradient clipping; bf16
autocast with TF32 matmuls; SDPA attention. The loss is the cross-entropy summed over masked
tokens and divided by the step's masked-token count, so long and short micro-batches weigh by
their masked tokens.

Writes <work_dir>/runs/<model>_<arm>_s<seed>/: epoch_<k>/ (model and tokenizer, loadable by
scripts 18 and 28), train_log.tsv (step, tokens, loss, lr, seconds), state.pt (for resuming;
deleted at the end), and results/27_training_<model>_<arm>_s<seed>.json with its .meta.json.
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


def encode(seqs, tokenizer):
    """Token arrays (uint8) with <cls> and <eos>, via the tokenizer's own vocabulary."""
    vocab = tokenizer.get_vocab()
    lut = np.full(256, vocab[tokenizer.unk_token], np.uint8)
    for tok, i in vocab.items():
        if len(tok) == 1:
            lut[ord(tok)] = i
    out = []
    for s in seqs:
        a = np.empty(len(s) + 2, np.uint8)
        a[0], a[-1] = tokenizer.cls_token_id, tokenizer.eos_token_id
        a[1:-1] = lut[np.frombuffer(s.encode(), np.uint8)]
        out.append(a)
    return out


def stream(replay, injected, cfg, max_tokens, seed):
    """[(source, index)] in training order: 0 = replay, 1 = injected."""
    epochs = cfg["injected_epochs"]
    rng = np.random.default_rng([seed, 1])
    inj_len = np.array([min(len(a), max_tokens) for a in injected], dtype=np.int64)
    inj_order = np.concatenate([rng.permutation(len(injected)) for _ in range(epochs)]) if injected else np.array([], int)
    inj_tokens = int(inj_len.sum()) * epochs
    budget = cfg["tokens_per_run"] - inj_tokens
    rep_len = np.array([min(len(a), max_tokens) for a in replay], dtype=np.int64)
    n_rep = int(np.searchsorted(np.cumsum(rep_len), budget)) + 1
    if n_rep > len(replay):
        raise RuntimeError("replay data too short for the token budget")
    # position of each item = the fraction of its own stream used before it; merged, they interleave evenly
    rep_pos = np.concatenate([[0], np.cumsum(rep_len[:n_rep])[:-1]]) / budget
    items = [(p, 0, i) for p, i in zip(rep_pos, range(n_rep))]
    if injected:
        inj_cum = np.concatenate([[0], np.cumsum(inj_len[inj_order])[:-1]]) / inj_tokens
        items += [(p, 1, int(i)) for p, i in zip(inj_cum, inj_order)]
    items.sort(key=lambda x: (x[0], x[1]))
    return [(s, i) for _, s, i in items], inj_tokens


def micro_batches(items, lengths, cfg, seed):
    """Length-bucketed micro-batches ([items]), grouped into steps, window by window."""
    rng = np.random.default_rng([seed, 2])
    step_tokens, mb_tokens, cells = cfg["tokens_per_step"], cfg["micro_batch_tokens"], cfg["max_attention_cells"]
    window = WINDOW_STEPS * step_tokens
    steps, start = [], 0
    while start < len(items):
        tokens, end = 0, start
        while end < len(items) and tokens < window:
            tokens += lengths[end]
            end += 1
        idx = sorted(range(start, end), key=lambda k: lengths[k])
        mbs, cur, longest = [], [], 0
        for k in idx:
            L = max(longest, lengths[k])
            if cur and ((len(cur) + 1) * L > mb_tokens or (len(cur) + 1) * L * L > cells):
                mbs.append(cur)
                cur, L = [], lengths[k]
            cur.append(k)
            longest = L
        if cur:
            mbs.append(cur)
        rng.shuffle(mbs)
        group, real = [], 0
        for mb in mbs:
            group.append(mb)
            real += sum(lengths[k] for k in mb)
            if real >= step_tokens:
                steps.append(group)
                group, real = [], 0
        if group:
            steps.append(group)
        start = end
    return steps


class Batcher:
    def __init__(self, arrays, items, mask_cfg, max_tokens, special, nucleotides, pad, mask_id, seed):
        self.arrays, self.items, self.cfg, self.max_tokens = arrays, items, mask_cfg, max_tokens
        self.special, self.nuc, self.pad, self.mask_id, self.seed = special, nucleotides, pad, mask_id, seed

    def __call__(self, mb_index, members):
        rng = np.random.default_rng([self.seed, 3, mb_index])
        toks = []
        for k in members:
            src, i = self.items[k]
            a = self.arrays[src][i]
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


def main():
    config = load_config()
    paths, exp = config["paths"], config["training_experiment"]
    cfg = exp["training"]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    key, arm, seed = args[0], args[1], int(args[2])
    bench = int(sys.argv[sys.argv.index("--benchmark") + 1]) if "--benchmark" in sys.argv else 0
    if key not in exp["models"] or arm not in ("A", "B", "0") or seed not in cfg["seeds"]:
        raise SystemExit("model must be in training_experiment.models, arm A, B or 0, seed in training.seeds")
    work = repo_path(exp["work_dir"].format(processed_dir=paths["processed_dir"]))
    run = work / "runs" / f"{key}_{arm}_s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    model_id = config["inference"]["implementations"]["multimolecule"][key]

    import multimolecule  # noqa: F401  (registers RiNALMo with transformers)
    from multimolecule import RiNALMoForMaskedLM, RnaTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    tokenizer = RnaTokenizer.from_pretrained(model_id)
    model = RiNALMoForMaskedLM.from_pretrained(model_id, attn_implementation="sdpa").cuda()
    if model.lm_head.decoder.weight.data_ptr() != model.model.embeddings.word_embeddings.weight.data_ptr():
        raise RuntimeError("MLM decoder is not tied to the input embeddings")

    max_tokens = exp["replay"]["crop"] + 2
    replay = encode(read_fasta(work / f"replay_seed{seed}.fasta"), tokenizer)
    injected = encode(read_fasta(work / f"injected_{arm}.fasta"), tokenizer) if arm != "0" else []
    items, inj_tokens = stream(replay, injected, cfg, max_tokens, seed)
    arrays = {0: replay, 1: injected}
    lengths = [min(len(arrays[s][i]), max_tokens) for s, i in items]
    steps = micro_batches(items, lengths, cfg, seed)
    n_steps = len(steps)
    # the step after which injected epoch k is complete (by position in the stream)
    inj_seen = np.cumsum([sum(lengths[k] for mb in g for k in mb if items[k][0] == 1) for g in steps])
    epochs = cfg["injected_epochs"]
    save_at = {}
    for k in cfg["checkpoints_at_epochs"]:
        s = int(np.searchsorted(inj_seen, inj_tokens * k / epochs)) if injected else round(n_steps * k / epochs) - 1
        save_at[min(s, n_steps - 1)] = k
    print(f"{key} {arm} s{seed}: {len(items)} items, {n_steps} steps, injected tokens {inj_tokens}, "
          f"checkpoints at steps {sorted(save_at)}", flush=True)

    vocab = tokenizer.get_vocab()
    special = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id]
    batcher = Batcher(arrays, items, cfg["mask"], max_tokens, special, [vocab[c] for c in "ACGU"],
                      tokenizer.pad_token_id, tokenizer.mask_token_id, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"], fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup_steps"]))

    start = 0
    state_path = run / "state.pt"
    if state_path.exists() and not bench:
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["step"] + 1
        print(f"resumed at step {start}", flush=True)
    mb_offsets = np.concatenate([[0], np.cumsum([len(g) for g in steps])])

    class Steps(torch.utils.data.Dataset):   # one item = one step's micro-batches, built in a worker
        def __len__(self):
            return n_steps - start

        def __getitem__(self, i):
            s = start + i
            return s, [batcher(int(mb_offsets[s]) + j, mb) for j, mb in enumerate(steps[s])]

    loader = torch.utils.data.DataLoader(Steps(), batch_size=None, num_workers=cfg["dataloader_workers"],
                                         prefetch_factor=4, persistent_workers=False)
    log = open(run / "train_log.tsv", "a")
    if start == 0:
        log.write("step\ttokens\tloss\tlr\tseconds\n")
    model.train()
    t0, last_state, tokens_done = time.time(), time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    for s, mbs in loader:
        n_masked = sum(int((lab != -100).sum()) for _, _, lab in mbs)
        total = 0.0
        for ids, attn, lab in mbs:
            ids, attn, lab = ids.cuda(non_blocking=True), attn.cuda(non_blocking=True), lab.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=attn).logits
            loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), lab.view(-1),
                                                     ignore_index=-100, reduction="sum") / n_masked
            loss.backward()
            total += float(loss)
            tokens_done += int(attn.sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if s % 20 == 0 or s == n_steps - 1:
            log.write(f"{s}\t{tokens_done}\t{total:.5f}\t{sched.get_last_lr()[0]:.3e}\t{time.time() - t0:.1f}\n")
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
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": s},
                       state_path.with_suffix(".tmp"))
            state_path.with_suffix(".tmp").replace(state_path)
            last_state = time.time()
    log.close()
    state_path.unlink(missing_ok=True)
    out = repo_path(paths["results_dir"]) / f"27_training_{key}_{arm}_s{seed}.json"
    train_log = np.genfromtxt(run / "train_log.tsv", names=True, delimiter="\t")
    out.write_text(json.dumps({
        "model": key, "arm": arm, "seed": seed, "steps": n_steps, "items": len(items),
        "injected_sequences": len(injected), "injected_tokens": inj_tokens, "checkpoint_steps": {str(v): k for k, v in save_at.items()},
        "final_loss_last_100_steps": float(np.mean(train_log["loss"][-5:])),
        "hours": round(float(train_log["seconds"][-1]) / 3600, 2)}, indent=2))
    write_meta(out, config, packages=["torch", "transformers", "multimolecule"])


if __name__ == "__main__":
    main()
