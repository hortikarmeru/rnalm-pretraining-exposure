"""RNA-km (Gong et al. 2024) attention maps from the authors' released weights.

The architecture is written out here from the authors' code (github.com/gongtiansu/RNA-km, MIT
license, commit 189a1cd), so their weights load without running their scripts. Each class names the
file and lines it follows; module names match theirs, so the checkpoint's keys load unchanged. Only
what the attention maps need is kept: the token embedding, the relative position bias and the 12
encoder layers. The final layer norm and the pair head (layer_norm_after, fc_pair_*) come after the
last attention map and are left out; load_weights checks that nothing else in the checkpoint is unused.

The model takes one token per nucleotide and no BOS/EOS tokens, so its maps are [L, L].
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

BASE = "ATCGX"   # model/pretrain_model.py:11; token id = position in this string
D_MODEL = 1024   # model/pretrain_model.py:32
N_LAYER = 12     # model/pretrain_model.py:33
N_HEAD = D_MODEL // 64   # model/pretrain_model.py:37
VOCAB_SIZE = 8   # model/pretrain_model.py:27 (five letters, unknown, mask, padding)
PAD_IDX = 7      # model/pretrain_model.py:15
DROPOUT = 0.1    # model/pretrain_model.py:39; inactive in eval mode
UNUSED = ("layer_norm_after.", "fc_pair_q.", "fc_pair_k.", "fc_pair_rp.", "fc_pair_cls.")


def encode(sequence):
    """Token ids [1, L] as in the authors' RNA-km.py:24-29: uppercase, U -> T (the model was trained
    on RNAcentral FASTA, which writes T). Any other letter becomes X, as in Wang et al. 2026's script
    (the authors' own script maps it to N, which is not in the vocabulary)."""
    seq = "".join(c if c in "ACGT" else "X" for c in sequence.upper().replace("U", "T"))
    return torch.tensor([[BASE.index(c) for c in seq]], dtype=torch.long)


class RelativePositionBias(nn.Module):
    """model/positional_encoding.py:7-45: one learned bias per head for each bucket of the signed
    distance i - j (exact buckets for short distances, log-spaced up to max_distance, then one bucket)."""

    def __init__(self, num_buckets=64, max_distance=256, n_head=2):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, n_head)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance):
        num_buckets //= 2
        ret = (relative_position < 0).to(relative_position) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        val_if_large = max_exact + (
            torch.log(relative_position / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, relative_position):
        return self.relative_attention_bias(
            self._relative_position_bucket(relative_position, self.num_buckets, self.max_distance))


class MultiHeadAttention(nn.Module):
    """model/attention.py:6-52: scaled dot-product attention with an additive bias, no bias terms in
    the projections."""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.scale = (d_model // n_head) ** 0.5
        self.fc_query = nn.Linear(d_model, d_model, bias=False)
        self.fc_key = nn.Linear(d_model, d_model, bias=False)
        self.fc_value = nn.Linear(d_model, d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        return x.view(*x.shape[:-1], self.n_head, -1).transpose(-3, -2)   # [B, L, C] -> [B, heads, L, C / heads]

    def forward(self, x, bias):
        q, k, v = (self.split_heads(f(x)) for f in (self.fc_query, self.fc_key, self.fc_value))
        attn = torch.matmul(q / self.scale, k.transpose(-1, -2))
        attn += bias
        attn = self.dropout(F.softmax(attn, dim=-1))
        x = torch.matmul(attn, v).transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        return self.dropout(self.fc_out(x)), attn


class FeedForward(nn.Module):
    """model/attention.py:55-67."""

    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.ff(x)


class TransformerEncoderLayer(nn.Module):
    """model/attention.py:70-84: layer norm before attention and before the feed-forward block."""

    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)

    def forward(self, x, bias):
        branch, attn = self.attn(self.norm1(x), bias)
        x = x + branch
        return x + self.ffn(self.norm2(x)), attn


class TransformerEncoder(nn.Module):
    """model/attention.py:87-97."""

    def __init__(self, n_layer, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**kwargs) for _ in range(n_layer)])

    def forward(self, x, bias):
        maps = []
        for layer in self.layers:
            x, attn = layer(x, bias)
            maps.append(attn)
        return torch.stack(maps)


class RnaKm(nn.Module):
    """model/pretrain_model.py:29-98, up to the encoder. forward returns the attention probabilities,
    [layers, batch, heads, L, L]. Inputs are single unpadded sequences, so the padding mask of
    pretrain_model.py:87-90 (zero without padding) is left out."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Sequential(nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_IDX), nn.Dropout(DROPOUT))
        self.embed_rp = RelativePositionBias(num_buckets=64, max_distance=256, n_head=N_HEAD)
        self.encoder = TransformerEncoder(N_LAYER, d_model=D_MODEL, n_head=N_HEAD, dim_feedforward=D_MODEL * 4,
                                          dropout=DROPOUT)

    def forward(self, src):
        x = self.embed_tokens(src)
        pos = torch.arange(src.shape[1], device=src.device)
        bias = self.embed_rp(pos.unsqueeze(1) - pos.unsqueeze(0)).permute(2, 0, 1).unsqueeze(0)
        return self.encoder(x, bias)


def used_weights(state_dict):
    """The checkpoint entries RnaKm needs, with the "module." prefix of nn.DataParallel removed
    (RNA-km.py:13-17). Raises if an entry is neither needed nor part of the left-out pair head."""
    weights = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    used = {k: v for k, v in weights.items() if not k.startswith(UNUSED)}
    unknown = sorted(k for k in used if not k.startswith(("embed_tokens.", "embed_rp.", "encoder.")))
    if unknown:
        raise ValueError(f"unexpected weights in the RNA-km checkpoint: {unknown}")
    return used


def load_weights(path):
    """RnaKm with the authors' weights. weights_only=True: the file is read as tensors, not as a pickle
    that could run code."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = RnaKm()
    model.load_state_dict(used_weights(checkpoint["state_dict"]))   # strict: every model weight must be in the file
    return model
