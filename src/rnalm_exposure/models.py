"""Load the RNA language models and get their attention maps.

Three implementations; the first two need their own conda envs because their transformers versions conflict:
  multimolecule  Hugging Face ports of RNA-FM, RiNALMo, RNA-MSM and ERNIE-RNA (transformers 5)
  genbio         GB.RNA (formerly AIDO.RNA) with its authors' model code from modelgenerator (transformers 4.38)
  rnakm          RNA-km: the authors' weights in the architecture written out in rnalm_exposure.rnakm (torch only)
"""
import os

import torch

from rnalm_exposure import rnakm
from rnalm_exposure.config import repo_path

# (tokens before, tokens after) the nucleotides in each implementation's attention maps: BOS/EOS, or none
SPECIAL_TOKENS = {"multimolecule": (1, 1), "genbio": (1, 1), "rnakm": (0, 0)}


class _DnaAlphabet:
    """GB.RNA was trained on RNAcentral FASTA, which writes T, so U is given as T (as in Wang et al. 2026)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences, **kwargs):
        return self.tokenizer([s.replace("U", "T") for s in sequences], **kwargs)


def load(implementation, model_id, device):
    if implementation == "multimolecule":
        import multimolecule  # noqa: F401  (registers the RNA models with transformers)
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    elif implementation == "genbio":
        import modelgenerator
        from modelgenerator.huggingface_models.rnabert import RNABertModel, RNABertTokenizer

        vocab = os.path.join(os.path.dirname(modelgenerator.__file__), "huggingface_models/rnabert/vocab.txt")
        tokenizer = _DnaAlphabet(RNABertTokenizer(vocab, version="v2"))   # version v2 adds [CLS] ... [SEP]
        model = RNABertModel.from_pretrained(model_id)
    elif implementation == "rnakm":
        model, tokenizer = rnakm.load_weights(repo_path(model_id)), rnakm.encode   # model_id: the weight file
    else:
        raise ValueError(f"unknown implementation {implementation}")
    return model.to(device).eval(), tokenizer


def load_mlm(implementation, model_id, device):
    """(model with its masked-LM head, tokenizer, ids of the four letters, mask id) for the masked-letter check."""
    if implementation == "multimolecule":
        import multimolecule  # noqa: F401
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model, info = AutoModelForMaskedLM.from_pretrained(model_id, output_loading_info=True)
        inner, letters = tokenizer, "ACGU"
    elif implementation == "genbio":
        import modelgenerator
        from modelgenerator.huggingface_models.rnabert import RNABertForMaskedLM, RNABertTokenizer

        vocab = os.path.join(os.path.dirname(modelgenerator.__file__), "huggingface_models/rnabert/vocab.txt")
        inner = RNABertTokenizer(vocab, version="v2")
        tokenizer, letters = _DnaAlphabet(inner), "ACGT"
        model, info = RNABertForMaskedLM.from_pretrained(model_id, output_loading_info=True)
    else:
        raise ValueError(f"unknown implementation {implementation}")
    # RiNALMo's hub weights have no lm_head.decoder.bias (it loads as zeros: one constant per letter)
    missing = set(info["missing_keys"]) - {"lm_head.decoder.bias"}
    if missing:
        raise ValueError(f"{model_id}: weights missing from the checkpoint: {sorted(missing)}")
    ids = inner.convert_tokens_to_ids(list(letters))
    if len(set(ids)) != 4 or inner.unk_token_id in ids or inner.mask_token_id is None:
        raise ValueError(f"{model_id}: letters {letters} map to {ids}")
    return model.to(device).eval(), tokenizer, ids, inner.mask_token_id


def attention_maps(model, tokenizer, sequence, device, autocast=True):
    """[layers, heads, T, T] attention maps for one sequence. T = L + 2 with BOS/EOS at both ends, or
    L for RNA-km, which has no special tokens (SPECIAL_TOKENS).

    The maps are attention probabilities, except for ERNIE-RNA. There they are the 13 maps Wang et al.
    2026 scored with the authors' code: map 0 is the model's pairing prior (a fixed AU / CG / GU score
    for every pair of positions, passed through a small learned projection to one value per head), which
    is added to the first layer's attention before the softmax; maps 1-12 are each layer's attention
    before the softmax, which the model passes on as the next layer's prior."""
    if isinstance(model, rnakm.RnaKm):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            return model(tokenizer(sequence).to(device))[:, 0]
    inputs = tokenizer([sequence], return_tensors="pt").to(device)
    if inputs["input_ids"].shape[-1] != len(sequence) + 2:   # one token per nucleotide, plus BOS and EOS
        raise ValueError(f"{len(sequence)} nt gave {inputs['input_ids'].shape[-1]} tokens")
    if getattr(model.config, "model_type", None) == "ernierna":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            out = model(**inputs, output_attention_biases=True)
            prior = model.pairwise_bias_proj(model.get_pairwise_bias(**inputs).unsqueeze(-1)).permute(0, 3, 1, 2)
        return torch.stack([prior[0], *(a[0] for a in out.attention_biases)])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        out = model(**inputs, output_attentions=True)
    # RNA-MSM is an MSA transformer: its pairwise (i, j) maps are the row attentions
    layers = out.row_attentions if hasattr(out, "row_attentions") else out.attentions
    return torch.stack([a[0] for a in layers])
