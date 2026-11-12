"""Load the RNA language models and get their attention maps.

Two implementations, each needing its own conda env because their transformers versions conflict:
  multimolecule  Hugging Face ports of RNA-FM, RiNALMo and RNA-MSM (transformers 5)
  genbio         GB.RNA (formerly AIDO.RNA) with its authors' model code from modelgenerator (transformers 4.38)
"""
import os

import torch


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
    else:
        raise ValueError(f"unknown implementation {implementation}")
    return model.to(device).eval(), tokenizer


def load_mlm(implementation, model_id, device):
    """(model with its masked-LM head, tokenizer, ids of the four letters, mask id) for the masked-letter check."""
    if implementation == "multimolecule":
        import multimolecule  # noqa: F401
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model, inner, letters = AutoModelForMaskedLM.from_pretrained(model_id), tokenizer, "ACGU"
    elif implementation == "genbio":
        import modelgenerator
        from modelgenerator.huggingface_models.rnabert import RNABertForMaskedLM, RNABertTokenizer

        vocab = os.path.join(os.path.dirname(modelgenerator.__file__), "huggingface_models/rnabert/vocab.txt")
        inner = RNABertTokenizer(vocab, version="v2")
        tokenizer, letters = _DnaAlphabet(inner), "ACGT"
        model = RNABertForMaskedLM.from_pretrained(model_id)
    else:
        raise ValueError(f"unknown implementation {implementation}")
    ids = inner.convert_tokens_to_ids(list(letters))
    if len(set(ids)) != 4 or inner.unk_token_id in ids or inner.mask_token_id is None:
        raise ValueError(f"{model_id}: letters {letters} map to {ids}")
    return model.to(device).eval(), tokenizer, ids, inner.mask_token_id


def attention_maps(model, tokenizer, sequence, device, autocast=True):
    """[layers, heads, L + 2, L + 2] attention probabilities for one sequence, with BOS/EOS at both ends."""
    inputs = tokenizer([sequence], return_tensors="pt").to(device)
    if inputs["input_ids"].shape[-1] != len(sequence) + 2:   # one token per nucleotide, plus BOS and EOS
        raise ValueError(f"{len(sequence)} nt gave {inputs['input_ids'].shape[-1]} tokens")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        out = model(**inputs, output_attentions=True)
    # RNA-MSM is an MSA transformer: its pairwise (i, j) maps are the row attentions
    layers = out.row_attentions if hasattr(out, "row_attentions") else out.attentions
    return torch.stack([a[0] for a in layers])
