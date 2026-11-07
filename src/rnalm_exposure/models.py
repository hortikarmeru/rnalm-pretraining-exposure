"""Load the RNA language models (Hugging Face ports from multimolecule) and get their attention maps."""
import torch


def load(model_id, device):
    import multimolecule  # noqa: F401  (registers the RNA models with transformers)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return model, tokenizer


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
