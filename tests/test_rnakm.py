import math

import pytest
import torch

from rnalm_exposure import rnakm


def test_encode_follows_the_authors_vocabulary():
    assert rnakm.encode("acguT").tolist() == [[0, 2, 3, 1, 1]]   # ATCGX order; U and T are the same token
    assert rnakm.encode("ANX").tolist() == [[0, 4, 4]]           # any other letter is X


def test_relative_position_buckets():
    """Buckets of the signed distance: 32 per sign; distances under 16 exact, then log-spaced up to 256."""
    bucket = lambda d: int(rnakm.RelativePositionBias._relative_position_bucket(torch.tensor([d]), 64, 256))
    for d in range(16):
        assert bucket(d) == d and (d == 0 or bucket(-d) == 32 + d)
    for d in (16, 20, 100, 255):
        assert bucket(d) == 16 + int(math.log(d / 16) / math.log(256 / 16) * 16)
        assert bucket(-d) == 32 + bucket(d)
    assert bucket(256) == bucket(5000) == 31 and bucket(-5000) == 63


def test_attention_maps_are_probabilities_and_match_a_direct_computation():
    torch.manual_seed(0)
    model = rnakm.RnaKm().eval()
    src = rnakm.encode("GGGAAACCCU")
    with torch.no_grad():
        maps = model(src)
        # first layer by hand: softmax(q k^T / sqrt(64) + position bias)
        layer = model.encoder.layers[0]
        x = layer.norm1(model.embed_tokens(src))
        q, k = (f(x).view(1, 10, 16, 64).transpose(1, 2) for f in (layer.attn.fc_query, layer.attn.fc_key))
        pos = torch.arange(10)
        bias = model.embed_rp(pos[:, None] - pos[None, :]).permute(2, 0, 1)
        expected = torch.softmax(q @ k.transpose(-1, -2) / 8 + bias, dim=-1)
    assert maps.shape == (12, 1, 16, 10, 10)
    assert torch.allclose(maps.sum(-1), torch.ones(12, 1, 16, 10), atol=1e-5)
    assert torch.allclose(maps[0], expected, atol=1e-6)


def test_used_weights_drops_the_pair_head_and_rejects_unknown_entries():
    state = {"module." + k: v for k, v in rnakm.RnaKm().state_dict().items()}
    state["module.fc_pair_q.weight"] = torch.zeros(1)
    state["module.layer_norm_after.bias"] = torch.zeros(1)
    used = rnakm.used_weights(state)
    rnakm.RnaKm().load_state_dict(used)   # strict: exactly the model's weights
    with pytest.raises(ValueError):
        rnakm.used_weights({**state, "module.something_else.weight": torch.zeros(1)})
