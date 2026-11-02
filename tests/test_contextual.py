"""§5 — contextual tier tests (shapes, gate behavior, gradients)."""
import pytest

torch = pytest.importorskip("torch")

from src.contextual.gated_transformer import (EntropyGatedAttention, ContextualEncoder,
                                              ConcatFusionEncoder)
from src.contextual.positional import sinusoidal_positions, TemporalPositionalEncoding


def test_sinusoidal_positions_shape_and_range():
    pe = sinusoidal_positions(8, 16)
    assert pe.shape == (8, 16)
    assert pe.abs().max() <= 1.0 + 1e-6


def test_temporal_encoding_additive():
    enc = TemporalPositionalEncoding(d_model=16, max_len=32)
    x = torch.zeros(2, 32, 16)
    assert torch.allclose(enc(x).squeeze(0), enc.pe.squeeze(0))


def test_attention_shapes():
    attn = EntropyGatedAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(2, 5, 64)
    conf = torch.rand(2, 5)
    out = attn(x, conf)
    assert out.shape == x.shape


def test_gate_changes_output_with_confidence():
    torch.manual_seed(0)
    attn = EntropyGatedAttention(d_model=64, n_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 4, 64)
    with torch.no_grad():
        out_confident = attn(x, torch.full((1, 4), 0.99))
        out_uncertain = attn(x, torch.full((1, 4), 0.05))
    assert not torch.allclose(out_confident, out_uncertain), \
        "entropy gate must modulate attention as confidence changes"


def test_encoder_forward_and_backward():
    enc = ContextualEncoder(d_model=64, n_heads=4, n_layers=2, in_dim=32)
    x = torch.randn(3, 6, 32, requires_grad=True)
    conf = torch.rand(3, 6)
    refined, ctx = enc(x, conf)
    assert refined.shape == (3, 6)
    assert ctx.shape == (3, 6, 64)
    assert (refined >= 0).all() and (refined <= 1).all()   # sigmoid head
    refined.mean().backward()
    assert x.grad is not None


def test_concat_ablation_variant():
    enc = ConcatFusionEncoder(d_model=64, in_dim=32)
    refined, ctx = enc(torch.randn(2, 5, 32), torch.rand(2, 5))
    assert refined.shape == (2, 5)
    assert ctx.shape == (2, 5, 64)
