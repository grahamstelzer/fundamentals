# test_attention.py
import torch
import pytest
import mylib

DEVICE = "cuda"

@pytest.mark.parametrize("batch,heads,seqlen,headdim", [
    (1, 1, 64, 32),
    (2, 8, 128, 64),
    (2, 8, 1000, 64),   # unaligned seqlen — will expose masking bugs
])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention(batch, heads, seqlen, headdim, causal):
    q = torch.randn(batch, heads, seqlen, headdim, device=DEVICE)
    k = torch.randn(batch, heads, seqlen, headdim, device=DEVICE)
    v = torch.randn(batch, heads, seqlen, headdim, device=DEVICE)
    out = mylib.attention(q, k, v, causal=causal)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)