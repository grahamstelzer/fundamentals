# test_reductions.py
import torch
import pytest
import mylib

DEVICE = "cuda"

@pytest.mark.parametrize("shape,axis", [
    ((128,), 0),
    ((33, 129), 0),
    ((33, 129), 1),
    ((4, 17, 65), 1),
    ((4, 17, 65), 2),
])
def test_sum(shape, axis):
    x = torch.randn(shape, device=DEVICE)
    out = mylib.sum(x, axis=axis)
    ref = x.sum(dim=axis)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

@pytest.mark.parametrize("shape", [(4, 128), (32, 4097), (2, 8, 4096)])
def test_softmax_numerically_stable_with_large_values(shape):
    # values large enough that naive exp() without max-subtraction overflows
    x = torch.randn(shape, device=DEVICE) * 100
    out = mylib.softmax(x, axis=-1)
    ref = torch.softmax(x, dim=-1)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    # rows must sum to 1
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(shape[:-1], device=DEVICE), atol=1e-4, rtol=1e-4)