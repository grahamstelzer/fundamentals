# test_layernorm.py
import torch
import pytest
import mylib

DEVICE = "cuda"

@pytest.mark.parametrize("shape", [(8, 64), (8, 768), (2, 33, 4097)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_layernorm(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    w = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    b = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    out = mylib.layernorm(x, w, b, eps=1e-5)
    ref = torch.nn.functional.layer_norm(x, (shape[-1],), w, b, eps=1e-5)
    atol, rtol = (2e-2, 2e-2) if dtype == torch.float16 else (1e-4, 1e-4)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)