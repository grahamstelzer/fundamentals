# test_matmul.py
import torch
import pytest
import mylib

DEVICE = "cuda"

SHAPES = [
    (1, 1, 1), (17, 33, 65),                  # tiny + unaligned
    (128, 128, 128), (512, 512, 512),         # aligned
    (1, 4096, 4096), (4096, 1, 4096),         # degenerate M or N
    (4097, 129, 33),                          # unaligned, all dims
]

@pytest.mark.parametrize("m,k,n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matmul(m, k, n, dtype):
    a = torch.randn(m, k, device=DEVICE, dtype=dtype)
    b = torch.randn(k, n, device=DEVICE, dtype=dtype)
    out = mylib.matmul(a, b)
    ref = a.float() @ b.float()
    atol, rtol = (1.0, 1e-2) if dtype == torch.float16 else (1e-2, 1e-4)
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)

def test_bmm():
    a = torch.randn(8, 64, 32, device=DEVICE)
    b = torch.randn(8, 32, 96, device=DEVICE)
    out = mylib.bmm(a, b)
    ref = torch.bmm(a, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-3)