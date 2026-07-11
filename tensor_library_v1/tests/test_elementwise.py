# test_elementwise.py
import torch
import pytest
import mylib  # your library

DEVICE = "cuda"
SHAPES = [
    (1,), (127,), (128,), (1024,),           # 1D: unaligned + aligned
    (32, 32), (33, 65), (256, 4096),          # 2D: unaligned + typical
    (2, 16, 128, 128),                        # 4D: attention-shaped
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add(shape, dtype):
    a = torch.randn(shape, device=DEVICE, dtype=dtype)
    b = torch.randn(shape, device=DEVICE, dtype=dtype)
    out = mylib.add(a, b)
    ref = a + b
    atol, rtol = (1e-2, 1e-2) if dtype != torch.float32 else (1e-5, 1e-5)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_relu(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    out = mylib.relu(x)
    ref = torch.relu(x)
    torch.testing.assert_close(out, ref)

@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_matches_exact_not_tanh_approx(shape):
    x = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    out = mylib.gelu(x)
    ref = torch.nn.functional.gelu(x, approximate="none")
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

def test_add_rejects_mismatched_shapes():
    a = torch.randn(4, 4, device=DEVICE)
    b = torch.randn(4, 5, device=DEVICE)
    with pytest.raises((RuntimeError, ValueError)):
        mylib.add(a, b)