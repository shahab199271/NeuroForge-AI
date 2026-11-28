# NeuroForge AI
# Maintainer: Shahab Baloochi

import importlib.util

import pytest
import torch

from neuroforge.kernels.advanced import matmul, residual_rmsnorm, rope


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    d = request.param
    if d == 'cuda' and (not torch.cuda.is_available() or torch.version.cuda is None
                        or importlib.util.find_spec('triton') is None
                        or torch.cuda.get_device_capability()[0] < 8):
        pytest.skip('Requires NVIDIA Ampere+ and Triton')
    return d


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_advanced(device, dtype):
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unsupported')
    torch.manual_seed(42)
    backend = 'triton' if device == 'cuda' else 'auto'
    x = torch.randn(3, 70, device=device, dtype=dtype)[:, ::2]
    r, w = torch.randn_like(x), torch.randn(35, device=device, dtype=dtype)
    tol = {torch.float32: 3e-5, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
    y, s = residual_rmsnorm(x, r, w, backend=backend)
    z = (x + r).double()
    ref = (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)*w.double()).to(dtype)
    torch.testing.assert_close(y, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(s, x+r, rtol=0, atol=0)
    x = torch.randn(3, 70, device=device, dtype=dtype)
    angle = torch.randn(3, 35, device=device)
    c, t = angle.cos().to(dtype), angle.sin().to(dtype)
    a, b = x.double().chunk(2, -1)
    ref = torch.cat((a*c.double()-b*t.double(), b*c.double()+a*t.double()), -1).to(dtype)
    torch.testing.assert_close(rope(x, c, t, backend=backend), ref, rtol=tol, atol=tol)
    a = torch.randn(17, 33, device=device, dtype=dtype)
    b = torch.randn(33, 19, device=device, dtype=dtype)
    torch.testing.assert_close(matmul(a, b, backend=backend), (a.double()@b.double()).to(dtype),
                               rtol=tol, atol=tol)


def test_grad_and_validation():
    x = torch.randn(2, 8, requires_grad=True)
    y, s = residual_rmsnorm(x, x, torch.ones(8))
    (y.sum()+s.sum()).backward()
    assert x.grad is not None
    with pytest.raises(ValueError):
        rope(torch.ones(3), torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError):
        matmul(torch.ones(2, 3), torch.ones(4, 2))
    with pytest.raises(RuntimeError):
        matmul(torch.ones(2, 3), torch.ones(3, 2), backend='triton')


@pytest.mark.parametrize('width', [1, 8192, 8193])
def test_residual_boundary_cpu(width):
    x = torch.linspace(-1, 1, width).reshape(1, width)
    w = torch.ones(width)
    y, summed = residual_rmsnorm(x, -x, w)
    torch.testing.assert_close(summed, torch.zeros_like(x))
    torch.testing.assert_close(y, torch.zeros_like(x))


def test_rope_identity_and_rotation():
    x = torch.tensor([[1., 2., 3., 4.]])
    c, s = torch.ones(1, 2), torch.zeros(1, 2)
    torch.testing.assert_close(rope(x, c, s), x)
    torch.testing.assert_close(rope(x, s, c), torch.tensor([[-3., -4., 1., 2.]]))


def test_noncontiguous_gemm_cpu():
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
    b = torch.arange(15, dtype=torch.float32).reshape(5, 3).T
    torch.testing.assert_close(matmul(a, b), (a.double() @ b.double()).float())


@pytest.mark.parametrize('eps', [0., -1., float('nan'), float('inf')])
def test_residual_invalid_epsilon(eps):
    with pytest.raises(ValueError):
        residual_rmsnorm(torch.ones(2, 4), torch.ones(2, 4), torch.ones(4), eps)
