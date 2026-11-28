# NeuroForge AI
# Maintainer: Shahab Baloochi

import importlib.util
import pytest
import torch
import torch.nn.functional as F
from neuroforge.kernels import rmsnorm, softmax, swiglu


@pytest.mark.parametrize('shape', [(1, 7), (3, 896), (2, 3, 4864)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_operators(shape, dtype, device):
    if device == 'cuda' and (not torch.cuda.is_available() or torch.version.cuda is None or importlib.util.find_spec('triton') is None):
        pytest.skip('NVIDIA CUDA and Triton required')
    if device == 'cuda' and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip('Phase 1 targets NVIDIA Ampere or newer')
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('bf16 unsupported')
    torch.manual_seed(42)
    x = torch.randn((*shape[:-1], shape[-1] * 2), device=device, dtype=dtype)[..., ::2]
    w = torch.randn(shape[-1], device=device, dtype=dtype)
    u = torch.randn_like(x)
    backend = 'triton' if device == 'cuda' else 'auto'
    z = x.double()
    tol = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    refs = [(z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6) * w.double()).to(dtype),
            z.softmax(-1).to(dtype), (F.silu(z) * u.double()).to(dtype)]
    with torch.inference_mode():
        out = [rmsnorm(x, w, backend=backend), softmax(x, backend=backend), swiglu(x, u, backend=backend)]
    for actual, expected in zip(out, refs):
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
        assert actual.dtype == dtype and actual.shape == x.shape and actual.device == x.device


def test_grad_fallback():
    x = torch.randn(2, 7, requires_grad=True)
    w = torch.ones(7, requires_grad=True)
    (rmsnorm(x, w).sum() + swiglu(x, x).sum() + softmax(x).sum()).backward()
    assert x.grad is not None and w.grad is not None


def test_explicit_backend_rejects_cpu():
    with pytest.raises(RuntimeError):
        softmax(torch.ones(2, 7), backend='triton')


def test_validation():
    with pytest.raises(ValueError):
        rmsnorm(torch.ones(2, 7), torch.ones(8))
    with pytest.raises(ValueError):
        swiglu(torch.ones(2, 7), torch.ones(7))
    with pytest.raises(ValueError):
        softmax(torch.ones(0, 7))


def test_stable_softmax():
    x = torch.tensor([[10000., 9999., -10000.]])
    torch.testing.assert_close(softmax(x).sum(-1), torch.ones(1))


@pytest.mark.parametrize('shape', [(7,), (2, 1), (2, 8192), (2, 8193)])
def test_cpu_edge_shapes(shape):
    x = torch.randn(shape)
    w = torch.randn(shape[-1])
    torch.testing.assert_close(rmsnorm(x, w), (x.double() / (x.double().square().mean(-1, keepdim=True) + 1e-6).sqrt() * w.double()).float())
    torch.testing.assert_close(softmax(x), x.softmax(-1))
    torch.testing.assert_close(swiglu(x, x), F.silu(x) * x)


@pytest.mark.parametrize('eps', [0, -1, float('inf'), float('nan')])
def test_invalid_eps(eps):
    with pytest.raises(ValueError):
        rmsnorm(torch.ones(2, 7), torch.ones(7), eps)


def test_forward_ad():
    x = torch.randn(2, 7)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
        result = swiglu(dual, dual)
        assert torch.autograd.forward_ad.unpack_dual(result).tangent is not None


def test_softmax_extreme_values():
    x = torch.tensor([[10000., 9999., -10000.], [-float('inf'), 0., 1.]])
    torch.testing.assert_close(softmax(x), x.double().softmax(-1).float())


def test_unknown_backend():
    with pytest.raises(ValueError):
        softmax(torch.ones(7), backend='invalid')
