# NeuroForge AI
# Maintainer: Shahab Baloochi

"""CPU contract, metamorphic and bounded stress tests for the kernel lab."""
import math

import numpy as np
import pytest
import torch

from neuroforge.kernels.ops import rmsnorm, softmax, swiglu
from neuroforge.kernels.advanced import matmul, residual_rmsnorm, rope
from neuroforge.kernels.int4 import quantize_int4, pack_int4, unpack_int4
from neuroforge.kernels.int4_gemm import int4_matmul
from neuroforge.kernels.group_quantization import (
    quantize_grouped, dequantize_grouped, quantize_channels, dequantize_channels,
)
from neuroforge.kernels.error_analysis import error_metrics


@pytest.mark.parametrize('bad', [
    torch.ones(3, dtype=torch.int64), torch.empty(0), torch.tensor(1.),
    torch.ones(3, dtype=torch.float64),
])
def test_basic_input_contract(bad):
    with pytest.raises(ValueError):
        softmax(bad, backend='torch')


@pytest.mark.parametrize('eps', [0., -1., math.inf, math.nan])
def test_rmsnorm_epsilon_contract(eps):
    with pytest.raises(ValueError, match='eps'):
        rmsnorm(torch.ones(2, 4), torch.ones(4), eps=eps)
    with pytest.raises(ValueError, match='eps'):
        residual_rmsnorm(torch.ones(2, 4), torch.ones(2, 4), torch.ones(4), eps=eps)


def test_basic_and_advanced_shape_contracts():
    a = torch.ones(2, 4)
    for w in (torch.ones(3), torch.ones(4, dtype=torch.float16)):
        with pytest.raises(ValueError):
            rmsnorm(a, w)
    for b in (torch.ones(2, 3), a.half()):
        with pytest.raises(ValueError):
            swiglu(a, b)
        with pytest.raises(ValueError):
            residual_rmsnorm(a, b, torch.ones(4))
    with pytest.raises(ValueError):
        residual_rmsnorm(a, a, torch.ones(3))
    with pytest.raises(ValueError):
        matmul(a, torch.ones(3, 5))
    with pytest.raises(ValueError):
        matmul(a, torch.ones(4, 5).half())
    with pytest.raises(ValueError):
        rope(a[:, :3], a[:, :1], a[:, :1])
    with pytest.raises(ValueError):
        rope(a, a, a)
    with pytest.raises(ValueError):
        rope(a, a[:, :2], a[:, :2].half())


def test_cpu_dispatch_and_autograd():
    a = torch.tensor([[1., 2., 3.]], requires_grad=True)
    b = softmax(a, backend='auto')
    torch.testing.assert_close(b, softmax(a, backend='torch'))
    b[0, 0].backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    for backend, exc in [('bad', ValueError), ('triton', RuntimeError)]:
        with pytest.raises(exc):
            softmax(a, backend=backend)


@pytest.mark.parametrize('kind', ['packed_shape', 'packed_dtype', 'scale_shape',
                                  'scale_dtype', 'nonfinite_a', 'zero_scale',
                                  'negative_scale', 'inf_scale', 'nan_scale',
                                  'padding', 'reserved', 'group', 'gradient',
                                  'scale_gradient', 'rank', 'backend'])
def test_int4_gemm_invalid_contract(kind):
    a = torch.ones(2, 5)
    p, s = quantize_int4(torch.ones(3, 5), 3, backend='torch')
    group, backend = 3, 'torch'
    if kind == 'packed_shape':
        p = p[:, :-1]
    elif kind == 'packed_dtype':
        p = p.int()
    elif kind == 'scale_shape':
        s = s[:, :-1]
    elif kind == 'scale_dtype':
        s = s.double()
    elif kind == 'nonfinite_a':
        a[0, 0] = float('nan')
    elif kind in ('zero_scale', 'negative_scale', 'inf_scale', 'nan_scale'):
        s[0, 0] = {'zero_scale': 0., 'negative_scale': -1.,
                   'inf_scale': math.inf, 'nan_scale': math.nan}[kind]
    elif kind == 'padding':
        p[0, -1] = 16
    elif kind == 'reserved':
        p[0, 0] = 8
    elif kind == 'group':
        group = 0
    elif kind == 'gradient':
        a.requires_grad_()
    elif kind == 'scale_gradient':
        s.requires_grad_()
    elif kind == 'rank':
        a = a[0]
    elif kind == 'backend':
        backend = 'bad'
    with pytest.raises(ValueError):
        int4_matmul(a, p, s, group, backend=backend)


def test_group_and_packed_validation():
    x = torch.ones(2, 5)
    q, s = quantize_grouped(x, 3, backend='torch')
    with pytest.raises(ValueError):
        quantize_grouped(x, True)
    with pytest.raises(ValueError):
        quantize_grouped(x * math.inf, 3)
    with pytest.raises(ValueError):
        dequantize_grouped(q.float(), s, 3)
    with pytest.raises(ValueError):
        dequantize_grouped(q, s[:, :1], 3)
    with pytest.raises(ValueError):
        dequantize_grouped(q, s * 0, 3)
    with pytest.raises(ValueError):
        quantize_channels(x, axis=2)
    with pytest.raises(ValueError):
        dequantize_channels(q, s, axis=0)
    with pytest.raises(ValueError):
        pack_int4(torch.tensor([8], dtype=torch.int8))
    with pytest.raises(ValueError):
        unpack_int4(torch.tensor([16], dtype=torch.uint8), 1)
    with pytest.raises(ValueError):
        unpack_int4(torch.tensor([8], dtype=torch.uint8), 1)


@pytest.mark.parametrize('group', [1, 7, 32, 128, 8193])
def test_group_sign_and_power_of_two_scaling(group):
    rng = torch.Generator().manual_seed(19)
    w = torch.randn(4, 137, generator=rng)
    p, s = quantize_int4(w, group, backend='torch')
    p2, s2 = quantize_int4(-2*w, group, backend='torch')
    q, q2 = unpack_int4(p, 137), unpack_int4(p2, 137)
    torch.testing.assert_close(q2, -q, rtol=0, atol=0)
    torch.testing.assert_close(s2, 2*s, rtol=0, atol=0)


@pytest.mark.parametrize('m,n,k,group', [(1, 33, 257, 31), (17, 65, 1025, 64),
                                          (33, 97, 513, 7), (3, 9, 8193, 128)])
def test_gemm_stress_and_metamorphic(m, n, k, group):
    rng = torch.Generator().manual_seed(51)
    a = torch.randn(m, 2*k, generator=rng)[:, ::2]
    w = torch.randn(n, k, generator=rng)
    p, s = quantize_int4(w, group, backend='torch')
    out = int4_matmul(a, p, s, group, backend='torch')
    codes = unpack_int4(p, k, backend='torch').numpy().astype(np.float32)
    weights = codes * s.numpy()[:, np.arange(k) // group]
    expected = a.numpy().astype(np.float64) @ weights.astype(np.float64).T
    # FP32 dot products may accumulate in different orders.
    np.testing.assert_allclose(out.numpy(), expected, rtol=5e-5, atol=5e-4)
    reverse = torch.arange(n-1, -1, -1)
    torch.testing.assert_close(int4_matmul(a, p[reverse], s[reverse], group,
                                          backend='torch'), out[:, reverse],
                               rtol=5e-5, atol=5e-4)
    torch.testing.assert_close(int4_matmul(-a, p, s, group, backend='torch'),
                               -out, rtol=5e-5, atol=5e-4)
    if m > 1:
        split = torch.cat([int4_matmul(chunk, p, s, group, backend='torch')
                           for chunk in a.split(3)])
        torch.testing.assert_close(split, out, rtol=5e-5, atol=5e-4)


def test_fp64_metric_overflow_contract():
    r = torch.tensor([1e308], dtype=torch.float64)
    with pytest.raises(ValueError, match='Difference'):
        error_metrics(r, -r)
    r = torch.tensor([1e-300], dtype=torch.float64)
    with pytest.raises(ValueError, match='Relative'):
        error_metrics(r, torch.tensor([1e300], dtype=torch.float64))
