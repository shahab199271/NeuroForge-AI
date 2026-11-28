# NeuroForge AI
# Maintainer: Shahab Baloochi

import importlib.util

import pytest
import torch

from neuroforge.kernels.int4 import quantize_int4, pack_int4
from neuroforge.kernels.int4_gemm import int4_matmul
from neuroforge.kernels.mixed_precision import mixed_int4_matmul, _auto_dtype


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    d = request.param
    if d == 'cuda' and (not torch.cuda.is_available() or torch.version.cuda is None
                        or importlib.util.find_spec('triton') is None
                        or torch.cuda.get_device_capability()[0] < 8):
        pytest.skip('NVIDIA Ampere+ and Triton required')
    return d


@pytest.mark.parametrize('precision,dtype', [('fp32', torch.float32),
                                            ('fp16', torch.float16),
                                            ('bf16', torch.bfloat16)])
@pytest.mark.parametrize('shape,group', [((3, 5, 7), 3), ((17, 33, 65), 7)])
def test_explicit_precision(device, precision, dtype, shape, group):
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unsupported')
    m, n, k = shape
    torch.manual_seed(17)
    a = torch.randn(m, 2*k, device=device)[:, ::2]
    codes = torch.randint(-7, 8, (n, k), device=device, dtype=torch.int8)
    scales = torch.rand(n, (k+group-1)//group, device=device) + .01
    packed = pack_int4(codes, backend='torch')
    p0, s0 = packed.clone(), scales.clone()
    y = mixed_int4_matmul(a, packed, scales, group, precision=precision,
                          backend='triton' if device == 'cuda' else 'torch')
    w = codes.float() * scales.repeat_interleave(group, -1)[:, :k]
    oracle = (a.to(dtype).double() @ w.to(dtype).double().T).float()
    assert y.dtype == torch.float32 and y.shape == (m, n)
    torch.testing.assert_close(y, oracle, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(packed, p0)
    torch.testing.assert_close(scales, s0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_default_compatibility(dtype):
    a = torch.randn(4, 17).to(dtype)
    p, s = quantize_int4(torch.randn(5, 17), 7)
    expected = int4_matmul(a, p, s, 7)
    torch.testing.assert_close(mixed_int4_matmul(a, p, s, 7), expected)
    torch.testing.assert_close(mixed_int4_matmul(a, p, s, 7, precision='auto'), expected)


@pytest.mark.parametrize('dtype,native,fits,cuda,expected', [
    (torch.float32, True, True, True, torch.bfloat16),
    (torch.float32, True, False, True, torch.float32),
    (torch.float32, False, True, True, torch.float32),
    (torch.float16, True, True, True, torch.float16),
    (torch.bfloat16, False, False, True, torch.float32),
    (torch.bfloat16, False, False, False, torch.bfloat16),
])
def test_auto_selection(dtype, native, fits, cuda, expected):
    assert _auto_dtype(dtype, native, fits, cuda) == expected


def test_auto_runtime(device):
    a = torch.randn(2, 9, device=device)
    p, s = quantize_int4(torch.randn(3, 9, device=device), 3, backend='torch')
    backend = 'triton' if device == 'cuda' else 'torch'
    expected_dtype = ('bf16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32')
    y = mixed_int4_matmul(a, p, s, 3, precision='auto', backend=backend)
    torch.testing.assert_close(y, mixed_int4_matmul(a, p, s, 3, precision=expected_dtype,
                                                   backend=backend))


def test_overflow_and_validation():
    a = torch.ones(1, 2)
    p, s = quantize_int4(torch.ones(1, 2), 2)
    with pytest.raises(ValueError, match='range'):
        mixed_int4_matmul(a * 65536, p, s, 2, precision='fp16')
    with pytest.raises(ValueError, match='range'):
        mixed_int4_matmul(a * torch.finfo(torch.float32).max, p, s, 2, precision='bf16')
    for bad in (float('inf'), float('nan')):
        with pytest.raises(ValueError, match='finite'):
            mixed_int4_matmul(a * bad, p, s, 2, precision='fp32')
    with pytest.raises(ValueError, match='precision'):
        mixed_int4_matmul(a, p, s, 2, precision='int8')
    with pytest.raises(ValueError, match='backend'):
        mixed_int4_matmul(a, p, s, 2, backend='bad')
    with pytest.raises(ValueError):
        mixed_int4_matmul(a.requires_grad_(), p, s, 2, precision='fp16')
    with pytest.raises(ValueError):
        mixed_int4_matmul(a.detach(), p, s.requires_grad_(), 2, precision='fp16')


def test_autocast_and_global_settings():
    a = torch.randn(2, 5)
    p, s = quantize_int4(torch.randn(3, 5), 3)
    old = torch.get_float32_matmul_precision()
    expected = mixed_int4_matmul(a, p, s, 3, precision='fp16')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = mixed_int4_matmul(a, p, s, 3, precision='fp16')
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, expected)
    assert torch.get_float32_matmul_precision() == old


def test_weight_saturation_and_output_overflow():
    p = pack_int4(torch.tensor([[7]], dtype=torch.int8))
    s = torch.tensor([[torch.finfo(torch.float32).max]])
    a = torch.ones(1, 1)
    y = mixed_int4_matmul(a, p, s, 1, precision='fp16')
    assert y.item() == torch.finfo(torch.float16).max
    y = mixed_int4_matmul(a * 2, p, s, 1, precision='fp32')
    assert bool(torch.isinf(y).all())


def test_forward_ad_rejected():
    a = torch.ones(1, 2)
    p, s = quantize_int4(a, 2)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(a, a)
        with pytest.raises(ValueError):
            mixed_int4_matmul(dual, p, s, 2, precision='fp16')


@pytest.mark.parametrize('precision,dtype', [('fp16', torch.float16), ('bf16', torch.bfloat16)])
def test_exact_activation_range(precision, dtype):
    p = pack_int4(torch.tensor([[1]], dtype=torch.int8))
    s = torch.ones(1, 1)
    limit = torch.tensor(torch.finfo(dtype).max, dtype=torch.float32)
    for sign in (-1, 1):
        a = (limit * sign).reshape(1, 1)
        y = mixed_int4_matmul(a, p, s, 1, precision=precision, backend='torch')
        torch.testing.assert_close(y, a, rtol=0, atol=0)
        outside = torch.nextafter(limit, torch.tensor(float('inf'))) * sign
        with pytest.raises(ValueError, match='range'):
            mixed_int4_matmul(outside.reshape(1, 1), p, s, 1, precision=precision)


@pytest.mark.parametrize('seed', [0, 29, 101])
@pytest.mark.parametrize('precision,dtype', [('fp32', torch.float32), ('fp16', torch.float16),
                                            ('bf16', torch.bfloat16)])
def test_randomized_independent_oracle(seed, precision, dtype):
    rng = torch.Generator().manual_seed(seed)
    m, n, k, group = 2 + seed % 5, 5 + seed % 3, 33 + seed % 9, 7
    a = torch.randn(m, k, generator=rng) * 2
    codes = torch.randint(-7, 8, (n, k), generator=rng, dtype=torch.int8)
    scales = torch.rand(n, (k + group - 1) // group, generator=rng) + 0.01
    p = pack_int4(codes, backend='torch')
    w = codes.float() * scales[:, torch.arange(k) // group]
    expected = (a.to(dtype).double() @ w.to(dtype).double().T).float()
    result = mixed_int4_matmul(a, p, scales, group, precision=precision, backend='torch')
    torch.testing.assert_close(result, expected, rtol=3e-4, atol=3e-4)


def test_invalid_policy_type_and_explicit_backend():
    a = torch.ones(1, 1)
    p, s = quantize_int4(a, 1)
    for policy in (None, [], 1):
        with pytest.raises(ValueError, match='precision'):
            mixed_int4_matmul(a, p, s, 1, precision=policy)
    with pytest.raises(RuntimeError):
        mixed_int4_matmul(a, p, s, 1, precision='fp16', backend='triton')


def test_auto_torch_preserves_input_without_hardware_probe(monkeypatch):
    from neuroforge.kernels import mixed_precision as mp

    def unexpected_probe(_):
        pytest.fail('Explicit torch backend must not probe GPU capability')

    monkeypatch.setattr(mp, '_native_bf16', unexpected_probe)
    a = torch.tensor([[1.25, -2.5]])
    p, s = quantize_int4(a, 2)
    torch.testing.assert_close(
        mixed_int4_matmul(a, p, s, 2, precision='auto', backend='torch'),
        int4_matmul(a, p, s, 2, backend='torch'), rtol=0, atol=0,
    )
