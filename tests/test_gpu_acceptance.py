# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Opt-in real CUDA acceptance tests; never silently skip when requested."""
import importlib.util
import os

import pytest
import torch

from neuroforge.kernels.ops import rmsnorm, softmax, swiglu
from neuroforge.kernels.advanced import matmul, residual_rmsnorm, rope
from neuroforge.kernels.quantization import quantize_int8, dequantize_int8
from neuroforge.kernels.group_quantization import quantize_grouped, dequantize_grouped
from neuroforge.kernels.int4 import pack_int4, unpack_int4, quantize_int4, dequantize_int4
from neuroforge.kernels.mixed_precision import mixed_int4_matmul

pytestmark = pytest.mark.skipif(
    os.environ.get('NEUROFORGE_REQUIRE_GPU') != '1',
    reason='Explicit GPU acceptance run not requested',
)


@pytest.fixture(scope='module')
def gpu():
    assert torch.cuda.is_available(), 'CUDA required for requested GPU acceptance'
    assert torch.version.cuda is not None, 'NVIDIA CUDA build required'
    assert importlib.util.find_spec('triton') is not None, 'Triton required'
    assert torch.cuda.get_device_capability()[0] >= 8, 'Ampere or newer required'
    assert torch.cuda.is_bf16_supported(), 'BF16 required for full acceptance suite'
    return torch.device('cuda')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_gpu_operator_acceptance(gpu, dtype):
    rng = torch.Generator().manual_seed(7)
    a = torch.randn(3, 34, generator=rng).to(gpu, dtype)
    b = torch.randn(34, 17, generator=rng).to(gpu, dtype)
    w = torch.ones(34, device=gpu, dtype=dtype)
    angle = torch.randn(3, 17, generator=rng).to(gpu)
    c, s = angle.cos().to(dtype), angle.sin().to(dtype)
    cases = [(rmsnorm, (a, w)), (softmax, (a,)), (swiglu, (a, a)),
             (rope, (a, c, s)), (residual_rmsnorm, (a, a, w))]
    for fn, args in cases:
        out = fn(*args, backend='triton')
        ref = fn(*args, backend='torch')
        if not isinstance(out, tuple):
            out, ref = (out,), (ref,)
        for actual, expected in zip(out, ref):
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    out = matmul(a, b, backend='triton')
    ref = (a.cpu().double() @ b.cpu().double()).to(dtype)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize('group', [7, 32, 128])
def test_gpu_quantization_acceptance(gpu, group):
    rng = torch.Generator().manual_seed(13)
    # Exact power-of-two values plus maxima make a robust cross-backend case.
    w = torch.randint(-7, 8, (33, 65), generator=rng).float().to(gpu)
    for quant, dequant, args in [
        (quantize_int8, dequantize_int8, ()),
        (quantize_grouped, dequantize_grouped, (group,)),
    ]:
        q, s = quant(w, *args, backend='triton')
        y = dequant(q, s, *args, backend='triton')
        torch.cuda.synchronize()
        assert bool(torch.isfinite(y).all())
        scales = s.repeat_interleave(w.shape[1] if not args else group, dim=-1)[..., :65]
        torch.testing.assert_close(y, q.float() * scales, rtol=1e-6, atol=1e-6)
        assert bool(((w - y).abs() <= scales / 2 + 1e-5).all())
    p, s = quantize_int4(w, group, backend='triton')
    q = unpack_int4(p, 65, backend='triton')
    torch.testing.assert_close(pack_int4(q, backend='triton'), p)
    y = dequantize_int4(p, s, 65, group, backend='triton')
    expanded = s.repeat_interleave(group, dim=-1)[..., :65]
    assert bool(((w - y).abs() <= expanded / 2 + 1e-5).all())
    a = torch.randn(17, 65, generator=rng).to(gpu)
    for precision, dtype in [('fp32', torch.float32), ('fp16', torch.float16),
                             ('bf16', torch.bfloat16)]:
        out = mixed_int4_matmul(a, p, s, group, precision=precision, backend='triton')
        ref = (a.to(dtype).cpu().double() @ y.to(dtype).cpu().double().T).float()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu(), ref, rtol=5e-4, atol=5e-4)
