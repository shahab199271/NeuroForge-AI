import importlib.util

import pytest
import torch

from neuroforge.kernels.group_quantization import (
    quantize_grouped, dequantize_grouped, quantize_channels, dequantize_channels,
)
from neuroforge.kernels.quantization import quantize_int8


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('size', [1, 7, 32, 64, 128, 256, 1024])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_group_roundtrip(device, size, dtype):
    if device == 'cuda' and (not torch.cuda.is_available() or torch.version.cuda is None
                            or importlib.util.find_spec('triton') is None
                            or torch.cuda.get_device_capability()[0] < 8):
        pytest.skip('NVIDIA Ampere+ and Triton required')
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unsupported')
    torch.manual_seed(42)
    x = torch.randn(2, 3, 514, device=device, dtype=dtype)[..., ::2]
    backend = 'triton' if device == 'cuda' else 'auto'
    q, s = quantize_grouped(x, size, backend=backend)
    y = dequantize_grouped(q, s, size, backend=backend)
    qr, sr = quantize_grouped(x, size, backend='torch')
    assert q.shape == x.shape and q.dtype == torch.int8
    assert s.shape == (2, 3, (257 + size - 1) // size) and s.dtype == torch.float32
    torch.testing.assert_close(s, sr, rtol=2e-6, atol=0)
    assert int((q.int() - qr.int()).abs().max()) <= 1
    bound = s.repeat_interleave(min(size, 257), -1)[..., :257] * .5001
    assert bool(((y - x.float()).abs() <= bound + 1e-6*x.float().abs()).all())
    torch.testing.assert_close(y, dequantize_grouped(q, s, size, backend='torch'))


@pytest.mark.parametrize('axis', [0, 1, 2, -1, -3])
def test_channels(axis):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) - 12
    q, s = quantize_channels(x, axis)
    y = dequantize_channels(q, s, axis)
    assert s.shape == (x.shape[axis],)
    z = x.movedim(axis, 0).reshape(x.shape[axis], -1)
    qr, sr = quantize_int8(z)
    torch.testing.assert_close(s, sr.squeeze(-1))
    torch.testing.assert_close(q.movedim(axis, 0).reshape_as(qr), qr)
    assert y.shape == x.shape


def test_tail_zero_and_ties():
    x = torch.tensor([127., .5, -.5, 0., 0., 0., 0.])
    q, s = quantize_grouped(x, 3)
    assert q.tolist() == [127, 1, -1, 0, 0, 0, 0]
    torch.testing.assert_close(s, torch.ones(3))


def test_outlier_example():
    x = torch.tensor([1270., 1., .25, -.25, .75, -.75])
    q, s = quantize_grouped(x, 2)
    y = dequantize_grouped(q, s, 2)
    qr, sr = quantize_grouped(x, 6)
    yr = dequantize_grouped(qr, sr, 6)
    assert (y-x).square().mean() < (yr-x).square().mean()


@pytest.mark.parametrize('size', [0, -1, True, 2.5])
def test_invalid_size(size):
    with pytest.raises(ValueError):
        quantize_grouped(torch.ones(3), size)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        quantize_grouped(torch.tensor([float('inf')]))
    with pytest.raises(ValueError):
        quantize_grouped(torch.ones(3, requires_grad=True))
    with pytest.raises(ValueError):
        dequantize_grouped(torch.tensor([-128], dtype=torch.int8), torch.ones(1))
    with pytest.raises(ValueError):
        dequantize_grouped(torch.ones(3, dtype=torch.int8), torch.zeros(1))
    with pytest.raises(ValueError):
        quantize_channels(torch.ones(3), 1)
    with pytest.raises(RuntimeError):
        quantize_grouped(torch.ones(3), backend='triton')


def test_extreme_and_wide():
    x = torch.tensor([torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max, 1e-42])
    q, s = quantize_grouped(x, 2)
    assert bool(torch.isfinite(dequantize_grouped(q, s, 2)).all())
    x = torch.linspace(-1, 1, 9001)
    q, s = quantize_grouped(x, 8193)
    assert dequantize_grouped(q, s, 8193).shape == x.shape


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('axis', [0, -1])
def test_channel_strided(device, axis):
    if device == 'cuda' and (not torch.cuda.is_available() or torch.version.cuda is None
                            or importlib.util.find_spec('triton') is None
                            or torch.cuda.get_device_capability()[0] < 8):
        pytest.skip('NVIDIA Ampere+ and Triton required')
    x = torch.arange(60, device=device, dtype=torch.float32).reshape(3, 4, 5)[:, ::2, :]
    backend = 'triton' if device == 'cuda' else 'torch'
    q, s = quantize_channels(x, axis, backend=backend)
    y = dequantize_channels(q, s, axis, backend=backend)
    shape = [1] * x.ndim
    shape[axis] = x.shape[axis]
    assert bool(((y - x).abs() <= .5001 * s.reshape(shape) + 1e-6 * x.abs()).all())


def test_group_size_larger_than_width():
    x = torch.tensor([1., -2., 3.])
    q, s = quantize_grouped(x, 10**9)
    qr, sr = quantize_int8(x)
    torch.testing.assert_close(q, qr)
    torch.testing.assert_close(s, sr)
    torch.testing.assert_close(dequantize_grouped(q, s, 10**9), q.float() * s)
