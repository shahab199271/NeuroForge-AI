import importlib.util

import pytest
import torch

from neuroforge.kernels.quantization import dequantize_int8, quantize_int8


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(7,), (3, 35), (2, 3, 896)])
def test_roundtrip(device, dtype, shape):
    if device == 'cuda' and (not torch.cuda.is_available() or torch.version.cuda is None
                            or importlib.util.find_spec('triton') is None
                            or torch.cuda.get_device_capability()[0] < 8):
        pytest.skip('NVIDIA Ampere+ and Triton required')
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unsupported')
    torch.manual_seed(42)
    x = torch.randn((*shape[:-1], shape[-1] * 2), device=device, dtype=dtype)[..., ::2]
    backend = 'triton' if device == 'cuda' else 'auto'
    q, s = quantize_int8(x, backend=backend)
    y = dequantize_int8(q, s, backend=backend)
    qref, sref = quantize_int8(x, backend='torch')
    assert q.dtype == torch.int8 and q.shape == x.shape
    assert s.dtype == torch.float32 and s.shape == (*x.shape[:-1], 1)
    assert y.dtype == torch.float32 and y.shape == x.shape
    assert int(q.min()) >= -127
    torch.testing.assert_close(s, sref, rtol=2e-6, atol=0)
    assert int((q.int() - qref.int()).abs().max()) <= 1
    assert bool(((y - x.float()).abs() <= 0.5001 * s + 1e-6 * x.float().abs()).all())
    torch.testing.assert_close(y, dequantize_int8(q, s, backend='torch'))


def test_zero_rows_and_ties():
    x = torch.tensor([[0., 0., 0., 0., 0.], [127., .5, -.5, 1.5, -1.5]])
    q, s = quantize_int8(x)
    torch.testing.assert_close(s, torch.ones(2, 1))
    assert q.tolist() == [[0]*5, [127, 1, -1, 2, -2]]


def test_near_half():
    x = torch.tensor([127., .5, -.5])
    x[1] = torch.nextafter(x[1], torch.tensor(0.))
    x[2] = torch.nextafter(x[2], torch.tensor(0.))
    assert quantize_int8(x)[0].tolist() == [127, 0, 0]


def test_extreme_finite():
    x = torch.tensor([[torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max],
                      [1e-42, -1e-42]])
    q, s = quantize_int8(x)
    y = dequantize_int8(q, s)
    assert bool(torch.isfinite(y).all()) and bool((s > 0).all())
    torch.testing.assert_close(y[0], x[0])
    assert q[1].tolist() == [0, 0]


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite(bad):
    with pytest.raises(ValueError):
        quantize_int8(torch.tensor([bad]))


@pytest.mark.parametrize('bad', [0., -1., 1e-42, float('nan'), float('inf')])
def test_invalid_scale(bad):
    with pytest.raises(ValueError):
        dequantize_int8(torch.tensor([1], dtype=torch.int8), torch.tensor([bad]))


def test_validation():
    with pytest.raises(ValueError):
        quantize_int8(torch.ones(0))
    with pytest.raises(ValueError):
        quantize_int8(torch.ones(3, requires_grad=True))
    with pytest.raises(ValueError):
        dequantize_int8(torch.tensor([-128], dtype=torch.int8), torch.ones(1))
    with pytest.raises(ValueError):
        dequantize_int8(torch.ones(3, dtype=torch.int8), torch.ones(3))
    with pytest.raises(RuntimeError):
        quantize_int8(torch.ones(3), backend='triton')


def test_large_width_fallback():
    x = torch.linspace(-1, 1, 8193)
    q, s = quantize_int8(x)
    assert bool(((dequantize_int8(q, s) - x).abs() <= s / 2 + 1e-6).all())


def test_cuda_ties_and_zero_rows():
    if (not torch.cuda.is_available() or torch.version.cuda is None
            or importlib.util.find_spec('triton') is None
            or torch.cuda.get_device_capability()[0] < 8):
        pytest.skip('NVIDIA Ampere+ and Triton required')
    x = torch.tensor([[0., 0., 0., 0., 0.], [127., .5, -.5, 1.5, -1.5]], device='cuda')
    q, s = quantize_int8(x, backend='triton')
    assert q.tolist() == [[0]*5, [127, 1, -1, 2, -2]]
    torch.testing.assert_close(s, torch.ones_like(s))
    torch.testing.assert_close(dequantize_int8(q, s, backend='triton'), q.float())
