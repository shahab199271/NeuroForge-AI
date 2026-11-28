# NeuroForge AI
# Maintainer: Shahab Baloochi

import importlib.util

import pytest
import torch

from neuroforge.kernels.int4 import pack_int4, unpack_int4, quantize_int4, dequantize_int4


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    d = request.param
    if d == 'cuda' and (not torch.cuda.is_available() or torch.version.cuda is None
                        or importlib.util.find_spec('triton') is None
                        or torch.cuda.get_device_capability()[0] < 8):
        pytest.skip('NVIDIA Ampere+ and Triton required')
    return d


@pytest.mark.parametrize('width', [1, 2, 15, 16, 257])
def test_pack(device, width):
    q = (torch.arange(2*width, device=device).reshape(2, width) % 15 - 7).to(torch.int8)
    backend = 'triton' if device == 'cuda' else 'torch'
    p = pack_int4(q, backend=backend)
    assert p.dtype == torch.uint8 and p.shape == (2, (width+1)//2)
    torch.testing.assert_close(p, pack_int4(q, backend='torch'))
    torch.testing.assert_close(unpack_int4(p, width, backend=backend), q)


def test_known_bytes():
    q = torch.tensor([-7, 7, -1, 0, 3], dtype=torch.int8)
    assert pack_int4(q).tolist() == [0x79, 0x0f, 0x03]


@pytest.mark.parametrize('size', [1, 7, 32, 128, 1024])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_quantization(device, size, dtype):
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unsupported')
    torch.manual_seed(42)
    x = torch.randn(2, 3, 514, device=device, dtype=dtype)[..., ::2]
    backend = 'triton' if device == 'cuda' else 'torch'
    p, s = quantize_int4(x, size, backend=backend)
    y = dequantize_int4(p, s, 257, size, backend=backend)
    pr, sr = quantize_int4(x, size, backend='torch')
    torch.testing.assert_close(s, sr, rtol=2e-6, atol=0)
    q, qr = unpack_int4(p, 257), unpack_int4(pr, 257)
    assert int((q.int()-qr.int()).abs().max()) <= 1
    assert p.shape == (2, 3, 129) and s.shape == (2, 3, (257+size-1)//size)
    bound = s.repeat_interleave(min(size, 257), -1)[..., :257] * .5001
    assert bool(((y-x.float()).abs() <= bound + 1e-6*x.float().abs()).all())
    torch.testing.assert_close(y, dequantize_int4(p, s, 257, size, backend='torch'))


def test_zero_ties_and_extremes():
    x = torch.tensor([[7., .5, -.5], [0., 0., 0.]])
    p, s = quantize_int4(x, 3)
    assert unpack_int4(p, 3).tolist() == [[7, 1, -1], [0, 0, 0]]
    torch.testing.assert_close(s, torch.ones(2, 1))
    x = torch.tensor([torch.finfo(torch.float32).max, -1e-42])
    p, s = quantize_int4(x, 1)
    assert bool(torch.isfinite(dequantize_int4(p, s, 2, 1)).all())


@pytest.mark.parametrize('bad', [-128, -8, 8, 127])
def test_invalid_codes(bad):
    with pytest.raises(ValueError):
        pack_int4(torch.tensor([bad], dtype=torch.int8))


def test_invalid_format():
    with pytest.raises(ValueError):
        unpack_int4(torch.tensor([8], dtype=torch.uint8), 1)
    with pytest.raises(ValueError):
        unpack_int4(torch.tensor([16], dtype=torch.uint8), 1)
    with pytest.raises(ValueError):
        unpack_int4(torch.tensor([0], dtype=torch.uint8), 3)
    with pytest.raises(ValueError):
        quantize_int4(torch.tensor([float('nan')]))
    with pytest.raises(ValueError):
        quantize_int4(torch.ones(3), 0)
    with pytest.raises(ValueError):
        quantize_int4(torch.ones(3, requires_grad=True))
    with pytest.raises(ValueError):
        dequantize_int4(torch.tensor([0], dtype=torch.uint8), torch.zeros(1), 2)


def test_noncontiguous_pack_and_huge_group():
    q = torch.arange(-7, 8, dtype=torch.int8)[::2]
    torch.testing.assert_close(unpack_int4(pack_int4(q), q.numel()), q)
    x = torch.tensor([7., -3., 1.])
    p, s = quantize_int4(x, 10**9)
    torch.testing.assert_close(dequantize_int4(p, s, 3, 10**9), x)
