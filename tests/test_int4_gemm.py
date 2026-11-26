import importlib.util

import pytest
import torch

from neuroforge.kernels.int4 import dequantize_int4, pack_int4, quantize_int4
from neuroforge.kernels.int4_gemm import int4_matmul


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    d = request.param
    if d == 'cuda' and (not torch.cuda.is_available() or torch.version.cuda is None
                        or importlib.util.find_spec('triton') is None
                        or torch.cuda.get_device_capability()[0] < 8):
        pytest.skip('NVIDIA Ampere+ and Triton required')
    return d


@pytest.mark.parametrize('m,n,k,group', [(1, 1, 1, 1), (3, 5, 7, 3), (17, 33, 65, 7),
                                        (16, 32, 64, 32), (2, 9, 129, 128), (2, 4, 33, 10**9)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_matmul(device, m, n, k, group, dtype):
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unsupported')
    torch.manual_seed(29)
    a = torch.randn(m, 2*k, device=device, dtype=dtype)[:, ::2]
    w = torch.randn(n, k, device=device)
    p, s = quantize_int4(w, group, backend='torch')
    # Explicit noncontiguous packed/scales views.
    pp = torch.empty(n, 2*p.shape[1], dtype=p.dtype, device=device)[:, ::2]
    ss = torch.empty(n, 2*s.shape[1], dtype=s.dtype, device=device)[:, ::2]
    pp.copy_(p)
    ss.copy_(s)
    backend = 'triton' if device == 'cuda' else 'torch'
    y = int4_matmul(a, pp, ss, group, backend=backend)
    decoded = dequantize_int4(p, s, k, group, backend='torch').to(dtype)
    # FP64 oracle on rounded operands avoids any TF32 setting in the test.
    expected = (a.double() @ decoded.double().T).float()
    assert y.dtype == torch.float32 and y.shape == (m, n)
    torch.testing.assert_close(y, expected, rtol=2e-4, atol=2e-4)


def test_known_orientation_and_group_boundary(device):
    q = torch.tensor([[-3, 5, 1], [7, -2, -1]], dtype=torch.int8, device=device)
    p = pack_int4(q, backend='torch')
    s = torch.tensor([[2., 3., 4.], [1., 2., 3.]], device=device)
    a = torch.tensor([[1., 2., -1.]], device=device)
    y = int4_matmul(a, p, s, 1, backend='triton' if device == 'cuda' else 'torch')
    torch.testing.assert_close(y, torch.tensor([[20., 2.]], device=device))


def test_saturation_and_autocast(device):
    p = pack_int4(torch.tensor([[7, -7]], dtype=torch.int8, device=device), backend='torch')
    s = torch.tensor([[torch.finfo(torch.float32).max]], device=device)
    a = torch.tensor([[1., 0.]], dtype=torch.float16, device=device)
    with torch.autocast(device_type=device, dtype=torch.float16 if device == 'cuda' else torch.bfloat16):
        y = int4_matmul(a, p, s, 2, backend='triton' if device == 'cuda' else 'torch')
    assert y.dtype == torch.float32
    assert y.item() == 65504.


def test_invalid_inputs():
    a = torch.ones(2, 3)
    p, s = quantize_int4(torch.ones(4, 3), 2)
    for aa, pp, ss, g in [(a.unsqueeze(0), p, s, 2), (a, p[:, :1], s, 2),
                          (a, p.int(), s, 2), (a, p, s.double(), 2),
                          (a, p, s[:, :1], 2), (a, p, s, 0),
                          (a, p, s*0, 2), (a, p, s*float('inf'), 2),
                          (a*float('nan'), p, s, 2), (a.requires_grad_().clone(), p, s, 2)]:
        with pytest.raises(ValueError):
            int4_matmul(aa, pp, ss, g)
    a = a.detach()
    bad = p.clone()
    bad[0, 0] = 8
    with pytest.raises(ValueError, match='Reserved'):
        int4_matmul(a, bad, s, 2)
    bad = p.clone()
    bad[0, -1] = 16
    with pytest.raises(ValueError, match='padding'):
        int4_matmul(a, bad, s, 2)
    with pytest.raises(RuntimeError):
        int4_matmul(a, p, s, 2, backend='triton')
