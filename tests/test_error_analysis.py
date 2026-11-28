import json
import math

import pytest
import torch

from neuroforge.kernels.error_analysis import analyze_quantization, error_metrics


def test_known_metrics():
    r = torch.tensor([1., -2., 3., -4.])
    y = torch.tensor([2., -2., 1., -3.])
    e = error_metrics(r, y)
    assert e.count == 4
    assert e.mae == pytest.approx(1.)
    assert e.rmse == pytest.approx(math.sqrt(1.5))
    assert e.max_abs == 2.
    assert e.bias == pytest.approx(0.)
    assert e.relative_l2 == pytest.approx(math.sqrt(6 / 30))
    assert e.cosine == pytest.approx(21 / math.sqrt(30 * 18))


def test_zero_norms():
    z = torch.zeros(3)
    e = error_metrics(z, z)
    assert e.rmse == 0 and e.relative_l2 is None and e.cosine is None
    e = error_metrics(torch.ones(3), z)
    assert e.relative_l2 == 1 and e.cosine is None


def test_large_and_tiny_metrics():
    for value in (1e300, 1e-300):
        r = torch.tensor([value, -value], dtype=torch.float64)
        e = error_metrics(r, -r)
        assert e.rmse == pytest.approx(2 * value, abs=0)
        assert e.relative_l2 == pytest.approx(2.)
        assert e.cosine == pytest.approx(-1.)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_metrics(bad):
    with pytest.raises(ValueError):
        error_metrics(torch.ones(1), torch.tensor([bad]))


def test_shape_and_dtype_metrics():
    for x, y in [(torch.ones(2), torch.ones(1)), (torch.empty(0), torch.empty(0)),
                 (torch.ones(1), torch.ones(1, dtype=torch.int64))]:
        with pytest.raises(ValueError):
            error_metrics(x, y)


@pytest.mark.parametrize('bits', [4, 8])
@pytest.mark.parametrize('group', [1, 3, 8, 128])
@pytest.mark.parametrize('precision', ['fp32', 'fp16', 'bf16'])
def test_analysis(bits, group, precision):
    gen = torch.Generator().manual_seed(19)
    w = torch.randn(5, 9, generator=gen)
    a = torch.randn(3, 9, generator=gen)
    report = analyze_quantization(w, a, bits=bits, group_size=group, precision=precision)
    storage = report['storage']
    assert storage['code_bytes'] == (25 if bits == 4 else 45)
    assert storage['scale_bytes'] == 5 * math.ceil(9 / group) * 4
    assert storage['effective_bits_per_weight'] == 8 * storage['total_bytes'] / 45
    assert report['weight_error']['count'] == 45
    assert report['total_output_error']['count'] == 15
    assert report['execution_error']['max_abs'] < 1e-4
    if precision == 'fp32':
        assert report['operand_output_error']['max_abs'] == 0
    if group == 1:
        assert report['weight_error']['max_abs'] < 1e-6
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('bits', [4, 8])
def test_exact_integer_case(bits):
    w = torch.tensor([[7., -7., 0.]]) if bits == 4 else torch.tensor([[127., -127., 0.]])
    a = torch.tensor([[1., 2., 3.]])
    report = analyze_quantization(w, a, bits=bits, group_size=3)
    for name in ('weight_error', 'quantization_output_error', 'operand_output_error',
                 'execution_error', 'total_output_error'):
        assert report[name]['max_abs'] == 0


def test_known_int4_error():
    w = torch.tensor([[7., 0.4, -0.6]])
    a = torch.tensor([[0., 1., 1.]])
    report = analyze_quantization(w, a, group_size=3)
    assert report['weight_error']['max_abs'] == pytest.approx(0.4)
    assert report['quantization_output_error']['max_abs'] == pytest.approx(0.8)
    assert report['total_output_error']['max_abs'] == pytest.approx(0.8)


def test_zero_report():
    report = analyze_quantization(torch.zeros(2, 5), torch.ones(3, 5))
    assert report['weight_error']['relative_l2'] is None
    assert report['total_output_error']['cosine'] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('kwargs', [{'bits': 3}, {'bits': True}, {'group_size': 0},
                                    {'precision': 'auto'}, {'backend': 'auto'}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        analyze_quantization(torch.ones(2, 3), torch.ones(1, 3), **kwargs)


def test_invalid_inputs():
    w = torch.ones(2, 3)
    for a in (torch.ones(1, 2), torch.ones(3), torch.ones(1, 3, requires_grad=True),
              torch.full((1, 3), float('nan'))):
        with pytest.raises(ValueError):
            analyze_quantization(w, a)
    with pytest.raises(ValueError, match='range'):
        analyze_quantization(w, torch.full((1, 3), 65536.), precision='fp16')


def test_operand_error_is_separate():
    w = torch.tensor([[7., 1., -2.]])
    a = torch.tensor([[1.0003, 0., 0.]])
    r = analyze_quantization(w, a, group_size=3, precision='fp16')
    assert r['quantization_output_error']['max_abs'] == 0
    assert r['operand_output_error']['max_abs'] > 0
    assert r['execution_error']['max_abs'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('bits', [4, 8])
def test_cuda_diagnostics(bits):
    import importlib.util
    if (torch.version.cuda is None or torch.cuda.get_device_capability()[0] < 8
            or importlib.util.find_spec('triton') is None):
        pytest.skip('NVIDIA Ampere+ and Triton required')
    w = torch.randn(5, 17, device='cuda')
    a = torch.randn(3, 17, device='cuda')
    r = analyze_quantization(w, a, bits=bits, group_size=7, backend='triton')
    assert r['execution_error']['max_abs'] < 0.01
