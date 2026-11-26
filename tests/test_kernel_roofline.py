import pytest

from neuroforge.kernel_roofline import effective_rates, gemm_cost, residual_cost, roofline, rope_cost


def test_gemm_cost():
    w = gemm_cost(2, 3, 4, 2)
    assert w.flops == 48
    assert w.logical_bytes == 52
    assert w.arithmetic_intensity == pytest.approx(48/52)


def test_residual_cost():
    w = residual_cost(2, 8, 2)
    assert (w.flops, w.logical_bytes) == (80, 144)


def test_rope_cost():
    w = rope_cost(2, 8, 2)
    assert (w.flops, w.logical_bytes) == (48, 96)


def test_roofline_limits():
    w = gemm_cost(1024, 1024, 1024, 2)
    r = roofline(w, 10, 1000)
    assert r['model_limit'] == 'compute'
    assert r['ideal_lower_bound_ms'] == r['compute_bound_ms']
    r = roofline(w, 1000, 100)
    assert r['model_limit'] == 'memory'
    assert r['ideal_lower_bound_ms'] == r['memory_bound_ms']
    assert r['ridge_flops_per_byte'] == 10000


def test_rates():
    w = gemm_cost(2, 3, 4, 2)
    r = effective_rates(w, 2)
    assert r['effective_tflops'] == pytest.approx(48/2e9)
    assert r['logical_gbs'] == pytest.approx(52/2e6)


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf')])
def test_bad_hardware_or_time(bad):
    w = gemm_cost(1, 1, 1, 2)
    with pytest.raises(ValueError):
        roofline(w, bad, 100)
    with pytest.raises(ValueError):
        roofline(w, 10, bad)
    with pytest.raises(ValueError):
        effective_rates(w, bad)


def test_invalid_shapes():
    with pytest.raises(ValueError):
        gemm_cost(0, 2, 3, 4)
    with pytest.raises(ValueError):
        rope_cost(2, 3, 4)
    with pytest.raises(ValueError):
        residual_cost(1, 1, 0)
