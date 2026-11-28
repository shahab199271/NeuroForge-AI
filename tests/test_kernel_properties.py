# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Bounded property tests with independent NumPy references on CPU."""
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from neuroforge.kernels.ops import rmsnorm, softmax, swiglu
from neuroforge.kernels.advanced import matmul, residual_rmsnorm, rope
from neuroforge.kernels.quantization import quantize_int8, dequantize_int8
from neuroforge.kernels.group_quantization import (
    quantize_grouped, dequantize_grouped, quantize_channels, dequantize_channels,
)
from neuroforge.kernels.int4 import pack_int4, unpack_int4, quantize_int4
from neuroforge.kernels.mixed_precision import mixed_int4_matmul
from neuroforge.kernels.error_analysis import error_metrics, analyze_quantization

# Each property has a bounded, repeatable example budget. Deadlines are
# disabled because CI scheduling and tensor library startup can vary.
property_settings = settings(max_examples=60, derandomize=True, deadline=None)


@st.composite
def matrices(draw, max_rows=5, max_width=41):
    shape = (draw(st.integers(1, max_rows)), draw(st.integers(1, max_width)))
    raw = draw(arrays(np.int16, shape, elements=st.integers(-128, 128)))
    return raw.astype(np.float32) / np.float32(8)


def _group_oracle(x, group, qmax):
    """Scalar-group NumPy reference with the documented FP32 scale contract."""
    rows, width = x.shape
    q = np.empty(x.shape, dtype=np.int8)
    scales = np.empty((rows, (width + group - 1) // group), dtype=np.float32)
    for i in range(rows):
        for j, start in enumerate(range(0, width, group)):
            v = x[i, start:start + group]
            peak = np.max(np.abs(v))
            scale = np.float32(1) if peak == 0 else np.maximum(
                peak / np.float32(qmax), np.finfo(np.float32).tiny)
            scales[i, j] = scale
            ratio = np.clip(v / scale, -qmax, qmax).astype(np.float64)
            q[i, start:start + group] = (np.sign(ratio)
                                        * np.floor(np.abs(ratio) + 0.5)).astype(np.int8)
    return q, scales


def _reconstruct(q, scales, group):
    return q.astype(np.float32) * scales[:, np.arange(q.shape[1]) // group]


def _close(actual, expected, rtol=2e-5, atol=2e-5):
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, rtol=rtol, atol=atol)


@property_settings
@given(x=matrices(), shift=st.integers(-16, 16))
def test_phase1_operators(x, shift):
    t = torch.from_numpy(x)
    z = x.astype(np.float64)
    ex = np.exp(z - z.max(axis=-1, keepdims=True))
    expected = ex / ex.sum(axis=-1, keepdims=True)
    _close(softmax(t, backend='torch'), expected)
    _close(softmax(t + shift, backend='torch'), expected)
    _close(rmsnorm(t, torch.ones(t.shape[-1]), backend='torch'),
           z / np.sqrt(np.mean(z**2, axis=-1, keepdims=True) + 1e-6))
    _close(swiglu(t, t, backend='torch'), z**2 / (1 + np.exp(-z)))


@property_settings
@given(x=matrices(max_width=20), seed=st.integers(0, 100000))
def test_phase2_operators(x, seed):
    rng = np.random.default_rng(seed)
    t = torch.from_numpy(x)
    rhs = rng.normal(size=(x.shape[1], 3)).astype(np.float32)
    _close(matmul(t, torch.from_numpy(rhs), backend='torch'),
           x.astype(np.float64) @ rhs.astype(np.float64), atol=1e-4)
    y, summed = residual_rmsnorm(t, -t, torch.ones(t.shape[-1]), backend='torch')
    assert torch.count_nonzero(summed) == torch.count_nonzero(y) == 0
    doubled = np.concatenate((x, x[:, ::-1]), axis=-1)
    theta = rng.uniform(-np.pi, np.pi, size=x.shape)
    c, s = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    out = rope(torch.from_numpy(doubled), torch.from_numpy(c), torch.from_numpy(s),
               backend='torch')
    expected = np.concatenate((x*c - x[:, ::-1]*s, x[:, ::-1]*c + x*s), axis=-1)
    _close(out, expected)
    np.testing.assert_allclose(np.linalg.norm(out.numpy(), axis=-1),
                               np.linalg.norm(doubled, axis=-1), rtol=2e-6, atol=2e-6)


@property_settings
@given(x=matrices())
def test_phase31_rowwise_int8(x):
    t = torch.from_numpy(x)
    q, s = quantize_int8(t, backend='torch')
    expected, scales = _group_oracle(x, x.shape[-1], 127)
    np.testing.assert_array_equal(q.numpy(), expected)
    np.testing.assert_array_equal(s.numpy(), scales)
    _close(dequantize_int8(q, s, backend='torch'), _reconstruct(expected, scales, x.shape[-1]),
           rtol=0, atol=0)


@property_settings
@given(x=matrices(), group=st.integers(1, 64))
def test_phase32_group_and_channel_int8(x, group):
    t = torch.from_numpy(x)
    q, s = quantize_grouped(t, group, backend='torch')
    expected, scales = _group_oracle(x, group, 127)
    np.testing.assert_array_equal(q.numpy(), expected)
    np.testing.assert_array_equal(s.numpy(), scales)
    _close(dequantize_grouped(q, s, group, backend='torch'), _reconstruct(expected, scales, group),
           rtol=0, atol=0)
    cq, cs = quantize_channels(t, axis=1, backend='torch')
    ec, es = _group_oracle(x.T, x.shape[0], 127)
    np.testing.assert_array_equal(cq.numpy(), ec.T)
    np.testing.assert_array_equal(cs.numpy(), es[:, 0])
    _close(dequantize_channels(cq, cs, axis=1, backend='torch'),
           _reconstruct(ec, es, x.shape[0]).T, rtol=0, atol=0)


@property_settings
@given(codes=st.lists(st.integers(-7, 7), min_size=1, max_size=129))
def test_phase33_packed_layout(codes):
    q = torch.tensor(codes, dtype=torch.int8)
    packed = pack_int4(q, backend='torch')
    expected = [(codes[i] & 15) | (((codes[i+1] if i+1 < len(codes) else 0) & 15) << 4)
                for i in range(0, len(codes), 2)]
    assert packed.tolist() == expected
    assert unpack_int4(packed, len(codes), backend='torch').tolist() == codes


@property_settings
@given(x=matrices(), group=st.integers(1, 64))
def test_phase33_grouped_int4_error_bound(x, group):
    packed, scales = quantize_int4(torch.from_numpy(x), group, backend='torch')
    q, s = _group_oracle(x, group, 7)
    np.testing.assert_array_equal(unpack_int4(packed, x.shape[1], backend='torch').numpy(), q)
    np.testing.assert_array_equal(scales.numpy(), s)
    error = np.abs(x.astype(np.float64) - _reconstruct(q, s, group).astype(np.float64))
    # Half a quantization step plus bounded FP32 roundoff.
    assert np.all(error <= s[:, np.arange(x.shape[1]) // group] / 2 + 5e-6)


@property_settings
@given(w=matrices(max_width=25), group=st.integers(1, 32),
       precision=st.sampled_from(['fp32', 'fp16', 'bf16']), seed=st.integers(0, 100000))
def test_phase34_35_gemm_numpy(w, group, precision, seed):
    a = np.random.default_rng(seed).normal(size=(3, w.shape[1])).astype(np.float32)
    packed, scales = quantize_int4(torch.from_numpy(w), group, backend='torch')
    q, s = _group_oracle(w, group, 7)
    wq = _reconstruct(q, s, group)
    if precision == 'fp16':
        ar, wr = a.astype(np.float16).astype(np.float64), wq.astype(np.float16).astype(np.float64)
    elif precision == 'bf16':
        # Independently round binary32 to bfloat16 using round-to-nearest-even.
        def bf16(x):
            u = x.view(np.uint32)
            return ((u + np.uint32(0x7fff) + ((u >> 16) & 1))
                    & np.uint32(0xffff0000)).view(np.float32).astype(np.float64)
        ar, wr = bf16(a), bf16(wq)
    else:
        ar, wr = a.astype(np.float64), wq.astype(np.float64)
    out = mixed_int4_matmul(torch.from_numpy(a), packed, scales, group,
                            precision=precision, backend='torch')
    assert out.dtype == torch.float32
    _close(out, ar @ wr.T, atol=1e-4)


@property_settings
@given(x=matrices(), shift=st.integers(-64, 64))
def test_phase36_metrics_numpy(x, shift):
    y = x + np.float32(shift / 8)
    r = x.astype(np.float64).ravel()
    e = y.astype(np.float64).ravel() - r
    m = error_metrics(torch.from_numpy(x), torch.from_numpy(y))
    assert m.mae == pytest.approx(np.abs(e).mean())
    assert m.rmse == pytest.approx(np.sqrt(np.mean(e**2)))
    assert m.max_abs == pytest.approx(np.abs(e).max())
    assert m.bias == pytest.approx(e.mean())
    if np.linalg.norm(r):
        assert m.relative_l2 == pytest.approx(np.linalg.norm(e) / np.linalg.norm(r))
    else:
        assert m.relative_l2 is None
    v = y.astype(np.float64).ravel()
    if np.linalg.norm(r) and np.linalg.norm(v):
        assert m.cosine == pytest.approx(np.dot(r, v) / (np.linalg.norm(r)*np.linalg.norm(v)))
    else:
        assert m.cosine is None


@property_settings
@given(w=matrices(max_width=19), group=st.integers(1, 32), bits=st.sampled_from([4, 8]))
def test_phase36_output_error_numpy(w, group, bits):
    a = np.arange(1, w.shape[1]+1, dtype=np.float32)[None, :] / np.float32(8)
    q, s = _group_oracle(w, group, 7 if bits == 4 else 127)
    wq = _reconstruct(q, s, group)
    y0 = a.astype(np.float64) @ w.astype(np.float64).T
    yq = a.astype(np.float64) @ wq.astype(np.float64).T
    report = analyze_quantization(torch.from_numpy(w), torch.from_numpy(a),
                                  bits=bits, group_size=group)
    assert report['quantization_output_error']['rmse'] == pytest.approx(
        np.sqrt(np.mean((yq-y0)**2)), abs=1e-10)
    assert report['operand_output_error']['max_abs'] == 0
    assert report['storage']['scale_bytes'] == s.nbytes
