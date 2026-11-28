# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Offline numerical diagnostics for grouped quantization and mixed precision.

All reference products and metric reductions run on CPU in FP64. Actual
quantization and GEMM use the requested device/backend. This is diagnostic
code, not a latency benchmark or a training operation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .group_quantization import quantize_grouped, dequantize_grouped
from .int4 import quantize_int4, dequantize_int4
from .mixed_precision import mixed_int4_matmul
from .ops import _validate
from .quantization import _inference_only


@dataclass(frozen=True)
class ErrorMetrics:
    count: int
    mae: float
    rmse: float
    max_abs: float
    bias: float
    relative_l2: float | None
    cosine: float | None


def _cpu64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device='cpu', dtype=torch.float64)


def error_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> ErrorMetrics:
    """Compare finite floating tensors of identical, nonempty shape.

    Relative L2 and cosine are undefined (None) when their denominators
    vanish. Bias is estimate minus reference. No per-element percentage
    errors or arbitrary denominator epsilons are used.
    """
    if (reference.shape != estimate.shape or reference.numel() == 0
            or not reference.is_floating_point() or not estimate.is_floating_point()):
        raise ValueError('Expected nonempty floating tensors of equal shape')
    r, y = _cpu64(reference).flatten(), _cpu64(estimate).flatten()
    if not bool(torch.isfinite(r).all() and torch.isfinite(y).all()):
        raise ValueError('Metrics require finite values')
    e = y - r
    if not bool(torch.isfinite(e).all()):
        raise ValueError('Difference exceeds FP64 range')
    # Rescale before reductions to avoid squaring huge values or underflow.
    peak = float(e.abs().max())
    er = e / peak if peak else e
    rms = peak * float(er.square().mean().sqrt())
    mae = peak * float(er.abs().mean())
    bias = peak * float(er.mean())
    rp, yp = float(r.abs().max()), float(y.abs().max())
    rn = float(torch.linalg.vector_norm(r / rp)) if rp else 0.0
    yn = float(torch.linalg.vector_norm(y / yp)) if yp else 0.0
    en = float(torch.linalg.vector_norm(er))
    relative = None
    if rp:
        relative = 0.0 if peak == 0 else (peak / rp) * (en / rn)
        if not math.isfinite(relative):
            raise ValueError('Relative error exceeds FP64 range')
    cosine = None
    if rp and yp:
        cosine = max(-1.0, min(1.0, float(torch.dot(r / rp / rn, y / yp / yn))))
    return ErrorMetrics(r.numel(), mae, rms, peak, bias, relative, cosine)


def _bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def analyze_quantization(weights: torch.Tensor, activations: torch.Tensor, *,
                         bits: int = 4, group_size: int = 128,
                         precision: str = 'fp32', backend: str = 'torch') -> dict:
    """Decompose weight, output quantization, operand and execution errors.

    Reference stages: A W.T, A Wq.T, Ar Wq.T, Ar Wr.T, actual output.
    All four reference products use FP64 on CPU. Wq is the actual FP32 reconstruction;
    Ar and Wr reproduce operand rounding and weight saturation. Stage
    differences are not additive as scalar norms; signed residuals are.
    INT8 analysis is a reconstruction-based reference, not fused INT8 GEMM.
    """
    for x in (weights, activations):
        _validate(x)
        _inference_only(x)
        if x.ndim != 2 or not bool(torch.isfinite(x).all()):
            raise ValueError('Expected finite two-dimensional matrices')
    if weights.shape[1] != activations.shape[1] or weights.device != activations.device:
        raise ValueError('Matrices must share inner dimension and device')
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError('bits must be 4 or 8')
    if type(group_size) is not int or group_size < 1:
        raise ValueError('group_size must be a positive integer')
    dtypes = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    if not isinstance(precision, str) or precision not in dtypes:
        raise ValueError('Use an explicit fp32, fp16 or bf16 precision')
    if backend not in ('torch', 'triton'):
        raise ValueError('Use an explicit torch or triton backend')
    dtype = dtypes[precision]
    limit = torch.finfo(dtype).max
    if bool((activations.float().abs() > limit).any()):
        raise ValueError('Activation exceeds requested precision range')

    if bits == 4:
        codes, scales = quantize_int4(weights, group_size, backend=backend)
        wq = dequantize_int4(codes, scales, weights.shape[1], group_size, backend=backend)
    else:
        codes, scales = quantize_grouped(weights, group_size, backend=backend)
        wq = dequantize_grouped(codes, scales, group_size, backend=backend)
    ar = activations.to(dtype)
    wr = wq.clamp(-limit, limit).to(dtype)
    a64, w64, q64 = map(_cpu64, (activations, weights, wq))
    y0 = a64 @ w64.T
    yq = a64 @ q64.T
    ar64, wr64 = _cpu64(ar), _cpu64(wr)
    ya = ar64 @ q64.T
    yr = ar64 @ wr64.T
    if bits == 4:
        actual = mixed_int4_matmul(activations, codes, scales, group_size,
                                   precision=precision, backend=backend)
    else:
        with torch.autocast(device_type=activations.device.type, enabled=False):
            actual = ar.float() @ wr.float().T

    return {
        'bits': bits, 'group_size': group_size, 'precision': precision,
        'backend': backend, 'device': str(weights.device),
        'weight_shape': list(weights.shape), 'activation_shape': list(activations.shape),
        'execution': 'packed_int4_gemm' if bits == 4 else 'reconstructed_int8_torch_gemm',
        'storage': {
            'code_bytes': _bytes(codes), 'scale_bytes': _bytes(scales),
            'total_bytes': _bytes(codes) + _bytes(scales),
            'fp32_baseline_bytes': weights.numel() * 4,
            'effective_bits_per_weight': 8 * (_bytes(codes) + _bytes(scales)) / weights.numel(),
        },
        'weight_error': asdict(error_metrics(w64, q64)),
        'quantization_output_error': asdict(error_metrics(y0, yq)),
        'activation_rounding_error': asdict(error_metrics(a64, ar64)),
        'weight_operand_error': asdict(error_metrics(q64, wr64)),
        'activation_output_error': asdict(error_metrics(yq, ya)),
        'weight_operand_output_error': asdict(error_metrics(ya, yr)),
        'operand_output_error': asdict(error_metrics(yq, yr)),
        'diagnostics': {
            'activation_rounded_to_zero': int(((a64 != 0) & (ar64 == 0)).sum()),
            'weight_rounded_to_zero': int(((q64 != 0) & (wr64 == 0)).sum()),
            'weight_saturated': int((q64.abs() > limit).sum()),
            'reconstructed_weight_zeros': int((q64 == 0).sum()),
        },
        'execution_error': asdict(error_metrics(yr, actual)),
        'total_output_error': asdict(error_metrics(y0, actual)),
    }
