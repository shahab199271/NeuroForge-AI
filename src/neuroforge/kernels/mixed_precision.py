"""Explicit operand precision for packed INT4 weight-only GEMM."""
from __future__ import annotations

from typing import Literal

import torch

from .int4_gemm import int4_matmul
from .ops import _validate
from .quantization import _inference_only

Precision = Literal['input', 'auto', 'fp32', 'fp16', 'bf16']
_DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def _native_bf16(a: torch.Tensor) -> bool:
    if not a.is_cuda or torch.version.cuda is None:
        return False
    with torch.cuda.device(a.device):
        return (torch.cuda.get_device_capability(a.device)[0] >= 8
                and torch.cuda.is_bf16_supported())


def _auto_dtype(dtype: torch.dtype, native_bf16: bool, fits_bf16: bool,
                is_cuda: bool) -> torch.dtype:
    if dtype == torch.float16:
        return dtype
    if dtype == torch.bfloat16:
        return torch.float32 if is_cuda and not native_bf16 else dtype
    if native_bf16 and fits_bf16:
        return torch.bfloat16
    return torch.float32


def mixed_int4_matmul(a: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                      group_size: int = 128, *, precision: Precision = 'input',
                      backend: str = 'auto') -> torch.Tensor:
    """Compute packed INT4 GEMM with an explicit floating-point operand dtype.

    ``input`` preserves the phase 3.4 contract. ``auto`` optionally reduces
    FP32 activations to BF16 on supported NVIDIA GPUs, never implicitly to
    FP16. Explicit modes select the requested operand dtype. The result and
    accumulator remain FP32; scales remain FP32 and stored bytes are unchanged.
    Activation conversion rejects overflow; weight saturation follows phase
    3.4. FP32 output overflow is not prevented. Inference only.
    """
    _validate(a)
    _inference_only(a)
    if a.ndim != 2:
        raise ValueError('Expected a two-dimensional activation matrix')
    if backend not in ('auto', 'torch', 'triton'):
        raise ValueError('Unknown backend')
    if not isinstance(precision, str) or (precision not in ('input', 'auto')
                                           and precision not in _DTYPES):
        raise ValueError('Unknown precision policy')
    if precision == 'input':
        return int4_matmul(a, packed, scales, group_size, backend=backend)
    if not bool(torch.isfinite(a).all()):
        raise ValueError('Expected finite activations')
    if precision == 'auto':
        native = backend != 'torch' and _native_bf16(a)
        fits = (bool((a.abs() <= torch.finfo(torch.bfloat16).max).all())
                if native and a.dtype == torch.float32 else False)
        dtype = _auto_dtype(a.dtype, native, fits, a.is_cuda and backend != 'torch')
    else:
        dtype = _DTYPES[precision]
    # Reject an out-of-range conversion rather than silently clip activations.
    if dtype != a.dtype and not bool((a.float().abs() <= torch.finfo(dtype).max).all()):
        raise ValueError('Activation exceeds requested precision range')
    operands = a.to(dtype=dtype)
    return int4_matmul(operands, packed, scales, group_size, backend=backend)
