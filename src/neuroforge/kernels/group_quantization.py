"""Symmetric grouped INT8 storage along the last axis."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .ops import _backend, _validate
from .quantization import _inference_only, _MIN_SCALE, _MAX_FLOAT


def _groups(width: int, group_size: int) -> int:
    if type(group_size) is not int or group_size < 1:
        raise ValueError('group_size must be a positive integer')
    return (width + group_size - 1) // group_size


def _use_triton(x: torch.Tensor, backend: str, group_size: int) -> bool:
    if group_size > 8192:
        if backend == 'triton':
            raise RuntimeError('Triton supports group sizes up to 8192')
        if backend != 'auto' and backend != 'torch':
            raise ValueError('Unknown backend')
        return False
    return _backend(x, backend, row_reduction=False)


def quantize_grouped(x: torch.Tensor, group_size: int = 128, *,
                     backend: str = 'auto') -> tuple[torch.Tensor, torch.Tensor]:
    """Return unpadded INT8 codes and FP32 scales (..., ceil(D/group_size))."""
    _validate(x)
    _inference_only(x)
    d = x.shape[-1]
    g = _groups(d, group_size)
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Expected finite input')
    # Avoid allocating padding beyond the actual width for a single group.
    size = min(group_size, d)
    if _use_triton(x, backend, size):
        from .group_quantization_triton import run_quantize
        return run_quantize(x, size, g)
    z = F.pad(x.float(), (0, g * size - d)).reshape(*x.shape[:-1], g, size)
    peak = z.abs().amax(-1, keepdim=True)
    scale = torch.where(peak == 0, 1.0, (peak / 127).clamp_min(_MIN_SCALE))
    ratio = (z / scale).clamp(-127, 127)
    mag = ratio.abs()
    whole = mag.floor()
    q = ((whole + (mag - whole >= 0.5).to(mag.dtype)) * ratio.sign()).to(torch.int8)
    return q.flatten(-2)[..., :d].contiguous(), scale.squeeze(-1)


def dequantize_grouped(q: torch.Tensor, scales: torch.Tensor, group_size: int = 128, *,
                       backend: str = 'auto') -> torch.Tensor:
    """Reconstruct FP32 values; scale layout must match the original group size."""
    if q.dtype != torch.int8 or q.ndim < 1 or q.numel() == 0:
        raise ValueError('Expected nonempty INT8 codes')
    d = q.shape[-1]
    g = _groups(d, group_size)
    if (scales.dtype != torch.float32 or scales.device != q.device
            or scales.shape != (*q.shape[:-1], g)):
        raise ValueError('Invalid scale shape, device or dtype')
    _inference_only(scales)
    if not bool((torch.isfinite(scales) & (scales >= _MIN_SCALE)).all()) or bool((q == -128).any()):
        raise ValueError('Invalid scale or reserved INT8 code')
    size = min(group_size, d)
    if _backend(scales, backend, row_reduction=False):
        from .group_quantization_triton import run_dequantize
        return run_dequantize(q, scales, size, g)
    expanded = scales.repeat_interleave(size, dim=-1)[..., :d]
    return (q.float() * expanded).clamp(-_MAX_FLOAT, _MAX_FLOAT)


def quantize_channels(x: torch.Tensor, axis: int = 0, *,
                      backend: str = 'auto') -> tuple[torch.Tensor, torch.Tensor]:
    """One scale per channel, reducing all axes except the selected axis."""
    _validate(x)
    if type(axis) is not int or not -x.ndim <= axis < x.ndim:
        raise ValueError('Invalid channel axis')
    moved = x.movedim(axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    q, s = quantize_grouped(flat, flat.shape[1], backend=backend)
    return q.reshape(moved.shape).movedim(0, axis).contiguous(), s.squeeze(-1)


def dequantize_channels(q: torch.Tensor, scales: torch.Tensor, axis: int = 0, *,
                        backend: str = 'auto') -> torch.Tensor:
    if q.ndim < 1 or q.numel() == 0 or type(axis) is not int or not -q.ndim <= axis < q.ndim:
        raise ValueError('Invalid channel axis or empty codes')
    moved = q.movedim(axis, 0)
    if scales.shape != (moved.shape[0],):
        raise ValueError('Expected one scale per channel')
    flat = moved.reshape(moved.shape[0], -1)
    y = dequantize_grouped(flat, scales.unsqueeze(-1), flat.shape[1], backend=backend)
    return y.reshape(moved.shape).movedim(0, axis).contiguous()
