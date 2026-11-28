# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Symmetric row-wise INT8 storage with FP32 scales and reconstruction."""
from __future__ import annotations

import torch

from .ops import _backend, _validate

_MIN_SCALE = torch.finfo(torch.float32).tiny
_MAX_FLOAT = torch.finfo(torch.float32).max


def _inference_only(x: torch.Tensor) -> None:
    if (torch.is_grad_enabled() and x.requires_grad) or (
        torch.autograd.forward_ad.unpack_dual(x).tangent is not None
    ):
        raise ValueError("INT8 storage is inference-only; detach inputs explicitly")


def quantize_int8(x: torch.Tensor, *, backend: str = "auto") -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last axis; return int8 codes and FP32 scales of shape (..., 1).

    Finite inputs only. Round ties away from zero, reserve -128, and use scale 1
    for zero rows. The minimum scale avoids subnormal divisors.
    """
    _validate(x)
    _inference_only(x)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Quantization requires finite inputs")
    if _backend(x, backend):
        from .quantization_triton import run_quantize
        return run_quantize(x)
    z = x.float()
    peak = z.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(peak == 0, 1.0, (peak / 127.0).clamp_min(_MIN_SCALE))
    ratio = (z / scale).clamp(-127, 127)
    # Compare the fractional part rather than adding 0.5 to avoid rounding
    # values just below a half-integer across the boundary.
    magnitude = ratio.abs()
    whole = magnitude.floor()
    rounded = whole + (magnitude - whole >= 0.5).to(magnitude.dtype)
    return (rounded * ratio.sign()).to(torch.int8), scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, *, backend: str = "auto") -> torch.Tensor:
    """Reconstruct FP32 activations, saturating at finite FP32 bounds."""
    if q.dtype != torch.int8 or q.ndim < 1 or q.numel() == 0:
        raise ValueError("Expected a nonempty int8 tensor")
    if (scale.dtype != torch.float32 or scale.shape != (*q.shape[:-1], 1)
            or scale.device != q.device):
        raise ValueError("Expected FP32 row scales on the same device")
    _inference_only(scale)
    if not bool((torch.isfinite(scale) & (scale >= _MIN_SCALE)).all()) or bool((q == -128).any()):
        raise ValueError("Expected finite normal positive scales and codes in [-127, 127]")
    if _backend(scale, backend, row_reduction=False):
        from .quantization_triton import run_dequantize
        return run_dequantize(q, scale)
    return (q.float() * scale).clamp(-_MAX_FLOAT, _MAX_FLOAT)
