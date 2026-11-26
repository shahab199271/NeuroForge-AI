"""Experimental forward-only fused operators and tiled matrix multiplication."""
import math

import torch

from .ops import _backend, _validate, rmsnorm


def _match(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.device != y.device or x.dtype != y.dtype:
        raise ValueError('Inputs must share dtype and device')


def residual_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                     eps: float = 1e-6, *, backend: str = 'auto'
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized activation and residual sum, rounded to input dtype."""
    _validate(x)
    _match(x, residual)
    _match(x, weight)
    if x.shape != residual.shape or weight.shape != (x.shape[-1],):
        raise ValueError('Invalid residual or weight shape')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    if _backend(x, backend, residual, weight):
        from .advanced_triton import run_residual
        return run_residual(x, residual, weight, eps)
    summed = x + residual
    return rmsnorm(summed, weight, eps, backend='torch'), summed


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
         *, backend: str = 'auto') -> torch.Tensor:
    """Split-half RoPE. cos/sin have shape (*x.shape[:-1], head_dim/2)."""
    _validate(x)
    _match(x, cos)
    _match(x, sin)
    d = x.shape[-1]
    if d % 2 or cos.shape != (*x.shape[:-1], d // 2) or sin.shape != cos.shape:
        raise ValueError('RoPE needs an even head dimension and matching half-width tables')
    if _backend(x, backend, cos, sin, row_reduction=False):
        from .advanced_triton import run_rope
        return run_rope(x, cos, sin)
    a, b = x.float().chunk(2, dim=-1)
    c, s = cos.float(), sin.float()
    return torch.cat((a*c-b*s, b*c+a*s), dim=-1).to(x.dtype)


def matmul(a: torch.Tensor, b: torch.Tensor, *, backend: str = 'auto') -> torch.Tensor:
    """2-D GEMM. Triton uses IEEE FP32; PyTorch follows caller precision settings."""
    _validate(a)
    _validate(b)
    _match(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('Expected compatible 2-D matrices')
    if _backend(a, backend, b, row_reduction=False):
        from .advanced_triton import run_matmul
        return run_matmul(a, b)
    return a @ b
