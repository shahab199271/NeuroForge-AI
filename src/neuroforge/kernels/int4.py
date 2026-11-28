# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Symmetric grouped INT4 storage. Low nibble precedes high nibble."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .group_quantization import _groups, _use_triton, dequantize_grouped
from .ops import _backend, _validate
from .quantization import _inference_only, _MIN_SCALE


def pack_int4(q: torch.Tensor, *, backend: str = 'auto') -> torch.Tensor:
    """Pack signed codes in [-7, 7] into uint8; zero-pad an odd final column."""
    if q.dtype != torch.int8 or q.ndim < 1 or q.numel() == 0:
        raise ValueError('Expected nonempty INT8 codes')
    if bool(((q < -7) | (q > 7)).any()):
        raise ValueError('INT4 codes must be in [-7, 7]')
    if _backend(q, backend, row_reduction=False):
        from .int4_triton import run_pack
        return run_pack(q)
    q = F.pad(q, (0, q.shape[-1] % 2)).to(torch.int16)
    return ((q[..., ::2] & 15) | ((q[..., 1::2] & 15) << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, width: int, *, backend: str = 'auto') -> torch.Tensor:
    """Decode two's-complement nibbles; reject reserved -8 and nonzero padding."""
    if (packed.dtype != torch.uint8 or packed.ndim < 1 or packed.numel() == 0
            or type(width) is not int or width < 1 or packed.shape[-1] != (width + 1)//2):
        raise ValueError('Invalid packed tensor or original width')
    if width % 2 and bool((packed[..., -1] & 240).any()):
        raise ValueError('Odd-width padding must be zero')
    if _backend(packed, backend, row_reduction=False):
        from .int4_triton import run_unpack
        q = run_unpack(packed, width)
    else:
        b = packed.to(torch.int16)
        n = torch.stack((b & 15, b >> 4), dim=-1).flatten(-2)[..., :width]
        q = torch.where(n >= 8, n - 16, n).to(torch.int8).contiguous()
    if bool((q == -8).any()):
        raise ValueError('Reserved INT4 code -8')
    return q


def quantize_int4(x: torch.Tensor, group_size: int = 128, *,
                  backend: str = 'auto') -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed uint8 codes and FP32 group scales; retain width/group_size."""
    _validate(x)
    _inference_only(x)
    d = x.shape[-1]
    g = _groups(d, group_size)
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Expected finite input')
    size = min(d, group_size)
    if _use_triton(x, backend, size):
        from .int4_triton import run_quantize
        q, scales = run_quantize(x, size, g)
    else:
        z = F.pad(x.float(), (0, g*size-d)).reshape(*x.shape[:-1], g, size)
        peak = z.abs().amax(-1, keepdim=True)
        s = torch.where(peak == 0, 1., (peak / 7.).clamp_min(_MIN_SCALE))
        r = (z / s).clamp(-7, 7)
        m = r.abs()
        w = m.floor()
        q = ((w + (m-w >= .5).to(m.dtype))*r.sign()).to(torch.int8)
        q = q.flatten(-2)[..., :d].contiguous()
        scales = s.squeeze(-1)
    return pack_int4(q, backend=backend), scales


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, width: int,
                    group_size: int = 128, *, backend: str = 'auto') -> torch.Tensor:
    """Decode then reconstruct FP32. Not a fused GEMM or hot-path implementation."""
    q = unpack_int4(packed, width, backend=backend)
    return dequantize_grouped(q, scales, group_size, backend=backend)
