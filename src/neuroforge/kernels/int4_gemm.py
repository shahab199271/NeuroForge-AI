"""Weight-only INT4 matrix multiplication with FP32 accumulation.

Weights use the phase 3.3 [N, ceil(K/2)] row-packed format.
"""
from __future__ import annotations

import torch

from .group_quantization import _groups
from .int4 import dequantize_int4
from .ops import _backend, _validate
from .quantization import _MIN_SCALE, _inference_only


def int4_matmul(a: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                group_size: int = 128, *, backend: str = 'auto') -> torch.Tensor:
    """Compute A @ W.T for A [M,K] and packed W [N,ceil(K/2)].

    Reconstructed weights saturate to the finite range of A.dtype, then round
    to A.dtype before the dot product. Returns FP32 (not clamped). Inference
    only. Validation synchronizes CUDA; the Triton dot path never materializes
    a full decoded weight matrix. The torch reference does.
    """
    _validate(a)
    _inference_only(a)
    if a.ndim != 2:
        raise ValueError('Expected a two-dimensional activation matrix')
    k = a.shape[1]
    g = _groups(k, group_size)
    if (packed.ndim != 2 or packed.dtype != torch.uint8 or packed.numel() == 0
            or packed.shape[1] != (k + 1) // 2 or packed.device != a.device):
        raise ValueError('Invalid packed weight shape, dtype or device')
    if (scales.shape != (packed.shape[0], g) or scales.dtype != torch.float32
            or scales.device != a.device):
        raise ValueError('Invalid group scale shape, dtype or device')
    _inference_only(scales)
    if not bool(torch.isfinite(a).all()):
        raise ValueError('Expected finite activations')
    if not bool((torch.isfinite(scales) & (scales >= _MIN_SCALE)).all()):
        raise ValueError('Expected finite positive normal FP32 scales')
    if k % 2 and bool((packed[:, -1] & 240).any()):
        raise ValueError('Odd-width padding must be zero')
    # Inspect packed bytes directly: no full decoded INT8 matrix on fused path.
    if bool((((packed & 15) == 8) | ((packed >> 4) == 8)).any()):
        raise ValueError('Reserved INT4 code -8')
    limit = torch.finfo(a.dtype).max
    if _backend(a, backend, scales, row_reduction=False):
        # Older dispatch versions may not check BF16 support themselves.
        if a.dtype == torch.bfloat16:
            with torch.cuda.device(a.device):
                bf16_ok = torch.cuda.is_bf16_supported()
            if not bf16_ok:
                if backend == 'triton':
                    raise RuntimeError('BF16 not supported on this device')
                return _reference(a, packed, scales, group_size, limit)
        from .int4_gemm_triton import run_matmul
        return run_matmul(a, packed, scales, min(group_size, k), limit)
    return _reference(a, packed, scales, group_size, limit)


def _reference(a: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
               group_size: int, limit: float) -> torch.Tensor:
    w = dequantize_int4(packed, scales, a.shape[1], group_size, backend='torch')
    w = w.clamp(-limit, limit).to(a.dtype)
    # Disable autocast: output and accumulation are explicitly FP32. CUDA
    # FP32 matmul precision still follows the caller's PyTorch settings.
    with torch.autocast(device_type=a.device.type, enabled=False):
        return a.float() @ w.float().T
