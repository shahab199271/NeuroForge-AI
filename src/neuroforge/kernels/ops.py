"""Last-axis operators. No automatic modification of existing models."""
import math

import torch
import torch.nn.functional as F


def _validate(x: torch.Tensor) -> None:
    if x.ndim < 1 or x.numel() == 0 or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Expected a nonempty floating-point tensor (fp32/fp16/bf16)")


def _backend(x: torch.Tensor, backend: str, *others: torch.Tensor,
             row_reduction: bool = True) -> bool:
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("Unknown backend")
    if backend == "torch":
        return False
    # Conservative resource limit for single-program reductions; larger rows
    # use the reference until a tiled reduction has been validated.
    tensors = (x, *others)
    has_forward_grad = any(torch.autograd.forward_ad.unpack_dual(t).tangent is not None
                           for t in tensors)
    eligible = (x.is_cuda and torch.version.cuda is not None
                and (not row_reduction or x.shape[-1] <= 8192)
                and not has_forward_grad
                and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors)))
    if eligible:
        major, _ = torch.cuda.get_device_capability(x.device)
        eligible = major >= 8  # NVIDIA Ampere+ target
        if eligible and x.dtype == torch.bfloat16:
            with torch.cuda.device(x.device):
                eligible = torch.cuda.is_bf16_supported()
    if eligible:
        try:
            from . import triton_ops  # noqa: F401
            return True
        except ImportError:
            pass
    if backend == "triton":
        raise RuntimeError("Triton requires NVIDIA CUDA, installed Triton, supported shape and inference")
    return False


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
            *, backend: str = "auto") -> torch.Tensor:
    _validate(x)
    if weight.shape != (x.shape[-1],) or weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError("Weight must match last dimension, dtype and device")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if _backend(x, backend, weight):
        from .triton_ops import run_rmsnorm
        return run_rmsnorm(x, weight, eps)
    z = x.float()
    return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def softmax(x: torch.Tensor, *, backend: str = "auto") -> torch.Tensor:
    _validate(x)
    if _backend(x, backend):
        from .triton_ops import run_softmax
        return run_softmax(x)
    return torch.softmax(x, dim=-1, dtype=torch.float32).to(x.dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor,
           *, backend: str = "auto") -> torch.Tensor:
    _validate(gate)
    if up.shape != gate.shape or up.dtype != gate.dtype or up.device != gate.device:
        raise ValueError("Inputs must share shape, dtype and device")
    if _backend(gate, backend, up, row_reduction=False):
        from .triton_ops import run_swiglu
        return run_swiglu(gate, up)
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)
