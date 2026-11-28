# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Forward-only row quantization and tiled FP32 reconstruction."""
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, Q, S, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    offset = row * D + col
    x = tl.load(X + offset, col < D, other=0).to(tl.float32)
    peak = tl.max(tl.abs(x), 0)
    scale = tl.where(peak == 0, 1.0, tl.maximum(tl.div_rn(peak, 127.0), 1.1754943508222875e-38))
    ratio = tl.minimum(tl.maximum(tl.div_rn(x, scale), -127.0), 127.0)
    mag = tl.abs(ratio)
    whole = tl.floor(mag)
    rounded = whole + (mag - whole >= 0.5).to(tl.float32)
    code = tl.where(ratio < 0, -rounded, rounded).to(tl.int8)
    tl.store(Q + offset, code, col < D)
    tl.store(S + row, scale)


@triton.jit
def _dequantize(Q, S, Y, N: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    q = tl.load(Q + i, i < N, other=0).to(tl.float32)
    scale = tl.load(S + i // D, i < N, other=1).to(tl.float32)
    y = tl.minimum(tl.maximum(q * scale, -3.4028234663852886e38), 3.4028234663852886e38)
    tl.store(Y + i, y, i < N)


def run_quantize(x):
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
    d = x.shape[-1]
    block = triton.next_power_of_2(d)
    with torch.cuda.device(x.device):
        _quantize[(x.numel() // d,)](x, q, scale, d, block,
                                    num_warps=4 if block <= 2048 else 8)
    return q, scale


def run_dequantize(q, scale):
    q, scale = q.contiguous(), scale.contiguous()
    y = torch.empty_like(q, dtype=torch.float32)
    n = q.numel()
    with torch.cuda.device(q.device):
        _dequantize[(triton.cdiv(n, 256),)](q, scale, y, n, q.shape[-1], 256)
    return y
