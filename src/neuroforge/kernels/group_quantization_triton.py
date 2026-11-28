# NeuroForge AI
# Maintainer: Shahab Baloochi

"""One reduction program per row/group; tiled reconstruction."""
import torch
import triton
import triton.language as tl


@triton.jit
def _quant(X, Q, S, D: tl.constexpr, G: tl.constexpr, SIZE: tl.constexpr, B: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    row, group = p // G, p % G
    i = tl.arange(0, B)
    col = group * SIZE + i
    mask = (i < SIZE) & (col < D)
    offset = row * D + col
    x = tl.load(X + offset, mask, other=0).to(tl.float32)
    peak = tl.max(tl.abs(x), 0)
    s = tl.where(peak == 0, 1., tl.maximum(tl.div_rn(peak, 127.), 1.1754943508222875e-38))
    r = tl.minimum(tl.maximum(tl.div_rn(x, s), -127.), 127.)
    m = tl.abs(r)
    w = tl.floor(m)
    v = w + (m - w >= 0.5).to(tl.float32)
    tl.store(Q + offset, tl.where(r < 0, -v, v).to(tl.int8), mask)
    tl.store(S + p, s)


@triton.jit
def _dequant(Q, S, Y, N: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
             SIZE: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    q = tl.load(Q + i, i < N, other=0).to(tl.float32)
    s = tl.load(S + (i // D) * G + (i % D) // SIZE, i < N, other=1.)
    y = tl.minimum(tl.maximum(q * s, -3.4028234663852886e38), 3.4028234663852886e38)
    tl.store(Y + i, y, i < N)


def run_quantize(x, size, groups):
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((*x.shape[:-1], groups), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(size)
    with torch.cuda.device(x.device):
        _quant[(s.numel(),)](x, q, s, x.shape[-1], groups, size, block,
                            num_warps=4 if block <= 2048 else 8)
    return q, s


def run_dequantize(q, scales, size, groups):
    q, scales = q.contiguous(), scales.contiguous()
    y = torch.empty_like(q, dtype=torch.float32)
    with torch.cuda.device(q.device):
        _dequant[(triton.cdiv(q.numel(), 256),)](q, scales, y, q.numel(), q.shape[-1],
                                               groups, size, 256)
    return y
