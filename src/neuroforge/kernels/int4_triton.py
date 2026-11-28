# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Grouped INT4 quantization and independent byte-level packing kernels."""
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
    s = tl.where(peak == 0, 1., tl.maximum(tl.div_rn(peak, 7.), 1.1754943508222875e-38))
    r = tl.minimum(tl.maximum(tl.div_rn(x, s), -7.), 7.)
    m = tl.abs(r)
    w = tl.floor(m)
    v = w + (m - w >= 0.5).to(tl.float32)
    tl.store(Q + offset, tl.where(r < 0, -v, v).to(tl.int8), mask)
    tl.store(S + p, s)


def run_quantize(x, size, groups):
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((*x.shape[:-1], groups), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(size)
    with torch.cuda.device(x.device):
        _quant[(s.numel(),)](x, q, s, x.shape[-1], groups, size, block,
                            num_warps=4 if block <= 2048 else 8)
    return q, s


@triton.jit
def _pack(Q, P, N: tl.constexpr, D: tl.constexpr, W: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*B + tl.arange(0, B)
    row, col = i // W, (i % W)*2
    a = tl.load(Q + row*D + col, i < N, other=0).to(tl.int32)
    b = tl.load(Q + row*D + col+1, (i < N) & (col+1 < D), other=0).to(tl.int32)
    tl.store(P+i, ((a & 15) | ((b & 15) << 4)).to(tl.uint8), i < N)


@triton.jit
def _unpack(P, Q, N: tl.constexpr, D: tl.constexpr, W: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*B + tl.arange(0, B)
    row, col = i // D, i % D
    byte = tl.load(P + row*W + col//2, i < N, other=0).to(tl.int32)
    nibble = (byte >> ((col % 2)*4)) & 15
    tl.store(Q+i, tl.where(nibble >= 8, nibble-16, nibble).to(tl.int8), i < N)


def run_pack(q):
    q = q.contiguous()
    d = q.shape[-1]
    w = (d+1)//2
    p = torch.empty((*q.shape[:-1], w), device=q.device, dtype=torch.uint8)
    with torch.cuda.device(q.device):
        _pack[(triton.cdiv(p.numel(), 256),)](q, p, p.numel(), d, w, 256)
    return p


def run_unpack(packed, width):
    packed = packed.contiguous()
    q = torch.empty((*packed.shape[:-1], width), device=packed.device, dtype=torch.int8)
    with torch.cuda.device(packed.device):
        _unpack[(triton.cdiv(q.numel(), 256),)](packed, q, q.numel(), width,
                                              packed.shape[-1], 256)
    return q
