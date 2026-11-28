# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Forward-only Triton kernels; fp32 arithmetic and masked tail loads."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms(X, W, Y, N: tl.constexpr, B: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    x = tl.load(X + row * N + i, i < N, other=0).to(tl.float32)
    w = tl.load(W + i, i < N, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    tl.store(Y + row * N + i, x * inv * w, i < N)


@triton.jit
def _soft(X, Y, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    x = tl.load(X + row * N + i, i < N, other=-float("inf")).to(tl.float32)
    e = tl.exp(x - tl.max(x, 0))
    tl.store(Y + row * N + i, e / tl.sum(e, 0), i < N)


@triton.jit
def _swi(G, U, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.load(G + i, i < N, other=0).to(tl.float32)
    u = tl.load(U + i, i < N, other=0).to(tl.float32)
    tl.store(Y + i, (g * tl.sigmoid(g)) * u, i < N)


def _row_launch(kernel, x, *args):
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.shape[-1]
    block = triton.next_power_of_2(n)
    warps = 4 if block <= 2048 else 8 if block <= 8192 else 16
    with torch.cuda.device(x.device):
        kernel[(x.numel() // n,)](x, *args, y, n, block, num_warps=warps)
    return y


def run_rmsnorm(x, weight, eps):
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.shape[-1]
    b = triton.next_power_of_2(n)
    with torch.cuda.device(x.device):
        _rms[(x.numel() // n,)](x, weight.contiguous(), y, n, b, eps,
                               num_warps=4 if b <= 2048 else 8 if b <= 8192 else 16)
    return y


def run_softmax(x):
    return _row_launch(_soft, x)


def run_swiglu(gate, up):
    gate, up = gate.contiguous(), up.contiguous()
    y = torch.empty_like(gate)
    n = gate.numel()
    with torch.cuda.device(gate.device):
        _swi[(triton.cdiv(n, 256),)](gate, up, y, n, 256, num_warps=4)
    return y
