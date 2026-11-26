"""Experimental NVIDIA Triton kernels; GPU execution not yet validated."""
import torch
import triton
import triton.language as tl


@triton.jit
def _residual(X, R, W, Y, S, D: tl.constexpr, B: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    p = row * D + i
    x = tl.load(X + p, i < D, other=0).to(tl.float32)
    r = tl.load(R + p, i < D, other=0).to(tl.float32)
    z = (x + r).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + i, i < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(z * z, 0) / D + EPS)
    tl.store(S + p, z, i < D)
    tl.store(Y + p, z * inv * w, i < D)


@triton.jit
def _rope(X, C, S, Y, H: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, col = i // H, i % H
    p = row * (2 * H) + col
    a = tl.load(X + p, i < N, other=0).to(tl.float32)
    b = tl.load(X + p + H, i < N, other=0).to(tl.float32)
    c = tl.load(C + i, i < N, other=0).to(tl.float32)
    s = tl.load(S + i, i < N, other=0).to(tl.float32)
    tl.store(Y + p, a*c-b*s, i < N)
    tl.store(Y + p + H, b*c+a*s, i < N)


@triton.autotune(configs=[
    triton.Config({'BM': 16, 'BN': 32, 'BK': 32}, num_warps=4, num_stages=2),
    triton.Config({'BM': 32, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
], key=['M', 'N', 'K', 'DTYPE', 'ARCH', 'DEVICE'])
@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          DTYPE: tl.constexpr, ARCH: tl.constexpr, DEVICE: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for t in range(tl.cdiv(K, BK)):
        q = t * BK + k
        a = tl.load(A + m[:, None]*K + q[None, :],
                    (m[:, None] < M) & (q[None, :] < K), other=0)
        b = tl.load(B + q[:, None]*N + n[None, :],
                    (q[:, None] < K) & (n[None, :] < N), other=0)
        acc += tl.dot(a, b, input_precision='ieee')
    tl.store(C + m[:, None]*N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def run_residual(x, residual, weight, eps):
    x, residual, weight = x.contiguous(), residual.contiguous(), weight.contiguous()
    y, s = torch.empty_like(x), torch.empty_like(x)
    d = x.shape[-1]
    block = triton.next_power_of_2(d)
    with torch.cuda.device(x.device):
        _residual[(x.numel()//d,)](x, residual, weight, y, s, d, block, eps,
                                  num_warps=4 if block <= 2048 else 8)
    return y, s


def run_rope(x, cos, sin):
    x, cos, sin = x.contiguous(), cos.contiguous(), sin.contiguous()
    y = torch.empty_like(x)
    n = cos.numel()
    with torch.cuda.device(x.device):
        _rope[(triton.cdiv(n, 256),)](x, cos, sin, y, x.shape[-1]//2, n, 256)
    return y


def run_matmul(a, b):
    a, b = a.contiguous(), b.contiguous()
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), dtype=a.dtype, device=a.device)
    with torch.cuda.device(a.device):
        arch = torch.cuda.get_device_capability(a.device)
        grid = lambda meta: (triton.cdiv(m, meta['BM']), triton.cdiv(n, meta['BN']))  # noqa: E731
        _gemm[grid](a, b, c, m, n, k, str(a.dtype), arch, a.device.index)
    return c
