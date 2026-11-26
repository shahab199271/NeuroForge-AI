"""Tiled fused INT4 decode, scale and GEMM; no full decoded weight buffer."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, P, S, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          W: tl.constexpr, G: tl.constexpr, SIZE: tl.constexpr, LIMIT: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    n = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK).to(tl.int64)
    acc = tl.zeros((BM, BN), tl.float32)
    for t in range(tl.cdiv(K, BK)):
        q = t * BK + k
        a = tl.load(A + m[:, None] * K + q[None, :],
                    (m[:, None] < M) & (q[None, :] < K), other=0)
        mask = (q[:, None] < K) & (n[None, :] < N)
        byte = tl.load(P + n[None, :] * W + q[:, None] // 2,
                       mask, other=0).to(tl.int32)
        nibble = (byte >> ((q[:, None] % 2) * 4).to(tl.int32)) & 15
        code = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
        scale = tl.load(S + n[None, :] * G + q[:, None] // SIZE,
                        mask, other=0).to(tl.float32)
        weight = tl.minimum(tl.maximum(code * scale, -LIMIT), LIMIT)
        b = weight.to(A.dtype.element_ty)
        acc = tl.dot(a, b, acc, input_precision='ieee')
    tl.store(C + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def run_matmul(a: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
               size: int, limit: float) -> torch.Tensor:
    a, packed, scales = a.contiguous(), packed.contiguous(), scales.contiguous()
    m, k = a.shape
    n, w = packed.shape
    c = torch.empty((m, n), dtype=torch.float32, device=a.device)
    bm, bn, bk = 16, 32, 32
    with torch.cuda.device(a.device):
        _gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            a, packed, scales, c, m, n, k, w, scales.shape[1], size, limit,
            bm, bn, bk, num_warps=4, num_stages=2,
        )
    return c
