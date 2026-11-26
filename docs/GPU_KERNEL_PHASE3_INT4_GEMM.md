# Fused packed INT4 weight-only GEMM

`int4_matmul(a, packed, scales, group_size=128, backend='auto')` computes
`A @ W.T` with activations `[M,K]`, packed weights `[N,ceil(K/2)]`, and FP32
scales `[N,ceil(K/group_size)]`. The output is `[M,N]` in FP32. Weights follow
phase 3.3: low nibble first, signed two's-complement, -8 reserved, odd-width
high nibble zero. The effective group width is min(group_size,K).

The fused Triton program assigns one output tile (16x32) per program and
iterates K in blocks of 32. For each block, it loads the relevant packed bytes,
extracts signed nibbles, loads their individual group scales, multiplies in
FP32 and clamps to the finite activation dtype range. Weights are then rounded
to the activation dtype and used by `tl.dot` with FP32 accumulation and IEEE
FP32 input precision. Arbitrary positive group sizes, including odd groups and
groups crossing a tile boundary, use elementwise scale indexing. All three
matrix dimensions are masked at their tails.

The public API accepts FP32, FP16 and BF16 activations, rejects nonfinite inputs,
invalid packed codes, invalid scales and active gradients. It checks bytes
without materializing decoded weights. Validation may allocate temporary packed-
size masks and synchronize CUDA; noncontiguous inputs may require copies.
The fused computation itself does not allocate a full decoded INT8 or FP32
weight matrix. This is weight-only compression, not native INT4 dot arithmetic.
The output may overflow FP32 for extreme operands; it is not saturated.

The PyTorch reference reconstructs the full weight matrix, applies the same
clamp and dtype rounding, disables autocast and uses FP32 matmul. CUDA reference
precision follows the caller's PyTorch FP32 matmul settings. Tests use an FP64
oracle on rounded operands. Auto dispatch falls back for unsupported devices;
explicit Triton dispatch raises. GPU compilation and execution remain unverified.

```python
import torch
from neuroforge.kernels.int4 import quantize_int4
from neuroforge.kernels.int4_gemm import int4_matmul

w = torch.randn(96, 129)
a = torch.randn(4, 129, dtype=torch.float16)
p, s = quantize_int4(w, 32)
y = int4_matmul(a, p, s, 32)
assert y.shape == (4, 96) and y.dtype == torch.float32
```

Run `pytest -q tests/test_int4_gemm.py` and `ruff check src tests`.
Fixed tiles are a conservative starting point; autotuning, performance and
memory measurements are deferred. No speedup or peak VRAM claim is made.
