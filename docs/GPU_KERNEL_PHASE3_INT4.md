# Packed symmetric INT4

The format stores signed two's-complement codes in [-7, 7], reserving -8.
Column 2j occupies the low nibble of byte j; column 2j+1 occupies the high
nibble. Each row is packed independently, with zero high-nibble padding for
odd widths. The original width and group size are required metadata.

Each last-axis group uses max(abs(x))/7, floored at the smallest normal FP32
scale, with scale 1 for zero groups. Rounding ties go away from zero. The
FP32 reconstruction saturates at finite FP32 bounds, using the existing
phase 3.2 grouped reconstruction. The approximate ordinary-input absolute
error bound is half the group scale plus floating-point rounding error.
This is not a model-accuracy guarantee.

Quantization produces temporary INT8 codes before byte packing. Separate
kernels avoid competing writes when an odd group boundary splits a byte.
Unpacking is exact for valid codes. Reconstruction currently materializes
INT8 and FP32 tensors; fused dequantization/GEMM is a later milestone.

```python
import torch
from neuroforge.kernels.int4 import quantize_int4, dequantize_int4

x = torch.randn(4, 897)
p, s = quantize_int4(x, group_size=128)
y = dequantize_int4(p, s, width=897, group_size=128)
```

Storage per row: ceil(D/2) + 4*ceil(D/group_size) bytes, excluding metadata
and intermediate tensors. For D=896 and group size 128 this is 476 bytes,
versus 1792 bytes for FP16. This is format size, not measured VRAM usage.

CPU fallback needs no Triton installation. Explicit Triton dispatch targets
NVIDIA Ampere or newer under the existing dispatch policy. Quantization
reductions have an effective group width limit of 8192; packing/unpacking
are tiled. Validation performs CUDA-synchronizing checks, so these checked
APIs are not yet optimized for low-latency serving.

```bash
pytest -q -rs tests/test_int4.py
ruff check src tests
```

Local combined kernel suite using prior phase archives: 125 passed, 65 skipped
(no GPU), 18 warnings. Python bytecode compilation succeeded. Ruff was not
installed locally. GPU compilation, numerical validation, benchmarking, and
full current-main CI remain pending. No performance gain is claimed.
