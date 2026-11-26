# Row-wise symmetric INT8 quantization

First milestone of the low-precision kernel lab. The public operators are
`quantize_int8(x, backend=...)` and `dequantize_int8(q, scale, backend=...)`
in `neuroforge.kernels.quantization`. Existing model paths are unchanged.

For each last-axis row, compute `s = max(abs(x))/127`, clamp nonzero scales
at the smallest normal FP32 value, and use `s = 1` for an all-zero row.
The code is `q = clamp(round(x/s), -127, 127)` with half-integer ties away
from zero. Reconstruction is `q*s` in FP32, saturated at finite FP32 bounds.
The format reserves -128 and stores FP32 scales of shape `(..., 1)`.
For ordinary finite inputs, absolute reconstruction error is approximately
bounded by half the row scale, up to FP32 rounding. This is not a bound on
model accuracy. Subnormal inputs can lose precision due to the scale floor.

The Triton quantizer uses one program per row, a masked max reduction and
FP32 division. Dequantization is tiled across flattened elements. Quantization
uses the existing phase 1 dispatch limit of 8192 columns; auto falls back
for wider rows. Dequantization has no such row reduction. The implementation
is inference-only, accepts finite FP32/FP16/BF16 inputs, and rejects gradient
tracking and invalid scales or reserved codes. Validation checks synchronize
CUDA; this is a checked API, not an optimized hot-path benchmark interface.
Noncontiguous tensors are copied by the Triton launchers when needed.

```python
import torch
from neuroforge.kernels.quantization import quantize_int8, dequantize_int8

x = torch.randn(4, 896)
q, scales = quantize_int8(x)
y = dequantize_int8(q, scales)
print((x - y).abs().max().item())
```

```bash
pytest -q -rs tests/test_int8_quantization.py
ruff check src tests
```

No new dependencies are required for CPU fallback. For explicit `backend="triton"`,
use an NVIDIA Ampere+ GPU and compatible PyTorch/Triton installation as in phase 1.
GPU compilation and runtime validation have not been performed. The tests compare
against PyTorch, check reconstruction error, and cover zero rows, rounding ties,
strided inputs, unusual widths, extreme finite values and invalid inputs.

Storage for R rows of width D is R(D+4) bytes, excluding metadata and temporary
buffers. That is not a measured runtime VRAM reduction: reconstruction allocates
an FP32 tensor. Group-wise scales, packed INT4, fused dequantization/GEMM, model
integration and performance measurement are subsequent milestones.
