# Group-wise and per-channel INT8 quantization

The grouped format stores signed INT8 codes in the original tensor shape and FP32
scales of shape `(..., ceil(D / group_size))`, where D is the last dimension.
Groups never cross row boundaries. Partial final groups are masked on Triton and
zero-padded only in the PyTorch reference; padding is not stored. Each group uses
symmetric max-absolute scaling, ties away from zero, a normal FP32 scale floor,
and scale 1 for all-zero groups, matching phase 3.1. Reconstruction saturates
at finite FP32 bounds. The caller must retain the original group size; scale
shape alone cannot identify it uniquely.

For ordinary finite inputs the approximate absolute error bound is s/2 plus
floating-point rounding. Smaller groups can reduce local quantization steps,
but do not guarantee lower error for every individual value or model task.
For R rows, stored bytes are R*(D + 4*ceil(D/group_size)), excluding metadata
and intermediate buffers. Per-channel mode reduces all axes except the selected
axis and uses one scale per channel; on a 2-D weight matrix [out, in], axis=0
means one scale per output channel. It may require layout copies.

```python
import torch
from neuroforge.kernels.group_quantization import (
    quantize_grouped, dequantize_grouped, quantize_channels, dequantize_channels,
)
x = torch.randn(4, 897)
q, s = quantize_grouped(x, 128)
y = dequantize_grouped(q, s, 128)
qc, sc = quantize_channels(x, axis=0)
yc = dequantize_channels(qc, sc, axis=0)
```

Triton is optional; the existing NVIDIA Ampere+ dispatch policy applies, with
an effective group-width limit of 8192 for quantization reductions. Wider groups
fall back to PyTorch in auto mode. Tiled dequantization has no reduction-width
limit. Explicit Triton dispatch raises if unsupported.
Public validation synchronizes CUDA; these are checked inference-only APIs,
not measured hot-path operations. No packed INT4 or quantized GEMM yet.

```bash
pytest -q -rs tests/test_group_quantization.py
ruff check src tests
```

CPU kernel tests using local phase 1–3.1 archives plus these files: 97 passed,
45 skipped (GPU unavailable). This does not certify the full current main branch,
Triton compilation, GPU correctness, latency or model accuracy. GPU tests are
included for later execution. No measured memory or speed improvement is claimed.
