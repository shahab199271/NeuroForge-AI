# Explicit mixed-precision policy for packed INT4 GEMM

Phase 3.5 adds an opt-in wrapper around the phase 3.4 GEMM. Existing
`int4_matmul` calls retain their behavior. The packed weight layout, FP32
scales and Triton tile geometry are unchanged. This is floating-point
operand selection with INT4 weight storage, not native INT4 arithmetic.

## API and selection policy

```python
import torch
from neuroforge.kernels.int4 import quantize_int4
from neuroforge.kernels.mixed_precision import mixed_int4_matmul

w = torch.randn(96, 129)
a = torch.randn(4, 129)
p, s = quantize_int4(w, 32)
y = mixed_int4_matmul(a, p, s, 32, precision='fp16', backend='torch')
assert y.shape == (4, 96) and y.dtype == torch.float32
```

| Policy | Floating-point operand dtype |
| --- | --- |
| `input` (default) | Preserve activation dtype and phase 3.4 behavior |
| `fp32` | FP32 |
| `fp16` | FP16 |
| `bf16` | BF16 |
| `auto` | Preserve FP16; preserve BF16 except on an unsupported NVIDIA/CUDA dispatch; choose BF16 for FP32 on supported NVIDIA Ampere+ when the activations fit its finite range, otherwise FP32 |

Explicit `backend='torch'` disables the hardware-driven auto downcast.
CPU auto preserves the input dtype. Auto is a heuristic, not a measured
speed or accuracy guarantee. Explicit Triton dispatch retains the existing
hardware and dependency restrictions. A supported GPU alone does not
establish that Triton is installed or that a kernel has compiled.

## Numerical contract

Activations must be finite and in range before conversion to the selected
dtype. Downcast underflow and rounding are possible. Reconstructed weights
are multiplied by FP32 scales, saturated to the selected operand dtype's
finite range and rounded to that dtype. The operation returns FP32 with
FP32 accumulation; overflow of the final result is possible and is not
silently saturated. FP32 accumulation is intentionally fixed rather than
introducing an unvalidated reduced-precision accumulator.

The Triton implementation is reused from phase 3.4. The PyTorch reference
uses rounded operands converted to FP32 for multiplication; CUDA reference
precision follows the caller's PyTorch FP32 matmul settings. The wrapper
never changes global precision settings. Explicit conversions and the
phase 3.4 reference are independent of ambient autocast. Gradients are
not supported by this inference API. Input validation can synchronize CUDA,
and activation conversion can allocate a new tensor. Weight buffers and
scales are not modified. No accuracy or latency improvement is claimed.

## Validation

```bash
pytest -q -rs tests/test_mixed_precision.py
ruff check src tests
```

Tests compare against independent FP64 multiplication of rounded operands,
including three additional seeded workloads; exercise odd shapes and group
boundaries; check unchanged packed data; and cover default compatibility,
auto selection, exact positive and negative activation range boundaries,
invalid precision types, gradients and autocast. CUDA tests require a
compatible NVIDIA device and Triton.

Local CPU validation (Python 3.13.5, PyTorch 2.10.0+cpu): three dedicated
runs each returned 33 passed, 7 skipped; two archive-based kernel regression
runs each returned 179 passed, 92 skipped, 18 warnings. Python bytecode
compilation succeeded. The direct phase 3.5 source and test blobs were
matched against the locally tested files after push. The local ops.py was
aligned with main; some older phase 2 archive files differ from main.
This is not a full checkout test. Full repository CI is required separately.
No GPU compilation, runtime or performance measurements are verified.
