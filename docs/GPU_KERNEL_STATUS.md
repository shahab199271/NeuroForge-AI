# Kernel capability and evidence register

Snapshot: phase 3.4, main commit
`1f38cf93777cda85e63bc9303d460883fd6d562e`, 2026-11-26.

| Phase | Capability | Evidence |
| --- | --- | --- |
| 1 | RMSNorm, softmax, SwiGLU | Implemented; CPU CI |
| 2 | Residual RMSNorm, split-half RoPE, tiled GEMM, analytical roofline | [PR 1](https://github.com/shahab199271/NeuroForge-AI/pull/1) |
| 3.1 | Row-wise symmetric INT8, FP32 scales | [CI 63](https://github.com/shahab199271/NeuroForge-AI/actions/runs/45885418262) |
| 3.2 | Group-wise and per-channel INT8 | [CI 70](https://github.com/shahab199271/NeuroForge-AI/actions/runs/45886296778) |
| 3.3 | Packed signed INT4, independent group scales | [CI 77](https://github.com/shahab199271/NeuroForge-AI/actions/runs/45888124106) |
| 3.4 | Tiled fused INT4 decode, scale, floating-point GEMM | [CI 84](https://github.com/shahab199271/NeuroForge-AI/actions/runs/45890194349) |

CI 63, 70, 77 and 84 completed successfully including Ruff and pytest.
These are pull-request CI results, not GPU runs. The CI workflow uses
Ubuntu and Python 3.11. GPU test skips do not count as validation.

## Data and numerical contracts

- INT8 codes are in [-127,127]; -128 is reserved.
- INT4 codes are in [-7,7]; -8 is reserved. Low nibble precedes high nibble,
  with zero high-nibble padding for odd widths.
- Group scales are FP32, symmetric max-absolute, with a normal FP32 floor
  and scale 1 for zero groups. Rounding ties go away from zero.
- Group sizes and original widths must be retained as metadata. Groups
  do not cross row boundaries.
- INT4 GEMM consumes [M,K] activations and [N,ceil(K/2)] packed weights,
  with scales [N,ceil(K/group_size)]. It rounds reconstructed weights to
  the activation dtype and returns [M,N] FP32; accumulation may overflow.
- Quantization and packed-weight APIs are inference-oriented. The fused
  dot path avoids a full decoded weight buffer, but input checks can
  allocate masks and synchronize CUDA; layout conversions can allocate copies.
- Packed INT4 is distinct from bitsandbytes NF4 QLoRA. The format is not
  automatically interoperable with other packed-weight libraries.

## Unverified and planned

GPU JIT compilation, GPU numerical correctness, memory safety and runtime
performance are unverified. The current fixed GEMM tiles are not established
as optimal. There is no demonstrated end-to-end Qwen replacement, trained
model accuracy gain, measured GPU speedup or peak VRAM saving. Storage
formulae and roofline estimates are analytical, not measurements.

Planned: 3.5 mixed-precision policy; 3.6 error analysis; 3.7 quantized
autotuning; 3.8 benchmarking. Model integration and full GPU validation
remain distinct work items. See [validation protocol](GPU_VALIDATION.md).
