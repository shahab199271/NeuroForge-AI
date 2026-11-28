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

## Pre-phase 3.5 verification (2026-11-27)

[CI 90](https://github.com/shahab199271/NeuroForge-AI/actions/runs/45891422700)
validated the documentation-only PR 6 before its merge into main at
`11bb767d67bf18efc171904c3ba16aa662695827`. The job log reports Ruff:
`All checks passed!` and pytest: `179 passed, 85 skipped, 4 warnings in 7.60s`.
The resolved environment included Python 3.11, PyTorch 2.15.0, Triton 3.9.0,
pytest 9.2.0 and Ruff 0.17.6. These are versions reported by that CI log,
not a GPU-tested compatibility matrix or pinned project requirements.

An independent local regression on the archived phase 1–3.4 kernel sources
reported `146 passed, 85 skipped, 18 warnings in 1.48s` under Python 3.13.5
and PyTorch 2.10.0+cpu. The skips were due to CUDA/Triton requirements.
Some archived phase 1–2 files differ from main, so this is complementary
CPU evidence, not a byte-identical checkout of main. Python bytecode
compilation also succeeded. No GPU validation was performed.

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

## Phase 3.5 addition

The opt-in mixed precision wrapper selects FP32, FP16 or BF16 operands
without modifying packed INT4 weights or FP32 scales. The default policy
preserves phase 3.4 semantics. FP32 accumulation and output remain fixed.
Auto selection is a hardware/range heuristic, not a measured optimum.
See [mixed precision contract](GPU_KERNEL_PHASE3_MIXED_PRECISION.md) for
repeated CPU tests (33 dedicated passes, 179 kernel regression passes).
[CI 104](https://github.com/shahab199271/NeuroForge-AI/actions/runs/46046434817)
reported 212 passed, 92 skipped, 4 warnings in 4.98s; Ruff passed.
PR 8 merged at `3de985822b8239dabbccd5feffcf50d10dd93270`.
GPU execution remains unverified. The snapshot above records the phase 3.4 baseline.

## Phase 3.6 addition

Offline FP64 error diagnostics separate reconstruction, operand rounding and
execution error. The seeded CLI sweeps INT4/INT8, precision and group size.
See [error analysis](GPU_KERNEL_PHASE3_ERROR_ANALYSIS.md). Validation of this
addition is pending; no model accuracy or performance claim is made.

## Unverified and planned

GPU JIT compilation, GPU numerical correctness, memory safety and runtime
performance are unverified. The current fixed GEMM tiles are not established
as optimal. There is no demonstrated end-to-end Qwen replacement, trained
model accuracy gain, measured GPU speedup or peak VRAM saving. Storage
formulae and roofline estimates are analytical, not measurements.

Planned: 3.7 quantized
autotuning; 3.8 benchmarking. Model integration and full GPU validation
remain distinct work items. See [validation protocol](GPU_VALIDATION.md).
