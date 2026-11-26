# Phase 2 — advanced kernels and performance analysis

Adds residual RMSNorm (returns both rounded residual sum and normalized output),
split-half RoPE with explicit half-width cosine/sine tables, and tiled 2-D GEMM.
GEMM tunes three tile configurations by shape, dtype and GPU capability. Initial
compilation and tuning must be excluded from steady-state measurements. No
persistent tuning database or paper reproduction is claimed. Operators are opt-in;
training and existing inference paths are unchanged. GPU kernels are forward-only.
The PyTorch GEMM baseline follows the caller's PyTorch precision settings; disable
TF32 for a comparable FP32 experiment. RoPE tables must be expanded to the input
leading dimensions before calling; this is not yet a drop-in Qwen adapter.

## Validation

From the repository root:

```bash
pytest -q -rs tests/test_gpu_kernels.py tests/test_advanced_kernels.py tests/test_kernel_roofline.py
ruff check src tests
```

Initial CPU-only run of the combined kernel test suite: 38 passed, 12 skipped.
Additional boundary tests are included; consult CI for current validation status.
Python bytecode compilation passed; this does not compile Triton GPU kernels.
GPU validation and Qwen integration remain pending. No speedup is claimed.

## Benchmark and analytical roofline

`advanced_benchmark.py` measures the first invocation (wall clock, potentially
including compilation and autotuning), then warm CUDA-event samples separately.
It checks every output against the PyTorch baseline, reports raw samples, median
speedup and maximum absolute error, and records hardware/software metadata.
GEMM benchmarking disables TF32 and reduced-precision FP16/BF16 reductions in the
PyTorch baseline, then restores the original precision settings.
The autotuning key includes the CUDA device index as well as shape, dtype and
compute capability; this is an in-process cache, not a persistent tuning database.

```bash
python -m neuroforge.advanced_benchmark --dtype float16 --output-json outputs/advanced_fp16.json
python -m neuroforge.advanced_benchmark --dtype float32 --output-json outputs/advanced_fp32.json
```

Optional `--peak-tflops` and `--bandwidth-gbs` must both be supplied, using ceilings
appropriate to the actual GPU and arithmetic mode. A single compute ceiling is
only illustrative for different operators (GEMM and scalar reductions may have
very different ceilings). Use separate analyses for architecture-specific conclusions.

The analytical model uses logical minimum tensor traffic, not measured DRAM bytes:

| Operator | Approximate FLOPs | Logical bytes (element size s) |
|---|---:|---:|
| GEMM M x K x N | 2 M N K | s(MK + KN + MN) |
| Residual RMSNorm R x D | 5 R D | s(4RD + D) |
| Split-half RoPE R x D | 3 R D | 3sRD |

Residual RMSNorm counts the rounded residual output and normalized output, and
excludes reciprocal-square-root cost. GEMM counts multiply-add as two FLOPs.
The minimum traffic assumes ideal reuse, excludes intermediate buffers, copies,
cache effects and repeated tile reads. Effective logical GB/s is not measured
memory bandwidth and may exceed DRAM specifications due to cache reuse.
The model computes intensity F/B and ideal time max(F/P, B/W), excluding launch
overhead and other hardware constraints. Use Nsight Compute or equivalent to
measure actual memory traffic, occupancy and achieved throughput later.

No GPU measurements or Qwen integration yet.
