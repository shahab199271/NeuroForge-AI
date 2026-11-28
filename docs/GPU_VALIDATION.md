# GPU validation protocol

## Scope and environment

This is a proposed reproducible starting configuration, not a certified
compatibility matrix. The existing CI runs Python 3.11 on Ubuntu without
GPU validation. The following upstream-compatible stack has not yet been
executed against NeuroForge:

| Component | Candidate configuration |
| --- | --- |
| OS | Ubuntu 22.04 x86-64 or Ubuntu 22.04 on WSL2 |
| Python | 3.11 |
| GPU | NVIDIA Ampere (SM 8.0/8.6) initially |
| PyTorch | 2.5.1+cu124 |
| CUDA runtime | 12.4, supplied by the PyTorch wheel |
| Triton | 3.1.0, matching the PyTorch distribution |
| NVIDIA driver | A current supported driver compatible with CUDA 12.4 |

This historical stack is a controlled starting point, not the latest or a
security-support recommendation. For other GPU generations use an upstream
supported stack and record it separately. The repository has not established
a minimum supported Triton version. The CUDA version printed by nvidia-smi
is the driver's supported CUDA level, not necessarily the wheel runtime.
On WSL2 install the NVIDIA Windows host driver; do not install a Linux display
driver inside WSL. Follow the vendor's instructions.

References:
- https://pytorch.org/get-started/previous-versions/
- https://pypi.org/project/triton/3.1.0/
- https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/wsl-user-guide/

## Installation

From a clean checkout in Linux/WSL2, using bash:

```bash
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
printf 'torch==2.5.1+cu124\ntriton==3.1.0\n' > /tmp/neuroforge-gpu-constraints.txt
python -m pip install -c /tmp/neuroforge-gpu-constraints.txt -e '.[dev]'
python -m pip check
```

If dependency resolution fails, preserve the error and select a compatible
environment; do not silently replace the PyTorch/Triton pairing. This constrains
the two GPU packages, not every transitive dependency. Capture the full
resolved environment before reporting results. No QLoRA dependency is needed
for the kernel tests.

## Preflight and provenance

```bash
mkdir -p outputs/gpu-validation
git rev-parse HEAD | tee outputs/gpu-validation/commit.txt
git status --short | tee outputs/gpu-validation/git-status.txt
nvidia-smi | tee outputs/gpu-validation/nvidia-smi.txt
python -m pip freeze > outputs/gpu-validation/pip-freeze.txt
python -m torch.utils.collect_env > outputs/gpu-validation/environment.txt
python - <<'PY'
import torch
import triton

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.version.cuda is not None, "Not an NVIDIA CUDA build"
assert torch.cuda.get_device_capability()[0] >= 8, "Ampere+ required"
print(torch.__version__, torch.version.cuda, triton.__version__)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
print("BF16:", torch.cuda.is_bf16_supported())
PY
```

Record whether BF16 is supported. A successful preflight is not a kernel pass.
Use a clean checkout for publishable results, and keep logs free of credentials.

## Correctness and compilation

```bash
set -e -o pipefail
ruff check src tests
pytest -q -rs | tee outputs/gpu-validation/full-pytest.txt
CUDA_LAUNCH_BLOCKING=1 pytest -vv -rs \
  tests/test_gpu_kernels.py tests/test_advanced_kernels.py \
  tests/test_kernel_roofline.py tests/test_int8_quantization.py \
  tests/test_group_quantization.py tests/test_int4.py tests/test_int4_gemm.py \
  tests/test_mixed_precision.py tests/test_error_analysis.py \
  | tee outputs/gpu-validation/kernel-pytest.txt
```

Run the correctness commands in a dedicated Bash shell or script: the
fail-fast options stop on a failed command or failed pipeline, including
pytest failures when output is piped to tee. Do not treat successful log
creation as a successful test run.

Inspect the verbose node IDs and skip reasons. Explicit Triton cases must
actually execute on CUDA; CPU fallback is not evidence of Triton correctness.
Unsupported BF16 cases must be identified rather than counted as passes.
First launches exercise JIT compilation for the tested specialization.
Review failures by operator, shape, dtype, device and backend. Repeat on each
target GPU; one device does not establish support across architectures.
Test odd dimensions, noncontiguous inputs, partial groups, zero groups, ties,
reserved codes, saturation, and GEMM against the independent FP64 oracle.
For mixed precision, record the requested policy and selected operand dtype;
exercise explicit FP32/FP16/BF16 and the auto selection on supported hardware.
Run the diagnostic sweep with explicit device and backend; see
[error analysis](GPU_KERNEL_PHASE3_ERROR_ANALYSIS.md). INT8 final GEMM is
PyTorch even when quantization and reconstruction use Triton.
Do not relax tolerances merely to make a run pass.

If NVIDIA Compute Sanitizer is installed and supported by the host:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest -q -rs tests/test_int4_gemm.py
```

Treat sanitizer availability as a separate check; WSL support depends on
driver and tooling. Do not benchmark under CUDA_LAUNCH_BLOCKING or sanitizer.

## Strict GPU acceptance (opt-in)

The dedicated acceptance suite intentionally fails if requested CUDA, Triton,
Ampere+ or BF16 support is absent. Ordinary CPU CI skips it explicitly.

```bash
NEUROFORGE_REQUIRE_GPU=1 python -m pytest -q -rs tests/test_gpu_acceptance.py
NEUROFORGE_REQUIRE_GPU=1 compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest -q -rs tests/test_gpu_acceptance.py
NEUROFORGE_REQUIRE_GPU=1 compute-sanitizer --tool racecheck --error-exitcode 1 \
  python -m pytest -q -rs tests/test_gpu_acceptance.py
NEUROFORGE_REQUIRE_GPU=1 compute-sanitizer --tool synccheck --error-exitcode 1 \
  python -m pytest -q -rs tests/test_gpu_acceptance.py
```

Run these only on a trusted GPU environment with supported sanitizer tooling.
This acceptance subset supplements rather than replaces the full GPU suite.
A clean sanitizer run is scoped to executed cases and the selected tool.
These commands have not been executed on repository hardware yet.

## Performance protocol

Existing phase 1–2 microbenchmarks:

```bash
python -m neuroforge.kernel_benchmark --dtype float16
python -m neuroforge.advanced_benchmark --dtype float16
```

Dedicated quantized benchmarks are planned for phase 3.8. For that work,
compare cached reconstructed floating-point GEMM, unpack/dequantize plus
GEMM, and fused INT4 GEMM. Match activation dtype, output dtype, rounding
policy and shapes. Separate prefill-like and decode-like M; include odd K,
partial groups and multiple group sizes. Report cold compilation separately
from warmed steady-state timings. Use CUDA events for GPU execution and
synchronized wall-clock timings for checked API latency. Report median and
tail latency with raw samples, warmup/repetition counts and seeds. Measure
peak allocated and reserved memory separately from theoretical packed bytes.
Keep validation, layout copies and quantization costs visible; do not compare
a checked end-to-end path to a kernel-only baseline without labeling them.
Validate correctness before reporting any timing. Record GPU, driver,
software versions, precision settings and workload; do not extrapolate
kernel speedup to model serving without end-to-end measurement.

## Evidence template

| Field | Result |
| --- | --- |
| Commit and clean working tree | Pending |
| GPU, compute capability and driver | Pending |
| Python / PyTorch / CUDA / Triton | Pending |
| CPU and GPU test counts and skip reasons | Pending |
| JIT compilation by dtype and operator | Pending |
| Compute Sanitizer result | Pending |
| Numerical errors and tolerances | Pending |
| Latency samples and memory peaks | Pending |

A GPU validation run is complete only for the recorded configuration and
executed cases. The current repository does not yet have such a run.
