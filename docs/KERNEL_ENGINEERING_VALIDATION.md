# Kernel engineering validation (phase 3.6+)

Maintainer: Shahab Baloochi.

This stage strengthens the evidence for phases 1–3.6 before kernel autotuning.
It adds Hypothesis, NumPy reference calculations and pytest-cov/coverage.py
to the existing pytest and Ruff workflow. It does not change the production
kernel algorithms.

## Property tests and independent references

The bounded CPU suite uses nine properties, each with a maximum of 60
generated examples and deterministic generation for a fixed Hypothesis
version. Hypothesis can shrink failures. Tensor sizes and values are
bounded to keep the CI workload manageable. The examples are not a proof
over all floating-point inputs.

| Phase | Independent check |
| --- | --- |
| 1 | Stable softmax and translation invariance, RMSNorm, SwiGLU |
| 2 | NumPy FP64 GEMM, cancelling residual, RoPE rotation and norm |
| 3.1 | Rowwise INT8 scales, nearest-code residual and reconstruction |
| 3.2 | Group tails, oversized groups, channel layout and reconstruction |
| 3.3 | Exact nibble layout and round trip, INT4 reconstruction error bound |
| 3.4–3.5 | NumPy decoding and FP64 products of FP32/FP16/BF16 operands |
| 3.6 | Independent error metrics and output error decomposition |

The quantization oracle allows one ULP for normal FP32 scale rounding and
a bounded nearest-code residual; exact ties remain covered by dedicated
unit tests. GEMM validation independently decodes the actual packed weights
so a legal near-tie difference does not contaminate the GEMM comparison.
BF16 rounding is implemented with NumPy integer bit operations for the
bounded finite inputs. NumPy products are CPU FP64, not exact real
arithmetic. Existing dedicated tests remain mandatory.

## Run locally

```bash
python -m pip install -e '.[dev]'
ruff check src tests
pytest -q --hypothesis-show-statistics tests/test_kernel_properties.py
mkdir -p outputs/validation
pytest -q --cov=neuroforge.kernels --cov-branch --cov-report=term-missing \
  --cov-report=json:outputs/validation/kernel-coverage.json \
  --junitxml=outputs/validation/pytest.xml
```

The CI workflow runs the full regression with line and branch coverage,
then the property tests again with statistics, then the existing repeated
error-analysis tests and two 144-row reproducibility sweeps. It uploads
the coverage JSON and JUnit XML as a workflow artifact.

Coverage is scoped to the kernel Python package, including Triton source
modules. CPU jobs cannot execute GPU-only paths. A percentage is not
evidence of GPU execution, device memory safety, or numerical correctness.
No percentage threshold is imposed before an observed baseline and
review of missed branches. Future changes can enforce a justified
CPU-specific threshold without hiding GPU-only paths.

## Evidence and remaining work

The baseline main commit is
`56a2f85e46de9c4b3c2ae23a3fd9ba1aa01ceefd`.
[CI 161](https://github.com/shahab199271/NeuroForge-AI/actions/runs/46135404161)
reported 258 passed, 94 skipped and 4 warnings before this addition.
The property/coverage workflow must complete successfully before merge.

GPU JIT execution, CUDA memory safety, and performance remain unverified.
The [GPU validation protocol](GPU_VALIDATION.md) documents the proposed
Compute Sanitizer and CUDA tests. Nsight profiling and quantitative GPU
benchmarks are separate tasks, not claimed results or installed project
dependencies. No CUDA C++ implementation is claimed.
