# Kernel engineering validation (phase 3.6+)

Maintainer: Shahab Baloochi.

This stage strengthens the evidence for phases 1–3.6 before kernel autotuning.
It adds Hypothesis, NumPy reference calculations and pytest-cov/coverage.py
to the existing pytest and Ruff workflow. It does not change the production
kernel algorithms.

## Property tests and independent references

The bounded CPU suite uses nine properties, each with a maximum of 200
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
CI 172 measured 56% combined line and branch coverage of the whole kernel
package. The new gates require at least 55% whole-package coverage and 80%
combined coverage of the nine CPU-facing modules (including their unexecuted
GPU dispatch branches). These are initial conservative regression floors,
not target completion percentages. Both scopes remain visible.

## Evidence and remaining work

The baseline main commit is
`56a2f85e46de9c4b3c2ae23a3fd9ba1aa01ceefd`.
[CI 161](https://github.com/shahab199271/NeuroForge-AI/actions/runs/46135404161)
reported 258 passed, 94 skipped and 4 warnings before this addition.
[CI 172](https://github.com/shahab199271/NeuroForge-AI/actions/runs/46136043998)
verified 267 passed, 94 skipped, 4 warnings, all nine property tests and
both reproducibility sweeps. PR 11 merged at `7deb8489098443b6cc52352062d8a5d1160dc04b`.
[CI 189](https://github.com/shahab199271/NeuroForge-AI/actions/runs/46139659756)
verified 308 passed, 100 skipped, 4 warnings; all 1,800 dedicated Hypothesis
examples passed, 41 contract/stress/report tests passed on repeat, and two
144-row sweeps were identical. CPU-facing kernel coverage was 86.16%; the
whole kernel package was 58.50% (combined lines and branches). Ruff,
scoped strict Mypy and pip check passed. SciPy and Matplotlib produced
48 matched configuration summaries and the error/storage plot. See the
[validation artifacts](https://github.com/shahab199271/NeuroForge-AI/actions/runs/46139659756/artifacts/15261422030).
PR 12 merged at `5172cafc80d64d7fd908124235844f87a16a924d`.
Actual GPU acceptance and Compute Sanitizer execution remain pending.

## Additional hardening

- 200 generated examples per property, with deterministic settings.
- Invalid inputs, sign/scale invariants, row permutation and batch splitting,
  bounded stress workloads up to K=8193, and FP64 metric overflow rejection.
- Strict Mypy checking scoped to the coverage utility; not repository-wide typing.
- SciPy IQR, NumPy median/worst errors and Matplotlib synthetic trade-off plots
  grouped by matched configuration. Three seeds provide descriptive summaries,
  not confidence intervals or statistical evidence of model-level improvement.
- CI artifacts include raw sweep, descriptive JSON, PNG, coverage and JUnit.
- Opt-in GPU acceptance requires CUDA, Triton, Ampere+ and BF16; it fails
  when requested hardware is absent rather than silently skipping.

Run the scientific report from a sweep JSON:

```bash
python -m neuroforge.quantization_report outputs/quantization_errors.json
```

Run strict GPU acceptance as described in the GPU validation protocol.

GPU JIT execution, CUDA memory safety, and performance remain unverified.
The [GPU validation protocol](GPU_VALIDATION.md) documents the proposed
Compute Sanitizer and CUDA tests. Nsight profiling and quantitative GPU
benchmarks are separate tasks, not claimed results or installed project
dependencies. No CUDA C++ implementation is claimed.
