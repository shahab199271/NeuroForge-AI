# Quantization error analysis (phase 3.6)

This offline diagnostic measures grouped INT8 and packed INT4 weight
reconstruction and matrix output errors, with explicit FP32, FP16 or BF16
operand policies. It does not change the existing kernels or claim GPU
performance. INT8 uses a reconstructed-weight PyTorch GEMM, not a fused
INT8 kernel. INT4 calls the phase 3.5 API with the chosen backend.

## Error decomposition

Let A and W denote the original floating-point inputs, Q the FP32 weight
reconstruction, and Ar and Wr the rounded operands (including weight
saturation). Four reference products are evaluated on CPU in FP64:

- Y0 = A W^T: original input reference.
- Yq = A Q^T: reconstruction-only reference.
- Ya = Ar Q^T: activation-rounded reference.
- Yr = Ar Wr^T: rounded operand reference.

The actual output Y is evaluated using the requested execution path.
Report weight reconstruction (Q versus W), quantization output (Yq versus
Y0), activation output (Ya versus Yq), weight operand output (Yr versus Ya),
combined operand output (Yr versus Yq), execution (Y versus Yr), and total
output (Y versus Y0). Signed residuals telescope, but their scalar norms
do not generally add. CPU FP64 is a high-precision reference, not exact
real arithmetic. All reference operands are copied to CPU; this is not
a hot-path diagnostic or latency benchmark.

Operand tensor metrics and counts of nonzero values rounded to zero, weight
saturation and reconstructed zeros are also recorded. Attribution is ordered:
activation rounding precedes weight rounding, so interaction effects depend on
that ordering.

Each comparison includes count, MAE, RMSE, maximum absolute error, signed
bias (estimate minus reference), relative L2 norm, and cosine similarity.
Undefined relative L2 or cosine due to zero norms is serialized as null,
not zero or NaN. Nonfinite inputs/outputs and FP64 metric overflow are
rejected rather than silently omitted. Activation range violations are
rejected as in phase 3.5.

Storage accounting includes packed bytes, odd-width padding and FP32 group
scales, but excludes Python metadata, allocator overhead, temporary buffers
and model-level memory. The FP32 baseline is four bytes per weight.
Effective bits per weight can exceed the nominal code width or even FP32
for small groups. It is not a measured VRAM saving.

## Example

```bash
python -m neuroforge.quantization_error_sweep --m 8 --n 64 --k 129 \
  --groups 16 32 64 128 --seeds 0 29 101 \
  --output outputs/quantization_errors.json
```

The CLI evaluates normal and synthetic outlier-heavy weights, both
bit widths, all explicit operand policies and each requested group/seed.
It records individual workloads, not pooled estimates. The default
configuration produces 144 diagnostic rows. Compare like-for-like rows;
smaller groups can reduce weight error but increase scale overhead.
No monotonic improvement is assumed for output error.

For a supported NVIDIA GPU, specify `--device cuda --backend triton`.
INT8 quantization and reconstruction can then use Triton, but its final
GEMM is still PyTorch. The actual runtime precision of the PyTorch FP32
product follows the caller's settings, which are recorded in the report.
No real model accuracy or GPU validation is implied by synthetic results.

## Validation

```bash
ruff check src tests
pytest -q -rs tests/test_error_analysis.py
pytest -q -rs
```

Tests include known analytical errors, zero norms, extreme FP64 values,
invalid/nonfinite inputs, odd groups and storage, explicit precisions,
and CUDA cases when available. CI 115 passed the initial implementation
(253 passed, 94 skipped, 4 warnings). The extended suite and repeated full
sweep are checked by the updated CI workflow; results for that revision
remain pending. GPU runtime validation remains pending.
