# Pre-results engineering audit

Audit date: 2026-11-20. Results are deliberately omitted: no claim of trained checkpoints or validated inference acceleration.

## Verified in the public GitHub repository
- Repository is public, contains Python package / training / evaluation / benchmarks / API / CI / docs.
- GitHub Actions CI (lint and unit tests) has reported success.
- No original prototype API key or user-specific hardcoded machine path was included in the rewrite.
- Baseline evaluation and JSON output support were added; invalid responses count as errors in all-sample accuracy.
- API masks unexpected raw errors and serializes single-model inference.

## Implementation-level limitations requiring real execution
1. Download the public base model and IMDb dataset; verify compatibility on a CUDA machine.
2. Complete LoRA training and QLoRA training; validate adapter save/reload and determinism.
3. Measure held-out baseline/LoRA/QLoRA accuracy and invalid-generation rate.
4. Run GPU inference benchmarks with fixed prompt lengths, input distributions and batch sizes.
5. Profile CUDA memory with process-level reporting; allocated memory is not total VRAM.
6. Verify behaviour under long inputs, edge cases, prompt injection attempts and load.
7. Pin resolved dependency versions in a lockfile or reproducible environment.
8. Add a genuinely implemented knowledge distillation pipeline and ONNX/other export pipeline *only when ready*, not as existing capabilities.
9. Audit dependency vulnerabilities, upstream dataset/model licences and Git history before further public releases.
10. Document deployment target, HTTP timeout, authentication/rate limits, observability, and concurrency before production use.

## Scientific reporting rules
Never conflate QLoRA **training** with quantized **inference**. Include a non-fine-tuned baseline, fixed held-out split, repeated seeds, and paired prompt lengths. Report quality alongside size, GPU peak memory, P50/P95 latency, and hardware details. Zero completed experiments means **no benchmark figures yet**.
