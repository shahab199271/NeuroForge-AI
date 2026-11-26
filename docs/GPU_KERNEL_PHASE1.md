# GPU Kernel Lab — Phase 1

Status: reviewed implementation candidate; 24 CPU tests passed, 9 GPU cases skipped. NVIDIA GPU execution and full repository regression testing pending.

This additive overlay targets the existing `src/neuroforge` layout. Existing training,
PEFT, distillation, evaluation, serving and benchmark entry points remain untouched.
Triton is optional; install a CUDA-compatible PyTorch/Triton pair in a Linux/WSL2
virtual environment. Do not replace a working CUDA environment blindly.

## Operators
- RMSNorm: one row per program, FP32 sum of squares and scaling.
- Softmax: one row per program, max subtraction, masked tails and FP32 reduction.
- SwiGLU: elementwise SiLU(gate) times up, fused in one program.

These are forward-only kernels. Automatic dispatch uses PyTorch when gradients are
required or CUDA/Triton is unavailable. Explicit Triton dispatch fails rather than
silently substituting a baseline. Noncontiguous inputs are copied when necessary.
Supported dtypes: FP32, FP16, BF16 (hardware dependent). Phase 1 conservatively targets NVIDIA Ampere or newer. Row reductions support
width <= 8192; larger rows use the PyTorch fallback in auto mode. SwiGLU is tiled
and does not have this row-width limit. Forward-mode AD also uses the fallback. No backward
kernel, native CUDA C++, Qwen module replacement or claimed speedup in this phase.

## Run from the repository root after copying the overlay

```bash
pip install -e '.[dev]'
# In a compatible CUDA environment only, if Triton is not already installed:
pip install triton
pytest -q -rs tests/test_gpu_kernels.py
ruff check src tests
python -m neuroforge.kernel_benchmark --dtype float16 --output-json outputs/kernels_fp16.json
python -m neuroforge.kernel_benchmark --dtype float32 --output-json outputs/kernels_fp32.json
```

Benchmark uses CUDA events, warm-up, numerical checks, raw timing samples and hardware
metadata. Compare against the explicit FP32-intermediate PyTorch reference, not a claim
of superiority over torch.compile or optimized vendor kernels. No model or dataset download.
For GPU acceptance require CUDA cases to pass, not skip; inspect JSON and rerun with
representative widths, devices and dtypes. Existing full test suite must also pass.

Background reading: official Triton tutorials on fused softmax and layer normalization
(https://triton-lang.org/main/getting-started/tutorials/). Independent implementation;
2026 paper reproduction and adaptive tuning are deferred to later phases.
