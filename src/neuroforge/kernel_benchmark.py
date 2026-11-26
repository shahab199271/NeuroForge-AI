"""CUDA-event microbenchmarks; excludes compilation and warm-up from timings."""
import argparse
import json
import statistics
import math
import platform
from pathlib import Path

import torch
from .kernels import rmsnorm, softmax, swiglu


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(iterations)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    ms = [a.elapsed_time(b) for a, b in events]
    if any(not math.isfinite(t) or t <= 0 for t in ms):
        raise RuntimeError("Invalid event timing; increase workload or repeat measurement")
    return {"median_ms": statistics.median(ms), "mean_ms": statistics.mean(ms),
            "min_ms": min(ms), "samples_ms": ms}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, nargs='+', default=[1, 32, 512])
    p.add_argument('--widths', type=int, nargs='+', default=[896, 4864])
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float16')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--output-json', type=Path)
    a = p.parse_args()
    if min(a.rows + a.widths + [a.warmup, a.iterations]) < 1 or max(a.widths) > 8192:
        p.error('Positive sizes/counts and width <= 8192 required')
    if not torch.cuda.is_available() or torch.version.cuda is None:
        p.error('NVIDIA CUDA required')
    if torch.cuda.get_device_capability()[0] < 8:
        p.error("Phase 1 targets NVIDIA Ampere or newer")
    try:
        import triton
    except ImportError:
        p.error("Triton is required for this benchmark")
    dtype = getattr(torch, a.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        p.error('bf16 unsupported')
    torch.manual_seed(42)
    report = {'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
              'torch': torch.__version__, 'triton': triton.__version__, 'cuda': torch.version.cuda,
              'python': platform.python_version(), 'platform': platform.platform(),
              'dtype': a.dtype, 'seed': 42, 'warmup': a.warmup, 'iterations': a.iterations,
              'scope': 'CUDA event elapsed time, allocating operator wrappers, not model inference',
              'results': []}
    with torch.inference_mode():
        for rows in a.rows:
            for width in a.widths:
                x = torch.randn(rows, width, device='cuda', dtype=dtype)
                w = torch.randn(width, device='cuda', dtype=dtype)
                u = torch.randn_like(x)
                for name, op, args in [('rmsnorm', rmsnorm, (x, w)), ('softmax', softmax, (x,)),
                                       ('swiglu', swiglu, (x, u))]:
                    ref = op(*args, backend='torch')
                    out = op(*args, backend='triton')
                    tol = {'float32': 2e-5, 'float16': 3e-3, 'bfloat16': 2e-2}[a.dtype]
                    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
                    item = {'operator': name, 'rows': rows, 'width': width,
                            'max_abs_error': (out.float() - ref.float()).abs().max().item()}
                    for backend in ('torch', 'triton'):
                        item[backend] = measure(lambda: op(*args, backend=backend), a.warmup, a.iterations)
                    item['median_speedup'] = item['torch']['median_ms'] / item['triton']['median_ms']
                    report['results'].append(item)
    result = json.dumps(report, indent=2, allow_nan=False)
    print(result)
    if a.output_json:
        a.output_json.parent.mkdir(parents=True, exist_ok=True)
        a.output_json.write_text(result + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
