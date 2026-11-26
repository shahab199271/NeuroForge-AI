"""Benchmark advanced forward operators; separate first call from steady state."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import torch

from .kernel_benchmark import measure
from .kernel_roofline import effective_rates, gemm_cost, residual_cost, roofline, rope_cost
from .kernels.advanced import matmul, residual_rmsnorm, rope


def _case(op, args, cost, warmup, iterations, tol, ceilings):
    result = {'workload': asdict(cost), 'arithmetic_intensity': cost.arithmetic_intensity}
    outputs = {}
    for backend in ('torch', 'triton'):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs[backend] = op(*args, backend=backend)
        torch.cuda.synchronize()
        first_ms = (time.perf_counter()-start)*1000
        stats = measure(lambda: op(*args, backend=backend), warmup, iterations)
        result[backend] = {'first_call_wall_ms': first_ms, **stats,
                           **effective_rates(cost, stats['median_ms'])}
    ref, actual = outputs['torch'], outputs['triton']
    if not isinstance(ref, tuple):
        ref, actual = (ref,), (actual,)
    errors = []
    for a, b in zip(actual, ref):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)
        errors.append((a.float()-b.float()).abs().max().item())
    result['max_absolute_errors'] = errors
    result['median_speedup'] = result['torch']['median_ms']/result['triton']['median_ms']
    if ceilings is not None:
        result['roofline'] = roofline(cost, *ceilings)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, nargs='+', default=[1, 32, 256])
    p.add_argument('--width', type=int, default=896)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--gemm-n', type=int, default=896)
    p.add_argument('--dtype', choices=['float32','float16','bfloat16'], default='float16')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--peak-tflops', type=float)
    p.add_argument('--bandwidth-gbs', type=float)
    p.add_argument('--output-json', type=Path)
    a = p.parse_args()
    if min(a.rows+[a.width,a.head_dim,a.gemm_n,a.warmup,a.iterations]) < 1:
        p.error('Positive sizes and iteration counts required')
    if a.width > 8192 or a.head_dim % 2:
        p.error('Width <=8192 and even head dimension required')
    ceilings = None
    if a.peak_tflops is not None or a.bandwidth_gbs is not None:
        if a.peak_tflops is None or a.bandwidth_gbs is None:
            p.error('Both hardware ceilings required')
        ceilings = (a.peak_tflops,a.bandwidth_gbs)
        try:
            roofline(gemm_cost(1,1,1,4), *ceilings)
        except ValueError as exc:
            p.error(str(exc))
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_capability()[0] < 8:
        p.error('NVIDIA Ampere or newer required')
    try:
        import triton
    except ImportError:
        p.error('Triton required')
    if a.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        p.error('BF16 unsupported')
    dtype = getattr(torch,a.dtype)
    size = torch.empty((),dtype=dtype).element_size()
    tol = {'float32': 5e-4, 'float16': 5e-3, 'bfloat16': 3e-2}[a.dtype]
    torch.manual_seed(42)
    report = {'gpu':torch.cuda.get_device_name(), 'capability':torch.cuda.get_device_capability(),
              'torch':torch.__version__, 'triton':triton.__version__, 'cuda':torch.version.cuda,
              'python':platform.python_version(), 'dtype':a.dtype, 'seed':42,
              'warmup':a.warmup, 'iterations':a.iterations, 'hardware_ceilings':ceilings,
              'scope':'allocating forward wrappers; logical bytes, not measured DRAM traffic',
              'results':[]}
    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            for rows in a.rows:
                x = torch.randn(rows,a.width,device='cuda',dtype=dtype)
                r,w = torch.randn_like(x),torch.randn(a.width,device='cuda',dtype=dtype)
                q = torch.randn(rows,a.head_dim,device='cuda',dtype=dtype)
                theta = torch.randn(rows,a.head_dim//2,device='cuda')
                b = torch.randn(a.width,a.gemm_n,device='cuda',dtype=dtype)
                cases = [(residual_rmsnorm,(x,r,w),residual_cost(rows,a.width,size)),
                         (rope,(q,theta.cos().to(dtype),theta.sin().to(dtype)),rope_cost(rows,a.head_dim,size)),
                         (matmul,(x,b),gemm_cost(rows,a.gemm_n,a.width,size))]
                for op,args,cost in cases:
                    item = _case(op,args,cost,a.warmup,a.iterations,tol,ceilings)
                    item['input_shapes'] = [list(t.shape) for t in args]
                    report['results'].append(item)
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    output = json.dumps(report,indent=2,allow_nan=False)
    print(output)
    if a.output_json:
        a.output_json.parent.mkdir(parents=True,exist_ok=True)
        a.output_json.write_text(output+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
