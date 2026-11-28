# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Analytical operator costs and an explicitly parameterized roofline model."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Workload:
    operator: str
    flops: int
    logical_bytes: int

    @property
    def arithmetic_intensity(self):
        return self.flops / self.logical_bytes


def _positive(*values):
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('Dimensions and element size must be positive integers')


def gemm_cost(m, n, k, element_bytes):
    _positive(m, n, k, element_bytes)
    return Workload('gemm', 2*m*n*k, (m*k + k*n + m*n)*element_bytes)


def residual_cost(rows, width, element_bytes):
    _positive(rows, width, element_bytes)
    # Add, square, reduction, scaling and weight multiply; excludes rsqrt.
    return Workload('residual_rmsnorm', rows*(5*width),
                    (4*rows*width + width)*element_bytes)


def rope_cost(rows, width, element_bytes):
    _positive(rows, width, element_bytes)
    if width % 2:
        raise ValueError('RoPE width must be even')
    return Workload('rope', 3*rows*width, 3*rows*width*element_bytes)


def roofline(work, peak_tflops, bandwidth_gbs):
    if any(not math.isfinite(v) or v <= 0 for v in (peak_tflops, bandwidth_gbs)):
        raise ValueError('Hardware ceilings must be finite and positive')
    if work.flops <= 0 or work.logical_bytes <= 0:
        raise ValueError('Workload costs must be positive')
    compute_ms = work.flops / (peak_tflops * 1e9)
    memory_ms = work.logical_bytes / (bandwidth_gbs * 1e6)
    return {'arithmetic_intensity': work.arithmetic_intensity,
            'compute_bound_ms': compute_ms, 'memory_bound_ms': memory_ms,
            'ideal_lower_bound_ms': max(compute_ms, memory_ms),
            'ridge_flops_per_byte': 1000*peak_tflops/bandwidth_gbs,
            'model_limit': 'compute' if compute_ms >= memory_ms else 'memory'}


def effective_rates(work, milliseconds):
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        raise ValueError('Latency must be finite and positive')
    return {'effective_tflops': work.flops/(milliseconds*1e9),
            'logical_gbs': work.logical_bytes/(milliseconds*1e6)}
