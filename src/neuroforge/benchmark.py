"""Reproducible single-request inference benchmark (not a server-load test).

Measures warm steady-state full pipeline latency, with optional cold startup
time and both GPU allocated/reserved and process RSS. No fabricated speedups.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .core import BASE_MODEL, load_base_model, load_model, predict


def latency_summary(samples_ms: list[float]) -> dict:
    """Empirical nearest-rank percentiles; does not silently interpolate."""
    if not samples_ms or any(x <= 0 or not math.isfinite(x) for x in samples_ms):
        raise ValueError("Need nonempty positive finite latency measurements")
    ordered = sorted(samples_ms)
    def percentile(p):
        return ordered[math.ceil(p * len(ordered)) - 1]
    return {
        "mean_ms": round(statistics.mean(ordered), 3),
        "p50_ms": round(percentile(0.50), 3),
        "p90_ms": round(percentile(0.90), 3),
        "p95_ms": round(percentile(0.95), 3),
        "p99_ms": round(percentile(0.99), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "requests_per_sec_sequential": round(1000.0 / statistics.mean(ordered), 3),
    }


def hardware_metadata() -> dict:
    import transformers

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_total_vram_mb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1e6, 2)
            if torch.cuda.is_available() else None
        ),
    }


def process_rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 2)
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="Local adapter path (omit for base model)")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--text", default="The movie had wonderful acting and an inspiring ending.")
    parser.add_argument("--input-file", help="UTF-8 text file: one test review per line")
    parser.add_argument("--output-json", help="Write machine-readable measured results")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations >= 1 and warmup >= 0 required")

    reviews = [args.text]
    if args.input_file:
        reviews = [
            line.strip() for line in Path(args.input_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not reviews:
            parser.error("input-file must contain at least one nonempty review")

    start = time.perf_counter()
    if args.adapter:
        tokenizer, model = load_model(args.adapter, args.model, args.load_in_4bit)
    else:
        tokenizer, model = load_base_model(args.model, args.load_in_4bit)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold_load_s = time.perf_counter() - start

    def timed(review: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        label, raw = predict(review, tokenizer, model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - started), label, raw

    for i in range(args.warmup):
        timed(reviews[i % len(reviews)])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rss_before = process_rss_mb()
    measurements = []
    invalid = 0
    for i in range(args.iterations):
        elapsed, label, _ = timed(reviews[i % len(reviews)])
        measurements.append(elapsed)
        if label is None:
            invalid += 1

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter": args.adapter,
        "four_bit_inference": args.load_in_4bit,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "distinct_reviews": len(reviews),
        "input_char_lengths": [len(x) for x in reviews[:20]],
        "cold_model_load_seconds": round(cold_load_s, 3),
        "invalid_prediction_rate": round(invalid / args.iterations, 4),
        "gpu_peak_allocated_mb_post_warmup": (
            round(torch.cuda.max_memory_allocated() / 1e6, 2)
            if torch.cuda.is_available() else None
        ),
        "gpu_peak_reserved_mb_post_warmup": (
            round(torch.cuda.max_memory_reserved() / 1e6, 2)
            if torch.cuda.is_available() else None
        ),
        "process_rss_before_mb": rss_before,
        "process_rss_after_mb": process_rss_mb(),
        "hardware": hardware_metadata(),
        "latency": latency_summary(measurements),
        "measurement_scope": (
            "Sequential tokenizer + autoregressive generation + decode; "
            "warm execution, batch=1. Not server concurrency or total VRAM consumption."
        ),
    }
    result = json.dumps(report, indent=2, default=str)
    print(result)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
