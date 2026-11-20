"""Benchmark end-to-end single-request inference; not concurrent server throughput."""
import argparse
import json
import statistics
import time

import torch

from .core import BASE_MODEL, load_model, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--text", default="The movie had wonderful acting and an inspiring ending.")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations >= 1 and warmup >= 0 required")
    tokenizer, model = load_model(args.adapter, args.model, args.load_in_4bit)

    def timed():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = predict(args.text, tokenizer, model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000, prediction

    for _ in range(args.warmup):
        timed()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    durations = []
    last_prediction = None
    for _ in range(args.iterations):
        ms, last_prediction = timed()
        durations.append(ms)

    ordered = sorted(durations)
    p95_idx = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.99999) - 1))
    report = {
        "model": args.model,
        "adapter": args.adapter,
        "four_bit": args.load_in_4bit,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "median_ms": round(statistics.median(durations), 2),
        "p95_ms": round(ordered[p95_idx], 2),
        "requests_per_sec_sequential": round(1000 / statistics.mean(durations), 3),
        "peak_gpu_allocated_mb": (
            round(torch.cuda.max_memory_allocated() / 1e6, 2)
            if torch.cuda.is_available() else None
        ),
        "last_prediction": last_prediction[0],
        "note": "Tokenization + generation; sequential, not server load testing",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
