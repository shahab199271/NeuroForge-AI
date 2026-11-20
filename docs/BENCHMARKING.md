# Benchmark methodology

`python -m neuroforge.benchmark` measures **sequential, batch=1, end-to-end client-side inference**, not FastAPI server load or concurrent throughput.

```bash
python -m neuroforge.benchmark --model Qwen/Qwen2.5-0.5B-Instruct \
  --iterations 50 --warmup 10 --output-json outputs/base_bench.json

python -m neuroforge.benchmark --adapter outputs/qwen-lora \
  --iterations 50 --warmup 10 --output-json outputs/lora_bench.json

python -m neuroforge.benchmark --adapter outputs/qwen-lora \
  --load-in-4bit --iterations 50 --warmup 10 \
  --output-json outputs/lora_nf4_bench.json
```

Use `--input-file path/to/reviews.txt` to cycle across nonempty reviews, one per line. Prefer several input lengths. Each timing wraps preprocessing, autoregressive generation, and decoding. CUDA synchronization occurs before/after timed calls. Loading time is separately recorded, and warm-ups are excluded from timing. P50/P90/P95/P99 are nearest-rank empirical percentiles.

A JSON file records environment, GPU, VRAM, process RSS (when `psutil` is installed), GPU allocated/reserved peaks after warm-up, invalid label rate, and per-run latency statistics.

## What the metrics do not mean

- `requests_per_sec_sequential` is the reciprocal of mean single-request duration; it is **not concurrency-tested server throughput**.
- `gpu_peak_allocated_mb_post_warmup` is a PyTorch allocator statistic. It does **not** represent total process VRAM or model weight size.
- RSS is process resident system memory, **not peak RAM**.
- A faster response on one short prompt does not imply robust speedups at longer lengths or larger batch sizes.
- Peak memory numbers include allocations that survive warm-up but may not capture short-lived allocations during model load. Report load-time memory separately where practical.
- 4-bit loading is an inference representation, not evidence the model was trained using QLoRA.

## Reproducibility checklist

Lock model revisions and software dependencies; use identical input sets and random seeds; run repeated trials; disclose device, precision, tokenizer settings and measured latency distribution; report accuracy and invalid-output rate alongside speed/size comparisons. **No measured results have been published yet.**
