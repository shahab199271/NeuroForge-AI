# NeuroForge AI

**Fine-tune · Compress · Benchmark · Deploy**

A reproducible, open-source LLM engineering starter focused on **parameter-efficient language-model adaptation**, sentiment classification, efficient inference and measurable trade-offs.

> **Project status — implementation, not benchmark claims.** LoRA/QLoRA training code and evaluation tooling are provided. A complete GPU training run, quantitative quality metrics, memory savings, and acceleration have **not yet been verified**. Do not report improvements without reproducing the experiments.

## System design

```mermaid
flowchart LR
    A[IMDb reviews] --> B[Chat-format instructions]
    B --> C[Qwen2.5 0.5B Instruct]
    C --> D[LoRA or 4-bit QLoRA]
    D --> E[PEFT adapter]
    E --> F[Held-out evaluation]
    E --> G[Latency & memory benchmark]
    E --> H[FastAPI inference]
```

## Features

- **Real LLM fine-tuning:** optimize LoRA adapter parameters on sentiment instructions, instead of merely calling a hosted LLM API.
- **Optional QLoRA:** NF4 4-bit base-model loading for CUDA-compatible training environments.
- **Evaluation:** held-out IMDb test set, accuracy, macro-F1 and invalid-answer rate.
- **Benchmark:** warm-up, median/p95 single-request latency, requests/sec and peak allocated GPU memory.
- **Serving:** local FastAPI endpoint with lazy model loading and typed requests.
- **Engineering:** Python package, tests, linting, GitHub Actions and Docker support.

## Setup

Python 3.10+; a CUDA-capable GPU is recommended for training. Downloading the base model and IMDb dataset requires internet access.

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest -q
```

For CUDA-enabled QLoRA, install the optional dependency: `pip install -e '.[dev,qlora]'`.

## Train

```bash
# Genuine LoRA fine-tuning:
python -m neuroforge.train --output-dir outputs/qwen-lora --train-samples 2000 --eval-samples 300 --epochs 1

# 4-bit QLoRA (compatible NVIDIA CUDA):
python -m neuroforge.train --qlora --output-dir outputs/qwen-qlora --train-samples 2000 --eval-samples 300 --epochs 1
```

The training objective masks prompt tokens and optimizes only the assistant's target sentiment label. A stratified validation subset comes from the original IMDb training split. The IMDb test split is reserved for reporting.

## Evaluate and benchmark

```bash
python -m neuroforge.evaluate --adapter outputs/qwen-lora --test-samples 200
python -m neuroforge.benchmark --adapter outputs/qwen-lora --iterations 30 --warmup 5
# To compare 4-bit inference: add --load-in-4bit
```

These are *example commands*, not evidence of completed experiments. Record GPU, software versions, seed, dataset subset, batch size, model revision, adapter configuration, F1, latency and memory for each experiment. See [experimental protocol](docs/EXPERIMENTS.md).

## Local API

```bash
export NEUROFORGE_ADAPTER=outputs/qwen-lora
uvicorn neuroforge.api:app --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' -d '{"text":"An unexpectedly delightful movie."}'
```

For a valid prediction, the API returns e.g. `{"sentiment":"positive","raw_output":"positive"}`; unexpected free-form responses return HTTP 422, not a fabricated classification. `GET /health` checks whether an adapter path is configured, not whether weights are actually loadable.

## Repository roadmap

1. Verify LoRA and QLoRA GPU training end-to-end with pinned versions and reproducible runs.
2. Compare FP16/BF16 against supported 4-bit inference and measure the accuracy–latency–memory trade-off.
3. Add a separate teacher–student **knowledge distillation** pipeline and evaluate its benefits. **Distillation is planned, not implemented in this release.**
4. Add additional model families, model/export formats, dashboards and deployment hardening.

## Responsible publication

This is a newly authored implementation motivated by a prior DistilBERT / LLaMA API sentiment prototype. **The original notebook, its third-party pretrained artifacts, and its API credentials are not published here.** Do not commit API keys, model checkpoints, IMDb dataset copies, or private files. Models and datasets retain their respective upstream licences. The MIT licence applies only to the original code in this repository.
