# Experiment plan

This document specifies **how to collect evidence**. It does not claim that any experiments have already run.

## Fixed conditions
- Base model / revision: `Qwen/Qwen2.5-0.5B-Instruct` (pin a specific revision in logged runs)
- Dataset: `stanfordnlp/imdb`; stratified train/validation, untouched IMDb test split
- Seed: 42 (then repeat with additional seeds)
- Training objective: answer-only supervised causal-language-model loss
- Tokenizer, sequence length, truncation, prompt template: keep fixed across comparisons

## Runs
| ID | Configuration | Accuracy | Macro F1 | Invalid % | P50 ms | P95 ms | Peak GPU MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Base model without training | TBD | TBD | TBD | TBD | TBD | TBD |
| B | LoRA adapter | TBD | TBD | TBD | TBD | TBD | TBD |
| C | QLoRA-trained adapter (NF4) | TBD | TBD | TBD | TBD | TBD | TBD |
| D | LoRA adapter, 4-bit inference | TBD | TBD | TBD | TBD | TBD | TBD |

**Important:** fine-tuning in QLoRA and quantized *inference* are distinct interventions. Compare each appropriately; do not infer speedups from smaller weight storage.

## Mandatory metadata
Date, OS, CPU, GPU, VRAM, device name, CUDA version, PyTorch/Transformers/PEFT versions, model and dataset revisions, seed, sample counts, tokens and batch size, LoRA ranks, precision, training steps, runtime duration.

## Next engineering steps
- Add optional validation metrics and deterministic snapshot/integration tests.
- Implement a distinct teacher–student distillation experiment.
- Add robust logging of model disk size and process / CPU memory.
- Export comparison figures only after recording genuine measurements.
