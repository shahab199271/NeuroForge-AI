# Teacher–student distillation protocol

## What is implemented

`python -m neuroforge.distillation` loads a frozen **Qwen2.5-1.5B-Instruct** teacher and a **Qwen2.5-0.5B-Instruct** student. It verifies that the model tokenizers share an identical token→ID mapping and that the output-vocabulary dimensions agree. A LoRA adapter is attached **only to the student**.

The IMDb training split provides hard target sentiment labels. The validation subset comes from IMDb training data; the held-out official IMDb test split is reserved for final evaluation. Input strings follow the same instruction prompt as the normal LoRA pipeline.

The loss is **answer-only causal next-token distillation**:

`L = (1 − alpha) * CE(student, gold_labels) + alpha * T^2 * KL(p_teacher^T || p_student^T)`

The logits are shifted against the next target token, the teacher probabilities are detached, and positions masked with `-100` (prompt and padding) contribute to neither term. The teacher is always in `eval()` mode and has frozen parameters.

This is **logit distillation** using the teacher's probability distribution at the supervised output positions. It is not sequence-level generation distillation, not teacher-generated synthetic data, and not a claim that the smaller model matches the teacher.

## Usage

```bash
pip install -e '.[dev]'
python -m neuroforge.distillation \
  --teacher Qwen/Qwen2.5-1.5B-Instruct \
  --student Qwen/Qwen2.5-0.5B-Instruct \
  --alpha 0.5 --temperature 2.0 \
  --train-samples 1000 --eval-samples 200 \
  --output-dir outputs/distilled-student
```

**GPU memory warning:** Both teacher and student must reside in memory for this baseline implementation. It does not currently offload teacher logits or provide tensor-parallel support. Use a sufficiently capable machine, or reduce sequence length and batch size. The teacher and student must have **exactly matching token IDs**, and CPU training can be slow.

## Verification ladder

1. `pytest -q tests/test_distillation.py tests/test_integration_offline.py`: offline tiny-tensor and tiny-random-transformer gradient tests.
2. Model compatibility smoke test: download both exact Qwen revisions and verify tokenizer vocabulary and output shapes.
3. Short actual run: train a small subset and confirm adapter save/reload.
4. Full training: record hardware, versions, model revisions, random seed and training duration.
5. Held-out comparison against the non-fine-tuned base model, direct LoRA, and QLoRA student under identical evaluation and tokenization conditions.
6. Measure **student inference only**, not teacher+student training memory, to quantify deployment tradeoffs.

Training implementation and offline tests **are not proof of a successful Qwen GPU run**. There are no measured improvements yet.
