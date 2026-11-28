# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Teacher–student logit distillation for shared-tokenizer causal language models.

Teacher and student MUST have identical token IDs/vocabulary mapping. For Qwen2.5
instruct models, use the exact student tokenizer and verify full vocabulary mapping.
The teacher is frozen. Only student LoRA adapters receive gradients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .core import BASE_MODEL
from .train import build_example


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return alpha * KD + (1-alpha) * hard-label CE on answer tokens.

    Causal shift: logits at t predict labels[t+1]. KL is teacher || student.
    Invalid (prompt/padded) targets are excluded *before* loss normalization.
    """
    if temperature <= 0 or not 0 <= alpha <= 1:
        raise ValueError("temperature > 0 and alpha in [0, 1] required")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher/student logits must share batch, time and vocabulary")
    if student_logits.shape[:2] != labels.shape:
        raise ValueError("Labels shape must match batch and sequence dimensions")
    s = student_logits[:, :-1, :].float()
    t = teacher_logits[:, :-1, :].detach().float()
    target = labels[:, 1:]
    valid = target.ne(-100)
    if not torch.any(valid):
        raise ValueError("No supervised answer tokens in batch")
    s_valid = s[valid]
    t_valid = t[valid]
    labels_valid = target[valid]
    ce = F.cross_entropy(s_valid, labels_valid)
    kd = F.kl_div(
        F.log_softmax(s_valid / temperature, dim=-1),
        F.softmax(t_valid / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    total = (1 - alpha) * ce + alpha * kd
    return total, {"ce": float(ce.detach()), "kd": float(kd.detach())}


class DistillationModule(nn.Module):
    """Pytorch training component; teacher has no trainable gradients."""

    def __init__(self, student: nn.Module, teacher: nn.Module, temperature=2.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, input_ids, attention_mask, labels):
        student_out = self.student(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            teacher_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
        return distillation_loss(
            student_out.logits,
            teacher_out.logits,
            labels,
            temperature=self.temperature,
            alpha=self.alpha,
        )


def main():
    """Distill a larger Qwen2.5 teacher into Qwen2.5-0.5B LoRA adapters."""
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Trainer, TrainingArguments, set_seed,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--student", default=BASE_MODEL)
    parser.add_argument("--output-dir", default="outputs/distilled-student")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_samples < 1 or args.eval_samples < 1 or args.max_length < 32:
        parser.error("samples must be positive and max-length >= 32")
    if args.epochs <= 0 or args.batch_size < 1 or args.grad_accum < 1:
        parser.error("epochs, batch size and gradient accumulation must be positive")
    if not 0 <= args.alpha <= 1 or args.temperature <= 0:
        parser.error("alpha in [0,1] and positive temperature required")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Teacher and student must have identical token-to-id vocabulary")
    if tokenizer.eos_token_id != teacher_tokenizer.eos_token_id:
        raise ValueError("Teacher and student EOS token IDs differ")
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device == "cuda" else torch.float32
    )
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype)
    if teacher.get_output_embeddings().weight.shape[0] != student.get_output_embeddings().weight.shape[0]:
        raise ValueError("Teacher/student output vocab sizes differ")
    teacher = teacher.to(device).eval()
    student = student.to(device)
    student.config.use_cache = False
    teacher.config.use_cache = False

    student = get_peft_model(
        student,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    raw = load_dataset("stanfordnlp/imdb")
    partition = raw["train"].train_test_split(
        test_size=0.1, seed=args.seed, stratify_by_column="label"
    )
    train_raw = partition["train"].shuffle(seed=args.seed).select(
        range(min(args.train_samples, len(partition["train"])))
    )
    val_raw = partition["test"].shuffle(seed=args.seed).select(
        range(min(args.eval_samples, len(partition["test"])))
    )

    def convert(batch):
        rows = [
            build_example(text, int(label), tokenizer, args.max_length)
            for text, label in zip(batch["text"], batch["label"])
        ]
        return {key: [row[key] for row in rows] for key in rows[0]}

    train_data = train_raw.map(convert, batched=True, remove_columns=train_raw.column_names)
    eval_data = val_raw.map(convert, batched=True, remove_columns=val_raw.column_names)

    module = DistillationModule(student, teacher, args.temperature, args.alpha)

    class KDTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, components = model(**inputs)
            # Avoid huge logits in Trainer predictions; metrics are computed separately.
            return (loss, {"loss": loss.detach()}) if return_outputs else loss

        def save_model(self, output_dir=None, _internal_call=False):
            target = output_dir or self.args.output_dir
            Path(target).mkdir(parents=True, exist_ok=True)
            self.model.student.save_pretrained(target)
            tokenizer.save_pretrained(target)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "distillation_config.json").write_text(json.dumps(vars(args), indent=2))
    trainer = KDTrainer(
        model=module,
        args=TrainingArguments(
            output_dir=str(out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            seed=args.seed,
            bf16=device == "cuda" and dtype == torch.bfloat16,
            fp16=device == "cuda" and dtype == torch.float16,
            remove_unused_columns=False,
        ),
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train()
    trainer.save_model(str(out))
    metrics = trainer.evaluate()
    (out / "validation_kd_loss.json").write_text(json.dumps(metrics, indent=2))
    print(f"Distilled student LoRA adapter saved to {out}. Evaluate on IMDb test next.")


if __name__ == "__main__":
    main()
