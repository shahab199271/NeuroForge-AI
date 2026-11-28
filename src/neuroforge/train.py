# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Fine-tune an instruction-following sentiment LLM with LoRA or 4-bit QLoRA."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .core import BASE_MODEL, make_messages


def build_example(review: str, label: int, tokenizer, max_length: int = 384) -> dict:
    """Keep the answer in the context and mask all prompt loss positions."""
    prompt = tokenizer.apply_chat_template(
        make_messages(review), tokenize=False, add_generation_prompt=True
    )
    answer = ("negative" if label == 0 else "positive") + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(answer_ids) >= max_length:
        raise ValueError("max_length must exceed answer token count")
    prompt_ids = prompt_ids[-(max_length - len(answer_ids)) :]
    ids = prompt_ids + answer_ids
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default="outputs/qwen-lora")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--eval-samples", type=int, default=300)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    if args.train_samples < 1 or args.eval_samples < 1:
        parser.error("sample counts must be positive")
    if args.qlora and not torch.cuda.is_available():
        parser.error("--qlora requires compatible NVIDIA CUDA")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset("stanfordnlp/imdb")
    split = raw["train"].train_test_split(
        test_size=0.1, seed=args.seed, stratify_by_column="label"
    )
    train_raw = split["train"].shuffle(seed=args.seed).select(
        range(min(args.train_samples, len(split["train"])))
    )
    val_raw = split["test"].shuffle(seed=args.seed).select(
        range(min(args.eval_samples, len(split["test"])))
    )

    def convert(batch):
        examples = [
            build_example(text, int(label), tokenizer, args.max_length)
            for text, label in zip(batch["text"], batch["label"])
        ]
        return {key: [example[key] for example in examples] for key in examples[0]}

    train_data = train_raw.map(convert, batched=True, remove_columns=train_raw.column_names)
    eval_data = val_raw.map(convert, batched=True, remove_columns=val_raw.column_names)

    kwargs = {}
    if args.qlora:
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16 if torch.cuda.is_available() else torch.float32
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = trainer.evaluate()
    (output_dir / "validation_loss.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    print(f"Adapter saved: {output_dir}. Next: held-out evaluation.")


if __name__ == "__main__":
    main()
