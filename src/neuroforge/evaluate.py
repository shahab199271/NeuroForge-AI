"""Evaluate base or PEFT-adapted model on held-out IMDb sentiment reviews."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from .core import BASE_MODEL, load_model, predict


def calculate_metrics(targets: list[str], predictions: list[str | None]) -> dict:
    """Treat invalid generations as incorrect for all-sample accuracy.

    Conditional F1 explicitly excludes invalid predictions; always show invalid rate.
    """
    if not targets or len(targets) != len(predictions):
        raise ValueError("Non-empty and equally sized targets/predictions required")
    valid = [i for i, label in enumerate(predictions) if label in {"negative", "positive"}]
    metrics = {
        "samples": len(targets),
        "valid_predictions": len(valid),
        "invalid_rate": (len(targets) - len(valid)) / len(targets),
        "accuracy_all": sum(a == b for a, b in zip(targets, predictions)) / len(targets),
        "macro_f1_valid_only": None,
        "accuracy_valid_only": None,
    }
    if valid:
        truth = [targets[i] for i in valid]
        predicted = [predictions[i] for i in valid]
        metrics["macro_f1_valid_only"] = f1_score(
            truth, predicted, labels=["negative", "positive"], average="macro", zero_division=0
        )
        metrics["accuracy_valid_only"] = accuracy_score(truth, predicted)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="Trained LoRA/QLoRA adapter; omit for untrained baseline")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-json", help="Save reproducible evaluation summary")
    args = parser.parse_args()
    if args.test_samples < 1:
        parser.error("--test-samples must be positive")
    if args.adapter:
        tokenizer, model = load_model(args.adapter, args.model, args.load_in_4bit)
    else:
        from .core import load_base_model
        tokenizer, model = load_base_model(args.model, args.load_in_4bit)

    dataset = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=args.seed)
    dataset = dataset.select(range(min(args.test_samples, len(dataset))))
    targets, predictions = [], []
    for example in dataset:
        label, _ = predict(example["text"], tokenizer, model)
        predictions.append(label)
        targets.append("positive" if example["label"] == 1 else "negative")

    report = calculate_metrics(targets, predictions)
    report.update({
        "base_model": args.model,
        "adapter": args.adapter,
        "four_bit_inference": args.load_in_4bit,
        "dataset": "stanfordnlp/imdb",
        "split": "test",
        "seed": args.seed,
    })
    formatted = json.dumps(report, indent=2)
    print(formatted)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
