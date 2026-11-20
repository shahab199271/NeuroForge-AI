"""Held-out IMDb evaluation: accuracy, macro F1, invalid-output rate."""
import argparse
import json

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from .core import BASE_MODEL, load_model, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    if args.test_samples < 1:
        parser.error("--test-samples must be positive")
    tokenizer, model = load_model(args.adapter, args.model, args.load_in_4bit)
    dataset = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=args.seed)
    dataset = dataset.select(range(min(args.test_samples, len(dataset))))
    targets, predictions = [], []
    for example in dataset:
        label, _ = predict(example["text"], tokenizer, model)
        predictions.append(label)
        targets.append("positive" if example["label"] == 1 else "negative")
    valid = [i for i, label in enumerate(predictions) if label is not None]
    metrics = {
        "samples": len(targets),
        "valid_predictions": len(valid),
        "invalid_rate": (len(targets) - len(valid)) / len(targets),
        "accuracy_all": sum(a == b for a, b in zip(targets, predictions)) / len(targets),
        "macro_f1_valid_only": (
            f1_score(
                [targets[i] for i in valid], [predictions[i] for i in valid],
                labels=["negative", "positive"], average="macro", zero_division=0
            ) if valid else None
        ),
        "accuracy_valid_only": (
            accuracy_score([targets[i] for i in valid], [predictions[i] for i in valid])
            if valid else None
        ),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
