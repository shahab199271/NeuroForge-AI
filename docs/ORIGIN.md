# Provenance and scope

The repository is a new implementation created from the design goals of an earlier sentiment project (DistilBERT classifier + a separately hosted LLaMA API). The original uploaded notebook and app contained environment-specific paths and potentially exposed credentials. They are **not** included here.

Calling an API is not fine-tuning. This project adds a local trainable PEFT adapter on an open-weight instruction model.

The original notebook referenced a third-party Hugging Face account for model artifacts. This repository does not reproduce or claim ownership of those artifacts. Third-party weights and datasets have separate licences.

**Current scope:** LoRA/QLoRA fine-tuning, held-out sentiment evaluation, local inference, and basic benchmarks. Knowledge distillation, ONNX optimization, and production-scale deployment are roadmap items rather than completed features.
