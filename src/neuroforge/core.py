"""Shared prompts, strict sentiment parsing, and model inference."""
from __future__ import annotations

import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "Classify the sentiment of the movie review. "
    "Answer with exactly one lowercase word: positive or negative."
)


def make_messages(review: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Movie review: {review}"},
    ]


def parse_sentiment(text: str) -> str | None:
    answer = text.strip().lower()
    return answer if answer in {"positive", "negative"} else None


def load_model(adapter: str, base_model: str = BASE_MODEL, four_bit: bool = False):
    """Load an already-trained PEFT adapter; never silently use an untrained base model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs = {}
    if four_bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit inference requires a compatible CUDA installation")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if not four_bit and torch.cuda.is_available():
        base = base.to("cuda")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return tokenizer, model


def predict(review: str, tokenizer, model, max_new_tokens: int = 8) -> tuple[str | None, str]:
    prompt = tokenizer.apply_chat_template(
        make_messages(review), tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    return parse_sentiment(reply), reply
