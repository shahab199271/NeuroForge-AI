# NeuroForge AI
# Maintainer: Shahab Baloochi

from neuroforge.train import build_example


class FakeTokenizer:
    eos_token = "<EOS>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT "

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def test_answer_only_labels():
    sample = build_example("Nice movie.", 1, FakeTokenizer(), max_length=32)
    assert len(sample["input_ids"]) == len(sample["labels"])
    assert sample["labels"][-1] != -100
    assert sample["labels"][0] == -100


def test_preserves_answer_when_truncating():
    sample = build_example("Long review", 0, FakeTokenizer(), max_length=20)
    assert len(sample["input_ids"]) == 20
    assert sum(value != -100 for value in sample["labels"]) > 0
