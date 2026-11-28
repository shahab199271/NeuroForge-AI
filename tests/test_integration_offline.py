# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Actual local integration: tiny random GPT-2, PEFT LoRA save/reload; no downloads."""
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from neuroforge.distillation import DistillationModule


def test_tiny_student_teacher_training_and_adapter_roundtrip(tmp_path):
    torch.manual_seed(10)
    config = GPT2Config(
        vocab_size=32, n_positions=32, n_ctx=32, n_embd=16,
        n_layer=1, n_head=2, bos_token_id=0, eos_token_id=1,
    )
    teacher = GPT2LMHeadModel(config)
    student = GPT2LMHeadModel(config)
    student = get_peft_model(
        student,
        LoraConfig(
            task_type="CAUSAL_LM", r=2, lora_alpha=4,
            target_modules=["c_attn"], lora_dropout=0.0
        ),
    )
    module = DistillationModule(student, teacher)
    input_ids = torch.tensor([[2, 3, 4, 5, 6]])
    mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, -100, 5, 6]])
    optimizer = torch.optim.AdamW(
        [p for p in module.parameters() if p.requires_grad], lr=0.01
    )
    trainable_before = {
        name: param.detach().clone()
        for name, param in student.named_parameters() if param.requires_grad
    }
    module.train()
    loss, _ = module(input_ids, mask, labels)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(trainable_before[name], param.detach())
        for name, param in student.named_parameters() if param.requires_grad
    )
    assert all(not p.requires_grad for p in teacher.parameters())
    student.save_pretrained(str(tmp_path))
    loaded = PeftModel.from_pretrained(GPT2LMHeadModel(config), str(tmp_path))
    loaded.eval()
    with torch.no_grad():
        output = loaded(input_ids=input_ids)
    assert output.logits.shape == (1, 5, 32)
