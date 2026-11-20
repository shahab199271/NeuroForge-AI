"""Distillation tests run entirely offline with tiny random models; no GPU required."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neuroforge.distillation import DistillationModule, distillation_loss


class TinyLM(nn.Module):
    def __init__(self, vocab=13, hidden=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def test_distillation_masks_prompt_and_padding_and_updates_only_student():
    torch.manual_seed(5)
    student = TinyLM()
    teacher = TinyLM()
    module = DistillationModule(student, teacher, alpha=0.7, temperature=2.0)
    module.train()
    assert not module.teacher.training
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 3, 4], [-100, -100, 7, -100]])
    loss, metrics = module(input_ids, mask, labels)
    assert torch.isfinite(loss) and loss.item() > 0
    assert metrics["kd"] >= 0 and metrics["ce"] > 0
    loss.backward()
    assert student.head.weight.grad is not None
    assert torch.count_nonzero(student.head.weight.grad) > 0
    assert all(param.grad is None for param in teacher.parameters())
    before = student.head.weight.detach().clone()
    torch.optim.SGD(student.parameters(), lr=0.01).step()
    assert not torch.equal(before, student.head.weight.detach())


def test_masked_positions_do_not_change_loss():
    torch.manual_seed(7)
    student = torch.randn(1, 5, 11)
    teacher = torch.randn(1, 5, 11)
    labels = torch.tensor([[-100, -100, -100, 5, 6]])
    original, _ = distillation_loss(student, teacher, labels)
    student[:, :2] += 100
    teacher[:, :2] -= 100
    modified, _ = distillation_loss(student, teacher, labels)
    assert torch.allclose(original, modified, atol=1e-6)


@pytest.mark.parametrize("temp,alpha", [(0, 0.5), (-1, 0.5), (1, -0.1), (2, 1.1)])
def test_invalid_hyperparameters(temp, alpha):
    with pytest.raises(ValueError):
        distillation_loss(torch.ones(1, 3, 4), torch.ones(1, 3, 4),
                          torch.tensor([[-100, 2, 3]]), temp, alpha)


def test_no_supervised_positions_is_an_error():
    with pytest.raises(ValueError, match="No supervised"):
        distillation_loss(torch.zeros(1, 3, 4), torch.zeros(1, 3, 4),
                          torch.full((1, 3), -100))
