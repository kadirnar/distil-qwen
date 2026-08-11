import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from distil_qwen.config import DistillationConfig
from distil_qwen.training.loss import chunked_distillation_loss


def test_chunked_loss_matches_full_projection() -> None:
    torch.manual_seed(7)
    student_hidden = torch.randn(2, 5, 4, requires_grad=True)
    teacher_hidden = torch.randn(2, 5, 4, requires_grad=True)
    student_head = nn.Linear(4, 11, bias=False)
    teacher_head = nn.Linear(4, 11, bias=False)
    labels = torch.tensor(
        [
            [-100, -100, 2, 3, 4],
            [-100, 5, 6, -100, 7],
        ]
    )
    config = DistillationConfig(temperature=1.7, logit_chunk_size=2)

    result = chunked_distillation_loss(
        student_hidden,
        teacher_hidden,
        labels,
        student_head,
        teacher_head,
        config,
    )

    targets = labels[:, 1:]
    mask = targets.ne(-100)
    selected_targets = targets[mask]
    student_logits = student_head(student_hidden[:, :-1][mask]).float()
    teacher_logits = teacher_head(teacher_hidden[:, :-1][mask]).float()
    expected_ce = functional.cross_entropy(student_logits, selected_targets)
    expected_kl = (
        functional.kl_div(
            functional.log_softmax(student_logits / config.temperature, dim=-1),
            functional.softmax(teacher_logits / config.temperature, dim=-1),
            reduction="sum",
        )
        / selected_targets.numel()
        * (config.temperature**2)
    )

    assert result.num_tokens == selected_targets.numel()
    torch.testing.assert_close(result.ce_loss, expected_ce)
    torch.testing.assert_close(result.kl_loss, expected_kl)
    torch.testing.assert_close(result.loss, expected_ce + 0.8 * expected_kl)

    result.loss.backward()
    assert student_hidden.grad is not None
    assert student_head.weight.grad is not None
    assert teacher_hidden.grad is None
    assert teacher_head.weight.grad is None


def test_loss_rejects_empty_supervision() -> None:
    hidden = torch.randn(1, 3, 2)
    labels = torch.full((1, 3), -100)
    with pytest.raises(ValueError, match="no supervised"):
        chunked_distillation_loss(hidden, hidden, labels, nn.Linear(2, 4), nn.Linear(2, 4))


def test_loss_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="expected hidden"):
        chunked_distillation_loss(
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.ones(2, 3, dtype=torch.long),
            nn.Linear(3, 4),
            nn.Linear(3, 4),
        )
    with pytest.raises(ValueError, match="incompatible"):
        chunked_distillation_loss(
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
            torch.ones(2, 2, dtype=torch.long),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        )
