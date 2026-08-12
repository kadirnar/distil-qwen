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


def test_frozen_head_path_matches_full_loss_and_hidden_gradient() -> None:
    torch.manual_seed(11)
    optimized_hidden = torch.randn(2, 5, 4, requires_grad=True)
    reference_hidden = optimized_hidden.detach().clone().requires_grad_(True)
    teacher_hidden = torch.randn(2, 5, 4)
    student_head = nn.Linear(4, 13, bias=True).requires_grad_(False)
    teacher_head = nn.Linear(4, 13, bias=True).requires_grad_(False)
    labels = torch.tensor([[-100, 2, 3, 4, 5], [-100, -100, 6, 7, 8]])
    config = DistillationConfig(
        temperature=1.5,
        ce_weight=0.7,
        kl_weight=0.3,
        logit_chunk_size=2,
        label_smoothing=0.1,
        loss_backend="torch",
    )

    result = chunked_distillation_loss(
        optimized_hidden,
        teacher_hidden,
        labels,
        student_head,
        teacher_head,
        config,
    )
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    selected_targets = targets[mask]
    student_logits = student_head(reference_hidden[:, :-1][mask]).float()
    teacher_logits = teacher_head(teacher_hidden[:, :-1][mask]).float()
    reference_ce = functional.cross_entropy(
        student_logits, selected_targets, label_smoothing=config.label_smoothing
    )
    reference_kl = (
        functional.kl_div(
            functional.log_softmax(student_logits / config.temperature, dim=-1),
            functional.softmax(teacher_logits / config.temperature, dim=-1),
            reduction="sum",
        )
        / selected_targets.numel()
        * (config.temperature**2)
    )
    reference_loss = config.ce_weight * reference_ce + config.kl_weight * reference_kl

    result.loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(result.ce_loss, reference_ce)
    torch.testing.assert_close(result.kl_loss, reference_kl)
    torch.testing.assert_close(result.loss, reference_loss)
    torch.testing.assert_close(optimized_hidden.grad, reference_hidden.grad)
    assert student_head.weight.grad is None


def test_liger_loss_request_rejects_cpu_tensors() -> None:
    hidden = torch.randn(1, 3, 2, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2]])
    with pytest.raises(ValueError, match="requires a supported accelerator"):
        chunked_distillation_loss(
            hidden,
            hidden.detach(),
            labels,
            nn.Linear(2, 4),
            nn.Linear(2, 4),
            DistillationConfig(loss_backend="liger"),
        )


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
