"""Exact, memory-bounded knowledge-distillation losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as functional

from distil_qwen.config import DistillationConfig


@dataclass
class DistillationLossOutput:
    """Scalar objective and detached metrics."""

    loss: torch.Tensor
    ce_loss: torch.Tensor
    kl_loss: torch.Tensor
    num_tokens: int

    def metrics(self) -> dict[str, Any]:
        return {
            "loss": self.loss.detach(),
            "ce_loss": self.ce_loss.detach(),
            "kl_loss": self.kl_loss.detach(),
            "num_tokens": self.num_tokens,
        }


def _supervised_hidden_states(
    hidden_states: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "expected hidden states [batch, sequence, hidden] and labels [batch, sequence]"
        )
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden states and labels have incompatible batch/sequence dimensions")
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)
    return hidden_states[:, :-1, :][mask], shifted_labels[mask]


def chunked_distillation_loss(
    student_hidden_states: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    labels: torch.Tensor,
    student_lm_head: torch.nn.Module,
    teacher_lm_head: torch.nn.Module,
    config: Optional[DistillationConfig] = None,
) -> DistillationLossOutput:
    """Compute causal CE and exact forward KL without materializing sequence-wide logits.

    Only positions with non-ignored next-token labels are projected to the vocabulary. Projection
    happens in small token chunks, so peak logit memory is bounded by
    ``logit_chunk_size * vocabulary_size`` rather than batch and sequence length.
    """

    config = config or DistillationConfig()
    student_tokens, target_tokens = _supervised_hidden_states(student_hidden_states, labels)
    teacher_tokens, teacher_targets = _supervised_hidden_states(teacher_hidden_states, labels)
    if not torch.equal(target_tokens, teacher_targets):
        raise ValueError("teacher and student supervised positions differ")
    if student_tokens.shape[0] == 0:
        raise ValueError("the batch contains no supervised target tokens")
    if student_tokens.shape[0] != teacher_tokens.shape[0]:
        raise ValueError("teacher and student have different supervised token counts")

    ce_sum = torch.zeros((), device=student_tokens.device, dtype=torch.float32)
    kl_sum = torch.zeros((), device=student_tokens.device, dtype=torch.float32)
    temperature = config.temperature

    for start in range(0, student_tokens.shape[0], config.logit_chunk_size):
        stop = start + config.logit_chunk_size
        targets = target_tokens[start:stop]
        with torch.no_grad():
            teacher_logits = teacher_lm_head(teacher_tokens[start:stop]).float()
            teacher_probabilities = functional.softmax(teacher_logits / temperature, dim=-1)
            del teacher_logits

        student_logits = student_lm_head(student_tokens[start:stop]).float()
        ce_sum = ce_sum + functional.cross_entropy(
            student_logits,
            targets,
            reduction="sum",
            label_smoothing=config.label_smoothing,
        )
        student_log_probabilities = functional.log_softmax(student_logits / temperature, dim=-1)
        kl_sum = kl_sum + functional.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="sum",
        )

    num_tokens = int(target_tokens.numel())
    ce_loss = ce_sum / num_tokens
    kl_loss = (kl_sum / num_tokens) * (temperature**2)
    loss = config.ce_weight * ce_loss + config.kl_weight * kl_loss
    return DistillationLossOutput(
        loss=loss,
        ce_loss=ce_loss,
        kl_loss=kl_loss,
        num_tokens=num_tokens,
    )
