"""Exact, memory-bounded knowledge-distillation losses."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as functional

from distil_qwen.config import DistillationConfig
from distil_qwen.errors import OptionalDependencyError


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


class _FrozenHeadDistillation(torch.autograd.Function):
    """Compute and discard chunk logits in forward, retaining only hidden-state gradients."""

    @staticmethod
    def forward(
        ctx: Any,
        student_tokens: torch.Tensor,
        student_weight: torch.Tensor,
        student_bias: Optional[torch.Tensor],
        teacher_tokens: torch.Tensor,
        teacher_weight: torch.Tensor,
        teacher_bias: Optional[torch.Tensor],
        target_tokens: torch.Tensor,
        temperature: float,
        ce_weight: float,
        kl_weight: float,
        chunk_size: int,
        label_smoothing: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = target_tokens.numel()
        gradients = torch.empty_like(student_tokens)
        ce_sum = torch.zeros((), device=student_tokens.device, dtype=torch.float32)
        kl_sum = torch.zeros((), device=student_tokens.device, dtype=torch.float32)

        for start in range(0, num_tokens, chunk_size):
            stop = start + chunk_size
            targets = target_tokens[start:stop]
            with torch.no_grad():
                teacher_logits = functional.linear(
                    teacher_tokens[start:stop], teacher_weight, teacher_bias
                ).float()
                teacher_probabilities = functional.softmax(teacher_logits / temperature, dim=-1)

            with torch.enable_grad():
                student_chunk = student_tokens[start:stop].detach().requires_grad_(True)
                student_logits = functional.linear(
                    student_chunk, student_weight, student_bias
                ).float()
                chunk_ce = functional.cross_entropy(
                    student_logits,
                    targets,
                    reduction="sum",
                    label_smoothing=label_smoothing,
                )
                student_log_probabilities = functional.log_softmax(
                    student_logits / temperature, dim=-1
                )
                chunk_kl = functional.kl_div(
                    student_log_probabilities,
                    teacher_probabilities,
                    reduction="sum",
                )
                chunk_loss = (
                    ce_weight * chunk_ce + kl_weight * (temperature**2) * chunk_kl
                ) / num_tokens
                chunk_gradient = torch.autograd.grad(chunk_loss, student_chunk)[0]

            gradients[start:stop].copy_(chunk_gradient)
            ce_sum.add_(chunk_ce.detach())
            kl_sum.add_(chunk_kl.detach())

        ce_loss = ce_sum / num_tokens
        kl_loss = (kl_sum / num_tokens) * (temperature**2)
        loss = ce_weight * ce_loss + kl_weight * kl_loss
        ctx.save_for_backward(gradients)
        ctx.mark_non_differentiable(ce_loss, kl_loss)
        return loss, ce_loss, kl_loss

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor, *_: Any) -> tuple[Any, ...]:
        (gradients,) = ctx.saved_tensors
        return (gradients * grad_loss.to(gradients.dtype),) + (None,) * 11


def _frozen_head_distillation_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    student_lm_head: torch.nn.Module,
    teacher_lm_head: torch.nn.Module,
    config: DistillationConfig,
) -> DistillationLossOutput:
    loss, ce_loss, kl_loss = _FrozenHeadDistillation.apply(
        student_tokens,
        student_lm_head.weight,
        student_lm_head.bias,
        teacher_tokens,
        teacher_lm_head.weight,
        teacher_lm_head.bias,
        target_tokens,
        config.temperature,
        config.ce_weight,
        config.kl_weight,
        config.logit_chunk_size,
        config.label_smoothing,
    )
    return DistillationLossOutput(
        loss=loss,
        ce_loss=ce_loss,
        kl_loss=kl_loss,
        num_tokens=target_tokens.numel(),
    )


def _liger_distillation_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    student_lm_head: torch.nn.Module,
    teacher_lm_head: torch.nn.Module,
    config: DistillationConfig,
) -> DistillationLossOutput:
    try:
        from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
    except ImportError as exc:
        raise OptionalDependencyError(
            "The Liger loss requires `pip install 'distil-qwen[kernels]'`."
        ) from exc

    if config.label_smoothing:
        raise ValueError("the Liger distillation loss does not support label smoothing")
    loss_function = LigerFusedLinearJSDLoss(
        weight_hard_loss=config.ce_weight,
        weight_soft_loss=config.kl_weight * (config.temperature**2),
        beta=0.0,
        temperature=config.temperature,
        compiled=True,
        chunk_size=config.logit_chunk_size,
        return_soft_hard_loss=True,
    )
    loss, soft_loss, hard_loss = loss_function(
        student_tokens,
        student_lm_head.weight,
        teacher_tokens,
        teacher_lm_head.weight,
        target_tokens,
        student_lm_head.bias,
        teacher_lm_head.bias,
    )
    return DistillationLossOutput(
        loss=loss,
        ce_loss=hard_loss,
        kl_loss=soft_loss * (config.temperature**2),
        num_tokens=target_tokens.numel(),
    )


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

    student_head_trainable = any(
        parameter.requires_grad for parameter in student_lm_head.parameters()
    )
    liger_available = importlib.util.find_spec("liger_kernel") is not None
    use_liger = config.loss_backend == "liger" or (
        config.loss_backend == "auto"
        and student_tokens.is_cuda
        and liger_available
        and student_head_trainable
        and not config.label_smoothing
    )
    if use_liger:
        if not student_tokens.is_cuda:
            raise ValueError("the Liger distillation loss requires a supported accelerator")
        if not student_head_trainable:
            raise ValueError(
                "the Liger loss computes an LM-head gradient; use the torch backend when the "
                "LM head is frozen"
            )
        return _liger_distillation_loss(
            student_tokens,
            teacher_tokens,
            target_tokens,
            student_lm_head,
            teacher_lm_head,
            config,
        )

    if not student_head_trainable and student_tokens.requires_grad:
        return _frozen_head_distillation_loss(
            student_tokens,
            teacher_tokens,
            target_tokens,
            student_lm_head,
            teacher_lm_head,
            config,
        )

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
