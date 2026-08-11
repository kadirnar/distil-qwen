"""Qwen3-ASR hidden-state forwarding for memory-efficient distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from distil_qwen.config import DistillationConfig
from distil_qwen.errors import ModelContractError
from distil_qwen.models.accessors import get_audio_tower, get_lm_head, get_text_model, get_thinker
from distil_qwen.training.loss import DistillationLossOutput, chunked_distillation_loss


@dataclass
class HiddenStatePair:
    student: torch.Tensor
    teacher: torch.Tensor


def freeze_module(module: torch.nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


def configure_student_trainability(
    student: torch.nn.Module,
    freeze_audio_tower: bool = True,
    freeze_embeddings: bool = True,
) -> int:
    """Freeze high-memory shared components and return the trainable parameter count."""

    if freeze_audio_tower:
        freeze_module(get_audio_tower(student))
    if freeze_embeddings:
        text_model = get_text_model(student)
        freeze_module(text_model.embed_tokens)
        freeze_module(get_lm_head(student))
    return sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)


def configure_teacher(teacher: torch.nn.Module) -> None:
    freeze_module(teacher)


def _audio_kwargs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = ("input_features", "feature_attention_mask", "audio_feature_lengths")
    return {key: batch[key] for key in keys if key in batch}


def _inject_audio(
    thinker: Any, input_ids: torch.Tensor, audio_features: torch.Tensor
) -> torch.Tensor:
    inputs_embeds = thinker.get_input_embeddings()(input_ids)
    audio_features = audio_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
    expected_values = audio_features.numel()
    if int(audio_mask.sum()) != expected_values:
        raise ModelContractError(
            "audio placeholder shape does not match encoded features: "
            f"mask={int(audio_mask.sum())}, features={expected_values}"
        )
    return inputs_embeds.masked_scatter(audio_mask, audio_features)


def _text_hidden_states(
    thinker: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = thinker.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=False,
    )
    return outputs[0]


def paired_hidden_states(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    reuse_audio_features: bool = True,
) -> HiddenStatePair:
    """Forward both text models, optionally encoding frozen audio exactly once."""

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    student_thinker = get_thinker(student)
    teacher_thinker = get_thinker(teacher)
    audio_inputs = _audio_kwargs(batch)

    if audio_inputs:
        if reuse_audio_features:
            with torch.no_grad():
                shared_audio = teacher_thinker.get_audio_features(**audio_inputs)
            student_embeds = _inject_audio(student_thinker, input_ids, shared_audio)
            teacher_embeds = _inject_audio(teacher_thinker, input_ids, shared_audio)
        else:
            student_audio = student_thinker.get_audio_features(**audio_inputs)
            with torch.no_grad():
                teacher_audio = teacher_thinker.get_audio_features(**audio_inputs)
            student_embeds = _inject_audio(student_thinker, input_ids, student_audio)
            teacher_embeds = _inject_audio(teacher_thinker, input_ids, teacher_audio)
    else:
        student_embeds = student_thinker.get_input_embeddings()(input_ids)
        teacher_embeds = teacher_thinker.get_input_embeddings()(input_ids)

    position_ids, _ = student_thinker.get_rope_index(attention_mask)
    student_hidden = _text_hidden_states(
        student_thinker, student_embeds, attention_mask, position_ids
    )
    with torch.no_grad():
        teacher_hidden = _text_hidden_states(
            teacher_thinker, teacher_embeds, attention_mask, position_ids
        )
    return HiddenStatePair(student=student_hidden, teacher=teacher_hidden)


def distillation_step(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    config: DistillationConfig,
) -> DistillationLossOutput:
    states = paired_hidden_states(
        student,
        teacher,
        batch,
        reuse_audio_features=config.reuse_audio_features,
    )
    return chunked_distillation_loss(
        student_hidden_states=states.student,
        teacher_hidden_states=states.teacher,
        labels=batch["labels"],
        student_lm_head=get_lm_head(student),
        teacher_lm_head=get_lm_head(teacher),
        config=config,
    )


def model_device(model: torch.nn.Module) -> Optional[torch.device]:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None
