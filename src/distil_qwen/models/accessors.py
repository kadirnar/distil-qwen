"""Centralized access to the nested Qwen3-ASR modules."""

from __future__ import annotations

from typing import Any

from distil_qwen.errors import ModelContractError


def get_thinker(model: Any) -> Any:
    """Return the conditional-generation component from a Qwen3-ASR model."""

    if hasattr(model, "module"):
        return get_thinker(model.module)
    if hasattr(model, "_orig_mod"):
        return get_thinker(model._orig_mod)
    if hasattr(model, "thinker"):
        return model.thinker
    wrapped = getattr(model, "model", None)
    if wrapped is not None and hasattr(wrapped, "thinker"):
        return wrapped.thinker
    raise ModelContractError("expected a Qwen3-ASR model exposing `.thinker`")


def get_text_model(model: Any) -> Any:
    thinker = get_thinker(model)
    text_model = getattr(thinker, "model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise ModelContractError("Qwen3-ASR thinker does not expose text decoder layers")
    return text_model


def get_audio_tower(model: Any) -> Any:
    thinker = get_thinker(model)
    tower = getattr(thinker, "audio_tower", None)
    if tower is None or not hasattr(tower, "layers"):
        raise ModelContractError("Qwen3-ASR thinker does not expose an audio tower")
    return tower


def get_lm_head(model: Any) -> Any:
    thinker = get_thinker(model)
    head = getattr(thinker, "lm_head", None)
    if head is None:
        raise ModelContractError("Qwen3-ASR thinker does not expose an LM head")
    return head
