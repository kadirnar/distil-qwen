"""Tools for distilling and serving Qwen3-ASR models."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from distil_qwen.config import DistillationConfig, InferenceConfig, StudentSpec
from distil_qwen.inference import OptimizedASR, Transcription

__all__ = [
    "DistillationConfig",
    "InferenceConfig",
    "OptimizedASR",
    "StudentSpec",
    "TrainingRunConfig",
    "Transcription",
    "chunked_distillation_loss",
    "initialize_student",
    "run_training",
    "uniform_layer_indices",
]

_LAZY_EXPORTS = {
    "TrainingRunConfig": ("distil_qwen.training.run", "TrainingRunConfig"),
    "chunked_distillation_loss": ("distil_qwen.training.loss", "chunked_distillation_loss"),
    "initialize_student": ("distil_qwen.models.initialization", "initialize_student"),
    "run_training": ("distil_qwen.training.run", "run_training"),
    "uniform_layer_indices": ("distil_qwen.models.initialization", "uniform_layer_indices"),
}


def __getattr__(name: str) -> Any:
    """Keep optional Transformers training modules out of server-only processes."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__version__ = "0.1.0"
