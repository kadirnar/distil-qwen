"""Tools for distilling and serving Qwen3-ASR models."""

from distil_qwen.config import DistillationConfig, InferenceConfig, StudentSpec
from distil_qwen.models.initialization import initialize_student, uniform_layer_indices
from distil_qwen.training.loss import chunked_distillation_loss

__all__ = [
    "DistillationConfig",
    "InferenceConfig",
    "StudentSpec",
    "chunked_distillation_loss",
    "initialize_student",
    "uniform_layer_indices",
]

__version__ = "0.1.0"
