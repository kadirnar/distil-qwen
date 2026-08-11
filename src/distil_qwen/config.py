"""Validated public configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class StudentSpec:
    """Architecture of a student initialized from Qwen3-ASR-1.7B."""

    text_layers: int = 14
    audio_layers: Optional[int] = None

    def __post_init__(self) -> None:
        if self.text_layers < 1:
            raise ValueError("text_layers must be at least 1")
        if self.audio_layers is not None and self.audio_layers < 1:
            raise ValueError("audio_layers must be at least 1 when specified")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistillationConfig:
    """Loss and memory settings for teacher-student training."""

    temperature: float = 2.0
    ce_weight: float = 1.0
    kl_weight: float = 0.8
    logit_chunk_size: int = 32
    label_smoothing: float = 0.0
    reuse_audio_features: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.ce_weight < 0 or self.kl_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if self.ce_weight == 0 and self.kl_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        if self.logit_chunk_size < 1:
            raise ValueError("logit_chunk_size must be at least 1")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceConfig:
    """Portable inference settings for Transformers or vLLM."""

    backend: str = "transformers"
    device: str = "auto"
    dtype: str = "auto"
    attention: str = "auto"
    batch_size: int = 8
    max_new_tokens: int = 256
    quantization: Optional[str] = None
    compile_model: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"transformers", "vllm"}:
            raise ValueError("backend must be 'transformers' or 'vllm'")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("unsupported dtype")
        if self.attention not in {"auto", "eager", "sdpa", "flash_attention_2"}:
            raise ValueError("unsupported attention implementation")
        if self.quantization not in {None, "4bit", "8bit"}:
            raise ValueError("quantization must be None, '4bit', or '8bit'")
        if self.batch_size < 1 or self.max_new_tokens < 1:
            raise ValueError("batch_size and max_new_tokens must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
