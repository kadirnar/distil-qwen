"""Unified optimized inference across Transformers, vLLM, SGLang, and llama.cpp."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch

from distil_qwen.backends.base import AudioBatch, Transcription, coerce_transcriptions
from distil_qwen.backends.transformers import (
    TransformersBackend,
    resolve_attention,
    resolve_device,
    resolve_dtype,
)
from distil_qwen.config import InferenceConfig


def _load_backend(
    model_name_or_path: Optional[str],
    config: InferenceConfig,
    **backend_kwargs: Any,
) -> Any:
    if config.backend == "transformers":
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required for Transformers inference")
        return TransformersBackend.from_pretrained(model_name_or_path, config, **backend_kwargs)
    if config.backend == "vllm":
        from distil_qwen.backends.vllm import VLLMBackend

        return VLLMBackend.from_pretrained(model_name_or_path, config, **backend_kwargs)
    if config.backend == "sglang":
        from distil_qwen.backends.sglang import create_sglang_backend

        return create_sglang_backend(model_name_or_path, config, **backend_kwargs)
    if config.backend == "llama_cpp":
        from distil_qwen.backends.llama_cpp import create_llama_cpp_backend

        return create_llama_cpp_backend(model_name_or_path, config, **backend_kwargs)
    raise AssertionError(f"validated backend was not dispatched: {config.backend}")


class OptimizedASR:
    """Stable library facade over all supported Qwen3-ASR inference runtimes."""

    def __init__(self, backend_model: Any, config: InferenceConfig) -> None:
        self.backend_model = backend_model
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
        **backend_kwargs: Any,
    ) -> OptimizedASR:
        resolved_config = config or InferenceConfig()
        backend = _load_backend(model_name_or_path, resolved_config, **backend_kwargs)
        return cls(backend, resolved_config)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: AudioBatch,
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[Transcription]:
        results = self.backend_model.transcribe(
            audio=audio,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )
        return coerce_transcriptions(results)

    def close(self) -> None:
        """Release an owned HTTP client; local runtimes are unaffected."""

        close = getattr(self.backend_model, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> OptimizedASR:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "OptimizedASR",
    "Transcription",
    "resolve_attention",
    "resolve_device",
    "resolve_dtype",
]
