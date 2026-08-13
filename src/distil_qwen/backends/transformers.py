"""Local Transformers inference for legacy and native Qwen3-ASR checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from distil_qwen.backends.base import AudioBatch, Transcription, coerce_transcriptions
from distil_qwen.compat import flash_attention_available, require_qwen_asr
from distil_qwen.config import InferenceConfig
from distil_qwen.errors import (
    BackendConfigurationError,
    BackendFeatureError,
    OptionalDependencyError,
)


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.startswith("cuda") or device == "mps":
        return torch.float16
    return torch.float32


def resolve_attention(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    if device.startswith("cuda") and flash_attention_available():
        return "flash_attention_2"
    return "sdpa"


def _quantization_config(mode: Optional[str], dtype: torch.dtype) -> Any:
    if mode is None:
        return None
    if not torch.cuda.is_available():
        raise ValueError("bitsandbytes quantization requires CUDA")
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise OptionalDependencyError(
            "quantized inference requires `pip install 'distil-qwen[quantization]'`"
        ) from exc
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _is_native_checkpoint(model_name_or_path: str) -> bool:
    if model_name_or_path.rstrip("/").endswith("-hf"):
        return True
    config_path = Path(model_name_or_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        version = str(json.loads(config_path.read_text())["transformers_version"])
        return int(version.split(".", 1)[0]) >= 5
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


class TransformersBackend:
    """Select the Qwen runtime or native Hugging Face implementation."""

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: InferenceConfig,
        **backend_kwargs: Any,
    ) -> TransformersBackend:
        implementation = backend_kwargs.pop("implementation", "auto")
        if implementation not in {"auto", "native", "qwen-asr"}:
            raise BackendConfigurationError(
                "Transformers implementation must be 'auto', 'native', or 'qwen-asr'"
            )
        use_native = implementation == "native" or (
            implementation == "auto" and _is_native_checkpoint(model_name_or_path)
        )
        backend_type = NativeTransformersBackend if use_native else QwenTransformersBackend
        return cls(backend_type.from_pretrained(model_name_or_path, config, **backend_kwargs))

    def transcribe(
        self,
        audio: AudioBatch,
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[Transcription]:
        return self.implementation.transcribe(
            audio=audio,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )

    def close(self) -> None:
        self.implementation.close()


class QwenTransformersBackend:
    """Official ``qwen-asr`` Transformers wrapper used by distilled checkpoints."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: InferenceConfig,
        **backend_kwargs: Any,
    ) -> QwenTransformersBackend:
        device = resolve_device(config.device)
        dtype = resolve_dtype(config.dtype, device)
        attention = resolve_attention(config.attention, device)
        quantization = _quantization_config(config.quantization, dtype)
        load_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": attention,
            "low_cpu_mem_usage": True,
            **backend_kwargs,
        }
        if quantization is not None:
            load_kwargs["quantization_config"] = quantization
            load_kwargs["device_map"] = device
        elif device != "cpu":
            load_kwargs["device_map"] = device

        model = require_qwen_asr().Qwen3ASRModel.from_pretrained(
            model_name_or_path,
            max_inference_batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
            **load_kwargs,
        )
        if device == "cpu":
            model.model.to(device)
        model.model.eval().requires_grad_(False)
        model.model.thinker.config.text_config.use_cache = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        if config.compile_model:
            model.model.thinker.model = torch.compile(
                model.model.thinker.model,
                dynamic=True,
                mode="reduce-overhead",
            )
        return cls(model)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: AudioBatch,
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[Transcription]:
        results = self.model.transcribe(
            audio=audio,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )
        return coerce_transcriptions(results)

    def close(self) -> None:
        return None


class NativeTransformersBackend:
    """Native Transformers 5 implementation for ``*-hf`` checkpoints."""

    def __init__(self, model: Any, processor: Any, max_new_tokens: int) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: InferenceConfig,
        **backend_kwargs: Any,
    ) -> NativeTransformersBackend:
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise OptionalDependencyError(
                "native Qwen3-ASR requires `pip install 'distil-qwen[transformers-native]'`"
            ) from exc

        device = resolve_device(config.device)
        dtype = resolve_dtype(config.dtype, device)
        attention = resolve_attention(config.attention, device)
        quantization = _quantization_config(config.quantization, dtype)
        processor_kwargs = backend_kwargs.pop("processor_kwargs", {})
        load_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": attention,
            "low_cpu_mem_usage": True,
            **backend_kwargs,
        }
        if quantization is not None:
            load_kwargs["quantization_config"] = quantization
            load_kwargs["device_map"] = device
        elif device != "cpu":
            load_kwargs["device_map"] = device

        processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)
        model = AutoModelForMultimodalLM.from_pretrained(model_name_or_path, **load_kwargs)
        if device == "cpu":
            model.to(device)
        model.eval().requires_grad_(False)
        model.config.use_cache = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        if config.compile_model:
            model = torch.compile(model, dynamic=True, mode="reduce-overhead")
        return cls(model=model, processor=processor, max_new_tokens=config.max_new_tokens)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: AudioBatch,
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[Transcription]:
        if return_time_stamps:
            raise BackendFeatureError(
                "native Transformers timestamps require a separate forced-aligner pipeline"
            )
        inputs = self.processor.apply_transcription_request(
            audio=audio,
            prompt=context,
            language=language,
        ).to(self.model.device, self.model.dtype)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = self.processor.decode(generated_ids, return_format="parsed")
        return [
            Transcription(
                language=str(result.get("language") or "auto"),
                text=str(result.get("transcription") or ""),
            )
            for result in parsed
        ]

    def close(self) -> None:
        return None
