"""Optimized Transformers and vLLM inference behind one small API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from distil_qwen.compat import flash_attention_available, require_qwen_asr
from distil_qwen.config import InferenceConfig
from distil_qwen.errors import OptionalDependencyError


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


@dataclass(frozen=True)
class Transcription:
    language: str
    text: str
    time_stamps: Optional[Any] = None


class OptimizedASR:
    """A stable library wrapper over the official Qwen3-ASR backends."""

    def __init__(self, backend_model: Any, config: InferenceConfig) -> None:
        self.backend_model = backend_model
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[InferenceConfig] = None,
        **backend_kwargs: Any,
    ) -> OptimizedASR:
        config = config or InferenceConfig()
        if config.backend == "vllm":
            qwen_model_class = require_qwen_asr().Qwen3ASRModel
            dtype_name = "bfloat16" if config.dtype == "auto" else config.dtype
            model = qwen_model_class.LLM(
                model=model_name_or_path,
                dtype=dtype_name,
                max_inference_batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens,
                **backend_kwargs,
            )
            return cls(model, config)

        device = resolve_device(config.device)
        dtype = resolve_dtype(config.dtype, device)
        attention = resolve_attention(config.attention, device)
        quantization = _quantization_config(config.quantization, dtype)
        qwen_model_class = require_qwen_asr().Qwen3ASRModel
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

        model = qwen_model_class.from_pretrained(
            model_name_or_path,
            max_inference_batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
            **load_kwargs,
        )
        if device == "cpu":
            model.model.to(device)
        model.model.eval().requires_grad_(False)
        text_config = model.model.thinker.config.text_config
        text_config.use_cache = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        if config.compile_model:
            model.model.thinker.model = torch.compile(
                model.model.thinker.model,
                dynamic=True,
                mode="reduce-overhead",
            )
        return cls(model, config)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[Any, Sequence[Any]],
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
        return [
            Transcription(
                language=result.language,
                text=result.text,
                time_stamps=result.time_stamps,
            )
            for result in results
        ]
