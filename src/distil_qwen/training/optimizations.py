"""Hardware-aware training optimizations for the custom Qwen3-ASR architecture."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable

import torch

from distil_qwen.errors import OptionalDependencyError
from distil_qwen.models.accessors import get_text_model

_LIGER_MODES = {"auto", "on", "off"}
_OPTIMIZERS = {
    "auto",
    "adamw_torch",
    "adamw_torch_fused",
    "adamw_bnb_8bit",
    "paged_adamw_8bit",
}
_RMS_NORM_NAMES = {"Qwen3ASRTextRMSNorm", "Qwen3ASRThinkerTextRMSNorm"}
_SWIGLU_NAMES = {"Qwen3ASRTextMLP", "Qwen3ASRThinkerTextMLP"}


@dataclass(frozen=True)
class LigerPatchReport:
    rms_norms: int
    swiglu_mlps: int
    rope: bool


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_liger(mode: str) -> bool:
    """Resolve an auto/on/off request without silently accepting an unavailable kernel."""

    if mode not in _LIGER_MODES:
        raise ValueError("liger must be 'auto', 'on', or 'off'")
    available = package_available("liger_kernel") and torch.cuda.is_available()
    if mode == "on" and not available:
        raise OptionalDependencyError(
            "Liger requires a supported accelerator and `pip install 'distil-qwen[kernels]'`."
        )
    return available if mode == "auto" else mode == "on"


def resolve_optimizer(requested: str) -> str:
    if requested not in _OPTIMIZERS:
        raise ValueError(f"unsupported optimizer: {requested}")
    if requested == "auto":
        return "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    if "8bit" in requested and not package_available("bitsandbytes"):
        raise OptionalDependencyError(
            "8-bit optimizers require `pip install 'distil-qwen[quantization]'`."
        )
    if requested == "adamw_torch_fused" and not torch.cuda.is_available():
        raise ValueError("adamw_torch_fused requires CUDA")
    return requested


def flash_attention_supported(dtype: torch.dtype) -> bool:
    """Return whether the installed runtime satisfies FlashAttention 2's training contract."""

    if not torch.cuda.is_available() or not package_available("flash_attn"):
        return False
    if dtype not in {torch.float16, torch.bfloat16}:
        return False
    if torch.version.hip is not None:
        return True
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def resolve_attention(requested: str, dtype: torch.dtype) -> str:
    if requested not in {"auto", "eager", "sdpa", "flash_attention_2"}:
        raise ValueError(f"unsupported attention implementation: {requested}")
    if requested == "auto":
        return "flash_attention_2" if flash_attention_supported(dtype) else "sdpa"
    if requested == "flash_attention_2" and not flash_attention_supported(dtype):
        raise OptionalDependencyError(
            "FlashAttention 2 training requires fp16/bf16, a supported GPU, and `flash-attn`."
        )
    return requested


def _load_liger_components() -> tuple[type[Any], type[Any], Callable[..., Any]]:
    from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    return LigerRMSNorm, LigerSwiGLUMLP, liger_rotary_pos_emb


def apply_liger_kernels(model: torch.nn.Module) -> LigerPatchReport:
    """Patch Qwen3-ASR instances directly because upstream has no ``qwen3_asr`` adapter."""

    try:
        liger_rms_norm, liger_swiglu, liger_rope = _load_liger_components()
    except ImportError as exc:
        raise OptionalDependencyError(
            "Liger requires `pip install 'distil-qwen[kernels]'`."
        ) from exc

    rms_norms = 0
    swiglu_mlps = 0
    for module in model.modules():
        module_name = type(module).__name__
        if module_name in _RMS_NORM_NAMES:
            module.offset = 0.0
            module.casting_mode = "llama"
            module.in_place = True
            module.row_mode = None
            module.forward = MethodType(liger_rms_norm.forward, module)
            rms_norms += 1
        elif module_name in _SWIGLU_NAMES:
            module.forward = MethodType(liger_swiglu.forward, module)
            swiglu_mlps += 1

    text_model = get_text_model(model)
    modeling = importlib.import_module(type(text_model).__module__)
    rope_patched = hasattr(modeling, "apply_rotary_pos_emb")
    if rope_patched:
        modeling.apply_rotary_pos_emb = liger_rope
    if not rms_norms or not swiglu_mlps:
        raise RuntimeError(
            "Liger found no Qwen3-ASR text kernels to patch; check qwen-asr compatibility"
        )
    return LigerPatchReport(rms_norms=rms_norms, swiglu_mlps=swiglu_mlps, rope=rope_patched)
