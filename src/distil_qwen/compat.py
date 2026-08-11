"""Small compatibility layer around optional Qwen dependencies."""

from __future__ import annotations

import importlib.util
from typing import Any

from distil_qwen.errors import OptionalDependencyError


def require_qwen_asr() -> Any:
    """Import qwen-asr so its Transformers auto classes are registered."""

    try:
        import qwen_asr
    except ImportError as exc:
        raise OptionalDependencyError(
            "Qwen support requires `pip install 'distil-qwen[qwen]'`."
        ) from exc
    return qwen_asr


def flash_attention_available() -> bool:
    """Return whether FlashAttention 2 can be selected safely."""

    return importlib.util.find_spec("flash_attn") is not None
