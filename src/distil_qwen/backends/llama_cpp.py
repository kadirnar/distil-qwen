"""Factory for llama.cpp's Qwen3-ASR GGUF server backend."""

from __future__ import annotations

from typing import Any, Optional

from distil_qwen.backends.server import LlamaCppBackend, ServerClient
from distil_qwen.config import InferenceConfig
from distil_qwen.errors import BackendConfigurationError


def create_llama_cpp_backend(
    model_name_or_path: Optional[str],
    config: InferenceConfig,
    **backend_kwargs: Any,
) -> LlamaCppBackend:
    if not config.server_url:
        raise BackendConfigurationError(
            "llama.cpp requires server_url; launch `llama-server` with a Qwen3-ASR GGUF first"
        )
    http_client = backend_kwargs.pop("http_client", None)
    validate_server = backend_kwargs.pop("validate_server", True)
    if backend_kwargs:
        unknown = ", ".join(sorted(backend_kwargs))
        raise BackendConfigurationError(f"unsupported llama.cpp backend arguments: {unknown}")
    server = ServerClient(
        config.server_url,
        config.request_timeout,
        config.api_key_env,
        client=http_client,
    )
    return LlamaCppBackend(
        model=model_name_or_path,
        server=server,
        max_new_tokens=config.max_new_tokens,
        max_concurrency=config.batch_size,
        validate_server=bool(validate_server),
    )
