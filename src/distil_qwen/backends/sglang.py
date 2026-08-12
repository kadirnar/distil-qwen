"""SGLang's OpenAI-compatible Qwen3-ASR transcription backend."""

from __future__ import annotations

from typing import Any, Optional

from distil_qwen.backends.server import OpenAITranscriptionBackend, ServerClient
from distil_qwen.config import InferenceConfig
from distil_qwen.errors import BackendConfigurationError


def create_sglang_backend(
    model_name_or_path: Optional[str],
    config: InferenceConfig,
    **backend_kwargs: Any,
) -> OpenAITranscriptionBackend:
    """Build a client without importing SGLang into the application process."""

    if not config.server_url:
        raise BackendConfigurationError(
            "SGLang requires server_url; launch `python -m sglang.launch_server` first"
        )
    http_client = backend_kwargs.pop("http_client", None)
    if backend_kwargs:
        unknown = ", ".join(sorted(backend_kwargs))
        raise BackendConfigurationError(f"unsupported SGLang backend arguments: {unknown}")
    server = ServerClient(
        config.server_url,
        config.request_timeout,
        config.api_key_env,
        client=http_client,
    )
    return OpenAITranscriptionBackend(
        model=model_name_or_path,
        server=server,
        max_concurrency=config.batch_size,
        backend_name="SGLang",
        # Upstream's current Qwen3-ASR adapter treats a non-empty prompt as a
        # complete model prompt and returns empty timestamp segments.
        supports_context=False,
        supports_language=False,
        supports_timestamps=False,
    )
