"""In-process and server-backed vLLM inference."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from distil_qwen.backends.base import AudioBatch, Transcription, coerce_transcriptions
from distil_qwen.backends.server import OpenAITranscriptionBackend, ServerClient
from distil_qwen.compat import require_qwen_asr
from distil_qwen.config import InferenceConfig
from distil_qwen.errors import BackendConfigurationError


class VLLMBackend:
    """Use the official local engine or an OpenAI-compatible vLLM server."""

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Optional[str],
        config: InferenceConfig,
        **backend_kwargs: Any,
    ) -> VLLMBackend:
        if config.server_url:
            http_client = backend_kwargs.pop("http_client", None)
            if backend_kwargs:
                unknown = ", ".join(sorted(backend_kwargs))
                raise BackendConfigurationError(
                    f"unsupported remote vLLM backend arguments: {unknown}"
                )
            server = ServerClient(
                config.server_url,
                config.request_timeout,
                config.api_key_env,
                client=http_client,
            )
            return cls(
                OpenAITranscriptionBackend(
                    model=model_name_or_path,
                    server=server,
                    max_concurrency=config.batch_size,
                    backend_name="vLLM",
                )
            )

        if not model_name_or_path:
            raise BackendConfigurationError("model_name_or_path is required for in-process vLLM")

        dtype = "auto" if config.dtype == "auto" else config.dtype
        model = require_qwen_asr().Qwen3ASRModel.LLM(
            model=model_name_or_path,
            dtype=dtype,
            max_inference_batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
            **backend_kwargs,
        )
        return cls(_LocalVLLM(model))

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


class _LocalVLLM:
    def __init__(self, model: Any) -> None:
        self.model = model

    def transcribe(
        self,
        audio: AudioBatch,
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[Transcription]:
        return coerce_transcriptions(
            self.model.transcribe(
                audio=audio,
                context=context,
                language=language,
                return_time_stamps=return_time_stamps,
            )
        )

    def close(self) -> None:
        return None
