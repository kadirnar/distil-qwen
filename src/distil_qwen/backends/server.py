"""OpenAI-compatible server clients for vLLM, SGLang, and llama.cpp."""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np

from distil_qwen.backends.base import (
    AudioBatch,
    AudioInput,
    Transcription,
    is_array_sample_rate_pair,
    normalize_requests,
    parse_qwen_output,
)
from distil_qwen.compat import require_httpx
from distil_qwen.errors import (
    BackendConfigurationError,
    BackendFeatureError,
    BackendRequestError,
)


def _endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        if path.startswith("/v1/"):
            return base + path[3:]
        return base[:-3] + path
    return base + path


def _wav_bytes(samples: Any, sample_rate: int) -> bytes:
    array = np.asarray(samples)
    if array.ndim not in {1, 2}:
        raise ValueError("audio samples must be mono or shaped as (samples, channels)")
    if array.ndim == 2:
        if array.shape[0] <= 8 < array.shape[1]:
            array = array.T
        if array.shape[1] > 8:
            raise ValueError("two-dimensional audio must be shaped as (samples, channels)")
    if np.issubdtype(array.dtype, np.floating):
        pcm = (np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")
    elif array.dtype == np.int16:
        pcm = array.astype("<i2", copy=False)
    elif np.issubdtype(array.dtype, np.integer):
        limit = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        pcm = (array.astype(np.float64) / limit * 32767.0).astype("<i2")
    else:
        raise ValueError(f"unsupported audio sample dtype: {array.dtype}")

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def _response_json(response: Any, backend: str) -> Dict[str, Any]:
    try:
        response.raise_for_status()
    except Exception as exc:
        status = getattr(response, "status_code", "unknown")
        body = str(getattr(response, "text", ""))[:500]
        raise BackendRequestError(f"{backend} request failed ({status}): {body}") from exc
    try:
        payload = response.json()
    except Exception as exc:
        raise BackendRequestError(f"{backend} returned a non-JSON response") from exc
    if not isinstance(payload, dict):
        raise BackendRequestError(f"{backend} returned an invalid response object")
    return payload


class ServerClient:
    """Own an authenticated HTTP client and normalize supported audio inputs."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        api_key_env: str,
        client: Optional[Any] = None,
    ) -> None:
        if not base_url:
            raise BackendConfigurationError("server_url is required for server inference")
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        if client is None:
            headers = {}
            api_key = os.getenv(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            client = require_httpx().Client(
                headers=headers,
                timeout=timeout,
                follow_redirects=True,
            )
        self.client = client

    def audio_bytes(self, audio: AudioInput) -> Tuple[str, bytes, str]:
        if isinstance(audio, bytes):
            return "audio.wav", audio, "audio/wav"
        if is_array_sample_rate_pair(audio):
            samples, sample_rate = audio
            return "audio.wav", _wav_bytes(samples, sample_rate), "audio/wav"
        if isinstance(audio, (str, os.PathLike)):
            value = os.fspath(audio)
            parsed = urlparse(value)
            if parsed.scheme in {"http", "https"}:
                response = self.client.get(value)
                payload = _response_bytes(response, "audio download")
                filename = Path(parsed.path).name or "audio.wav"
                content_type = response.headers.get("content-type", "").split(";", 1)[0]
            else:
                path = Path(value).expanduser()
                if not path.is_file():
                    raise FileNotFoundError(f"audio file does not exist: {path}")
                payload = path.read_bytes()
                filename = path.name
                content_type = ""
            mime = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
            return filename, payload, mime
        raise TypeError("server backends accept paths, URLs, bytes, or (samples, sample_rate)")

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def model_ids(self) -> List[str]:
        """Return the model identifiers advertised by the OpenAI-compatible server."""

        response = self.client.get(_endpoint(self.base_url, "/v1/models"))
        payload = _response_json(response, "server model discovery")
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise BackendRequestError("server model discovery returned invalid data")
        return [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]


def _response_bytes(response: Any, source: str) -> bytes:
    try:
        response.raise_for_status()
    except Exception as exc:
        status = getattr(response, "status_code", "unknown")
        raise BackendRequestError(f"{source} failed with HTTP {status}") from exc
    return bytes(response.content)


class OpenAITranscriptionBackend:
    """Client for the OpenAI ``/audio/transcriptions`` endpoint."""

    def __init__(
        self,
        model: Optional[str],
        server: ServerClient,
        max_concurrency: int,
        backend_name: str,
        supports_context: bool = True,
        supports_language: bool = True,
        supports_timestamps: bool = True,
    ) -> None:
        self.server = server
        self.model = model or self._discover_model()
        self.max_concurrency = max_concurrency
        self.backend_name = backend_name
        self.supports_context = supports_context
        self.supports_language = supports_language
        self.supports_timestamps = supports_timestamps

    def _discover_model(self) -> str:
        models = self.server.model_ids()
        if len(models) != 1:
            raise BackendConfigurationError(
                "model must be provided when the inference server does not "
                "advertise exactly one model"
            )
        return models[0]

    def _transcribe_one(
        self,
        request: Tuple[AudioInput, str, Optional[str], bool],
    ) -> Transcription:
        audio, context, language, return_time_stamps = request
        if context and not self.supports_context:
            raise BackendFeatureError(f"{self.backend_name} does not support ASR context prompts")
        if language and not self.supports_language:
            raise BackendFeatureError(f"{self.backend_name} does not support forced ASR language")
        if return_time_stamps and not self.supports_timestamps:
            raise BackendFeatureError(f"{self.backend_name} does not support ASR timestamps")
        filename, payload, mime = self.server.audio_bytes(audio)
        data = {"model": self.model}
        if context:
            data["prompt"] = context
        if language:
            data["language"] = language
        if return_time_stamps:
            data["response_format"] = "verbose_json"
        response = self.server.client.post(
            _endpoint(self.server.base_url, "/v1/audio/transcriptions"),
            files={"file": (filename, payload, mime)},
            data=data,
        )
        result = _response_json(response, self.backend_name)
        return Transcription(
            language=str(result.get("language") or language or "auto"),
            text=str(result.get("text") or ""),
            time_stamps=result.get("segments") if return_time_stamps else None,
        )

    def transcribe(
        self,
        audio: AudioBatch,
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[Transcription]:
        audio_items, contexts, languages = normalize_requests(audio, context, language)
        requests = [
            (item, item_context, item_language, return_time_stamps)
            for item, item_context, item_language in zip(audio_items, contexts, languages)
        ]
        workers = min(self.max_concurrency, len(requests))
        if workers == 1:
            return [self._transcribe_one(requests[0])]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._transcribe_one, requests))

    def close(self) -> None:
        self.server.close()


class LlamaCppBackend:
    """Qwen3-ASR client for llama.cpp's multimodal chat endpoint."""

    def __init__(
        self,
        model: Optional[str],
        server: ServerClient,
        max_new_tokens: int,
        max_concurrency: int,
        validate_server: bool = True,
    ) -> None:
        self.server = server
        self.model = model or self._discover_model()
        self.max_new_tokens = max_new_tokens
        self.max_concurrency = max_concurrency
        if validate_server:
            self._validate_server()

    def _discover_model(self) -> str:
        models = self.server.model_ids()
        if len(models) != 1:
            raise BackendConfigurationError(
                "model must be provided when llama.cpp does not advertise exactly one model"
            )
        return models[0]

    def _validate_server(self) -> None:
        response = self.server.client.get(_endpoint(self.server.base_url, "/props"))
        props = _response_json(response, "llama.cpp capability check")
        modalities = props.get("modalities")
        if not isinstance(modalities, dict) or not modalities.get("audio"):
            raise BackendConfigurationError(
                "llama.cpp server has no audio projector; launch Qwen3-ASR GGUF with MTMD enabled"
            )

    def _transcribe_one(
        self,
        request: Tuple[AudioInput, str, Optional[str]],
    ) -> Transcription:
        audio, context, language = request
        filename, audio_bytes, _ = self.server.audio_bytes(audio)
        audio_format = Path(filename).suffix.lower().lstrip(".") or "wav"
        messages: List[Dict[str, Any]] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                            "format": audio_format,
                        },
                    }
                ],
            }
        )
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_new_tokens,
            "stream": False,
        }
        if language:
            messages.append({"role": "assistant", "content": f"language {language}<asr_text>"})
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False

        response = self.server.client.post(
            _endpoint(self.server.base_url, "/v1/chat/completions"),
            json=body,
        )
        result = _response_json(response, "llama.cpp")
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendRequestError("llama.cpp response has no assistant content") from exc
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        return parse_qwen_output(str(content), fallback_language=language)

    def transcribe(
        self,
        audio: AudioBatch,
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[Transcription]:
        if return_time_stamps:
            raise BackendFeatureError(
                "llama.cpp Qwen3-ASR does not expose forced-alignment timestamps"
            )
        audio_items, contexts, languages = normalize_requests(audio, context, language)
        requests = list(zip(audio_items, contexts, languages))
        workers = min(self.max_concurrency, len(requests))
        if workers == 1:
            return [self._transcribe_one(requests[0])]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._transcribe_one, requests))

    def close(self) -> None:
        self.server.close()
