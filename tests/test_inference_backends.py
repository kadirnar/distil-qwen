import base64
import json
from pathlib import Path

import numpy as np
import pytest

from distil_qwen.backends.base import (
    Transcription,
    coerce_transcriptions,
    normalize_audio_batch,
    normalize_requests,
    parse_qwen_output,
)
from distil_qwen.backends.server import (
    LlamaCppBackend,
    OpenAITranscriptionBackend,
    ServerClient,
    _endpoint,
    _wav_bytes,
)
from distil_qwen.errors import BackendConfigurationError, BackendFeatureError, BackendRequestError


class FakeResponse:
    def __init__(self, payload=None, status_code=200, content=b"", headers=None):
        self.payload = payload
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")


class FakeClient:
    def __init__(self, posts=None, gets=None):
        self.posts = list(posts or [])
        self.gets = list(gets or [])
        self.requests = []
        self.closed = False

    def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        return self.posts.pop(0)

    def get(self, url, **kwargs):
        self.requests.append(("GET", url, kwargs))
        return self.gets.pop(0)

    def close(self):
        self.closed = True


class EchoAudioClient(FakeClient):
    def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        text = kwargs["files"]["file"][1].decode()
        return FakeResponse({"text": text, "segments": [{"start": 0.0}]})


def test_normalize_requests_broadcasts_scalar_metadata() -> None:
    audio, contexts, languages = normalize_requests(["a.wav", "b.wav"], "names", "en")
    assert audio == ["a.wav", "b.wav"]
    assert contexts == ["names", "names"]
    assert languages == ["en", "en"]
    with pytest.raises(ValueError, match="context"):
        normalize_requests(["a.wav", "b.wav"], ["only one"])


def test_audio_batch_and_result_normalization() -> None:
    sample = (np.zeros(4), 16_000)
    assert normalize_audio_batch(sample) == [sample]
    with pytest.raises(ValueError, match="at least one"):
        normalize_audio_batch([])
    results = coerce_transcriptions(
        [
            Transcription("en", "one"),
            {"language": "tr", "transcription": "iki", "segments": [1]},
            SimpleResult(),
        ]
    )
    assert [(item.language, item.text) for item in results] == [
        ("en", "one"),
        ("tr", "iki"),
        ("auto", "three"),
    ]


class SimpleResult:
    text = "three"


def test_parse_qwen_output_handles_raw_and_prefilled_responses() -> None:
    parsed = parse_qwen_output("language English<asr_text>Hello")
    assert parsed.language == "English"
    assert parsed.text == "Hello"
    assert parse_qwen_output("Merhaba", "Turkish").language == "Turkish"


def test_openai_transcription_backend_posts_multipart_audio() -> None:
    client = FakeClient(posts=[FakeResponse({"text": "hello", "language": "en"})])
    backend = OpenAITranscriptionBackend(
        "student",
        ServerClient("http://localhost:8000/v1", 30, "KEY", client=client),
        max_concurrency=1,
        backend_name="vLLM",
    )
    result = backend.transcribe(b"RIFF", context="Qwen", language="en")[0]
    assert result.text == "hello"
    _, url, kwargs = client.requests[0]
    assert url == "http://localhost:8000/v1/audio/transcriptions"
    assert kwargs["data"] == {"model": "student", "prompt": "Qwen", "language": "en"}
    assert kwargs["files"]["file"][1] == b"RIFF"


def test_openai_backend_verbose_response_and_parallel_ordering() -> None:
    client = EchoAudioClient()
    backend = OpenAITranscriptionBackend(
        "student",
        ServerClient("http://localhost:8000", 30, "KEY", client=client),
        max_concurrency=2,
        backend_name="vLLM",
    )
    results = backend.transcribe([b"one", b"two"], return_time_stamps=True)
    assert [result.text for result in results] == ["one", "two"]
    assert results[0].time_stamps == [{"start": 0.0}]
    assert all(
        request[2]["data"]["response_format"] == "verbose_json" for request in client.requests
    )


def test_sglang_capability_errors_are_explicit() -> None:
    backend = OpenAITranscriptionBackend(
        "student",
        ServerClient("http://localhost:30000", 30, "KEY", client=FakeClient()),
        max_concurrency=1,
        backend_name="SGLang",
        supports_context=False,
        supports_language=False,
        supports_timestamps=False,
    )
    with pytest.raises(BackendFeatureError, match="context"):
        backend.transcribe(b"audio", context="hotword")
    with pytest.raises(BackendFeatureError, match="timestamps"):
        backend.transcribe(b"audio", return_time_stamps=True)
    with pytest.raises(BackendFeatureError, match="language"):
        backend.transcribe(b"audio", language="en")


def test_llama_cpp_validates_audio_and_builds_multimodal_request() -> None:
    client = FakeClient(
        gets=[FakeResponse({"modalities": {"audio": True}})],
        posts=[FakeResponse({"choices": [{"message": {"content": "Merhaba"}}]})],
    )
    backend = LlamaCppBackend(
        "student",
        ServerClient("http://localhost:8080", 30, "KEY", client=client),
        max_new_tokens=64,
        max_concurrency=1,
    )
    result = backend.transcribe((np.zeros(160, dtype=np.float32), 16_000), language="Turkish")[0]
    assert result.language == "Turkish"
    method, url, kwargs = client.requests[1]
    assert method == "POST"
    assert url.endswith("/v1/chat/completions")
    assert kwargs["json"]["continue_final_message"] is True
    encoded = kwargs["json"]["messages"][0]["content"][0]["input_audio"]["data"]
    assert base64.b64decode(encoded).startswith(b"RIFF")


def test_llama_cpp_rejects_server_without_audio_projector() -> None:
    client = FakeClient(gets=[FakeResponse({"modalities": {"audio": False}})])
    with pytest.raises(BackendConfigurationError, match="audio projector"):
        LlamaCppBackend(
            "student",
            ServerClient("http://localhost:8080", 30, "KEY", client=client),
            max_new_tokens=64,
            max_concurrency=1,
        )


def test_llama_cpp_parses_content_parts_and_rejects_timestamps() -> None:
    client = FakeClient(
        posts=[
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"text": "language English<asr_text>"},
                                    {"text": "hello"},
                                ]
                            }
                        }
                    ]
                }
            )
        ]
    )
    backend = LlamaCppBackend(
        "student",
        ServerClient("http://localhost:8080/v1", 30, "KEY", client=client),
        max_new_tokens=64,
        max_concurrency=1,
        validate_server=False,
    )
    assert backend.transcribe(b"audio")[0].text == "hello"
    with pytest.raises(BackendFeatureError, match="timestamps"):
        backend.transcribe(b"audio", return_time_stamps=True)


def test_llama_cpp_rejects_malformed_response() -> None:
    backend = LlamaCppBackend(
        "student",
        ServerClient(
            "http://localhost:8080", 30, "KEY", client=FakeClient(posts=[FakeResponse({})])
        ),
        max_new_tokens=64,
        max_concurrency=1,
        validate_server=False,
    )
    with pytest.raises(BackendRequestError, match="assistant content"):
        backend.transcribe(b"audio")


def test_server_error_does_not_silently_return_empty_text() -> None:
    client = FakeClient(posts=[FakeResponse({"error": "bad"}, status_code=500)])
    backend = OpenAITranscriptionBackend(
        "student",
        ServerClient("http://localhost:8000", 30, "KEY", client=client),
        max_concurrency=1,
        backend_name="vLLM",
    )
    with pytest.raises(BackendRequestError, match="500"):
        backend.transcribe(b"audio")


def test_server_rejects_non_json_response() -> None:
    response = FakeResponse()
    response.json = lambda: (_ for _ in ()).throw(ValueError("bad json"))
    backend = OpenAITranscriptionBackend(
        "student",
        ServerClient("http://localhost:8000", 30, "KEY", client=FakeClient(posts=[response])),
        max_concurrency=1,
        backend_name="vLLM",
    )
    with pytest.raises(BackendRequestError, match="non-JSON"):
        backend.transcribe(b"audio")


def test_server_client_reads_paths_urls_and_sample_arrays(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.flac"
    audio_path.write_bytes(b"fLaC")
    client = FakeClient(
        gets=[FakeResponse(content=b"remote", headers={"content-type": "audio/mpeg"})]
    )
    server = ServerClient("http://localhost:8000", 30, "KEY", client=client)
    filename, payload, mime = server.audio_bytes(audio_path)
    assert (filename, payload) == ("sample.flac", b"fLaC")
    assert mime in {"audio/flac", "audio/x-flac"}
    assert server.audio_bytes("https://example.com/audio.mp3") == (
        "audio.mp3",
        b"remote",
        "audio/mpeg",
    )
    assert server.audio_bytes((np.zeros((2, 16)), 16_000))[1].startswith(b"RIFF")
    with pytest.raises(FileNotFoundError):
        server.audio_bytes(tmp_path / "missing.wav")
    with pytest.raises(TypeError):
        server.audio_bytes(object())


def test_server_model_discovery() -> None:
    client = FakeClient(gets=[FakeResponse({"data": [{"id": "student"}]})])
    server = ServerClient("http://localhost:8000/v1/", 30, "KEY", client=client)
    assert server.base_url == "http://localhost:8000/v1"
    assert server.model_ids() == ["student"]


def test_endpoint_and_wav_validation() -> None:
    assert _endpoint("http://localhost:8000", "/v1/models") == ("http://localhost:8000/v1/models")
    assert _endpoint("http://localhost:8000/v1", "/props") == "http://localhost:8000/props"
    assert _wav_bytes(np.array([0, 1], dtype=np.int32), 16_000).startswith(b"RIFF")
    with pytest.raises(ValueError, match="mono"):
        _wav_bytes(np.zeros((2, 2, 2)), 16_000)
