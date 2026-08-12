from types import SimpleNamespace

import pytest
import torch

from distil_qwen.backends.base import Transcription
from distil_qwen.backends.llama_cpp import create_llama_cpp_backend
from distil_qwen.backends.sglang import create_sglang_backend
from distil_qwen.backends.transformers import (
    NativeTransformersBackend,
    QwenTransformersBackend,
    TransformersBackend,
    _is_native_checkpoint,
)
from distil_qwen.backends.vllm import VLLMBackend, _LocalVLLM
from distil_qwen.config import InferenceConfig
from distil_qwen.errors import BackendConfigurationError
from distil_qwen.inference import OptimizedASR


class NoopClient:
    def close(self):
        return None


class Delegate:
    def __init__(self):
        self.closed = False

    def transcribe(self, **kwargs):
        return [Transcription("en", str(kwargs["audio"]))]

    def close(self):
        self.closed = True


def test_transformers_facade_delegates_and_validates_implementation(monkeypatch) -> None:
    delegate = Delegate()
    monkeypatch.setattr(
        "distil_qwen.backends.transformers.QwenTransformersBackend.from_pretrained",
        lambda *args, **kwargs: delegate,
    )
    backend = TransformersBackend.from_pretrained("student", InferenceConfig())
    assert backend.transcribe("a.wav")[0].text == "a.wav"
    backend.close()
    assert delegate.closed is True
    with pytest.raises(BackendConfigurationError, match="implementation"):
        TransformersBackend.from_pretrained("student", InferenceConfig(), implementation="invalid")


def test_native_checkpoint_detection_uses_name_or_config(tmp_path) -> None:
    assert _is_native_checkpoint("Qwen/Qwen3-ASR-1.7B-hf") is True
    (tmp_path / "config.json").write_text('{"transformers_version":"5.13.0"}')
    assert _is_native_checkpoint(str(tmp_path)) is True
    (tmp_path / "config.json").write_text("invalid")
    assert _is_native_checkpoint(str(tmp_path)) is False


def test_server_factories_validate_urls_and_unknown_arguments() -> None:
    with pytest.raises(BackendConfigurationError, match="server_url"):
        create_sglang_backend("student", InferenceConfig(backend="sglang"))
    with pytest.raises(BackendConfigurationError, match="server_url"):
        create_llama_cpp_backend("student", InferenceConfig(backend="llama_cpp"))

    sglang_config = InferenceConfig(backend="sglang", server_url="http://localhost:30000")
    backend = create_sglang_backend("student", sglang_config, http_client=NoopClient())
    assert backend.backend_name == "SGLang"
    with pytest.raises(BackendConfigurationError, match="unsupported SGLang"):
        create_sglang_backend("student", sglang_config, unsupported=True)

    llama_config = InferenceConfig(backend="llama_cpp", server_url="http://localhost:8080")
    backend = create_llama_cpp_backend(
        "student", llama_config, http_client=NoopClient(), validate_server=False
    )
    assert backend.model == "student"
    with pytest.raises(BackendConfigurationError, match="unsupported llama.cpp"):
        create_llama_cpp_backend(
            "student", llama_config, http_client=NoopClient(), unsupported=True
        )


def test_remote_factory_discovers_single_advertised_model() -> None:
    client = SimpleNamespace(
        get=lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"id": "served-student"}]},
        ),
        close=lambda: None,
    )
    config = InferenceConfig(backend="vllm", server_url="http://localhost:8000")
    backend = VLLMBackend.from_pretrained(None, config, http_client=client)
    assert backend.implementation.model == "served-student"


def test_vllm_remote_factory_and_unknown_arguments() -> None:
    config = InferenceConfig(backend="vllm", server_url="http://localhost:8000")
    backend = VLLMBackend.from_pretrained("student", config, http_client=NoopClient())
    assert backend.implementation.backend_name == "vLLM"
    with pytest.raises(BackendConfigurationError, match="unsupported remote vLLM"):
        VLLMBackend.from_pretrained("student", config, invalid=True)


def test_local_vllm_and_optimized_asr_context_manager() -> None:
    model = SimpleNamespace(
        transcribe=lambda **kwargs: [SimpleNamespace(language="en", text="hello")]
    )
    local = _LocalVLLM(model)
    assert local.transcribe("a.wav")[0].text == "hello"
    local.close()

    delegate = Delegate()
    with OptimizedASR(delegate, InferenceConfig()) as asr:
        assert asr.transcribe("inside")[0].text == "inside"
    assert delegate.closed is True


def test_qwen_transformers_result_conversion() -> None:
    model = SimpleNamespace(
        transcribe=lambda **kwargs: [SimpleNamespace(language="English", text="hello")]
    )
    backend = QwenTransformersBackend(model)
    assert backend.transcribe("a.wav")[0] == Transcription("English", "hello")
    backend.close()


class FakeBatch(dict):
    def to(self, *args):
        self.to_args = args
        return self


def test_native_transformers_generation_and_timestamp_contract() -> None:
    inputs = FakeBatch(input_ids=torch.tensor([[1, 2]]))
    processor = SimpleNamespace(
        apply_transcription_request=lambda **kwargs: inputs,
        decode=lambda *args, **kwargs: [{"language": "Turkish", "transcription": "merhaba"}],
    )
    model = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        generate=lambda **kwargs: torch.tensor([[1, 2, 3]]),
    )
    backend = NativeTransformersBackend(model, processor, max_new_tokens=32)
    assert backend.transcribe("a.wav")[0].text == "merhaba"
    with pytest.raises(NotImplementedError, match="forced-aligner"):
        backend.transcribe("a.wav", return_time_stamps=True)
    backend.close()


def test_optimized_asr_factory_uses_selected_backend(monkeypatch) -> None:
    delegate = Delegate()
    monkeypatch.setattr(
        TransformersBackend,
        "from_pretrained",
        lambda *args, **kwargs: delegate,
    )
    asr = OptimizedASR.from_pretrained("student", InferenceConfig())
    assert asr.backend_model is delegate
