from types import SimpleNamespace

import pytest
import torch

from distil_qwen.config import InferenceConfig
from distil_qwen.inference import (
    OptimizedASR,
    resolve_attention,
    resolve_device,
    resolve_dtype,
)


class FakeBackend:
    def transcribe(self, **kwargs):
        assert kwargs["audio"] == "sample.wav"
        return [SimpleNamespace(language="English", text="hello", time_stamps=None)]


def test_optimized_asr_converts_backend_results() -> None:
    model = OptimizedASR(FakeBackend(), InferenceConfig())
    result = model.transcribe("sample.wav")[0]
    assert result.language == "English"
    assert result.text == "hello"


def test_runtime_resolution_on_explicit_values() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_dtype("bfloat16", "cpu") == torch.bfloat16
    assert resolve_dtype("auto", "cpu") == torch.float32
    assert resolve_attention("eager", "cpu") == "eager"
    assert resolve_attention("auto", "cpu") == "sdpa"


def test_quantization_rejects_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        OptimizedASR.from_pretrained(
            "unused",
            InferenceConfig(device="cpu", quantization="4bit"),
        )
