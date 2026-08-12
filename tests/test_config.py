import pytest

from distil_qwen.config import DistillationConfig, InferenceConfig, StudentSpec


def test_default_configs_are_valid() -> None:
    assert StudentSpec().text_layers == 14
    assert DistillationConfig().kl_weight == 0.8
    assert InferenceConfig().backend == "transformers"


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: StudentSpec(text_layers=0), "text_layers"),
        (lambda: StudentSpec(audio_layers=0), "audio_layers"),
        (lambda: DistillationConfig(temperature=0), "temperature"),
        (lambda: DistillationConfig(ce_weight=0, kl_weight=0), "at least one"),
        (lambda: DistillationConfig(logit_chunk_size=0), "logit_chunk_size"),
        (lambda: DistillationConfig(label_smoothing=1), "label_smoothing"),
        (lambda: DistillationConfig(loss_backend="magic"), "loss_backend"),
        (lambda: InferenceConfig(backend="invalid"), "backend"),
        (lambda: InferenceConfig(dtype="int8"), "dtype"),
        (lambda: InferenceConfig(attention="magic"), "attention"),
        (lambda: InferenceConfig(quantization="2bit"), "quantization"),
        (lambda: InferenceConfig(batch_size=0), "positive"),
        (lambda: InferenceConfig(request_timeout=0), "request_timeout"),
        (lambda: InferenceConfig(api_key_env=""), "api_key_env"),
    ],
)
def test_invalid_configs_raise(factory, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_configs_serialize_to_plain_dicts() -> None:
    assert StudentSpec(audio_layers=12).to_dict() == {"text_layers": 14, "audio_layers": 12}
    assert DistillationConfig().to_dict()["reuse_audio_features"] is True
    assert InferenceConfig().to_dict()["quantization"] is None


def test_inference_backend_aliases_are_normalized() -> None:
    assert InferenceConfig(backend="llama.cpp").backend == "llama_cpp"
    assert InferenceConfig(backend="llama-cpp").backend == "llama_cpp"
