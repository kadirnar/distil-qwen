import numpy as np
import pytest
import torch

from distil_qwen.training.cache import (
    AudioFeatureCache,
    audio_cache_key,
    audio_cache_namespace,
)


def test_audio_cache_key_uses_content_and_sampling_rate() -> None:
    audio = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    assert audio_cache_key(audio, 16_000) == audio_cache_key(audio.copy(), 16_000)
    assert audio_cache_key(audio, 8_000) != audio_cache_key(audio, 16_000)
    assert audio_cache_key(audio + 0.1, 16_000) != audio_cache_key(audio, 16_000)


def test_audio_cache_namespace_tracks_local_checkpoint_metadata(tmp_path) -> None:
    config = type("Config", (), {"to_dict": lambda self: {"layers": 2}})()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"first")
    first = audio_cache_namespace(str(tmp_path), config)
    checkpoint.write_bytes(b"second-version")
    assert audio_cache_namespace(str(tmp_path), config) != first


def test_audio_feature_cache_persists_safe_tensor_entries(tmp_path) -> None:
    value = torch.randn(3, 4, dtype=torch.float16)
    cache = AudioFeatureCache(cache_dir=str(tmp_path), namespace="teacher@commit")
    assert cache.enabled
    assert cache.get("clip") is None
    cache.put("clip", value)

    reloaded = AudioFeatureCache(cache_dir=str(tmp_path), namespace="teacher@commit")
    torch.testing.assert_close(reloaded.get("clip"), value)
    assert reloaded.stats().hit_rate == 1.0

    isolated = AudioFeatureCache(cache_dir=str(tmp_path), namespace="different-teacher")
    assert isolated.get("clip") is None


def test_audio_feature_cache_bounds_memory_and_validates_limit() -> None:
    cache = AudioFeatureCache(max_memory_mb=1)
    cache.put("first", torch.zeros(160_000))
    cache.put("second", torch.ones(160_000))
    assert cache.stats().memory_entries == 1
    assert cache.get("first") is None
    assert cache.get("second") is not None
    assert cache.stats().memory_bytes <= 1024 * 1024

    with pytest.raises(ValueError, match="negative"):
        AudioFeatureCache(max_memory_mb=-1)
