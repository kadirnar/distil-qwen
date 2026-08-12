from typing import Any

import numpy as np
import pytest
import torch

from distil_qwen.training.collator import (
    Qwen3ASRDataCollator,
    feature_lengths_after_encoder,
    load_audio,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, texts, padding: bool, return_tensors: str):
        assert padding and return_tensors == "pt"
        return {"attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])}


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool):
        assert add_generation_prompt and not tokenize
        return ["P<audio>"]

    def replace_multimodal_special_tokens(self, prefixes, lengths):
        list(lengths)
        return prefixes

    def __call__(self, **kwargs: Any):
        assert len(kwargs["audio"]) == 2
        return {
            "input_ids": torch.tensor([[10, 11, 12, 20, 99], [0, 10, 11, 12, 21]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]),
            "input_features": torch.randn(2, 4, 3),
            "feature_attention_mask": torch.ones(2, 4, dtype=torch.long),
        }


def test_collator_masks_prompts_with_left_padding() -> None:
    collator = Qwen3ASRDataCollator(FakeProcessor())
    batch = collator(
        [
            {"audio": np.zeros(16), "text": "first", "prompt": ""},
            {"audio": np.zeros(8), "text": "second"},
        ]
    )
    assert batch["labels"].tolist() == [
        [-100, -100, -100, 20, 99],
        [-100, -100, -100, -100, 21],
    ]


def test_collator_emits_stable_or_explicit_audio_cache_keys() -> None:
    features = [
        {"audio": np.zeros(16), "text": "first", "id": "a"},
        {"audio": np.ones(8), "text": "second", "id": "b"},
    ]
    hashed = Qwen3ASRDataCollator(FakeProcessor(), include_audio_cache_keys=True)(features)[
        "audio_cache_keys"
    ]
    assert len(hashed) == 2
    assert hashed[0] != hashed[1]

    explicit = Qwen3ASRDataCollator(
        FakeProcessor(), include_audio_cache_keys=True, cache_key_column="id"
    )(features)["audio_cache_keys"]
    assert explicit == ["a", "b"]


def test_feature_length_formula() -> None:
    values = feature_lengths_after_encoder(torch.tensor([100, 200, 3000]))
    assert values.tolist() == [13, 26, 390]


def test_load_audio_handles_arrays_and_stereo() -> None:
    mono = np.linspace(-0.5, 0.5, 4, dtype=np.float32)
    assert np.array_equal(load_audio(mono), mono)
    stereo = np.stack([mono, mono + 0.2])
    np.testing.assert_allclose(load_audio((stereo, 16_000)), mono + 0.1, atol=1e-7)
    np.testing.assert_allclose(load_audio(np.array([-4, 2], dtype=np.float32)), [-1, 0.5])
    with pytest.raises(TypeError, match="unsupported"):
        load_audio(object())


def test_collator_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        Qwen3ASRDataCollator(FakeProcessor())([])
