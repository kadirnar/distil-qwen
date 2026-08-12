"""Dynamic multimodal batching for Qwen3-ASR teacher-forced training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from distil_qwen.training.cache import audio_cache_key


def feature_lengths_after_encoder(input_lengths: torch.Tensor) -> torch.Tensor:
    """Mirror Qwen3-ASR's convolutional feature-length calculation."""

    remainder = input_lengths % 100
    feature_lengths = (remainder - 1) // 2 + 1
    return ((feature_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def _array_and_rate(value: Any, default_sampling_rate: int) -> Tuple[np.ndarray, int]:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False), default_sampling_rate
    if isinstance(value, dict):
        if "array" in value:
            return np.asarray(value["array"], dtype=np.float32), int(
                value.get("sampling_rate", default_sampling_rate)
            )
        if "path" in value and value["path"]:
            value = value["path"]
    if isinstance(value, (tuple, list)) and len(value) == 2:
        first, second = value
        if np.isscalar(first):
            return np.asarray(second, dtype=np.float32), int(first)
        return np.asarray(first, dtype=np.float32), int(second)
    if isinstance(value, (str, Path)):
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("audio paths require `pip install 'distil-qwen[train]'`") from exc
        audio, sampling_rate = librosa.load(str(value), sr=None, mono=True)
        return np.asarray(audio, dtype=np.float32), int(sampling_rate)
    raise TypeError(f"unsupported audio value: {type(value).__name__}")


def load_audio(value: Any, sampling_rate: int = 16_000) -> np.ndarray:
    """Load, mono-mix, and resample a dataset audio value."""

    audio, original_rate = _array_and_rate(value, sampling_rate)
    if audio.ndim == 2:
        channel_axis = 0 if audio.shape[0] <= 8 else 1
        audio = audio.mean(axis=channel_axis)
    if audio.ndim != 1:
        raise ValueError(f"audio must be one-dimensional after mono mixing, got {audio.shape}")
    if original_rate != sampling_rate:
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("resampling requires `pip install 'distil-qwen[train]'`") from exc
        audio = librosa.resample(audio, orig_sr=original_rate, target_sr=sampling_rate)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


@dataclass
class Qwen3ASRDataCollator:
    """Create inputs and mask every prompt token from the causal objective.

    The audio feature extractor runs once per sample. Prefix lengths are found by expanding the
    already-computed audio placeholder lengths and invoking only the tokenizer for the second pass.
    """

    processor: Any
    audio_column: str = "audio"
    text_column: str = "text"
    prompt_column: str = "prompt"
    sampling_rate: int = 16_000
    include_audio_cache_keys: bool = False
    cache_key_column: Optional[str] = None

    def _prefix(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]
        rendered = self.processor.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=False
        )
        return rendered[0] if isinstance(rendered, list) else rendered

    def _prefix_lengths(
        self,
        prefix_texts: Sequence[str],
        feature_attention_mask: torch.Tensor,
    ) -> List[int]:
        audio_lengths = feature_lengths_after_encoder(feature_attention_mask.sum(dim=-1)).tolist()
        if hasattr(self.processor, "replace_multimodal_special_tokens"):
            expanded = self.processor.replace_multimodal_special_tokens(
                list(prefix_texts), iter(audio_lengths)
            )
            tokenized = self.processor.tokenizer(
                expanded,
                padding=True,
                return_tensors="pt",
            )
        else:
            raise TypeError(
                "processor must expose replace_multimodal_special_tokens; "
                "install the pinned qwen-asr version"
            )
        return [int(length) for length in tokenized["attention_mask"].sum(dim=-1)]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        audios = [load_audio(item[self.audio_column], self.sampling_rate) for item in features]
        prefixes = [self._prefix(str(item.get(self.prompt_column, "") or "")) for item in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_text = [
            prefix + str(item[self.text_column]) + eos for prefix, item in zip(prefixes, features)
        ]
        batch = self.processor(
            text=full_text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_lengths = self._prefix_lengths(prefixes, batch["feature_attention_mask"])

        labels = batch["input_ids"].clone()
        labels.masked_fill_(batch["attention_mask"].eq(0), -100)
        for row, prefix_length in enumerate(prefix_lengths):
            active_positions = batch["attention_mask"][row].nonzero(as_tuple=False).flatten()
            if prefix_length >= active_positions.numel():
                raise ValueError("target text produced no trainable tokens")
            labels[row, active_positions[:prefix_length]] = -100
        batch["labels"] = labels
        if self.include_audio_cache_keys:
            batch["audio_cache_keys"] = [
                str(item[self.cache_key_column])
                if self.cache_key_column is not None
                else audio_cache_key(audio, self.sampling_rate)
                for item, audio in zip(features, audios)
            ]
        return dict(batch)
