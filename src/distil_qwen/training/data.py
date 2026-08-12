"""Dataset loading helpers shared by the CLI and notebooks."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np


def _array_duration(value: Any, sampling_rate: int) -> float:
    shape = getattr(value, "shape", None)
    if not shape:
        value = np.asarray(value)
        shape = value.shape
    if not shape:
        return 0.0
    sample_axis = -1 if len(shape) == 1 or shape[0] <= 8 else 0
    return float(shape[sample_axis]) / sampling_rate


def audio_duration_seconds(value: Any, default_sampling_rate: int = 16_000) -> float:
    """Read audio duration without decoding paths when metadata is available."""

    if isinstance(value, Mapping):
        sampling_rate = int(value.get("sampling_rate", default_sampling_rate))
        if "array" in value and value["array"] is not None:
            return _array_duration(value["array"], sampling_rate)
        if value.get("path"):
            value = value["path"]
        elif value.get("bytes"):
            try:
                import soundfile
            except ImportError as exc:
                raise ImportError(
                    "audio metadata requires `pip install 'distil-qwen[train]'`"
                ) from exc
            return float(soundfile.info(io.BytesIO(value["bytes"])).duration)

    if isinstance(value, (tuple, list)) and len(value) == 2:
        first, second = value
        if np.isscalar(first):
            return _array_duration(second, int(first))
        return _array_duration(first, int(second))
    if isinstance(value, np.ndarray) or hasattr(value, "shape"):
        return _array_duration(value, default_sampling_rate)
    if isinstance(value, (str, Path)):
        try:
            import soundfile
        except ImportError as exc:
            raise ImportError("audio metadata requires `pip install 'distil-qwen[train]'`") from exc
        return float(soundfile.info(str(value)).duration)
    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        return _array_duration(samples.data, int(samples.sample_rate))
    raise TypeError(f"cannot determine duration for audio value: {type(value).__name__}")


def estimate_training_length(
    audio: Any,
    text: Any,
    prompt: Any = "",
    sampling_rate: int = 16_000,
) -> int:
    """Estimate post-encoder audio plus text tokens for padding-aware sampling."""

    audio_tokens = math.ceil(audio_duration_seconds(audio, sampling_rate) * 13)
    text_tokens = len(str(text)) + len(str(prompt or ""))
    return max(1, audio_tokens + text_tokens)


def _add_length_to_example(
    example: Mapping[str, Any],
    *,
    audio_column: str,
    text_column: str,
    prompt_column: str,
    length_column: str,
) -> dict[str, int]:
    return {
        length_column: estimate_training_length(
            example[audio_column],
            example[text_column],
            example.get(prompt_column, ""),
        )
    }


def add_length_column(
    dataset: Any,
    *,
    audio_column: str,
    text_column: str,
    prompt_column: str,
    length_column: str,
    num_proc: Optional[int] = None,
) -> Any:
    """Add a cached sampling-cost column once, preserving an existing column."""

    if length_column in getattr(dataset, "column_names", ()):
        return dataset
    map_kwargs = {
        "fn_kwargs": {
            "audio_column": audio_column,
            "text_column": text_column,
            "prompt_column": prompt_column,
            "length_column": length_column,
        },
        "desc": "Estimating Qwen3-ASR sequence lengths",
    }
    if num_proc is not None and num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    return dataset.map(_add_length_to_example, **map_kwargs)


def load_splits(
    dataset: str,
    dataset_config: Optional[str] = None,
    train_split: str = "train",
    eval_split: Optional[str] = None,
) -> Tuple[Any, Optional[Any]]:
    """Load Hub datasets, saved datasets, or local JSON/JSONL manifests."""

    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError("training requires `pip install 'distil-qwen[train]'`") from exc

    path = Path(dataset)
    if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
        data_files = {"train": str(path)}
        loaded = load_dataset("json", data_files=data_files)
        train_data = loaded["train"]
        return train_data, None
    if path.is_dir() and (path / "dataset_dict.json").exists():
        loaded = load_from_disk(str(path))
        return loaded[train_split], loaded[eval_split] if eval_split else None

    train_data = load_dataset(dataset, dataset_config, split=train_split)
    eval_data = (
        load_dataset(dataset, dataset_config, split=eval_split) if eval_split is not None else None
    )
    return train_data, eval_data
