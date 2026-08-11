"""Dataset loading helpers shared by the CLI and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple


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
