"""Batched, atomic pseudo-label generation for JSONL manifests."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from distil_qwen.config import InferenceConfig
from distil_qwen.data.filtering import _atomic_write_jsonl, _read_jsonl
from distil_qwen.inference import OptimizedASR


def pseudo_label_jsonl(
    input_path: str,
    output_path: str,
    model_name_or_path: str = "Qwen/Qwen3-ASR-1.7B",
    inference_config: Optional[InferenceConfig] = None,
    audio_column: str = "audio",
    prompt_column: str = "prompt",
    language: Optional[str] = None,
) -> int:
    """Transcribe a manifest atomically and return the number of labeled rows."""

    inference_config = inference_config or InferenceConfig()
    model = OptimizedASR.from_pretrained(model_name_or_path, inference_config)
    batch_size = inference_config.batch_size

    count = 0

    def label_records() -> Iterable[Dict[str, Any]]:
        nonlocal count
        batch: List[Dict[str, Any]] = []
        for record in _read_jsonl(input_path):
            batch.append(record)
            if len(batch) == batch_size:
                yield from label_batch(batch)
                count += len(batch)
                batch = []
        if batch:
            yield from label_batch(batch)
            count += len(batch)

    def label_batch(batch: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        results = model.transcribe(
            audio=[record[audio_column] for record in batch],
            context=[str(record.get(prompt_column, "") or "") for record in batch],
            language=language,
        )
        for record, result in zip(batch, results):
            enriched = dict(record)
            enriched["teacher_text"] = result.text
            enriched["teacher_language"] = result.language
            yield enriched

    try:
        _atomic_write_jsonl(output_path, label_records())
    finally:
        model.close()
    return count
