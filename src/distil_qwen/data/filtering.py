"""Language-agnostic pseudo-label quality filters."""

from __future__ import annotations

import json
import os
import tempfile
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def normalize_text(text: str) -> str:
    """Apply conservative Unicode, case, punctuation, and whitespace normalization."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters = [
        " " if unicodedata.category(character).startswith(("P", "S")) else character
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, metric: str = "auto") -> float:
    """Compute WER or CER, selecting CER for unsegmented scripts in auto mode."""

    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if metric not in {"auto", "wer", "cer"}:
        raise ValueError("metric must be auto, wer, or cer")
    use_words = metric == "wer" or (metric == "auto" and " " in reference)
    reference_units = reference.split() if use_words else list(reference.replace(" ", ""))
    hypothesis_units = hypothesis.split() if use_words else list(hypothesis.replace(" ", ""))
    if not reference_units:
        return 0.0 if not hypothesis_units else 1.0
    return _edit_distance(reference_units, hypothesis_units) / len(reference_units)


def repeated_ngram_ratio(text: str, n: int = 5) -> float:
    units = normalize_text(text).split()
    if len(units) < n:
        return 0.0
    ngrams = [tuple(units[index : index + n]) for index in range(len(units) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


@dataclass(frozen=True)
class FilterReport:
    total: int
    kept: int
    empty: int
    high_error: int
    repetitive: int

    @property
    def keep_rate(self) -> float:
        return self.kept / self.total if self.total else 0.0


def filter_records(
    records: Iterable[Dict[str, Any]],
    reference_column: str = "text",
    pseudo_column: str = "teacher_text",
    metric: str = "auto",
    max_error_rate: float = 0.2,
    max_repeated_ngram_ratio: float = 0.3,
) -> tuple[List[Dict[str, Any]], FilterReport]:
    kept: List[Dict[str, Any]] = []
    counters = {"total": 0, "empty": 0, "high_error": 0, "repetitive": 0}
    for record in records:
        counters["total"] += 1
        pseudo = str(record.get(pseudo_column, "") or "")
        if not normalize_text(pseudo):
            counters["empty"] += 1
            continue
        record_error_rate = error_rate(str(record[reference_column]), pseudo, metric)
        if record_error_rate > max_error_rate:
            counters["high_error"] += 1
            continue
        if repeated_ngram_ratio(pseudo) > max_repeated_ngram_ratio:
            counters["repetitive"] += 1
            continue
        enriched = dict(record)
        enriched["pseudo_label_error_rate"] = record_error_rate
        kept.append(enriched)
    return kept, FilterReport(kept=len(kept), **counters)


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc


def _atomic_write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_path)
        raise


def filter_jsonl(input_path: str, output_path: str, **kwargs: Any) -> FilterReport:
    records, report = filter_records(_read_jsonl(input_path), **kwargs)
    _atomic_write_jsonl(output_path, records)
    return report
