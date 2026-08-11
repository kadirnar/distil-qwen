import json

import pytest

from distil_qwen.data.filtering import (
    error_rate,
    filter_jsonl,
    filter_records,
    normalize_text,
    repeated_ngram_ratio,
)


def test_normalization_and_error_rates() -> None:
    assert normalize_text("  Hello, WORLD! ") == "hello world"
    assert error_rate("one two three", "one too three", "wer") == pytest.approx(1 / 3)
    assert error_rate("你好世界", "你好世", "auto") == 0.25
    assert error_rate("", "", "cer") == 0
    assert error_rate("", "extra", "cer") == 1
    with pytest.raises(ValueError, match="metric"):
        error_rate("a", "b", "invalid")


def test_repetition_ratio_detects_hallucinated_loops() -> None:
    clean = "one two three four five six seven"
    repeated = "one two three one two three one two three"
    assert repeated_ngram_ratio(clean, n=3) == 0
    assert repeated_ngram_ratio(repeated, n=3) > 0


def test_filter_records_reports_each_rejection_reason() -> None:
    records = [
        {"text": "hello world", "teacher_text": "hello world"},
        {"text": "hello", "teacher_text": ""},
        {"text": "one two", "teacher_text": "completely wrong"},
        {
            "text": "a b c a b c a b c",
            "teacher_text": "a b c a b c a b c",
        },
    ]
    kept, report = filter_records(
        records,
        max_error_rate=0.2,
        max_repeated_ngram_ratio=0.1,
    )
    assert len(kept) == 1
    assert kept[0]["pseudo_label_error_rate"] == 0
    assert report.total == 4
    assert report.kept == 1
    assert report.empty == 1
    assert report.high_error == 1
    assert report.repetitive == 1
    assert report.keep_rate == 0.25


def test_filter_jsonl_writes_valid_output_atomically(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "nested" / "filtered.jsonl"
    source.write_text(
        json.dumps({"text": "Merhaba dünya", "teacher_text": "Merhaba dünya"}) + "\n",
        encoding="utf-8",
    )
    report = filter_jsonl(str(source), str(output))
    assert report.kept == 1
    assert json.loads(output.read_text(encoding="utf-8"))["pseudo_label_error_rate"] == 0


def test_invalid_jsonl_has_line_context(tmp_path) -> None:
    source = tmp_path / "broken.jsonl"
    source.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        filter_jsonl(str(source), str(tmp_path / "out.jsonl"))
