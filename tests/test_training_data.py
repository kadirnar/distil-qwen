from types import SimpleNamespace

import numpy as np

from distil_qwen.training.data import (
    add_length_column,
    audio_duration_seconds,
    estimate_training_length,
)


def test_audio_duration_handles_common_in_memory_layouts() -> None:
    mono = np.zeros(32_000, dtype=np.float32)
    channels_first = np.zeros((2, 32_000), dtype=np.float32)
    channels_last = np.zeros((32_000, 2), dtype=np.float32)

    assert audio_duration_seconds({"array": mono, "sampling_rate": 16_000}) == 2.0
    assert audio_duration_seconds((16_000, channels_first)) == 2.0
    assert audio_duration_seconds((channels_last, 16_000)) == 2.0


def test_audio_duration_supports_lazy_audio_decoder() -> None:
    decoder = SimpleNamespace(
        get_all_samples=lambda: SimpleNamespace(
            data=np.zeros((1, 8_000), dtype=np.float32), sample_rate=16_000
        )
    )
    assert audio_duration_seconds(decoder) == 0.5


def test_training_length_combines_audio_and_text_cost() -> None:
    audio = np.zeros(32_000, dtype=np.float32)
    assert estimate_training_length(audio, "four", "go") == 32


def test_add_length_column_is_cached_by_the_dataset() -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.column_names = ["audio", "text", "prompt"]
            self.rows = [
                {
                    "audio": np.zeros(16_000, dtype=np.float32),
                    "text": "hello",
                    "prompt": "",
                }
            ]
            self.calls = 0

        def map(self, function, **kwargs):
            self.calls += 1
            for row in self.rows:
                row.update(function(row, **kwargs["fn_kwargs"]))
            self.column_names.append("cost")
            return self

    dataset = FakeDataset()
    result = add_length_column(
        dataset,
        audio_column="audio",
        text_column="text",
        prompt_column="prompt",
        length_column="cost",
        num_proc=1,
    )
    assert result.rows[0]["cost"] == 18
    assert result.calls == 1

    add_length_column(
        result,
        audio_column="audio",
        text_column="text",
        prompt_column="prompt",
        length_column="cost",
    )
    assert result.calls == 1
