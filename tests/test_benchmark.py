import numpy as np
import pytest

from distil_qwen.benchmark import benchmark


class FakeASR:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return ["ok"] * len(audio)


def test_benchmark_reports_real_time_metrics() -> None:
    model = FakeASR()
    result = benchmark(
        model,
        [np.zeros(16_000, dtype=np.float32), np.zeros(8_000, dtype=np.float32)],
        warmup_runs=1,
        measured_runs=2,
    )
    assert model.calls == 3
    assert result.samples == 2
    assert result.audio_seconds == 1.5
    assert result.real_time_factor > 0
    assert result.audio_seconds_per_second > 0
    assert result.to_dict()["samples"] == 2


@pytest.mark.parametrize(
    ("audio", "warmup", "runs", "match"),
    [
        ([], 0, 1, "at least one"),
        ([np.zeros(1)], -1, 1, "warmup"),
        ([np.zeros(1)], 0, 0, "measured"),
    ],
)
def test_benchmark_validates_inputs(audio, warmup: int, runs: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        benchmark(FakeASR(), audio, warmup_runs=warmup, measured_runs=runs)
