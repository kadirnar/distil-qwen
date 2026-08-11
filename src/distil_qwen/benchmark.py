"""Small reproducible latency benchmark for ASR inference."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from distil_qwen.inference import OptimizedASR
from distil_qwen.training.collator import load_audio


@dataclass(frozen=True)
class BenchmarkResult:
    samples: int
    audio_seconds: float
    median_latency_seconds: float
    mean_latency_seconds: float
    real_time_factor: float
    audio_seconds_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark(
    model: OptimizedASR,
    audio: Sequence[Any],
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> BenchmarkResult:
    if not audio:
        raise ValueError("at least one audio input is required")
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measured_runs must be positive")
    total_audio_seconds = sum(len(load_audio(item)) / 16_000 for item in audio)
    for _ in range(warmup_runs):
        model.transcribe(audio=list(audio))
    _synchronize()

    latencies = []
    for _ in range(measured_runs):
        started = time.perf_counter()
        model.transcribe(audio=list(audio))
        _synchronize()
        latencies.append(time.perf_counter() - started)
    median = statistics.median(latencies)
    mean = statistics.fmean(latencies)
    return BenchmarkResult(
        samples=len(audio),
        audio_seconds=total_audio_seconds,
        median_latency_seconds=median,
        mean_latency_seconds=mean,
        real_time_factor=median / total_audio_seconds,
        audio_seconds_per_second=total_audio_seconds / median,
    )
