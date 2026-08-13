"""Shared inference backend contracts and input normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

AudioInput = Union[str, PathLike[str], bytes, Tuple[Any, int]]
AudioBatch = Union[AudioInput, Sequence[AudioInput]]
StringBatch = Union[str, Sequence[str]]
LanguageBatch = Union[str, Sequence[Optional[str]]]

_ASR_MARKER = "<asr_text>"
_LANGUAGE_PREFIX = re.compile(r"^\s*language\s+(.+?)\s*<asr_text>", re.IGNORECASE)


@dataclass(frozen=True)
class Transcription:
    """Backend-independent transcription result."""

    language: str
    text: str
    time_stamps: Optional[Any] = None


@runtime_checkable
class ASRBackend(Protocol):
    """Minimal contract implemented by local and server inference backends."""

    def transcribe(
        self,
        audio: AudioBatch,
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[Transcription]: ...

    def close(self) -> None: ...


def is_array_sample_rate_pair(value: Any) -> bool:
    """Return whether ``value`` is the supported ``(samples, sample_rate)`` form."""

    return (
        isinstance(value, tuple)
        and len(value) == 2
        and not isinstance(value[0], (str, bytes, PathLike))
        and isinstance(value[1], int)
    )


def normalize_audio_batch(audio: AudioBatch) -> List[AudioInput]:
    """Normalize scalar and batched audio without splitting paths or sample tuples."""

    if isinstance(audio, (str, bytes, PathLike)) or is_array_sample_rate_pair(audio):
        return [audio]
    if isinstance(audio, Sequence):
        values = list(audio)
        if not values:
            raise ValueError("at least one audio input is required")
        return values
    return [audio]


def _broadcast(
    value: Optional[Union[str, Sequence[Optional[str]]]],
    size: int,
    name: str,
    default: Optional[str],
) -> List[Optional[str]]:
    if value is None:
        return [default] * size
    if isinstance(value, str):
        return [value] * size
    values = list(value)
    if len(values) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    return values


def normalize_requests(
    audio: AudioBatch,
    context: Union[str, Sequence[str]] = "",
    language: Optional[Union[str, Sequence[Optional[str]]]] = None,
) -> Tuple[List[AudioInput], List[str], List[Optional[str]]]:
    """Broadcast optional request fields and validate batch cardinality."""

    audio_items = normalize_audio_batch(audio)
    contexts = _broadcast(context, len(audio_items), "context", "")
    languages = _broadcast(language, len(audio_items), "language", None)
    return audio_items, [str(item or "") for item in contexts], languages


def parse_qwen_output(raw_text: str, fallback_language: Optional[str] = None) -> Transcription:
    """Parse Qwen3-ASR's ``language ...<asr_text>...`` wire format."""

    raw_text = raw_text.strip()
    match = _LANGUAGE_PREFIX.match(raw_text)
    if match:
        language = match.group(1).strip()
        text = raw_text[match.end() :].strip()
    elif _ASR_MARKER in raw_text:
        prefix, text = raw_text.split(_ASR_MARKER, 1)
        language = re.sub(r"^\s*language\s+", "", prefix, flags=re.IGNORECASE).strip()
        text = text.strip()
    else:
        language = fallback_language or "auto"
        text = raw_text
    return Transcription(language=language or fallback_language or "auto", text=text)


def coerce_transcriptions(results: Sequence[Any]) -> List[Transcription]:
    """Convert official runtime result objects into the public result type."""

    converted = []
    for result in results:
        if isinstance(result, Transcription):
            converted.append(result)
            continue
        if isinstance(result, dict):
            converted.append(
                Transcription(
                    language=str(result.get("language") or "auto"),
                    text=str(result.get("text") or result.get("transcription") or ""),
                    time_stamps=result.get("time_stamps") or result.get("segments"),
                )
            )
            continue
        converted.append(
            Transcription(
                language=str(getattr(result, "language", "auto") or "auto"),
                text=str(getattr(result, "text", "")),
                time_stamps=getattr(result, "time_stamps", None),
            )
        )
    return converted
