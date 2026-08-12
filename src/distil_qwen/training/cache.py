"""Exact, bounded caching for outputs of a frozen Qwen3-ASR audio tower."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

LOGGER = logging.getLogger(__name__)


def audio_cache_key(audio: np.ndarray, sampling_rate: int) -> str:
    """Return a stable content key for a normalized waveform."""

    contiguous = np.ascontiguousarray(audio, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(sampling_rate).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def audio_cache_namespace(model_name_or_path: str, config: object) -> str:
    """Fingerprint the frozen acoustic model strongly enough to isolate persistent caches."""

    path = Path(model_name_or_path).expanduser()
    local_files = []
    if path.is_dir():
        patterns = ("config.json", "*.safetensors", "pytorch_model*.bin")
        for pattern in patterns:
            for candidate in sorted(path.glob(pattern)):
                stat = candidate.stat()
                local_files.append((candidate.name, stat.st_size, stat.st_mtime_ns))
    payload = {
        "model": model_name_or_path,
        "commit": getattr(config, "_commit_hash", None),
        "config": config.to_dict() if hasattr(config, "to_dict") else str(config),
        "local_files": local_files,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AudioCacheStats:
    hits: int
    misses: int
    memory_entries: int
    memory_bytes: int

    @property
    def hit_rate(self) -> float:
        requests = self.hits + self.misses
        return self.hits / requests if requests else 0.0


class AudioFeatureCache:
    """LRU memory cache with an optional process-safe persistent tier.

    Values are detached and stored on CPU. Disk entries use safetensors and atomic replacement, so
    dataloader or distributed-training processes never deserialize executable Python objects.
    """

    def __init__(
        self,
        *,
        max_memory_mb: int = 0,
        cache_dir: Optional[str] = None,
        namespace: str = "default",
    ) -> None:
        if max_memory_mb < 0:
            raise ValueError("max_memory_mb cannot be negative")
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        namespace_hash = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        self.cache_dir = Path(cache_dir).expanduser() / namespace_hash if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._memory_bytes = 0
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_memory_bytes > 0 or self.cache_dir is not None

    def _path(self, key: str) -> Path:
        assert self.cache_dir is not None
        normalized = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / normalized[:2] / f"{normalized}.safetensors"

    def _remember(self, key: str, value: torch.Tensor) -> None:
        if self.max_memory_bytes <= 0 or value.nbytes > self.max_memory_bytes:
            return
        existing = self._memory.pop(key, None)
        if existing is not None:
            self._memory_bytes -= existing.nbytes
        self._memory[key] = value
        self._memory_bytes += value.nbytes
        while self._memory_bytes > self.max_memory_bytes:
            _, evicted = self._memory.popitem(last=False)
            self._memory_bytes -= evicted.nbytes

    def get(self, key: str) -> Optional[torch.Tensor]:
        value = self._memory.pop(key, None)
        if value is not None:
            self._memory[key] = value
            self._hits += 1
            return value

        if self.cache_dir is not None:
            path = self._path(key)
            if path.is_file():
                try:
                    value = load_file(path.as_posix(), device="cpu")["audio_features"]
                except (OSError, RuntimeError, SafetensorError, ValueError) as exc:
                    LOGGER.warning("Ignoring unreadable audio cache entry %s: %s", path, exc)
                else:
                    self._remember(key, value)
                    self._hits += 1
                    return value

        self._misses += 1
        return None

    def put(self, key: str, value: torch.Tensor) -> None:
        stored = value.detach().to(device="cpu").contiguous()
        self._remember(key, stored)
        if self.cache_dir is None:
            return

        destination = self._path(key)
        if destination.is_file():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        try:
            save_file({"audio_features": stored}, temporary_name)
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def stats(self) -> AudioCacheStats:
        return AudioCacheStats(
            hits=self._hits,
            misses=self._misses,
            memory_entries=len(self._memory),
            memory_bytes=self._memory_bytes,
        )
