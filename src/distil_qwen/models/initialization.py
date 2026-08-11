"""Initialize a smaller Qwen3-ASR model from maximally spaced teacher layers."""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from distil_qwen.compat import require_qwen_asr
from distil_qwen.config import StudentSpec
from distil_qwen.errors import ModelContractError
from distil_qwen.models.accessors import get_audio_tower, get_text_model

LOGGER = logging.getLogger(__name__)
_LAYER_KEY = re.compile(r"^thinker\.(?:model|audio_tower)\.layers\.\d+\.")


@dataclass(frozen=True)
class InitializationReport:
    """Summary persisted next to an initialized checkpoint."""

    teacher_text_layers: int
    student_text_layers: int
    teacher_audio_layers: int
    student_audio_layers: int
    text_layer_map: Tuple[int, ...]
    audio_layer_map: Tuple[int, ...]
    teacher_parameters: int
    student_parameters: int

    @property
    def parameter_reduction(self) -> float:
        return 1.0 - (self.student_parameters / self.teacher_parameters)

    def to_dict(self) -> Dict[str, Any]:
        result = dict(self.__dict__)
        result["text_layer_map"] = list(self.text_layer_map)
        result["audio_layer_map"] = list(self.audio_layer_map)
        result["parameter_reduction"] = self.parameter_reduction
        return result


def uniform_layer_indices(teacher_layers: int, student_layers: int) -> Tuple[int, ...]:
    """Select maximally spaced layers, always including the first and last."""

    if teacher_layers < 1 or student_layers < 1:
        raise ValueError("layer counts must be positive")
    if student_layers > teacher_layers:
        raise ValueError("student cannot have more layers than its teacher")
    if student_layers == 1:
        return (teacher_layers - 1,)
    if student_layers == teacher_layers:
        return tuple(range(teacher_layers))

    denominator = student_layers - 1
    indices = tuple(
        (index * (teacher_layers - 1)) // denominator for index in range(student_layers)
    )
    if len(set(indices)) != student_layers:
        raise AssertionError("uniform layer selection produced duplicate indices")
    return indices


def _nested_configs(config: Any) -> Tuple[Any, Any]:
    thinker_config = getattr(config, "thinker_config", None)
    if thinker_config is None:
        raise ModelContractError("model config does not contain thinker_config")
    text_config = getattr(thinker_config, "text_config", None)
    audio_config = getattr(thinker_config, "audio_config", None)
    if text_config is None or audio_config is None:
        raise ModelContractError("thinker_config must contain text_config and audio_config")
    return text_config, audio_config


def _copy_layer_modules(
    source_layers: Sequence[torch.nn.Module],
    destination_layers: Sequence[torch.nn.Module],
    mapping: Sequence[int],
) -> None:
    if len(destination_layers) != len(mapping):
        raise ModelContractError("destination layer count does not match the layer map")
    for destination_index, source_index in enumerate(mapping):
        destination_layers[destination_index].load_state_dict(
            source_layers[source_index].state_dict(), strict=True
        )


def initialize_student(
    teacher: torch.nn.Module,
    spec: Optional[StudentSpec] = None,
) -> Tuple[torch.nn.Module, InitializationReport]:
    """Build a student of the same class and copy uniformly selected teacher layers."""

    spec = spec or StudentSpec()
    teacher_text = get_text_model(teacher)
    teacher_audio = get_audio_tower(teacher)
    teacher_text_count = len(teacher_text.layers)
    teacher_audio_count = len(teacher_audio.layers)
    student_audio_count = spec.audio_layers or teacher_audio_count
    text_map = uniform_layer_indices(teacher_text_count, spec.text_layers)
    audio_map = uniform_layer_indices(teacher_audio_count, student_audio_count)

    student_config = copy.deepcopy(teacher.config)
    text_config, audio_config = _nested_configs(student_config)
    text_config.num_hidden_layers = spec.text_layers
    audio_config.encoder_layers = student_audio_count
    audio_config.num_hidden_layers = student_audio_count

    teacher_dtype = next(teacher.parameters()).dtype
    from_config = getattr(teacher.__class__, "_from_config", None)
    if from_config is not None:
        student = from_config(student_config, dtype=teacher_dtype)
    else:
        student = teacher.__class__(student_config).to(dtype=teacher_dtype)
    non_layer_state = {
        name: tensor for name, tensor in teacher.state_dict().items() if not _LAYER_KEY.match(name)
    }
    incompatible = student.load_state_dict(non_layer_state, strict=False)
    if incompatible.unexpected_keys:
        raise ModelContractError(
            f"unexpected non-layer weights while creating student: {incompatible.unexpected_keys}"
        )

    _copy_layer_modules(teacher_text.layers, get_text_model(student).layers, text_map)
    _copy_layer_modules(teacher_audio.layers, get_audio_tower(student).layers, audio_map)
    if hasattr(student, "tie_weights"):
        student.tie_weights()
    if hasattr(teacher, "generation_config"):
        student.generation_config = copy.deepcopy(teacher.generation_config)

    report = InitializationReport(
        teacher_text_layers=teacher_text_count,
        student_text_layers=spec.text_layers,
        teacher_audio_layers=teacher_audio_count,
        student_audio_layers=student_audio_count,
        text_layer_map=text_map,
        audio_layer_map=audio_map,
        teacher_parameters=sum(parameter.numel() for parameter in teacher.parameters()),
        student_parameters=sum(parameter.numel() for parameter in student.parameters()),
    )
    return student, report


def initialize_from_pretrained(
    teacher_name_or_path: str,
    output_dir: str,
    spec: Optional[StudentSpec] = None,
    dtype: str = "auto",
    attention: str = "sdpa",
    revision: str = "main",
) -> InitializationReport:
    """Load a teacher, create a student, and save a complete Hub-compatible checkpoint."""

    spec = spec or StudentSpec()
    require_qwen_asr()
    from transformers import AutoModel, AutoProcessor

    load_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "revision": revision,
    }
    if dtype != "auto":
        load_kwargs["dtype"] = getattr(torch, dtype)
    if attention != "auto":
        load_kwargs["attn_implementation"] = attention

    teacher = AutoModel.from_pretrained(teacher_name_or_path, **load_kwargs)
    processor = AutoProcessor.from_pretrained(
        teacher_name_or_path,
        revision=revision,
        fix_mistral_regex=True,
    )
    student, report = initialize_student(teacher, spec)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(destination, safe_serialization=True)
    processor.save_pretrained(destination)
    metadata = {
        "format_version": 1,
        "teacher": teacher_name_or_path,
        "revision": revision,
        "student_spec": spec.to_dict(),
        "initialization": report.to_dict(),
    }
    (destination / "distil_qwen_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Initialized %d text / %d audio layer student (%.1f%% fewer parameters)",
        report.student_text_layers,
        report.student_audio_layers,
        report.parameter_reduction * 100,
    )
    return report


def parse_layer_map(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated indices for future custom initialization workflows."""

    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]
