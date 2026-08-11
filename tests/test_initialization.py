from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from distil_qwen.config import StudentSpec
from distil_qwen.models.initialization import initialize_student, uniform_layer_indices


class FakeText(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 3)
        self.layers = nn.ModuleList([nn.Linear(3, 3, bias=False) for _ in range(layers)])
        self.norm = nn.LayerNorm(3)


class FakeAudio(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 3, bias=False) for _ in range(layers)])
        self.projection = nn.Linear(3, 3)


class FakeThinker(nn.Module):
    def __init__(self, text_layers: int, audio_layers: int) -> None:
        super().__init__()
        self.model = FakeText(text_layers)
        self.audio_tower = FakeAudio(audio_layers)
        self.lm_head = nn.Linear(3, 8, bias=False)


class FakeASR(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = deepcopy(config)
        self.thinker = FakeThinker(
            config.thinker_config.text_config.num_hidden_layers,
            config.thinker_config.audio_config.encoder_layers,
        )
        self.generation_config = {"eos": 7}
        self.tied = False

    def tie_weights(self) -> None:
        self.tied = True


def make_teacher() -> FakeASR:
    config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=6),
            audio_config=SimpleNamespace(encoder_layers=4, num_hidden_layers=4),
        )
    )
    teacher = FakeASR(config)
    for index, layer in enumerate(teacher.thinker.model.layers):
        layer.weight.data.fill_(index + 1)
    for index, layer in enumerate(teacher.thinker.audio_tower.layers):
        layer.weight.data.fill_(10 + index)
    return teacher


@pytest.mark.parametrize(
    ("teacher", "student", "expected"),
    [
        (6, 3, (0, 2, 5)),
        (28, 14, (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27)),
        (4, 4, (0, 1, 2, 3)),
        (4, 1, (3,)),
    ],
)
def test_uniform_layer_indices(teacher: int, student: int, expected: tuple[int, ...]) -> None:
    assert uniform_layer_indices(teacher, student) == expected


def test_uniform_layer_indices_validate_counts() -> None:
    with pytest.raises(ValueError, match="positive"):
        uniform_layer_indices(0, 1)
    with pytest.raises(ValueError, match="more layers"):
        uniform_layer_indices(2, 3)


def test_initialize_student_copies_selected_layers_and_shared_weights() -> None:
    teacher = make_teacher()
    teacher.thinker.model.embed_tokens.weight.data.fill_(42)
    student, report = initialize_student(teacher, StudentSpec(text_layers=3, audio_layers=2))

    assert len(student.thinker.model.layers) == 3
    assert len(student.thinker.audio_tower.layers) == 2
    assert report.text_layer_map == (0, 2, 5)
    assert report.audio_layer_map == (0, 3)
    assert [layer.weight[0, 0].item() for layer in student.thinker.model.layers] == [1, 3, 6]
    assert [layer.weight[0, 0].item() for layer in student.thinker.audio_tower.layers] == [10, 13]
    assert torch.all(student.thinker.model.embed_tokens.weight == 42)
    assert student.tied is True
    assert report.parameter_reduction > 0
    assert report.to_dict()["text_layer_map"] == [0, 2, 5]


def test_initialize_student_keeps_full_audio_tower_by_default() -> None:
    student, report = initialize_student(make_teacher(), StudentSpec(text_layers=2))
    assert len(student.thinker.audio_tower.layers) == 4
    assert report.audio_layer_map == (0, 1, 2, 3)


def test_initialize_student_preserves_teacher_dtype() -> None:
    teacher = make_teacher().double()
    student, _ = initialize_student(teacher, StudentSpec(text_layers=2))
    assert next(student.parameters()).dtype == torch.float64
