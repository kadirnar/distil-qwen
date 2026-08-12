from types import SimpleNamespace

import pytest
import torch
from torch import nn

from distil_qwen.config import DistillationConfig
from distil_qwen.errors import ModelContractError
from distil_qwen.models.accessors import get_audio_tower, get_lm_head, get_text_model, get_thinker
from distil_qwen.training.cache import AudioFeatureCache
from distil_qwen.training.engine import (
    configure_student_trainability,
    configure_teacher,
    distillation_step,
    model_device,
    paired_hidden_states,
)


class ToyTextModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs_embeds, **kwargs):
        del kwargs
        hidden = self.layers[0](inputs_embeds)
        return (self.norm(hidden),)


class ToyAudioTower(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)])


class ToyThinker(nn.Module):
    def __init__(self, hidden_size: int = 4, vocabulary: int = 16) -> None:
        super().__init__()
        self.model = ToyTextModel(hidden_size)
        self.audio_tower = ToyAudioTower(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocabulary, bias=False)
        self.audio_calls = 0

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_audio_features(self, input_features, **kwargs):
        del kwargs
        self.audio_calls += 1
        return input_features.reshape(-1, input_features.shape[-1])

    def get_placeholder_mask(self, input_ids, inputs_embeds):
        return input_ids.eq(1).unsqueeze(-1).expand_as(inputs_embeds)

    def get_rope_index(self, attention_mask):
        positions = attention_mask.cumsum(-1).sub(1).clamp_min(0)
        return positions.unsqueeze(0).expand(3, -1, -1), torch.zeros(attention_mask.shape[0], 1)


class ToyASR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.thinker = ToyThinker()


def make_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[2, 1, 3, 4], [2, 1, 5, 6]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "input_features": torch.randn(2, 1, 4),
        "feature_attention_mask": torch.ones(2, 1, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4], [-100, -100, 5, 6]]),
    }


def test_paired_hidden_states_reuses_teacher_audio_once() -> None:
    student, teacher = ToyASR(), ToyASR()
    states = paired_hidden_states(student, teacher, make_batch(), reuse_audio_features=True)
    assert states.student.shape == (2, 4, 4)
    assert states.teacher.shape == (2, 4, 4)
    assert student.thinker.audio_calls == 0
    assert teacher.thinker.audio_calls == 1


def test_paired_hidden_states_reuses_persistent_audio_cache() -> None:
    student, teacher = ToyASR(), ToyASR()
    batch = make_batch()
    batch["audio_cache_keys"] = ["first", "second"]
    cache = AudioFeatureCache(max_memory_mb=1)

    paired_hidden_states(student, teacher, batch, audio_cache=cache)
    paired_hidden_states(student, teacher, batch, audio_cache=cache)

    assert teacher.thinker.audio_calls == 1
    assert student.thinker.audio_calls == 0
    assert cache.stats().hits == 2
    assert cache.stats().misses == 2


def test_audio_cache_rejects_mismatched_keys() -> None:
    student, teacher = ToyASR(), ToyASR()
    batch = make_batch()
    batch["audio_cache_keys"] = ["only-one"]
    with pytest.raises(ModelContractError, match="cache keys"):
        paired_hidden_states(
            student,
            teacher,
            batch,
            audio_cache=AudioFeatureCache(max_memory_mb=1),
        )


def test_paired_hidden_states_can_train_audio_tower_separately() -> None:
    student, teacher = ToyASR(), ToyASR()
    paired_hidden_states(student, teacher, make_batch(), reuse_audio_features=False)
    assert student.thinker.audio_calls == 1
    assert teacher.thinker.audio_calls == 1


def test_distillation_step_backpropagates_only_student() -> None:
    student, teacher = ToyASR(), ToyASR()
    configure_teacher(teacher)
    output = distillation_step(
        student,
        teacher,
        make_batch(),
        DistillationConfig(logit_chunk_size=1),
    )
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_configure_student_trainability_freezes_shared_components() -> None:
    model = ToyASR()
    trainable = configure_student_trainability(model)
    assert trainable > 0
    assert all(not parameter.requires_grad for parameter in model.thinker.audio_tower.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.thinker.model.embed_tokens.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.thinker.lm_head.parameters())
    assert model_device(model) == torch.device("cpu")


def test_text_only_forward_and_accessor_wrappers() -> None:
    student, teacher = ToyASR(), ToyASR()
    batch = make_batch()
    batch.pop("input_features")
    batch.pop("feature_attention_mask")
    states = paired_hidden_states(student, teacher, batch)
    assert states.student.shape == states.teacher.shape
    assert get_thinker(SimpleNamespace(module=student)) is student.thinker
    assert get_text_model(student) is student.thinker.model
    assert get_audio_tower(student) is student.thinker.audio_tower
    assert get_lm_head(student) is student.thinker.lm_head


def test_accessors_and_audio_injection_validate_contracts() -> None:
    with pytest.raises(ModelContractError, match="thinker"):
        get_thinker(object())
    with pytest.raises(ModelContractError, match="text decoder"):
        get_text_model(SimpleNamespace(thinker=SimpleNamespace(model=object())))
    with pytest.raises(ModelContractError, match="audio tower"):
        get_audio_tower(SimpleNamespace(thinker=SimpleNamespace()))
    with pytest.raises(ModelContractError, match="LM head"):
        get_lm_head(SimpleNamespace(thinker=SimpleNamespace()))

    student, teacher = ToyASR(), ToyASR()
    batch = make_batch()
    batch["input_features"] = torch.randn(2, 2, 4)
    with pytest.raises(ModelContractError, match="placeholder shape"):
        paired_hidden_states(student, teacher, batch)
