from types import SimpleNamespace

import pytest
import torch
from torch import nn

from distil_qwen.errors import OptionalDependencyError
from distil_qwen.training import optimizations


def test_attention_auto_selects_flash_only_for_supported_runtime(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(optimizations, "package_available", lambda name: name == "flash_attn")
    assert optimizations.resolve_attention("auto", torch.bfloat16) == "flash_attention_2"
    assert optimizations.resolve_attention("auto", torch.float32) == "sdpa"

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    with pytest.raises(OptionalDependencyError, match="FlashAttention"):
        optimizations.resolve_attention("flash_attention_2", torch.float16)


def test_liger_and_optimizer_resolution_never_silently_ignore_requests(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(optimizations, "package_available", lambda name: False)
    assert optimizations.resolve_liger("auto") is False
    assert optimizations.resolve_optimizer("auto") == "adamw_torch"
    with pytest.raises(OptionalDependencyError, match="Liger"):
        optimizations.resolve_liger("on")
    with pytest.raises(ValueError, match="unsupported optimizer"):
        optimizations.resolve_optimizer("paged_adamw_8bit")
    with pytest.raises(ValueError, match="unsupported optimizer"):
        optimizations.resolve_optimizer("mystery")


class Qwen3ASRThinkerTextRMSNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.variance_epsilon = 1e-6


class Qwen3ASRThinkerTextMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Qwen3ASRThinkerTextMLP()])
        self.norm = Qwen3ASRThinkerTextRMSNorm()


class ASRModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.thinker = nn.Module()
        self.thinker.model = TextModel()


def test_qwen_specific_liger_adapter_patches_existing_instances(monkeypatch) -> None:
    class FakeRMS:
        def forward(self, value):
            return value

    class FakeSwiGLU:
        def forward(self, value):
            return value

    fake_modeling = SimpleNamespace(apply_rotary_pos_emb=lambda *args: args)
    monkeypatch.setattr(
        optimizations,
        "_load_liger_components",
        lambda: (FakeRMS, FakeSwiGLU, lambda *args: args),
    )
    monkeypatch.setattr(optimizations.importlib, "import_module", lambda name: fake_modeling)

    model = ASRModel()
    report = optimizations.apply_liger_kernels(model)
    assert report.rms_norms == 1
    assert report.swiglu_mlps == 1
    assert report.rope is True
    assert model.thinker.model.norm.in_place is True


def test_compile_targets_the_modules_executed_by_distillation() -> None:
    class CompilableLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.compile_kwargs = None

        def compile(self, **kwargs) -> None:
            self.compile_kwargs = kwargs

    class CompilableTextModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([CompilableLayer(), CompilableLayer()])
            self.compile_kwargs = None

        def compile(self, **kwargs) -> None:
            self.compile_kwargs = kwargs

    model = nn.Module()
    model.thinker = nn.Module()
    model.thinker.model = CompilableTextModel()

    report = optimizations.compile_training_modules(model, scope="text_model")
    assert report.modules == 1
    assert model.thinker.model.compile_kwargs["backend"] == "inductor"
    assert all(layer.compile_kwargs is None for layer in model.thinker.model.layers)

    report = optimizations.compile_training_modules(
        model, scope="decoder_layers", mode="reduce-overhead", dynamic=False
    )
    assert report.modules == 2
    assert all(
        layer.compile_kwargs["mode"] == "reduce-overhead" for layer in model.thinker.model.layers
    )


def test_compile_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="compile scope"):
        optimizations.compile_training_modules(ASRModel(), scope="everything")
