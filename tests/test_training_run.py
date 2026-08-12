import pytest

from distil_qwen.training.run import TrainingRunConfig


def test_training_optimization_defaults_preserve_full_precision() -> None:
    config = TrainingRunConfig(student="student")
    assert config.optimizer == "auto"
    assert config.group_by_length is True
    assert config.gradient_checkpointing_preserve_rng_state is False
    assert config.dataloader_non_blocking is True
    assert config.ddp_broadcast_buffers is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"compile_scope": "all"}, "compile_scope"),
        ({"dataloader_prefetch_factor": 0}, "prefetch"),
        ({"length_preprocessing_workers": 0}, "length_preprocessing_workers"),
        ({"ddp_bucket_cap_mb": 0}, "ddp_bucket_cap_mb"),
    ],
)
def test_training_optimization_configuration_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        TrainingRunConfig(student="student", **kwargs)
