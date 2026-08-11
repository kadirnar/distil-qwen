import pytest
import torch
from transformers import AutoModel

pytestmark = pytest.mark.qwen_contract
pytest.importorskip("qwen_asr")

from qwen_asr.core.transformers_backend import (  # noqa: E402
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
)

from distil_qwen import DistillationConfig, StudentSpec, initialize_student  # noqa: E402
from distil_qwen.training.engine import (  # noqa: E402
    configure_student_trainability,
    configure_teacher,
    distillation_step,
)


def tiny_qwen_config() -> Qwen3ASRConfig:
    return Qwen3ASRConfig(
        thinker_config={
            "audio_config": {
                "num_mel_bins": 8,
                "encoder_layers": 2,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "d_model": 8,
                "max_source_positions": 100,
                "n_window": 10,
                "n_window_infer": 20,
                "conv_chunksize": 10,
                "output_dim": 8,
                "downsample_hidden_size": 4,
            },
            "text_config": {
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "max_position_embeddings": 128,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 0]},
                "pad_token_id": 0,
                "tie_word_embeddings": True,
            },
            "audio_token_id": 1,
            "audio_start_token_id": 2,
        },
        support_languages=["English"],
        pad_token_id=0,
    )


def test_real_qwen_initialization_and_shared_audio_distillation(tmp_path) -> None:
    teacher = Qwen3ASRForConditionalGeneration(tiny_qwen_config())
    student, report = initialize_student(teacher, StudentSpec(text_layers=2, audio_layers=1))
    assert report.text_layer_map == (0, 3)
    assert report.audio_layer_map == (1,)
    assert len(student.thinker.model.layers) == 2
    assert len(student.thinker.audio_tower.layers) == 1
    student.save_pretrained(tmp_path, safe_serialization=True)
    student = AutoModel.from_pretrained(tmp_path, local_files_only=True)
    assert len(student.thinker.model.layers) == 2

    configure_student_trainability(student)
    configure_teacher(teacher)
    batch = {
        "input_ids": torch.tensor([[3, 1, 1, 1, 4, 5]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "input_features": torch.randn(1, 8, 20),
        "feature_attention_mask": torch.ones(1, 20, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, -100, 4, 5]]),
    }
    output = distillation_step(
        student,
        teacher,
        batch,
        DistillationConfig(logit_chunk_size=1),
    )
    output.loss.backward()
    assert output.num_tokens == 2
    assert torch.isfinite(output.loss)
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
