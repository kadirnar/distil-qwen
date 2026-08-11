import torch
from torch import nn
from transformers import TrainingArguments

from distil_qwen.config import DistillationConfig
from distil_qwen.training.trainer import DistillationTrainer


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)

    def forward(self, input_features=None, labels=None):
        del labels
        return self.projection(input_features)


def test_trainer_casts_only_floating_inputs_to_model_dtype(tmp_path) -> None:
    student = TinyModel().to(torch.float16)
    teacher = TinyModel().to(torch.float16)
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        distillation_config=DistillationConfig(),
        args=TrainingArguments(output_dir=str(tmp_path), report_to="none"),
    )
    prepared = trainer._prepare_inputs(
        {
            "input_features": torch.ones(1, 2, dtype=torch.float32),
            "input_ids": torch.ones(1, 2, dtype=torch.long),
        }
    )
    assert prepared["input_features"].dtype == torch.float16
    assert prepared["input_ids"].dtype == torch.long
