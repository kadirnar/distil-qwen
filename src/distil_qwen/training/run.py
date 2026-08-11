"""High-level assembly of optimized Qwen3-ASR distillation training."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

from distil_qwen.compat import flash_attention_available, require_qwen_asr
from distil_qwen.config import DistillationConfig
from distil_qwen.training.collator import Qwen3ASRDataCollator
from distil_qwen.training.data import load_splits
from distil_qwen.training.engine import configure_student_trainability
from distil_qwen.training.trainer import DistillationTrainer

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingRunConfig:
    student: str
    teacher: str = "Qwen/Qwen3-ASR-1.7B"
    dataset: str = ""
    output_dir: str = "outputs/distil-qwen"
    dataset_config: Optional[str] = None
    train_split: str = "train"
    eval_split: Optional[str] = None
    audio_column: str = "audio"
    text_column: str = "text"
    prompt_column: str = "prompt"
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    num_train_epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    dataloader_num_workers: int = 4
    attention: str = "auto"
    dtype: str = "auto"
    teacher_quantization: Optional[str] = None
    gradient_checkpointing: bool = True
    freeze_audio_tower: bool = True
    freeze_embeddings: bool = True
    compile_model: bool = False
    seed: int = 42
    report_to: str = "none"
    resume_from_checkpoint: Optional[str] = None


def _runtime_dtype(requested: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _attention_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available() and flash_attention_available():
        return "flash_attention_2"
    return "sdpa"


def _quantization_config(mode: Optional[str], dtype: torch.dtype) -> Any:
    if mode is None:
        return None
    if not torch.cuda.is_available():
        raise ValueError("teacher quantization requires CUDA")
    from transformers import BitsAndBytesConfig

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError("teacher_quantization must be None, '4bit', or '8bit'")


def run_training(
    run: TrainingRunConfig,
    distillation: Optional[DistillationConfig] = None,
) -> DistillationTrainer:
    """Load all components, train, save, and return the Trainer for inspection."""

    distillation = distillation or DistillationConfig()
    if not run.dataset:
        raise ValueError("dataset is required")
    require_qwen_asr()
    from transformers import AutoModel, AutoProcessor, TrainingArguments, set_seed

    set_seed(run.seed)
    dtype = _runtime_dtype(run.dtype)
    attention = _attention_backend(run.attention)
    common_kwargs = {
        "dtype": dtype,
        "attn_implementation": attention,
        "low_cpu_mem_usage": True,
    }
    student = AutoModel.from_pretrained(run.student, **common_kwargs)

    teacher_kwargs = dict(common_kwargs)
    quantization_config = _quantization_config(run.teacher_quantization, dtype)
    if quantization_config is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        teacher_kwargs.update(
            quantization_config=quantization_config,
            device_map={"": local_rank},
        )
    teacher = AutoModel.from_pretrained(run.teacher, **teacher_kwargs)
    processor = AutoProcessor.from_pretrained(run.student, fix_mistral_regex=True)

    trainable = configure_student_trainability(
        student,
        freeze_audio_tower=run.freeze_audio_tower,
        freeze_embeddings=run.freeze_embeddings,
    )
    if distillation.reuse_audio_features and not run.freeze_audio_tower:
        raise ValueError("reuse_audio_features requires a frozen student audio tower")
    if run.gradient_checkpointing:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    LOGGER.info("Trainable student parameters: %s", f"{trainable:,}")
    train_data, eval_data = load_splits(
        dataset=run.dataset,
        dataset_config=run.dataset_config,
        train_split=run.train_split,
        eval_split=run.eval_split,
    )
    collator = Qwen3ASRDataCollator(
        processor=processor,
        audio_column=run.audio_column,
        text_column=run.text_column,
        prompt_column=run.prompt_column,
    )

    use_bf16 = dtype == torch.bfloat16
    use_fp16 = dtype == torch.float16 and not torch.backends.mps.is_available()
    training_args = TrainingArguments(
        output_dir=run.output_dir,
        per_device_train_batch_size=run.per_device_batch_size,
        per_device_eval_batch_size=run.per_device_batch_size,
        gradient_accumulation_steps=run.gradient_accumulation_steps,
        learning_rate=run.learning_rate,
        num_train_epochs=run.num_train_epochs,
        max_steps=run.max_steps,
        warmup_ratio=run.warmup_ratio,
        weight_decay=run.weight_decay,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=run.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_compile=run.compile_model,
        torch_compile_mode="default",
        torch_compile_backend="inductor",
        eval_strategy="steps" if eval_data is not None else "no",
        eval_steps=run.eval_steps if eval_data is not None else None,
        save_strategy="steps",
        save_steps=run.save_steps,
        save_total_limit=run.save_total_limit,
        logging_steps=run.logging_steps,
        dataloader_num_workers=run.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=run.dataloader_num_workers > 0,
        dataloader_prefetch_factor=2 if run.dataloader_num_workers > 0 else None,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        prediction_loss_only=True,
        save_safetensors=True,
        report_to=run.report_to,
        seed=run.seed,
    )
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        distillation_config=distillation,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=run.resume_from_checkpoint)
    trainer.save_model(run.output_dir)
    return trainer
