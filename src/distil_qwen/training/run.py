"""High-level assembly of optimized Qwen3-ASR distillation training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from distil_qwen.compat import require_qwen_asr
from distil_qwen.config import DistillationConfig
from distil_qwen.models.accessors import get_audio_tower
from distil_qwen.training.cache import AudioFeatureCache, audio_cache_namespace
from distil_qwen.training.collator import Qwen3ASRDataCollator
from distil_qwen.training.data import add_length_column, load_splits
from distil_qwen.training.engine import configure_student_trainability
from distil_qwen.training.optimizations import (
    apply_liger_kernels,
    compile_training_modules,
    resolve_attention,
    resolve_liger,
    resolve_optimizer,
)
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
    cache_key_column: Optional[str] = None
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
    dataloader_prefetch_factor: int = 2
    dataloader_non_blocking: bool = True
    group_by_length: bool = True
    length_column_name: str = "__distil_qwen_length"
    length_preprocessing_workers: Optional[int] = None
    attention: str = "auto"
    dtype: str = "auto"
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    gradient_checkpointing_preserve_rng_state: bool = False
    freeze_audio_tower: bool = True
    freeze_embeddings: bool = True
    compile_model: bool = False
    compile_mode: str = "default"
    compile_scope: str = "text_model"
    liger: str = "auto"
    optimizer: str = "auto"
    tf32: bool = True
    activation_offload: bool = False
    ddp_bucket_cap_mb: Optional[int] = None
    ddp_broadcast_buffers: bool = False
    audio_cache_dir: Optional[str] = None
    audio_cache_memory_mb: int = 0
    seed: int = 42
    report_to: str = "none"
    resume_from_checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.liger not in {"auto", "on", "off"}:
            raise ValueError("liger must be 'auto', 'on', or 'off'")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError(f"unsupported compile mode: {self.compile_mode}")
        if self.compile_scope not in {"text_model", "decoder_layers"}:
            raise ValueError("compile_scope must be 'text_model' or 'decoder_layers'")
        if self.audio_cache_memory_mb < 0:
            raise ValueError("audio_cache_memory_mb cannot be negative")
        if self.dataloader_prefetch_factor < 1:
            raise ValueError("dataloader_prefetch_factor must be positive")
        if self.length_preprocessing_workers is not None and self.length_preprocessing_workers < 1:
            raise ValueError("length_preprocessing_workers must be positive")
        if self.ddp_bucket_cap_mb is not None and self.ddp_bucket_cap_mb < 1:
            raise ValueError("ddp_bucket_cap_mb must be positive")
        if self.per_device_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch size and gradient accumulation must be positive")


def _runtime_dtype(requested: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _audio_tower_config(model: torch.nn.Module) -> Any:
    config = getattr(get_audio_tower(model), "config", None)
    return config.to_dict() if hasattr(config, "to_dict") else config


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
    attention = resolve_attention(run.attention, dtype)
    use_liger = resolve_liger(run.liger)
    optimizer = resolve_optimizer(run.optimizer)
    if run.activation_offload and not torch.cuda.is_available():
        raise ValueError("activation offload is only useful with a CUDA training device")
    common_kwargs = {
        "dtype": dtype,
        "attn_implementation": attention,
        "low_cpu_mem_usage": True,
    }
    student = AutoModel.from_pretrained(run.student, **common_kwargs)

    teacher = AutoModel.from_pretrained(run.teacher, **common_kwargs)
    processor = AutoProcessor.from_pretrained(run.student, fix_mistral_regex=True)

    if use_liger:
        student_report = apply_liger_kernels(student)
        teacher_report = apply_liger_kernels(teacher)
        LOGGER.info("Liger student patch: %s", student_report)
        LOGGER.info("Liger teacher patch: %s", teacher_report)

    trainable = configure_student_trainability(
        student,
        freeze_audio_tower=run.freeze_audio_tower,
        freeze_embeddings=run.freeze_embeddings,
    )
    if distillation.reuse_audio_features and not run.freeze_audio_tower:
        raise ValueError("reuse_audio_features requires a frozen student audio tower")
    if distillation.reuse_audio_features and _audio_tower_config(student) != _audio_tower_config(
        teacher
    ):
        raise ValueError(
            "reuse_audio_features requires identical student and teacher audio architectures; "
            "disable it for a reduced audio tower"
        )
    cache_enabled = run.audio_cache_memory_mb > 0 or run.audio_cache_dir is not None
    if cache_enabled and not distillation.reuse_audio_features:
        raise ValueError("audio feature caching requires reuse_audio_features")
    checkpointing_kwargs = {
        "use_reentrant": run.gradient_checkpointing_use_reentrant,
        "preserve_rng_state": run.gradient_checkpointing_preserve_rng_state,
    }
    if run.gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing_kwargs)
    if run.compile_model:
        student_compile = compile_training_modules(
            student,
            scope=run.compile_scope,
            mode=run.compile_mode,
        )
        LOGGER.info("Compiled student modules: %s", student_compile)
        teacher_compile = compile_training_modules(
            teacher,
            scope=run.compile_scope,
            mode=run.compile_mode,
        )
        LOGGER.info("Compiled teacher modules: %s", teacher_compile)
    LOGGER.info("Trainable student parameters: %s", f"{trainable:,}")
    train_data, eval_data = load_splits(
        dataset=run.dataset,
        dataset_config=run.dataset_config,
        train_split=run.train_split,
        eval_split=run.eval_split,
    )
    if run.group_by_length:
        train_data = add_length_column(
            train_data,
            audio_column=run.audio_column,
            text_column=run.text_column,
            prompt_column=run.prompt_column,
            length_column=run.length_column_name,
            num_proc=run.length_preprocessing_workers,
        )
    collator = Qwen3ASRDataCollator(
        processor=processor,
        audio_column=run.audio_column,
        text_column=run.text_column,
        prompt_column=run.prompt_column,
        include_audio_cache_keys=cache_enabled,
        cache_key_column=run.cache_key_column,
    )
    audio_cache = None
    if cache_enabled:
        audio_cache = AudioFeatureCache(
            max_memory_mb=run.audio_cache_memory_mb,
            cache_dir=run.audio_cache_dir,
            namespace=audio_cache_namespace(run.teacher, getattr(teacher, "config", None)),
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
        optim=optimizer,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=run.tf32 and torch.cuda.is_available(),
        gradient_checkpointing=run.gradient_checkpointing,
        gradient_checkpointing_kwargs=checkpointing_kwargs,
        # Compilation is applied in-place to the inner modules used by distillation_step.
        torch_compile=False,
        eval_strategy="steps" if eval_data is not None else "no",
        eval_steps=run.eval_steps if eval_data is not None else None,
        save_strategy="steps",
        save_steps=run.save_steps,
        save_total_limit=run.save_total_limit,
        logging_steps=run.logging_steps,
        dataloader_num_workers=run.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=run.dataloader_num_workers > 0,
        dataloader_prefetch_factor=(
            run.dataloader_prefetch_factor if run.dataloader_num_workers > 0 else None
        ),
        group_by_length=run.group_by_length,
        length_column_name=run.length_column_name,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=run.ddp_bucket_cap_mb,
        ddp_broadcast_buffers=run.ddp_broadcast_buffers,
        accelerator_config={
            "non_blocking": run.dataloader_non_blocking and torch.cuda.is_available()
        },
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
        audio_feature_cache=audio_cache,
        activation_offload=run.activation_offload,
    )
    trainer.train(resume_from_checkpoint=run.resume_from_checkpoint)
    trainer.save_model(run.output_dir)
    return trainer
