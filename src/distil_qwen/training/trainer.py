"""Transformers Trainer integration for Qwen3-ASR distillation."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import Trainer

from distil_qwen.config import DistillationConfig
from distil_qwen.training.cache import AudioFeatureCache
from distil_qwen.training.engine import configure_teacher, distillation_step


class DistillationTrainer(Trainer):
    """Train only the student while keeping a colocated, frozen teacher."""

    def __init__(
        self,
        *args: Any,
        teacher_model: torch.nn.Module,
        distillation_config: DistillationConfig,
        audio_feature_cache: Optional[AudioFeatureCache] = None,
        activation_offload: bool = False,
        **kwargs: Any,
    ) -> None:
        self.teacher_model = teacher_model
        self.distillation_config = distillation_config
        self.audio_feature_cache = audio_feature_cache
        self.activation_offload = activation_offload
        self._latest_distillation_metrics: Dict[str, float] = {}
        configure_teacher(self.teacher_model)
        super().__init__(*args, **kwargs)
        if not getattr(self.teacher_model, "hf_device_map", None):
            self._move_model_to_device(self.teacher_model, self.args.device)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Any:
        del num_items_in_batch
        offload_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if self.activation_offload and model.training
            else nullcontext()
        )
        with offload_context:
            output = distillation_step(
                student=model,
                teacher=self.teacher_model,
                batch=inputs,
                config=self.distillation_config,
                audio_cache=self.audio_feature_cache,
            )
        self._latest_distillation_metrics = {
            "ce_loss": output.ce_loss.detach().float().item(),
            "kl_loss": output.kl_loss.detach().float().item(),
            "supervised_tokens": float(output.num_tokens),
        }
        if self.audio_feature_cache is not None:
            stats = self.audio_feature_cache.stats()
            self._latest_distillation_metrics["audio_cache_hit_rate"] = stats.hit_rate
        if return_outputs:
            return output.loss, {"loss": output.loss.detach()}
        return output.loss

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is None:
            model_dtype = next(self.model.parameters()).dtype
        return {
            key: value.to(dtype=model_dtype)
            if torch.is_tensor(value) and value.is_floating_point()
            else value
            for key, value in inputs.items()
        }

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if "loss" in logs and self._latest_distillation_metrics:
            logs = {**logs, **self._latest_distillation_metrics}
        super().log(logs, start_time=start_time)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if self.args.should_save:
            destination = Path(output_dir or self.args.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "distillation_config.json").write_text(
                json.dumps(self.distillation_config.to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
