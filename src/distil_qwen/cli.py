"""Command-line entry point; all behavior also remains available as library APIs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from typing import Optional, Sequence

from distil_qwen import __version__
from distil_qwen.config import DistillationConfig, InferenceConfig, StudentSpec


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--attention",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--quantization", choices=("4bit", "8bit"))
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)


def _inference_config(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        attention=args.attention,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        quantization=args.quantization,
        compile_model=args.compile_model,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distil-qwen", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="initialize a reduced student checkpoint")
    initialize.add_argument("--teacher", default="Qwen/Qwen3-ASR-1.7B")
    initialize.add_argument("--output-dir", required=True)
    initialize.add_argument("--text-layers", type=int, default=14)
    initialize.add_argument("--audio-layers", type=int)
    initialize.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    initialize.add_argument(
        "--attention",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    initialize.add_argument("--revision", default="main")

    train = commands.add_parser("train", help="run online teacher-student distillation")
    train.add_argument("--student", required=True)
    train.add_argument("--teacher", default="Qwen/Qwen3-ASR-1.7B")
    train.add_argument("--dataset", required=True)
    train.add_argument("--dataset-config")
    train.add_argument("--train-split", default="train")
    train.add_argument("--eval-split")
    train.add_argument("--output-dir", default="outputs/distil-qwen")
    train.add_argument("--audio-column", default="audio")
    train.add_argument("--text-column", default="text")
    train.add_argument("--prompt-column", default="prompt")
    train.add_argument("--cache-key-column")
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--epochs", type=float, default=3.0)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--warmup-ratio", type=float, default=0.03)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--logging-steps", type=int, default=10)
    train.add_argument("--save-steps", type=int, default=500)
    train.add_argument("--eval-steps", type=int, default=500)
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument(
        "--attention",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    train.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    train.add_argument("--teacher-quantization", choices=("4bit", "8bit"))
    train.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    train.add_argument(
        "--gradient-checkpointing-use-reentrant",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train.add_argument("--freeze-audio-tower", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument(
        "--reuse-audio-features", action=argparse.BooleanOptionalAction, default=True
    )
    train.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    train.add_argument("--liger", choices=("auto", "on", "off"), default="auto")
    train.add_argument(
        "--optimizer",
        choices=(
            "auto",
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_bnb_8bit",
            "paged_adamw_8bit",
        ),
        default="auto",
    )
    train.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--activation-offload", action="store_true")
    train.add_argument("--audio-cache-dir")
    train.add_argument("--audio-cache-memory-mb", type=int, default=0)
    train.add_argument("--temperature", type=float, default=2.0)
    train.add_argument("--ce-weight", type=float, default=1.0)
    train.add_argument("--kl-weight", type=float, default=0.8)
    train.add_argument("--logit-chunk-size", type=int, default=32)
    train.add_argument("--label-smoothing", type=float, default=0.0)
    train.add_argument("--distillation-backend", choices=("auto", "torch", "liger"), default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--report-to", default="none")
    train.add_argument("--resume-from-checkpoint")

    transcribe = commands.add_parser("transcribe", help="transcribe one or more audio files")
    transcribe.add_argument("audio", nargs="+")
    transcribe.add_argument("--model", required=True)
    transcribe.add_argument("--language")
    transcribe.add_argument("--context", default="")
    _add_inference_arguments(transcribe)

    pseudo = commands.add_parser("pseudo-label", help="add teacher labels to a JSONL manifest")
    pseudo.add_argument("--input", required=True)
    pseudo.add_argument("--output", required=True)
    pseudo.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    pseudo.add_argument("--audio-column", default="audio")
    pseudo.add_argument("--prompt-column", default="prompt")
    pseudo.add_argument("--language")
    _add_inference_arguments(pseudo)

    quality_filter = commands.add_parser(
        "filter", help="filter pseudo-labels by error rate and repetition"
    )
    quality_filter.add_argument("--input", required=True)
    quality_filter.add_argument("--output", required=True)
    quality_filter.add_argument("--reference-column", default="text")
    quality_filter.add_argument("--pseudo-column", default="teacher_text")
    quality_filter.add_argument("--metric", choices=("auto", "wer", "cer"), default="auto")
    quality_filter.add_argument("--max-error-rate", type=float, default=0.2)
    quality_filter.add_argument("--max-repeated-ngram-ratio", type=float, default=0.3)

    benchmark = commands.add_parser("benchmark", help="measure latency and real-time factor")
    benchmark.add_argument("audio", nargs="+")
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--warmup-runs", type=int, default=1)
    benchmark.add_argument("--measured-runs", type=int, default=3)
    _add_inference_arguments(benchmark)
    return parser


def _run_init(args: argparse.Namespace) -> None:
    from distil_qwen.models.initialization import initialize_from_pretrained

    report = initialize_from_pretrained(
        teacher_name_or_path=args.teacher,
        output_dir=args.output_dir,
        spec=StudentSpec(text_layers=args.text_layers, audio_layers=args.audio_layers),
        dtype=args.dtype,
        attention=args.attention,
        revision=args.revision,
    )
    print(json.dumps(report.to_dict(), indent=2))


def _run_train(args: argparse.Namespace) -> None:
    from distil_qwen.training.run import TrainingRunConfig, run_training

    run = TrainingRunConfig(
        student=args.student,
        teacher=args.teacher,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        output_dir=args.output_dir,
        audio_column=args.audio_column,
        text_column=args.text_column,
        prompt_column=args.prompt_column,
        cache_key_column=args.cache_key_column,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.workers,
        attention=args.attention,
        dtype=args.dtype,
        teacher_quantization=args.teacher_quantization,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_use_reentrant=args.gradient_checkpointing_use_reentrant,
        freeze_audio_tower=args.freeze_audio_tower,
        freeze_embeddings=args.freeze_embeddings,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        liger=args.liger,
        optimizer=args.optimizer,
        tf32=args.tf32,
        activation_offload=args.activation_offload,
        audio_cache_dir=args.audio_cache_dir,
        audio_cache_memory_mb=args.audio_cache_memory_mb,
        seed=args.seed,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    distillation = DistillationConfig(
        temperature=args.temperature,
        ce_weight=args.ce_weight,
        kl_weight=args.kl_weight,
        logit_chunk_size=args.logit_chunk_size,
        label_smoothing=args.label_smoothing,
        reuse_audio_features=args.reuse_audio_features,
        loss_backend=args.distillation_backend,
    )
    run_training(run, distillation)


def _run_transcribe(args: argparse.Namespace) -> None:
    from distil_qwen.inference import OptimizedASR

    model = OptimizedASR.from_pretrained(args.model, _inference_config(args))
    results = model.transcribe(args.audio, context=args.context, language=args.language)
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


def _run_pseudo_label(args: argparse.Namespace) -> None:
    from distil_qwen.data.pseudo_label import pseudo_label_jsonl

    count = pseudo_label_jsonl(
        input_path=args.input,
        output_path=args.output,
        model_name_or_path=args.model,
        inference_config=_inference_config(args),
        audio_column=args.audio_column,
        prompt_column=args.prompt_column,
        language=args.language,
    )
    print(json.dumps({"labeled": count, "output": args.output}, indent=2))


def _run_filter(args: argparse.Namespace) -> None:
    from distil_qwen.data.filtering import filter_jsonl

    report = filter_jsonl(
        input_path=args.input,
        output_path=args.output,
        reference_column=args.reference_column,
        pseudo_column=args.pseudo_column,
        metric=args.metric,
        max_error_rate=args.max_error_rate,
        max_repeated_ngram_ratio=args.max_repeated_ngram_ratio,
    )
    payload = {**asdict(report), "keep_rate": report.keep_rate, "output": args.output}
    print(json.dumps(payload, indent=2))


def _run_benchmark(args: argparse.Namespace) -> None:
    from distil_qwen.benchmark import benchmark
    from distil_qwen.inference import OptimizedASR

    model = OptimizedASR.from_pretrained(args.model, _inference_config(args))
    result = benchmark(
        model,
        args.audio,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )
    print(json.dumps(result.to_dict(), indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    handlers = {
        "init": _run_init,
        "train": _run_train,
        "transcribe": _run_transcribe,
        "pseudo-label": _run_pseudo_label,
        "filter": _run_filter,
        "benchmark": _run_benchmark,
    }
    handlers[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
