import pytest

from distil_qwen.cli import build_parser, main


def test_parser_accepts_minimal_commands() -> None:
    parser = build_parser()
    init_args = parser.parse_args(["init", "--output-dir", "out"])
    assert init_args.text_layers == 14
    train_args = parser.parse_args(["train", "--student", "student", "--dataset", "data.jsonl"])
    assert train_args.freeze_audio_tower is True
    assert train_args.group_by_length is True
    assert train_args.gradient_checkpointing_preserve_rng_state is False
    assert train_args.compile_scope == "text_model"
    assert train_args.ddp_broadcast_buffers is False
    inference_args = parser.parse_args(["transcribe", "a.wav", "--model", "model"])
    assert inference_args.batch_size == 8
    server_args = parser.parse_args(
        [
            "transcribe",
            "a.wav",
            "--model",
            "model",
            "--backend",
            "sglang",
            "--server-url",
            "http://localhost:30000/v1",
        ]
    )
    assert server_args.server_url == "http://localhost:30000/v1"


def test_main_reports_version(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    assert "distil-qwen" in capsys.readouterr().out
