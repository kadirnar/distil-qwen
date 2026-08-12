import json
from types import SimpleNamespace

from distil_qwen.config import InferenceConfig
from distil_qwen.data import pseudo_label as pseudo_module


class FakeASR:
    def __init__(self) -> None:
        self.batch_sizes = []

    def transcribe(self, audio, context, language):
        assert language == "English"
        assert len(audio) == len(context)
        self.batch_sizes.append(len(audio))
        return [SimpleNamespace(text=f"label-{item}", language="English") for item in audio]

    def close(self):
        return None


def test_pseudo_label_jsonl_streams_in_requested_batches(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text(
        "".join(
            json.dumps({"audio": f"{index}.wav", "prompt": f"p{index}"}) + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    fake = FakeASR()
    monkeypatch.setattr(
        pseudo_module.OptimizedASR,
        "from_pretrained",
        lambda *args, **kwargs: fake,
    )

    count = pseudo_module.pseudo_label_jsonl(
        str(source),
        str(output),
        inference_config=InferenceConfig(batch_size=2),
        language="English",
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 3
    assert fake.batch_sizes == [2, 1]
    assert records[2]["teacher_text"] == "label-2.wav"
    assert records[2]["teacher_language"] == "English"
