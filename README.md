# distil-qwen

Efficient knowledge distillation for **Qwen3-ASR-1.7B**.

`distil-qwen` keeps Qwen3-ASR's acoustic tower intact by default and reduces its autoregressive
decoder with uniformly selected teacher layers. Training combines pseudo-label cross-entropy with
exact logit KL divergence.

> This repository provides the distillation library, not a pretrained student checkpoint. An
> initialized student must be trained before it is useful for transcription.

## Install

```bash
pip install -e ".[train]"
```

On a CUDA training host, add Liger and FlashAttention:

```bash
pip install -e ".[train,kernels]"
MAX_JOBS=4 pip install flash-attn==2.8.3.post1 --no-build-isolation
```

For inference only, use `pip install -e ".[inference]"`. Add `quantization` or `vllm` when needed.

## Quick start

Initialize the balanced 14-layer student from the 28-layer teacher:

```bash
distil-qwen init \
  --teacher Qwen/Qwen3-ASR-1.7B \
  --text-layers 14 \
  --output-dir outputs/student-init
```

Train from a JSONL manifest or Hugging Face dataset:

```bash
accelerate launch -m distil_qwen.cli train \
  --student outputs/student-init \
  --dataset data/train.jsonl \
  --output-dir outputs/student
```

Each row needs `audio` and `text`; `prompt` is optional:

```json
{"audio":"audio/example.wav","text":"The transcript.","prompt":""}
```

Transcribe with optimized greedy decoding:

```bash
distil-qwen transcribe audio/example.wav --model outputs/student --language English
```

Python uses the same API:

```python
from distil_qwen import InferenceConfig
from distil_qwen.inference import OptimizedASR

asr = OptimizedASR.from_pretrained(
    "outputs/student",
    InferenceConfig(batch_size=8, attention="auto"),
)
result = asr.transcribe("audio/example.wav", language="English")[0]
print(result.text)
```

## What is optimized

- Frozen audio features are shared by teacher and student and can be cached across epochs.
- Only supervised tokens are projected to the 151k-token vocabulary.
- Frozen-head KL/CE retains hidden gradients instead of vocabulary logits; an optional Liger fused
  loss covers trainable heads.
- Liger RMSNorm/SwiGLU/RoPE, BF16, FlashAttention 2, fused AdamW, non-reentrant checkpointing, and
  TF32 are selected when supported.
- Inference supports KV caching, optional compilation, 4/8-bit loading, and vLLM.
- Pseudo-labels can be filtered by WER/CER and repeated n-grams.

See [training.md](docs/training.md) for the full workflow and [research.md](docs/research.md) for the
design rationale.

## License

Apache-2.0. Qwen3-ASR is also distributed under Apache-2.0.
