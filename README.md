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

For local inference use `pip install -e ".[inference]"`. Add `vllm` for the in-process vLLM
engine, `server` for vLLM/SGLang/llama.cpp endpoints, or `quantization` when needed.

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

## Inference backends

Every runtime uses `OptimizedASR`, `InferenceConfig`, and the same ordered list of
`Transcription` results:

| Backend | Mode | Model | Notes |
|---|---|---|---|
| `transformers` | local | distilled checkpoint or `Qwen/Qwen3-ASR-1.7B` | default; SDPA/FA2, KV cache, compile |
| `vllm` | local or server | distilled checkpoint | continuous batching; server mode uses `/v1/audio/transcriptions` |
| `sglang` | server | distilled checkpoint | high-throughput `/v1/audio/transcriptions` |
| `llama_cpp` | server | converted GGUF | CPU/GPU GGUF inference through multimodal chat |

Start a server, then point the same CLI at it:

```bash
# vLLM (the qwen-asr wrapper passes through vLLM serve arguments)
qwen-asr-serve outputs/student --host 0.0.0.0 --port 8000
distil-qwen transcribe audio/example.wav --model outputs/student \
  --backend vllm --server-url http://localhost:8000/v1

# SGLang
python -m sglang.launch_server --model-path outputs/student --host 0.0.0.0 --port 30000
distil-qwen transcribe audio/example.wav --model outputs/student \
  --backend sglang --server-url http://localhost:30000/v1

# llama.cpp (use a converted student GGUF + multimodal projector)
llama-server -m student.gguf --mmproj mmproj-student.gguf --host 0.0.0.0 --port 8080
distil-qwen transcribe audio/example.wav --model student \
  --backend llama_cpp --server-url http://localhost:8080/v1
```

For server backends, `--model` may be omitted when `/v1/models` advertises exactly one loaded
model. Keep it explicit for routers serving multiple models.

For an immediate llama.cpp test, its official pre-converted teacher can be launched with
`llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF`. The client validates the server's audio
capability before inference. See [inference.md](docs/inference.md) for native Transformers 5,
batching, context, timestamps, conversion, and server tuning.

## What is optimized

- Frozen audio features are shared by teacher and student and can be cached across epochs.
- Training batches are grouped by estimated post-encoder audio and transcript length to reduce
  padding, with non-blocking device transfer and persistent worker prefetching.
- Only supervised tokens are projected to the 151k-token vocabulary.
- Frozen-head KL/CE retains hidden gradients instead of vocabulary logits; an optional Liger fused
  loss covers trainable heads.
- Liger RMSNorm/SwiGLU/RoPE, BF16, FlashAttention 2, fused AdamW, non-reentrant checkpointing,
  reduced RNG bookkeeping, and TF32 are selected when supported.
- Optional compilation targets the Qwen text modules actually invoked by distillation and keeps
  checkpoint state-dict keys unchanged.
- Inference supports Transformers, vLLM, SGLang, and llama.cpp; KV caching, optional compilation,
  4/8-bit loading, continuous/server batching, and bounded concurrent requests are available where
  the backend supports them.
- Pseudo-labels can be filtered by WER/CER and repeated n-grams.

See [training.md](docs/training.md) for the full workflow and [research.md](docs/research.md) for the
design rationale.

## License

Apache-2.0. Qwen3-ASR is also distributed under Apache-2.0.
