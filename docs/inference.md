# Inference

`OptimizedASR` keeps one API across local PyTorch, GPU servers, and GGUF deployment. Backend-only
packages are imported lazily, so installing the lightweight server client does not install vLLM or
SGLang into the application process.

## Python API

```python
from distil_qwen import InferenceConfig, OptimizedASR

config = InferenceConfig(
    backend="vllm",
    server_url="http://localhost:8000/v1",
    batch_size=16,  # local batch limit or remote request concurrency
    request_timeout=120,
)

with OptimizedASR.from_pretrained("outputs/student", config) as asr:
    results = asr.transcribe(
        ["audio/one.wav", "audio/two.wav"],
        context="Vocabulary: Qwen, Istanbul.",
        language="Turkish",
    )
```

Server backends accept local paths, HTTP(S) URLs, bytes, or `(numpy_array, sample_rate)` tuples.
Scalar context/language values are broadcast across a batch; lists must exactly match its size.
Set the environment variable named by `api_key_env` when a server requires bearer authentication.
The model name is optional when `/v1/models` advertises exactly one loaded model.

## Transformers

The default implementation loads distilled and original `Qwen/Qwen3-ASR-1.7B` checkpoints through
the official `qwen-asr` runtime. It enables the KV cache, inference mode, TF32 on CUDA, automatic
BF16/FP16 selection, FlashAttention 2 when installed, and optional `torch.compile`.

Native Qwen3-ASR support is also available in Transformers 5 for `*-hf` checkpoints. Because
`qwen-asr==0.0.6` pins Transformers 4.57.6, install this mode in a separate environment:

```bash
python -m venv .venv-native
. .venv-native/bin/activate
pip install -e ".[transformers-native]"
```

Then load `Qwen/Qwen3-ASR-1.7B-hf`; the backend is detected automatically. Pass
`implementation="native"` only for a checkpoint already stored in the native Transformers format.
The distilled checkpoints produced by this repository use the `qwen-asr` format by default.

## vLLM

`backend="vllm"` without `server_url` uses the official `qwen-asr` in-process vLLM wrapper:

```python
config = InferenceConfig(backend="vllm", batch_size=64, dtype="bfloat16")
asr = OptimizedASR.from_pretrained(
    "outputs/student",
    config,
    gpu_memory_utilization=0.85,
)
```

With `server_url`, the backend uses the OpenAI transcription endpoint. This separates heavyweight
GPU dependencies from clients and lets vLLM schedule concurrent requests continuously. Context is
sent as the OpenAI `prompt` field; timestamps require a server configured with forced alignment.

## SGLang

Install SGLang 0.5.17 or newer in its own CUDA environment, then use `backend="sglang"` plus
`server_url`. The current Qwen3-ASR adapter supports the OpenAI transcription and realtime routes.
Its HTTP transcription adapter does not yet map context to the Qwen system prompt, force a requested
language, or return forced-alignment segments, so this library raises an explicit
`BackendFeatureError` if one of these is requested instead of silently ignoring it.

## llama.cpp

llama.cpp build b10375 and newer include Qwen3-ASR's audio projector and publish
`ggml-org/Qwen3-ASR-1.7B-GGUF`. For a distilled checkpoint, convert both parts with the current
llama.cpp conversion scripts:

```bash
python convert_hf_to_gguf.py outputs/student --outfile student-f16.gguf
python convert_hf_to_gguf.py outputs/student --mmproj --outfile student-f16.gguf
llama-quantize student-f16.gguf student-q8_0.gguf Q8_0
llama-server -m student-q8_0.gguf --mmproj mmproj-student-f16.gguf -c 8192 -ngl 99
```

The client sends base64 audio to `/v1/chat/completions`, supports language prefill and context, and
preserves batch ordering while issuing at most `batch_size` concurrent requests. It checks `/props`
at startup and refuses a server without audio capability. Timestamps are not available through this
backend. Quantizing exported weights changes inference storage/compute only; training remains
BF16/FP16/FP32 as configured.

## Compatibility references

- [Qwen3-ASR official runtime](https://github.com/QwenLM/Qwen3-ASR) documents its Transformers and
  vLLM engines.
- [Transformers Qwen3-ASR documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_asr)
  defines native processing, language prefill, batching, and context behavior.
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/) includes
  `Qwen3ASRForConditionalGeneration` and the OpenAI speech-to-text API.
- [SGLang's Qwen3-ASR implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_asr.py)
  includes its model, multimodal processor, and transcription adapter.
- [llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
  lists the Qwen3-ASR GGUF models and audio-capable server interface.
