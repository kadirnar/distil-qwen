# Training guide

## 1. Prepare labels

Human transcripts work directly. For unlabeled audio, create teacher transcripts:

```bash
distil-qwen pseudo-label \
  --input data/unlabeled.jsonl \
  --output data/pseudo.jsonl \
  --model Qwen/Qwen3-ASR-1.7B \
  --batch-size 16
```

When reference text is available, retain high-quality teacher labels. `auto` uses WER for segmented
text and CER for unsegmented text.

```bash
distil-qwen filter \
  --input data/pseudo.jsonl \
  --output data/filtered.jsonl \
  --max-error-rate 0.2
```

Use `teacher_text` as `--text-column` when training from the filtered manifest.

## 2. Choose a student

The teacher has 28 text layers and 24 audio layers. Keep all audio layers unless acoustic encoder
latency or memory is the actual bottleneck.

| Preset | Text layers | Audio layers | Purpose |
| --- | ---: | ---: | --- |
| Conservative | 21 | 24 | Highest retention |
| Balanced | 14 | 24 | Default starting point |
| Compact | 8 | 24 | Low-latency experiments |

Always compare WER/CER, real-time factor, peak memory, and hallucination/repetition rate. A smaller
layer count is a hypothesis until it is trained and evaluated on the target languages.

## 3. Launch distillation

```bash
accelerate config

accelerate launch -m distil_qwen.cli train \
  --student outputs/student-init \
  --teacher Qwen/Qwen3-ASR-1.7B \
  --dataset data/filtered.jsonl \
  --text-column teacher_text \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --output-dir outputs/student
```

Defaults freeze the acoustic tower and tied input/output embeddings. This avoids optimizer states
for the large vocabulary and preserves the teacher's acoustic representation. Disable either
freeze only for a measured domain-adaptation need.

Shared audio features require identical frozen student and teacher audio towers. This is true for
the default student, which retains all 24 audio layers. A student initialized with fewer audio
layers must use `--no-reuse-audio-features`; otherwise the launcher stops before training.

The objective is:

```text
loss = 1.0 * pseudo_label_cross_entropy + 0.8 * KL(teacher || student)
```

Both distributions use temperature 2.0 and the KL term is scaled by the squared temperature. The
loss is exact; `--logit-chunk-size` changes memory and kernel-launch overhead, not its value.

## 4. Optimize the training runtime

The default `auto` settings use BF16 when supported, FlashAttention 2 when it is installed and the
GPU/dtype are compatible, Liger when it is installed on an accelerator, fused AdamW on CUDA,
non-reentrant gradient checkpointing, TF32 matrix multiplication, padding-aware length grouping,
non-blocking pinned-memory loading, and persistent dataloader workers. Qwen3-ASR's checkpointed
decoder blocks have no dropout, so RNG-state preservation is disabled by default to avoid needless
bookkeeping. Unsupported explicit kernel requests fail instead of silently falling back.

For the fast CUDA path:

```bash
pip install -e ".[train,kernels]"
MAX_JOBS=4 pip install flash-attn==2.8.3.post1 --no-build-isolation

accelerate launch -m distil_qwen.cli train \
  --student outputs/student-init \
  --dataset data/filtered.jsonl \
  --liger on \
  --attention flash_attention_2 \
  --optimizer adamw_torch_fused \
  --output-dir outputs/student
```

FlashAttention 2 needs fp16/bf16 and supported hardware (Ampere, Ada, Hopper, or a supported ROCm
GPU). `auto` falls back to PyTorch SDPA. Liger is applied directly to Qwen3-ASR's custom thinker
RMSNorm, SwiGLU, and RoPE operations. The generic Transformers `use_liger_kernel` switch is not
used because upstream Liger does not register the `qwen3_asr` model type.

When the audio tower is frozen, cache its deterministic output on local NVMe and optionally retain
a bounded hot set in CPU memory:

```bash
distil-qwen train \
  --student outputs/student-init \
  --dataset data/filtered.jsonl \
  --audio-cache-dir /local-nvme/distil-qwen-audio \
  --audio-cache-memory-mb 4096 \
  --output-dir outputs/student
```

Waveform content is hashed by default. `--cache-key-column id` avoids hashing when the dataset has
stable, globally unique audio IDs. Disk entries are isolated by teacher checkpoint and written as
atomic safetensors files. The first pass computes a miss; later epochs reuse it. Do not enable this
cache while training or structurally reducing the audio tower.

Length grouping performs a one-time, dataset-cached duration scan and estimates the combined
post-encoder audio and transcript cost. It is enabled by default. Use multiple preprocessing
workers for large manifests, or disable it when a custom batch sampler already performs packing:

```bash
distil-qwen train \
  --student outputs/student-init \
  --dataset data/filtered.jsonl \
  --length-preprocessing-workers 8 \
  --output-dir outputs/student
```

Compilation is opt-in. It is applied directly to the text module invoked by the custom
distillation engine, rather than the unused outer forward. In-place module compilation also keeps
saved checkpoint keys stable:

```bash
distil-qwen train \
  --student outputs/student-init \
  --dataset data/filtered.jsonl \
  --compile-model \
  --compile-scope text_model \
  --output-dir outputs/student
```

`decoder_layers` is an alternative regional scope with lower compilation startup cost. Benchmark
both scopes because the best choice depends on sequence-length variance and PyTorch/CUDA versions.

Choose additional memory levers deliberately:

| Setting | Memory | Throughput | Notes |
| --- | --- | --- | --- |
| `--activation-offload` | Much lower activation VRAM | Lower | Moves saved tensors to pinned CPU memory |
| smaller `--logit-chunk-size` | Lower loss workspace | Lower | Objective remains exact |
| `--compile-model --compile-scope text_model` | Workload-dependent | Higher after warmup | Compiles the executed text path |
| `--no-gradient-checkpointing` | Higher | Higher | Useful only when activation memory fits |

Training intentionally stays in FP32, FP16, or BF16. It does not use FP8, integer-quantized
teachers, or integer optimizer states, so the teacher distribution and trainable weights retain
their floating-point representation.

For distributed data parallel training, immutable model buffers do not need to be broadcast every
iteration, so `--no-ddp-broadcast-buffers` is the default. `--ddp-bucket-cap-mb` is exposed for
interconnect-specific tuning; leave it unset until profiling shows communication is a bottleneck.

The default frozen-head loss is specially optimized: it retains only the gradient for each decoder
hidden-state chunk and immediately releases the vocabulary logits. If embeddings/LM head are made
trainable, install Liger and leave `--distillation-backend auto`; it selects Liger's fused CE +
forward-KL path. Decoder KV caching is always disabled during teacher-forced training.

## 5. Benchmark

```bash
distil-qwen benchmark audio/*.wav \
  --model outputs/student \
  --batch-size 8 \
  --measured-runs 5
```

Report the GPU, dtype, attention backend, batch size, input duration distribution, decoding limit,
median latency, and real-time factor with every result.
