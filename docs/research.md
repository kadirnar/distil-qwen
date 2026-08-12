# Design notes

The implementation follows four observations from the upstream work.

1. [Distil-Whisper](https://arxiv.org/abs/2311.00430) copies maximally spaced teacher layers,
   freezes the acoustic encoder, and trains with pseudo-label CE plus forward KL. Its ablations show
   that decoder depth is the strongest latency lever and that WER-filtered, diverse pseudo-labels
   matter more than hidden-state MSE.
2. The [Distil-Whisper code](https://github.com/huggingface/distil-whisper) reuses frozen encoder
   states between teacher and student, uses temperature 2, BF16, gradient clipping, and a linear
   warmup/decay schedule.
3. The [Qwen3-ASR report](https://arxiv.org/abs/2601.21337) describes a pretrained acoustic encoder
   feeding a Qwen3 causal LM. The 1.7B checkpoint supports 30 languages plus 22 Chinese dialects,
   and its official runtime supports offline, streaming, Transformers, and vLLM inference.
4. The official [Qwen3-ASR code](https://github.com/QwenLM/Qwen3-ASR) fixes the 1.7B interface used
   here: a 24-layer audio tower, a 28-layer text model, audio-placeholder injection, and a 151,936
   token vocabulary.
5. [Liger Kernel](https://github.com/linkedin/Liger-Kernel) supplies exact Triton RMSNorm, RoPE,
   SwiGLU, and fused distillation operators. Its generic Transformers integration does not list the
   custom `qwen3_asr` model type, so this library patches the actual Qwen3-ASR thinker modules and
   verifies the number of replacements instead of relying on a no-op model-type dispatch.
6. [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) supplies exact, IO-aware fp16
   and bf16 attention on supported CUDA/ROCm hardware. Qwen3-ASR's audio tower already exposes its
   variable-length sequence boundaries, while the causal text tower routes through the same
   Transformers attention backend.

## Qwen-specific changes

Materializing teacher and student logits for a padded audio batch is wasteful at Qwen's vocabulary
size. The trainer therefore:

- forwards the frozen acoustic tower once;
- injects the shared acoustic features into each model's own token embeddings;
- obtains teacher and student decoder hidden states;
- selects only positions whose shifted label is not `-100`; and
- projects and compares small token chunks.

With the default frozen LM head, each chunk computes its hidden-state gradient immediately and
discards its 151,936-way logits. This preserves exact forward KL while genuinely bounding retained
logit memory. If the LM head is trainable, the optional Liger fused CE/KL path calculates the
required head gradient without sequence-wide logits. Freezing tied embeddings and the LM head also
removes their gradients and Adam states; gradients still flow through the fixed student projection
into the decoder.

Training never creates a decoder KV cache: both teacher and student text forwards pass
`use_cache=False`. KV caching is an autoregressive inference optimization and would waste memory in
full-sequence teacher forcing. The separate audio-feature cache is valid because the acoustic tower
is frozen; it is content-addressed, namespaced by the teacher checkpoint, bounded in RAM, and uses
atomic safetensors files for an optional persistent tier.

No unverified speed or accuracy claim is encoded in the project. The balanced 14-layer default is a
starting point based on the teacher's 28-layer decoder. A released checkpoint should include
multilingual WER/CER, long-form repetition/insertion metrics, real-time factor, peak memory, and
teacher comparisons.
