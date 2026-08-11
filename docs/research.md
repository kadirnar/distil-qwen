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

## Qwen-specific changes

Materializing teacher and student logits for a padded audio batch is wasteful at Qwen's vocabulary
size. The trainer therefore:

- forwards the frozen acoustic tower once;
- injects the shared acoustic features into each model's own token embeddings;
- obtains teacher and student decoder hidden states;
- selects only positions whose shifted label is not `-100`; and
- projects and compares small token chunks.

This preserves the exact forward KL while bounding peak logit memory. Freezing tied embeddings and
the LM head also removes their gradients and Adam states; gradients still flow through the fixed
student projection into the decoder.

No unverified speed or accuracy claim is encoded in the project. The balanced 14-layer default is a
starting point based on the teacher's 28-layer decoder. A released checkpoint should include
multilingual WER/CER, long-form repetition/insertion metrics, real-time factor, peak memory, and
teacher comparisons.

