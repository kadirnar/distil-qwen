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

The objective is:

```text
loss = 1.0 * pseudo_label_cross_entropy + 0.8 * KL(teacher || student)
```

Both distributions use temperature 2.0 and the KL term is scaled by the squared temperature. The
loss is exact; `--logit-chunk-size` changes memory and kernel-launch overhead, not its value.

For constrained CUDA memory, try `--teacher-quantization 8bit`, then `4bit`. Quantization changes
the teacher distribution, so validate quality before a full run.

## 4. Benchmark

```bash
distil-qwen benchmark audio/*.wav \
  --model outputs/student \
  --batch-size 8 \
  --measured-runs 5
```

Report the GPU, dtype, attention backend, batch size, input duration distribution, decoding limit,
median latency, and real-time factor with every result.

