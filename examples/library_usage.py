from distil_qwen import InferenceConfig
from distil_qwen.inference import OptimizedASR

asr = OptimizedASR.from_pretrained(
    "outputs/student",
    InferenceConfig(batch_size=8, attention="auto"),
)

for result in asr.transcribe(["audio/one.wav", "audio/two.wav"]):
    print(result.language, result.text)
