# Upstream provenance

Runtime: [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts/tree/a39b80660bdc5a4394d5e1d41af6dbfd9c443a54), MIT, commit a39b80660bdc5a4394d5e1d41af6dbfd9c443a54. Dense Python runtime and frontend assets are vendored unchanged in src/_upstream; original LICENSE retained there. Tiny/MoE code is excluded. The service adapter is src/engine.py.

Model: [inclusionAI/Ming-3.0-tts-0.5b](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5b), dense 0.5B. Updates tracks the configured Hub revision (default main). The upstream README, infer.py and MingTTS class were inspected. MingTTS loads a local snapshot, returns a (1, samples) waveform, uses torch bfloat16, and has no streaming method. Weights include speaker encoder/audio tokenizer in addition to the language backbone; parameter count does not establish total download size or VRAM.

Dependencies follow upstream requirements.txt: torch/torchaudio 2.6.0, transformers 4.46.3, accelerate 1.3.0, diffusers 0.33.0 and x-transformers 2.9.2. Docker uses pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime. Optional flash-attn omitted; upstream supports SDPA fallback. Optional pynini normalization is off by default.

Real GPU synthesis, minimum VRAM, total weight size, Docker build and deployment are not verified. No weights were downloaded during creation.
