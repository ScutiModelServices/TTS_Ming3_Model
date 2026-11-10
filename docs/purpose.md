# Purpose

**Kind:** text → audio

**Group:** Text-to-Speech

**Tags:** `gpu`, `voice-cloning`, `multilingual`

| | |
|---|---|
| Input | Text, optional built-in or cloned voice, language, voice description or task controls |
| Output | Mono s16 WAV; optional completed-WAV streaming transport |
| Runs on | NVIDIA GPU, Ming dense 0.5B, torch 2.6 / CUDA 12.4; about 3.7 GB VRAM at peak and 2.96 GB of weights measured on a 6 GB RTX 4050 Laptop (one machine, one replica) |

**Use for:** multilingual speech, voice cloning/design, paralinguistic and delivery control.
**Not for:** tiny/MoE, podcast dialogue, real-time audio streaming, CPU inference.
