# TTS_Ming3_Model

Ming-3.0-tts-0.5b dense text to speech behind the shared model-service HTTP/admin API. Port 29324. Built for callers such as Bard and Mimic; their connections are configured separately. Only `inclusionAI/Ming-3.0-tts-0.5b` is served; no tiny/MoE runtime.

## API

| Method | Route | Result |
|---|---|---|
| GET | `/health` | Public status, ready, model, device, num_thread, voices |
| GET | `/v1/voices` | Allowed voices and default |
| POST | `/v1/speak` | Mono 16-bit WAV and audio metadata headers |
| GET | `/metrics` | Admin-authenticated Prometheus metrics |
| GET | `/admin/` | Shared admin page |
| GET, POST | `/admin/tokens` | List/create public tokens |
| PUT, DELETE | `/admin/tokens/{token_id}` | Edit/revoke token |
| GET, PUT | `/admin/settings` | Inspect/update settings |
| PUT | `/admin/threads` | Apply/save num_thread |
| GET | `/admin/workers` | Replica count and parallel=1 |
| GET | `/admin/updates` | Download/check status |
| POST | `/admin/updates/check`, `/admin/updates/weights` | Check/download weights |
| GET | `/admin/voices` | Built-in/cloned voices |
| PUT, DELETE | `/admin/voices/{name}` | Save/delete clone |

**Input:** JSON body `text` (1–5000 characters), optional `voice`, `seed` (0–2**62), `temperature` (0–1.5), `normalize` (default false), `stream` (default false). Unknown fields return 422. `preset` remains in the shared schema but Ming has no equivalent; non-null values return 422.

**Voice cloning is off.** `CLONING = False` in `src/engine.py`. One-call references and saved clones answer `422`, and `PUT /admin/voices/{name}` is refused. Measured on an RTX 4050 Laptop: cloned speech starts mid-word (the first word is cut off), and a reference that ends in silence makes most sentences stop early. Use the built-in voices, `voice_design`, `instruction` and inline tags such as `[Uhm]`, `[Sigh]`, `[Laughter]` (case matters). Language, emotion and speed/pitch/volume need a reference, so they are refused too.

**Body options:** One-call cloning uses paired `reference_audio_base64` (WAV, at most 10 MB decoded; data URLs accepted) and `reference_text` (1–2000 characters), without `voice`. Admin clones use `{audio_base64, transcript}`. Temporary references are removed on completion/failure/cancellation, never cached; upstream prompt caching is disabled. Saved clone names use the sibling ASCII naming rules. Chinese built-in names are accepted in requests and token scopes.

| Optional Ming extension | Meaning and supported combination |
|---|---|
| `language_id` | Chinese/English/Japanese/Korean/German/French/Spanish or zh/en/ja/ko/de/fr/es. Requires a reference voice; selects TTS or MULTILINGUAL |
| `voice` | Existing shared field; built-ins US_Emma (default), UK_George, UK_Charlotte, 灵小柔, 灵楷 select IP; saved clone names select TTS |
| `voice_design` | Description, 1–2000 characters; VOICE_DESIGN, no voice/reference |
| `instruction` | Freeform delivery description; FREEFORM, no voice/reference |
| `speed`, `pitch`, `volume` | BASIC mode: all three required, reference voice required; speed 慢速/中速/快速, pitch and volume 低/中/高 |
| `emotion` | EMOTION mode, reference voice required: 高兴/悲伤/愤怒/恐惧/厌恶/惊讶/中性 |

These optional extensions preserve existing callers. Modes are separate: design, freeform, emotion and basic attributes cannot be mixed. `language_id` belongs to reference TTS/MULTILINGUAL only. Inline paralinguistic tags remain in `text`; use reference TTS mode, e.g. `[Sigh]Hello [Laughter]`. All 14 upstream tags pass through unchanged. Dialect and podcast APIs are not exposed. Only five documented built-ins are advertised; other upstream catalog voices are not exposed yet.

**Result:** `audio/wav`, `X-Audio-Seconds`, `X-Duration-Ms`, `X-Voice`, `X-Seed`, `X-Cache: hit|miss`. Non-ASCII X-Voice values are UTF-8 percent-encoded. Errors use shared FastAPI `detail` format: 401 auth, 403 scope, 404 voice, 422 validation, 429 admission/rate limit with Retry-After, 503 unavailable weights/GPU.

**Streaming:** `stream: true` keeps the sibling unknown-length WAV transport. Ming generates the entire utterance before its single PCM piece is sent; no incremental generation or latency benefit. Complete cached WAVs may answer streamed calls.

**Cache:** Seeded requests only, complete request controls in key, per replica, at most 1 MB each; one-call references excluded. Model reload invalidates cache generation. Saved clone replacement retains sibling cache semantics: use a new seed or name to hear it immediately.

**Tokens:** X-Internal-Token is a public token for synthesis, admin password for admin routes and metrics. Without an admin password public calls have no auth. Voice-limited tokens cannot send references or design voices. Chinese voice scopes work. Admin static page is public; admin operations require the password.

## Admin page

Shared Tokens, usage/metrics, CPU threads, Settings, replicas, Updates, Cloned voices, Connect and Try it panels. Threads apply on the next generation without reload; use physical core count. Replicas poll saved settings and weight markers every second. Each replica loads one model and serializes generation. `SERVICE_MAX_INFLIGHT` / max_inflight bounds admitted active plus waiting requests (0 means 4); overflow gets 429, Retry-After 5. Startup loads downloaded weights in the background while health/admin remain available.

## Run

Docker uses the upstream CUDA base and pinned torch 2.6.0/cu124, torchaudio 2.6.0, transformers 4.46.3. NVIDIA GPU required. No weights download at startup or inference: start the API, then explicitly press Download in admin Updates. Native defaults use shared `AI/.cache/huggingface`; Compose bind-mounts the same location. Never clean that cache. See [deployment](deployment.md).

```bash
docker compose up -d --build
```

```bash
uv sync --extra inference
uv run --extra inference python src/app.py
```

```bash
curl http://127.0.0.1:29324/v1/speak -H "X-Internal-Token: <SERVICE_TOKEN>" -H "Content-Type: application/json" -d '{"text":"Hello from Ming.","voice":"US_Emma","seed":42}' --output speech.wav
```

The asynchronous Python client example is in [CLI guide](docs/guide-cli.md).

## Tests

```bash
uv sync --frozen
uv run --no-sync pytest -q
```

Default install contains HTTP and test dependencies only. The inference extra installs torch and the upstream runtime dependencies. Tests inject a fake speaker: no GPU or weights. Browser tests need `uv run playwright install chromium` and skip if Chromium is absent.

## Limits worth knowing

**Measured on one machine** (native Windows, RTX 4050 Laptop GPU, 6 GB, plain PyTorch SDPA attention, no flash-attn, one replica):
- Weights: `model.safetensors` is 2.96 GB (about 2.8 GB in the cache), plus a 0.03 GB speaker model. The file holds 1.48B BF16 parameters.
- VRAM: about 3.0 GB after the model loads and about 3.7 GB at the peak of the requests tried (built-in voice, one-call clones of 10.8 s and 27 s references). Other programs on the GPU used about 0.75 GB before the model loaded, so allow about 4.5 GB in total. VRAM stays near the peak after requests because PyTorch keeps its cache. Each extra replica loads its own copy, so a second replica does not fit on 6 GB.
- Speed: upstream's CLI generated about 3 s of audio in 4 to 5 s (about 7.4 steps/s), roughly 1.5 times slower than real time, excluding model load. This is one rough run, not a benchmark.

**Reference clips.** A 10.8 s clip cloned correctly. A 27 s clip with mixed delivery gave no audible speech, with the service and with upstream's own `infer.py`, even with an exact Whisper transcript. The limit is not measured. Use a clean 8 to 12 second clip with its exact transcript. If the output is silent, the service returns `422` (`The model produced no audible speech`) instead of a near-empty WAV; a reference longer than 15 s adds a hint. The saved WAV limit is 10 MB.

**Still unverified:** Docker image build and deployment, audio quality beyond listening to a few clips, other GPUs, and exact CUDA reproducibility (seeds govern global RNG only inside the serialized worker). GPU availability is checked before constructing MingTTS; there is no CPU fallback. Upstream tested H800/H20 only. `normalize: true` needs optional pynini; install the normalization extra on Linux; native Windows normalization is unverified.

## Try it from the admin page

Paste a public token, load allowed voices, choose one request mode, submit Speak. Use reference clip + transcript for language/emotion/basic control; clear Voice for cloning/design/freeform. Stream sends the finished utterance only.

## Documentation

- [Deployment](deployment.md)
- [Changelog](CHANGELOG.md)
- [Purpose](docs/purpose.md)
- [Admin guide](docs/guide-web-ui.md)
- [Docker on this device](docs/guide-docker-same-device.md)
- [Docker on deployment device](docs/guide-docker-deployment-device.md)
- [Native run](docs/guide-no-docker-same-device.md)
- [CLI and async example](docs/guide-cli.md)
- [Upstream provenance](docs/upstream.md)
