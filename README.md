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

Real GPU inference, minimum VRAM, latency, audio quality and weight size are unverified. Docker engine was unavailable during creation, so image build/deployment are unverified. GPU availability is checked before constructing MingTTS; there is no CPU fallback. Upstream tested H800/H20, which is not evidence of minimum VRAM. `normalize: true` needs optional pynini; install the normalization extra on Linux. Native Windows normalization and GPU runtime are unverified. Seeds govern global RNG only inside the serialized worker; exact CUDA reproducibility is not guaranteed.

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
