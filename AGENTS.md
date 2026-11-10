# TTS_Ming3_Model

Ming dense 0.5B text to speech, built from the Kitten2 HTTP/admin scaffold and pinned upstream MIT runtime. Port 29324; SERVICE_ env prefix, ming3_ token prefix.

## Agent instructions

Edit AGENTS.md; CLAUDE.md imports it with @AGENTS.md.

## Commands

- Install tests: `uv sync --frozen`; runtime: `uv sync --frozen --extra inference`.
- Test: `uv run --no-sync pytest -q`; optional browser: `uv run playwright install chromium`.
- Run: `run.bat run` or `uv run --extra inference python src/app.py`; --dotenv explicitly enables local env loading.
- Windows sandbox: set UV_CACHE_DIR and UV_PYTHON_INSTALL_DIR to writable D: folders, UV_LINK_MODE=copy; redirect TEMP/TMP if needed.

## Rules

- Follow model-service-parity, service-layout, service-docs and service-groups skills. Shared admin/settings/token/update/audio files stay identical apart from allowed CONFIG/PREFIX lines.
- Serve only inclusionAI/Ming-3.0-tts-0.5b; never tiny/MoE. Upstream provenance is in docs/upstream.md.
- Startup and requests load local snapshots only; only admin Updates Download fetches weights. Never delete or clean AI/.cache.
- No real model run without confirmed CUDA GPU; no CPU fallback. Inference stays in a serialized worker; cancellation must await worker cleanup before releasing the slot.
- Voice cloning is off (`CLONING` in src/engine.py); do not re-enable without fixing the cut first word. Inline tags like `[Uhm]` are case-sensitive.
- Keep optional task modes distinct; reject incompatible controls with 422. One-call references never persist or enter audio/prompt caches.
- Tests use fake backends; no weights/GPU/network. Do not commit or publish unless asked.
- Never read/print .env files, secrets, tokens.json or settings.json. Synthetic test files are isolated.
- Update operator docs and CHANGELOG with behavior changes. GPU, VRAM and weight size are measured on one 6 GB laptop (see README limits); Docker build stays unverified until run.
