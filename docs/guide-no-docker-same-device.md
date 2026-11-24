# Native run

## Requirements

Python 3.11 or 3.12, uv, NVIDIA GPU and compatible driver. Native Windows CUDA inference ran on an RTX 4050 Laptop with 6 GB (about 3.7 GB VRAM at peak, one replica). Normalization defaults false, avoiding pynini. See [Docker requirements](guide-docker-same-device.md#requirements).

## Start

```bash
uv sync --frozen --extra inference
uv run --no-sync python src/app.py
```

`run.bat run` syncs the inference extra and passes --dotenv explicitly. Only that flag reads .env.local then .env; existing environment wins. Native cache defaults to AI/.cache/huggingface. HF_HOME or SERVICE_MODELS_DIR overrides it. Open http://127.0.0.1:29324/admin/ and explicitly Download model weights.

## Check

```bash
curl http://127.0.0.1:29324/health
```

## Stop

Press Ctrl+C. Running inference completes before its worker releases model/reference state.

## Tests

```bash
uv sync --frozen
uv run --no-sync pytest -q
```

No inference extra, GPU or weight downloads needed. On this Windows sandbox set UV_CACHE_DIR and UV_PYTHON_INSTALL_DIR to writable folders on D:, UV_LINK_MODE=copy, and TEMP/TMP to a writable local test folder if required. Browser tests skip without Chromium.

## Problems

See [Docker problems](guide-docker-same-device.md#problems).
