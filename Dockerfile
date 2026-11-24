# syntax=docker/dockerfile:1
# Uses an NVIDIA GPU when one is reserved (host driver + NVIDIA Container Toolkit); the CUDA libraries come with the torch wheel.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
# libsndfile supports the upstream WAV reference loader.
ENV SERVICE_WEB_DIR=/app/web SERVICE_TOKENS_FILE=/data/tokens.json SERVICE_VOICES_DIR=/voices PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH" HF_HOME=/models SERVICE_HOST=0.0.0.0 SERVICE_PORT=29324
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra inference
COPY src ./src
COPY web ./web

EXPOSE 29324
CMD ["python", "src/app.py"]
