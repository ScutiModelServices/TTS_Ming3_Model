# Docker on another device

## Requirements

GPU host and runtime requirements: [Docker guide](guide-docker-same-device.md#requirements). No deployment has been verified.

## Steps

Use `run.bat deploy` to preview publishing or deploy when authorized. Configure the git-ignored tools/deploy.local.ini using tools/deploy.local.example.ini. The repository origin is a configured future destination; no remote repository is created or pushed automatically. Initial local source has no commit.

On the deployment device, create citadel_core if absent, provision the password yourself, and start Compose. Its ../../.cache/huggingface path resolves to AI/.cache/huggingface with the workspace layout; otherwise set an absolute shared cache path in SERVICE_MODELS_DIR.

```bash
docker compose up -d --build
```

## Check

```bash
curl http://127.0.0.1:29324/health
```

Use [admin guide](guide-web-ui.md) to download weights.

## Scaling

Set SERVICE_REPLICAS then recreate Compose and restart tts_ming3_model_lb. Each replica holds one model, with one active inference; measure GPU memory before increasing replicas. Settings, clone storage and weight markers are shared; audio caches are per replica.

## Updating

Pull authorized releases and rebuild. Download model changes only from admin Updates. Never delete AI/.cache.

## Releasing

Keep VERSION and pyproject version equal; use run.bat deploy release only after explicit authorization. No commits, publishing or releases were made during creation.

## Problems

See [Docker problems](guide-docker-same-device.md#problems).
