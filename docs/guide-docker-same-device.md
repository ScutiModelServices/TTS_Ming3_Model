# Docker on this device

## Requirements

Docker Desktop with WSL2 NVIDIA support, or Linux Docker with NVIDIA Container Toolkit. Upstream CUDA 12.4 base; compatible driver and GPU required. Docker build, GPU run, VRAM and weight size are not yet verified.

## Settings

Copy the example to a local Compose env file and fill the password yourself. No secrets are included in source. Only the launcher with --dotenv reads .env.local then .env; Docker and plain Python never do. Real environment variables win. Do not share local env files.

| Variable | Default | Purpose |
|---|---|---|
| SERVICE_ADMIN_PASSWORD | empty | Admin password and public token enforcement |
| SERVICE_PORT | 29324 | Published host port |
| SERVICE_REPLICAS | 1 | One model process per replica |
| SERVICE_MODELS_DIR | ../../.cache/huggingface | Shared host cache bind-mounted at /models; never clean it |
| SERVICE_MAX_INFLIGHT | 0 | Active plus waiting requests, 0 = 4; overflow 429 |
| SERVICE_MEM_LIMIT | 0 | Compose container RAM cap, 0 = unlimited |

Settings card controls cache_size, max_inflight, update_check_hours; device, model_revision, max_new_tokens and warm need restart. Saved settings override environment/defaults. Threads apply live on the next call and replicate within about a second. See [admin guide](guide-web-ui.md).

## Start

```bash
docker network create citadel_core
docker compose --env-file .env.compose up -d --build
```

Skip network creation if it already exists. Health starts before model loading finishes. Open port 29324 /admin/, press Download explicitly, then wait for ready=true. No implicit downloads; unavailable weights return 503.

## Check

```bash
curl http://127.0.0.1:29324/health
```

## Updating

Rebuild for code/dependency changes; use admin Updates for weights. The cache is shared with native runs and other services, never pruned by this service.

## Problems

| Symptom | Action |
|---|---|
| 503 / ready false | Download weights explicitly; inspect container logs and GPU support |
| CUDA unavailable | Check host driver, Toolkit and GPU reservation; no CPU fallback |
| normalize fails | Leave false, or install optional pynini normalization extra on Linux |
| 422 controls | Follow separate task modes in README |
| 429 busy | Lower caller load or increase bounded max_inflight; more replicas need enough VRAM |
