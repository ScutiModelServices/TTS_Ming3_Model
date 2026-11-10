# Deploying TTS_Ming3_Model

Port 29324, replicas behind tts_ming3_model_lb; callers use the host port or that alias on citadel_core.

## Ways to run it

| Path | Guide |
|---|---|
| Docker on this PC | [Docker same device](docs/guide-docker-same-device.md) |
| Docker on another machine | [Deployment device](docs/guide-docker-deployment-device.md) |
| Direct native run | [No Docker](docs/guide-no-docker-same-device.md) |

See [admin guide](docs/guide-web-ui.md) for tokens, settings and explicit weight downloads.
