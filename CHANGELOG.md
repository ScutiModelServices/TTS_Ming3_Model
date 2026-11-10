# Changelog

## Unreleased

- Add Ming-3.0-tts-0.5b dense speech service on port 29324 with shared admin, tokens, settings, metrics and update controls.
- Add built-in voices, saved/one-call cloning, language selection, voice design, freeform delivery, inline tags, categorical speed/pitch/volume and emotion as optional fields.
- Serialize inference outside the event loop, bound active/waiting admission, and keep canceled workers isolated until cleanup.
- Load downloaded weights only from shared AI/.cache; explicit admin Download updates weights and invalidates replicas.
- Add CPU-only fake-backend tests; real GPU, minimum VRAM, download size, Docker build and deployment remain unverified.
