# Changelog

## Unreleased

- Add Ming-3.0-tts-0.5b dense speech service on port 29324 with shared admin, tokens, settings, metrics and update controls.
- Add built-in voices, saved/one-call cloning, language selection, voice design, freeform delivery, inline tags, categorical speed/pitch/volume and emotion as optional fields.
- Serialize inference outside the event loop, bound active/waiting admission, and keep canceled workers isolated until cleanup.
- Load downloaded weights only from shared AI/.cache; explicit admin Download updates weights and invalidates replicas.
- Add CPU-only fake-backend tests.
- Return `422` ("The model produced no audible speech") instead of a near-silent WAV when a clone produces no audible speech; a hint appears for references over 15 s.
- Document measured weights (2.96 GB), VRAM (about 3.7 GB peak), speed and reference-clip behaviour on an RTX 4050 Laptop; Docker build and deployment remain unverified.
