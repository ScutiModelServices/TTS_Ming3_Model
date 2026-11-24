# Admin page

## Before you open it

Start the API, open http://127.0.0.1:29324/admin/. Set SERVICE_ADMIN_PASSWORD before exposing it beyond localhost. All deployment/runtime options are in [Docker settings](guide-docker-same-device.md#settings).

## Sign in and status

Use the admin password. Health stays available while downloaded weights load in the background. Starting is expected before download; only Updates Download fetches weights, never startup/inference. Updates checks online but does not download automatically. All replicas invalidate their model/audio generation when the weight marker changes and reload downloaded weights. Shared cache is AI/.cache/huggingface; never clean it.

## Tokens

Create one public token per caller, copy it once. Scope projects means allowed voices; rate and expiry work as on siblings. Voice-limited tokens cannot send one-call references or design voices. Chinese built-in voice scopes are accepted. Cloned voices accepts a WAV plus exact transcript; built-in names return 409. Deleting removes saved clones only.

## CPU threads

Set physical core count. Applies on next generation without model reload; other replicas poll saved count every second. GPU still performs inference.

## Settings

Live cache_size, max_inflight and update_check_hours apply within about a second. 0 max_inflight = 4 admitted active/waiting calls; excess requests get 429. One active generation per replica. Restart options are marked. Replicas card is read-only. Updates shows cached/remote Hub revisions; Download downloads/replaces weights explicitly and invalidates replicas. Failed downloads retain loaded model.

## Try it

Paste a public token; it stays in browser memory. Connect lists host and Docker aliases.

### This service's request

Speak takes text, optional voice/seed/temperature and separate Ming modes. Clear voice for one-call reference (WAV and exact transcript; use a clean 8 to 12 second clip, see README limits) or voice design/freeform. Language and basic/emotion control require a reference. Use Chinese categorical attribute/emotion values documented in README. Inline tags go in text. Stream sends one completed waveform; it does not start generation audio early. Normalize defaults false; true needs optional pynini.

## Problems

See [Docker problems](guide-docker-same-device.md#problems). Real GPU, VRAM, weight size, Docker and deployment remain unverified.
