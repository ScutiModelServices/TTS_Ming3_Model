"""HTTP face of Ming dense 0.5B speech synthesis on an NVIDIA GPU.

POST /v1/speak {"text", "voice", "seed", "temperature", "preset", "normalize"} returns a WAV file (audio/wav); X-Audio-Seconds, X-Duration-Ms,
X-Voice and X-Seed describe it. GET /v1/voices lists the voices (built-in and cloned). PUT /admin/voices/{name} adds
one from a reference clip and its transcript. GET /health reports whether the model is loaded."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import logging
import os
from pathlib import Path
import secrets
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from disconnect import CallerGone, Measure, first_piece
from audio import AudioCache, wav_bytes, wav_header
from engine import DEFAULT_REVISION, ONE_CALL_VOICE, EngineError, Ming3Engine, default_marker
from options import SETTINGS
import settings
from tokens import LatencyHistogram, TokenStore
from updates import Updates
from voices import MAX_WAV_BYTES, NAME, check_clip

_log = logging.getLogger(__name__)

NAME_PATTERN = NAME.pattern  # a voice name; a token's "projects" list limits the voices it may use
MAX_SEED = 2**62


class SpeakRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=5000)
    voice: str | None = Field(None, min_length=1, max_length=64, pattern=r"^[^\W_][\w.-]*$")
    seed: int | None = Field(None, ge=0, le=MAX_SEED)  # same text, voice, preset, temperature, normalize and seed = same speech; only these calls are cached
    temperature: float | None = Field(None, ge=0, le=1.5)
    preset: str | None = Field(None, max_length=32)  # shared compatibility field; Ming rejects non-null presets
    normalize: bool = False  # true requires optional pynini; no text-normalization dependency by default
    language_id: Literal['Chinese','English','Japanese','Korean','German','French','Spanish','zh','en','ja','ko','de','fr','es'] | None = None
    voice_design: str | None = Field(None, min_length=1, max_length=2000)
    instruction: str | None = Field(None, min_length=1, max_length=2000)
    speed: Literal['慢速','中速','快速'] | None = None
    pitch: Literal['低','中','高'] | None = None
    volume: Literal['低','中','高'] | None = None
    emotion: Literal['高兴','悲伤','愤怒','恐惧','厌恶','惊讶','中性'] | None = None
    stream: bool = False  # completed waveform transport only; Ming has no incremental generation
    # A cloned voice for this call only (instead of `voice`): used while the audio is made, then deleted; never stored or cached.
    reference_audio_base64: str | None = Field(None, min_length=1, max_length=MAX_WAV_BYTES * 4 // 3 + 64)  # a WAV clip, base64 or a data: URL (the Try it panel sends one)
    reference_text: str | None = Field(None, min_length=1, max_length=2000)  # exactly what the clip says


class VoiceUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_base64: str = Field(min_length=1, max_length=MAX_WAV_BYTES * 4 // 3 + 8)  # the reference clip, a WAV file
    transcript: str = Field(min_length=1, max_length=2000)  # exactly what the clip says


class SettingsUpdate(BaseModel):
    values: dict[str, Any]  # None resets an option to its environment variable or default


class NewToken(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    projects: list[Annotated[str, Field(pattern=r"^[\w][\w.-]{0,63}$")]] | None = Field(None, max_length=32)
    rate: int | None = Field(None, ge=1, le=100_000)  # calls per minute
    expires_days: int | None = Field(None, ge=1, le=3650)


class TokenScope(BaseModel):
    projects: list[Annotated[str, Field(pattern=r"^[\w][\w.-]{0,63}$")]] | None = Field(None, max_length=32)
    rate: int | None = Field(None, ge=1, le=100_000)


class Threads(BaseModel):
    num_thread: int = Field(ge=1, le=256)


def create_app(client: Ming3Engine | None = None, admin_password: str | None = None,
               tokens: TokenStore | None = None, warm: bool = False,
               updates: Updates | None = None, check_hours: float = 0, max_inflight: int | None = None,
               cache_size: int | None = None, config: settings.Config | None = None) -> FastAPI:
    """`admin_password` (SERVICE_ADMIN_PASSWORD) unlocks /admin/, where each caller gets a token of its own; when it
    is set, every call except /health must send one of those tokens in the X-Internal-Token header. Unset = no
    auth at all (local use only). `tokens` holds the tokens (SERVICE_TOKENS_FILE). `cache_size` > 0 keeps that
    many answers per replica (SERVICE_CACHE_SIZE). `config` (the admin page's settings) replaces `warm`,
    `check_hours`, `max_inflight` and `cache_size` and keeps them live; without it they stay fixed."""
    tokens = tokens or TokenStore(os.environ.get("SERVICE_TOKENS_FILE", "tokens.json"))
    client = client or Ming3Engine(config.get("model_revision") if config else os.environ.get("SERVICE_MODEL_REVISION", DEFAULT_REVISION),
                                    marker=default_marker())
    voices = client.store
    inflight = 0
    cache = AudioCache()

    def cache_limit() -> int:
        """Answers this replica keeps; 0 = off."""
        return config.get("cache_size") if config else (
            cache_size if cache_size is not None else int(os.environ.get("SERVICE_CACHE_SIZE") or 0))

    def inflight_limit() -> int:
        """Calls this replica lets wait for the model at once; more get 429. 0 = 4."""
        value = config.get("max_inflight") if config else (
            max_inflight if max_inflight is not None else int(os.environ.get("SERVICE_MAX_INFLIGHT") or 0))
        return value or 4

    updates = updates or Updates(client.model, cached_sha=lambda: client.cached_sha(), remote_sha=lambda: client.remote_sha(),
                                 download=lambda: client.pull(), unload=lambda: client.unload())

    async def check_forever() -> None:
        """Checks every `update_check_hours` (0 = off). The value is read again every second, so the admin page can change it."""
        last = None
        while True:
            wait = (config.get("update_check_hours") if config else check_hours) * 3600
            now = time.monotonic()
            if wait > 0 and (last is None or now - last >= wait):
                last = now
                try:
                    status = await updates.check()
                    if status["error"]:
                        _log.warning("Update check failed: %s", status["error"])
                except Exception:
                    _log.exception("Update check failed")
            await asyncio.sleep(min(wait, 1.0) if wait > 0 else 1.0)

    async def sync_forever() -> None:
        """Applies what another replica saved (CPU threads) and drops models it replaced with new files."""
        while True:
            saved = await asyncio.to_thread(settings.load, tokens.settings_path)
            if saved.get("num_thread") and saved["num_thread"] != client.num_thread:
                client.num_thread = int(saved["num_thread"])
            await client.sync()
            # A dropped model (download, threads change, sync) leaves /health "Starting" until a request loads it: reload once per drop.
            if config.get("warm") and not client.ready and client.generation != warmed:
                await warm_up()
            await asyncio.sleep(1)

    warmed = 0  # generation of the last warm-up attempt

    async def warm_up() -> None:
        nonlocal warmed
        warmed = client.generation
        try:
            await client.warm()
        except Exception:
            _log.exception("Model warm-up failed; the first request will try again")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        background: list[asyncio.Task] = []
        try:
            if config.get("warm") if config else warm:
                # In the background: cached weights load while /health remains available.
                background.append(asyncio.create_task(warm_up()))
            if config or check_hours > 0:
                background.append(asyncio.create_task(check_forever()))
            if config:
                background.append(asyncio.create_task(sync_forever()))
            yield
        finally:
            for task in background:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await client.close()

    def authorize(request: Request, x_internal_token: str = Header("")) -> list[str] | None:
        """The voices the token is limited to, or None for all. Unset admin password = no auth."""
        if not admin_password:
            return None
        row = tokens.lookup(x_internal_token)
        if row is None:
            raise HTTPException(401, "Invalid token")
        wait = tokens.throttle(row)
        if wait:
            raise HTTPException(429, "Too many requests for this token", headers={"Retry-After": str(wait)})
        request.state.token_id = row["id"]
        return row.get("projects")

    def check_scope(allowed: list[str] | None, voice: str) -> None:
        if allowed is not None and voice not in allowed:
            raise HTTPException(403, f"This token may not use voice {voice!r}")

    def admin(x_internal_token: str = Header("")) -> None:
        if not admin_password or not hmac.compare_digest(x_internal_token, admin_password):
            raise HTTPException(401, "Admin password required (set SERVICE_ADMIN_PASSWORD)")

    app = FastAPI(title="Ming 3 dense 0.5B text-to-speech service", lifespan=lifespan)

    async def slot():
        nonlocal inflight
        limit = inflight_limit()
        if limit and inflight >= limit:
            raise HTTPException(429, "Ming3 is busy: too many requests waiting", headers={"Retry-After": "5"})
        inflight += 1
        try:
            yield
        finally:
            inflight -= 1

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": client.model, "ready": client.ready,
                "num_thread": client.num_thread, "device": client.device, "voices": await asyncio.to_thread(lambda: client.voices)}

    @app.get("/v1/voices")
    async def list_voices(allowed: list[str] | None = Depends(authorize)) -> dict[str, Any]:
        names = [name for name in await asyncio.to_thread(lambda: client.voices) if allowed is None or name in allowed]
        return {"voices": names, "default": client.default_voice}

    @app.post("/v1/speak")
    async def speak(body: SpeakRequest, request: Request, allowed: list[str] | None = Depends(authorize), _slot: None = Depends(slot)) -> Response:
        clip = await asyncio.to_thread(one_call_clip, body, allowed)
        if (body.voice_design or body.instruction) and allowed is not None:
            raise HTTPException(403, "Voice-limited tokens may not design a voice")
        name = ONE_CALL_VOICE if clip else body.voice or client.default_voice
        if not clip:
            check_scope(allowed, name)
        started = time.perf_counter()
        size = cache_limit()
        seed = body.seed if body.seed is not None else secrets.randbelow(MAX_SEED + 1)
        key = (client.generation, name, body.preset, body.temperature, body.normalize, seed, body.text,
               tuple(features(body).items()))
        audio = cache.get(key, size) if body.seed is not None and not clip else None
        if audio is not None:
            return wav_reply(audio, started, "hit")
        # Without a seed the call can never repeat, so keeping its answer would only fill the cache.
        # A one-call voice is never cached: nothing made from it stays on the server.
        keep = size if body.seed is not None and not clip else 0
        try:
            if body.stream:
                return await stream_reply(request, body, name, seed, key, keep, clip)
            audio = await client.synthesize(body.text, body.voice, seed, body.temperature, body.preset, body.normalize, clip=clip, **features(body))
        except EngineError as error:
            raise HTTPException(error.status or 500, str(error)) from None
        except Exception:
            _log.exception("Ming3 could not speak")
            raise HTTPException(500, "Ming3 could not speak") from None
        audio["seed"] = seed
        cache.put(key, audio, keep)
        return wav_reply(audio, started, "miss")

    def features(body: SpeakRequest) -> dict[str, Any]:
        return {key: getattr(body, key) for key in ('language_id', 'voice_design', 'instruction', 'speed', 'pitch', 'volume', 'emotion')}

    def one_call_clip(body: SpeakRequest, allowed: list[str] | None) -> tuple[bytes, str] | None:
        """The (WAV, transcript) of a one-call cloned voice, or None. Needs both fields, no `voice`, and a token that
        is not limited to certain voices (a limited token may only use the voices listed for it)."""
        if body.reference_audio_base64 is None and body.reference_text is None:
            return None
        if body.reference_audio_base64 is None or body.reference_text is None:
            raise HTTPException(422, "reference_audio_base64 and reference_text go together")
        if body.voice is not None:
            raise HTTPException(422, "Send either voice or a one-call reference, not both")
        if allowed is not None:
            raise HTTPException(403, "This token is limited to certain voices and may not send its own reference")
        try:
            data = body.reference_audio_base64
            data = data.partition(",")[2] if data.startswith("data:") else data
            wav = base64.b64decode(data, validate=True)
            check_clip(wav, body.reference_text)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(422, str(error)) from None
        return wav, body.reference_text.strip()

    def wav_reply(audio: dict[str, Any], started: float, cached: str) -> Response:
        return Response(audio["wav"], media_type="audio/wav", headers={
            "X-Audio-Seconds": f"{audio['seconds']:.3f}", "X-Voice": quote(audio["voice"]), "X-Seed": str(audio["seed"]), "X-Cache": cached,
            "X-Duration-Ms": str(round((time.perf_counter() - started) * 1000))})

    async def stream_reply(request: Request, body: SpeakRequest, name: str, seed: int, key: tuple, size: int,
                           clip: tuple[bytes, str] | None = None) -> Response:
        """Starts the audio before the text is finished. The first piece is made here, so an unknown voice or a
        failed model still gives a proper HTTP error; once the stream has begun an error can only cut it off."""
        pieces = client.stream(body.text, body.voice, seed, body.temperature, body.preset, body.normalize, clip=clip, **features(body))
        try:
            rate, first = await first_piece(request, pieces)
        except CallerGone:
            await pieces.aclose()
            return Response(status_code=499)  # nobody is listening; the generation is already stopped
        except BaseException:
            await pieces.aclose()
            raise

        async def chunks():
            kept: bytearray | None = bytearray(first) if size else None
            try:
                yield wav_header(rate)
                yield first
                async for _, piece in pieces:
                    if kept is not None:
                        kept += piece
                        if len(kept) > cache.MAX_BYTES:
                            kept = None  # too long to keep; stop collecting it
                    yield piece
            finally:
                await pieces.aclose()
            if kept is not None:  # only a stream that reached its end
                cache.put(key, {"wav": wav_bytes(bytes(kept), rate), "seconds": len(kept) / 2 / rate,
                                "voice": 'designed' if body.voice_design or body.instruction else name, "seed": seed}, size)

        shown_voice = 'designed' if body.voice_design or body.instruction else name
        return StreamingResponse(chunks(), media_type="audio/wav", headers={"X-Voice": quote(shown_voice), "X-Seed": str(seed), "X-Cache": "miss"})

    @app.get("/admin/voices", dependencies=[Depends(admin)])
    def admin_voices() -> dict[str, Any]:
        """Every voice with its origin, for the admin page: cloned ones can be deleted, the others cannot."""
        cloned = set(voices.names())
        return {"voices": [{"name": name, "cloned": name in cloned} for name in client.voices], "default": client.default_voice}

    @app.put("/admin/voices/{name}", dependencies=[Depends(admin)])
    def put_voice(name: str, body: VoiceUpload) -> dict[str, Any]:
        """Adds or replaces a voice: a WAV clip of a few seconds and exactly what it says. Every replica sees it at once.
        Answers kept for the old clip stay in the cache until they age out, so use a seed-less call or a new name to hear the change."""
        try:
            if name in client.builtin:
                raise HTTPException(409, f"{name!r} is a built-in voice; pick another name")
            audio = base64.b64decode(body.audio_base64, validate=True)
            voices.put(name, audio, body.transcript)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(422, str(error)) from None
        return {"voice": name, "bytes": len(audio)}

    @app.delete("/admin/voices/{name}", dependencies=[Depends(admin)])
    def delete_voice(name: str) -> dict[str, bool]:
        if not voices.delete(name):
            raise HTTPException(404, "No such voice")
        return {"ok": True}

    @app.put("/admin/threads", dependencies=[Depends(admin)])
    def set_threads(body: Threads) -> dict[str, Any]:
        """Applies to this replica now and is saved; the other replicas apply it within a few seconds. It beats SERVICE_NUM_THREAD.
        torch reads it on every call, so the model is not reloaded. Set the physical core count."""
        client.num_thread = body.num_thread
        settings.save(tokens.settings_path, num_thread=body.num_thread)
        return {"num_thread": client.num_thread}

    @app.get("/admin/workers", dependencies=[Depends(admin)])
    def workers() -> dict[str, Any]:
        """Read-only: the replica count is a Docker Compose setting the service cannot change."""
        def number(name: str) -> int | None:
            value = os.environ.get(name, "").strip()
            return int(value) if value.isdigit() else None

        return {"replicas": number("SERVICE_REPLICAS"), "parallel": 1}

    @app.get("/admin/updates", dependencies=[Depends(admin)])
    async def update_status() -> dict[str, Any]:
        """Return the last check without I/O; zero check hours disables only the timer."""
        return {"enabled": True, **updates.status()}

    @app.post("/admin/updates/check", dependencies=[Depends(admin)])
    async def check_updates() -> dict[str, Any]:
        return {"enabled": True, **await updates.check()}

    @app.post("/admin/updates/weights", dependencies=[Depends(admin)])
    async def update_weights() -> dict[str, Any]:
        try:
            return {"enabled": True, **await updates.update_weights()}
        except Exception as error:
            _log.exception("Weight update failed")
            raise HTTPException(502, f"Weight update failed: {error}") from None

    @app.get("/admin/tokens", dependencies=[Depends(admin)])
    def list_tokens() -> dict[str, Any]:
        return {"tokens": tokens.list()}

    @app.post("/admin/tokens", dependencies=[Depends(admin)])
    def create_token(body: NewToken) -> dict[str, Any]:
        return tokens.create(body.name.strip(), body.projects, body.rate, body.expires_days)

    @app.get("/admin/settings", dependencies=[Depends(admin)])
    def get_settings() -> dict[str, Any]:
        return {"settings": config.describe() if config else []}

    @app.put("/admin/settings", dependencies=[Depends(admin)])
    def put_settings(body: SettingsUpdate) -> dict[str, Any]:
        """Saves options; null resets one. Every replica reads the shared file, so a change applies everywhere within
        a second, except options marked `restart`."""
        if config is None:
            raise HTTPException(409, "Settings are fixed in this setup")
        try:
            config.set(body.values)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return {"settings": config.describe()}

    @app.put("/admin/tokens/{token_id}", dependencies=[Depends(admin)])
    def update_token(token_id: str, body: TokenScope) -> dict[str, bool]:
        if not tokens.update(token_id, body.projects, body.rate):
            raise HTTPException(404, "No such token")
        return {"ok": True}

    latency = LatencyHistogram()

    app.add_middleware(Measure, observe=latency.observe,
                       record=lambda token_id, ms, failed: run_in_threadpool(tokens.record, token_id, ms, failed))

    @app.get("/metrics", dependencies=[Depends(admin)])
    def metrics() -> PlainTextResponse:
        """Prometheus text per token, from the shared token file (up to a minute behind). Send the admin password."""
        return PlainTextResponse(tokens.metrics("ming3") + latency.render("ming3"))

    @app.delete("/admin/tokens/{token_id}", dependencies=[Depends(admin)])
    def revoke_token(token_id: str) -> dict[str, bool]:
        if not tokens.revoke(token_id):
            raise HTTPException(404, "No such token")
        return {"ok": True}

    web = Path(os.environ.get("SERVICE_WEB_DIR", Path(__file__).resolve().parents[1] / "web"))
    if web.is_dir():
        app.mount("/admin", StaticFiles(directory=web, html=True))

    return app


def main() -> None:
    import sys

    import uvicorn

    if "--dotenv" in sys.argv[1:]:  # run.bat asks for it; a plain `python src/app.py` and Docker never read .env files
        loaded = settings.load_env_files(Path(__file__).resolve().parents[1])
        for name, names in loaded.items():
            print(f"{name}: loaded {', '.join(names) or 'no variables'} (names only; a real environment variable wins)", flush=True)
        if not loaded:
            print(".env.local / .env: no file in the service folder", flush=True)
    settings.apply_models_dir()  # SERVICE_MODELS_DIR -> HF_HOME / OLLAMA_MODELS

    store = TokenStore(os.environ.get("SERVICE_TOKENS_FILE", "tokens.json"))
    config = settings.Config(store.settings_path, SETTINGS)
    client = Ming3Engine(config.get("model_revision"), config.get("device"), config.get("max_new_tokens"),
                          marker=default_marker())
    saved = settings.load(store.settings_path)
    threads = saved.get("num_thread") or os.environ.get("SERVICE_NUM_THREAD", "").strip()
    if threads:
        client.num_thread = int(threads)
    app = create_app(client=client, admin_password=os.environ.get("SERVICE_ADMIN_PASSWORD") or None,
                     tokens=store, config=config)
    host, port = os.environ.get("SERVICE_HOST", "127.0.0.1"), int(os.environ.get("SERVICE_PORT", "29324"))

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)  # after uvicorn's own "running on" line
            shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
            print(f"INFO:     Admin page: http://{shown}:{port}/admin/", flush=True)

    Server(uvicorn.Config(app, host=host, port=port)).run()


if __name__ == "__main__":
    main()
