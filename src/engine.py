"""Offline Ming dense inference; one worker owns model state and global RNG per replica."""
from __future__ import annotations

import asyncio
import os
import random
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from audio import wav_bytes
from voices import VoiceStore

MODEL_ID = "inclusionAI/Ming-3.0-tts-0.5b"
DEFAULT_REVISION = "main"
DEFAULT_VOICE = "US_Emma"
BUILTIN = ("US_Emma", "UK_George", "UK_Charlotte", "灵小柔", "灵楷")
ONE_CALL_VOICE = "custom"
DESIGN_VOICE = "designed"
UPSTREAM_COMMIT = "a39b80660bdc5a4394d5e1d41af6dbfd9c443a54"
SILENT_PEAK = 100  # of 32767 (about -50 dB); a failed 27 s clone measured -70 dB, speech peaks far above this
LONG_REFERENCE_SECONDS = 15  # one 10.8 s clip worked, a 27 s clip gave silence; the exact limit is not measured
sys.path.insert(0, str(Path(__file__).parent / "_upstream"))


class EngineError(RuntimeError):
    """Failure mapped to the shared HTTP detail response."""

    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.status = status


def cache_home() -> Path:
    """Native shared AI cache; Docker mounts the same cache at /models."""
    return Path(os.environ.get("HF_HOME") or os.environ.get("SERVICE_MODELS_DIR") or
                Path(__file__).resolve().parents[3] / ".cache" / "huggingface").expanduser()


def default_marker() -> Path:
    return Path(os.environ.get("SERVICE_WEIGHTS_MARKER") or cache_home() / ".ming3-weights-version")


def _snapshot(model: str, revision: str, *, online: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(model, revision=revision, cache_dir=str(cache_home() / "hub"),
                                    local_files_only=not online, token=False))
    for name in ("config.json", "model.safetensors", "campplus.onnx", "tokenizer.json"):
        if not (folder / name).is_file():
            raise OSError(f"Incomplete dense checkpoint: missing {name}")
    return folder


def _cached_sha(model: str, revision: str) -> str | None:
    try:
        return _snapshot(model, revision).name
    except OSError:
        return None


def _remote_sha(model: str, revision: str) -> str:
    from huggingface_hub import HfApi
    return HfApi().model_info(model, revision=revision, token=False).sha


def _download(model: str, revision: str) -> None:
    from progress import hub_blobs, hub_total, tracking
    with tracking(model, hub_total(model, revision, None), hub_blobs(model, cache_home() / 'hub')):
        _snapshot(model, revision, online=True)


class Speaker:
    """Adapt real MingTTS.synthesize waveform to mono s16le PCM."""

    def __init__(self, model: Any):
        self.model = model
        self.rate = int(model.sample_rate)

    def speak(self, text: str, seed: int, temperature: float | None, normalize: bool,
              max_new_tokens: int, threads: int | None, arguments: dict[str, Any]) -> bytes:
        import numpy as np
        import torch

        if threads:
            torch.set_num_threads(threads)
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model.text_normalization = normalize
        wave = self.model.synthesize(text=text, temperature=temperature or 0.0,
                                     max_decode_steps=max_new_tokens, **arguments)
        return (np.clip(wave.detach().cpu().float().numpy().reshape(-1), -1, 1) * 32767).astype('<i2').tobytes()


def load(model: str, revision: str, device: str) -> Speaker:
    folder = _snapshot(model, revision)
    import torch
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise EngineError("Ming requires a confirmed NVIDIA GPU; CUDA is unavailable", 503)
    from ming_tts import MingTTS
    return Speaker(MingTTS(str(folder), device=device, text_normalization=False, cache_prompts=False))


class Ming3Engine:
    """Injectable backend with serialized loading, synthesis, updates and safe cancellation."""

    def __init__(self, revision: str = DEFAULT_REVISION, device: str = "cuda:0", max_new_tokens: int = 1000,
                 *, store: VoiceStore | None = None, marker: Path | None = None,
                 loader: Callable = load, remote_sha: Callable = _remote_sha,
                 cached_sha: Callable = _cached_sha, download: Callable = _download):
        self.model, self.revision, self.device = MODEL_ID, revision, device
        self.max_new_tokens = max_new_tokens
        self.store = store or VoiceStore()
        self._loader, self._remote_sha_of, self._cached_sha_of, self._download_of = loader, remote_sha, cached_sha, download
        self._marker = marker
        self._seen = self._read_marker()
        self._loaded: Speaker | None = None
        self._load_lock = threading.Lock()
        self._gate = asyncio.Lock()
        self._threads: int | None = None
        self.generation = 0

    @property
    def builtin(self) -> list[str]:
        return list(BUILTIN)

    @property
    def voices(self) -> list[str]:
        return sorted({*BUILTIN, *self.store.names()})

    @property
    def default_voice(self) -> str:
        return DEFAULT_VOICE

    @property
    def ready(self) -> bool:
        return self._loaded is not None

    @property
    def num_thread(self) -> int | None:
        return self._threads

    @num_thread.setter
    def num_thread(self, count: int):
        self._threads = count

    async def _work(self, operation: Callable, *args):
        """Cancellation cannot release slot while worker still uses model or temp files."""
        async with self._gate:
            task = asyncio.create_task(asyncio.to_thread(operation, *args))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not task.cancelled():
                    task.exception()
                raise

    def _speaker(self) -> Speaker:
        with self._load_lock:
            if self._loaded is None:
                try:
                    self._loaded = self._loader(self.model, self.revision, self.device)
                except OSError:
                    raise EngineError("Model weights are not downloaded. Ask an admin to press Download on the Updates card.", 503) from None
            return self._loaded

    async def warm(self):
        await self._work(self._speaker)

    async def close(self):
        await self._work(self._forget)

    def _synthesize(self, text, voice, seed, temperature, preset, normalize, clip, features):
        if not text.strip():
            raise EngineError("text must not be empty", 422)
        if preset is not None:
            raise EngineError("Ming has no preset; use task controls", 422)
        reference = None if clip else self.store.get(voice or DEFAULT_VOICE)
        name = ONE_CALL_VOICE if clip else voice or DEFAULT_VOICE
        if not clip and not reference and name not in BUILTIN:
            raise EngineError(f"Voice {name!r} is not served here; GET /v1/voices lists them", 404)
        with tempfile.TemporaryDirectory(prefix="ming3-ref-") as folder:
            if clip:
                path = Path(folder) / "reference.wav"
                path.write_bytes(clip[0])
                reference = (path, clip[1])
            args = task_arguments(voice, reference, features)
            if features.get("voice_design") or features.get("instruction"):
                name = DESIGN_VOICE
            loaded = self._speaker()
            try:
                pcm = loaded.speak(text, seed, temperature, normalize, self.max_new_tokens, self._threads, args)
            except ValueError as error:
                raise EngineError(str(error), 422) from None
            if not audible(pcm):
                hint = " A reference clip over about 15 seconds is a known cause; use a clean 8-12 second clip with its exact transcript." \
                    if reference and clip_seconds(reference[0]) > LONG_REFERENCE_SECONDS else ""
                raise EngineError("The model produced no audible speech." + hint, 422 if reference else 500)
            return {"wav": wav_bytes(pcm, loaded.rate), "seconds": len(pcm) / 2 / loaded.rate, "voice": name}

    async def synthesize(self, text, voice, seed, temperature, preset=None, normalize=False, clip=None, **features):
        return await self._work(self._synthesize, text, voice, seed if seed is not None else random.getrandbits(62),
                                temperature, preset, normalize, clip, features)

    async def stream(self, *args, **kwargs) -> AsyncIterator[tuple[int, bytes]]:
        """Transport compatibility only: upstream returns one complete waveform."""
        audio = await self.synthesize(*args, **kwargs)
        rate = int.from_bytes(audio['wav'][24:28], 'little')
        yield rate, audio['wav'][44:]

    async def cached_sha(self):
        return await asyncio.to_thread(self._cached_sha_of, self.model, self.revision)

    async def remote_sha(self):
        return await asyncio.to_thread(self._remote_sha_of, self.model, self.revision)

    async def pull(self):
        await self._work(self._download_of, self.model, self.revision)

    async def unload(self):
        def invalidate():
            if self._marker:
                self._marker.parent.mkdir(parents=True, exist_ok=True)
                self._seen = uuid.uuid4().hex
                self._marker.write_text(self._seen, 'utf-8')
            self._forget()
        await self._work(invalidate)

    async def sync(self):
        current = await asyncio.to_thread(self._read_marker)
        if current != self._seen:
            self._seen = current
            await self._work(self._forget)

    def _forget(self):
        with self._load_lock:
            self._loaded = None
            self.generation += 1

    def _read_marker(self):
        try:
            return self._marker.read_text('utf-8') if self._marker else None
        except OSError:
            return None


def audible(pcm: bytes) -> bool:
    """False when 16-bit little-endian `pcm` is empty or never rises above SILENT_PEAK."""
    return len(pcm) >= 2 and max(map(abs, memoryview(pcm[:len(pcm) // 2 * 2]).cast("h"))) > SILENT_PEAK


def clip_seconds(path: Path) -> float:
    """Length of a WAV file in seconds; 0 when it cannot be read."""
    import wave
    try:
        with wave.open(str(path), "rb") as clip:
            return clip.getnframes() / clip.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return 0.0


def task_arguments(voice: str | None, reference: tuple[Path, str] | None, features: dict) -> dict:
    """Map optional shared-schema extensions to disjoint upstream task modes."""
    language = features.get('language_id')
    style, instruction, emotion = (features.get(k) for k in ('voice_design', 'instruction', 'emotion'))
    basic = {k: features[k] for k in ('speed', 'pitch', 'volume') if features.get(k) is not None}
    if sum(bool(x) for x in (style, instruction, emotion, basic)) > 1:
        raise EngineError("voice_design, instruction, emotion and basic attributes are separate task modes", 422)
    if (style or instruction) and (reference or voice):
        raise EngineError("Voice design/freeform cannot be combined with voice or reference audio", 422)
    if (basic or emotion) and not reference:
        raise EngineError("Attribute/emotion control requires a reference voice", 422)
    if (style or instruction or basic or emotion) and language:
        raise EngineError("language_id belongs to TTS/MULTILINGUAL mode only", 422)
    args: dict[str, Any] = {'use_zero_spk_emb': reference is None}
    if style:
        args.update(task='VOICE_DESIGN', controls={'style': style})
    elif instruction:
        args.update(task='FREEFORM', instruction=instruction)
    elif basic:
        if len(basic) != 3:
            raise EngineError("speed, pitch and volume must all be supplied", 422)
        args.update(task='BASIC', controls=basic)
    elif emotion:
        args.update(task='EMOTION', controls={'emotion': emotion})
    elif reference:
        args.update(task='MULTILINGUAL' if language and language.lower() not in ('zh','zh-cn','chinese','en','english') else 'TTS', language_id=language)
    else:
        if language:
            raise EngineError("language_id requires a reference voice; built-in voices use IP mode", 422)
        args.update(task='IP', controls={'ip': voice or DEFAULT_VOICE})
    if reference:
        args.update(use_spk_emb=True, use_zero_spk_emb=False, prompt_wav_path=str(reference[0]))
        if not basic and not emotion:
            args['prompt_text'] = reference[1]
    return args
