"""Cloned voices: a WAV clip and its exact transcript, kept in a folder every replica shares (the voices volume).

Built-in IP voices are model controls and are not stored here."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_WAV_BYTES = 10_000_000  # about 1 minute of 44.1 kHz 16-bit mono; the model wants 5-30 seconds


def voices_dir() -> Path:
    return Path(os.environ.get("SERVICE_VOICES_DIR") or "_voices").expanduser()


def check_clip(wav: bytes, transcript: str) -> None:
    """Raises ValueError unless `wav` is a usable reference clip and `transcript` is not empty."""
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("The reference clip must be a WAV file")
    if not 0 < len(wav) <= MAX_WAV_BYTES:
        raise ValueError(f"The reference clip must be 1 byte to {MAX_WAV_BYTES} bytes")
    if not transcript.strip():
        raise ValueError("The transcript must not be empty")


class VoiceStore:
    """A folder of `<name>.wav` + `<name>.txt`. Read again on every call, so a voice added through one replica is
    visible to the others at once."""

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder or voices_dir()

    def names(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted(path.stem for path in self.folder.glob("*.wav") if path.with_suffix(".txt").is_file())

    def get(self, name: str) -> tuple[Path, str] | None:
        """The clip path and its transcript, or None."""
        if not NAME.match(name):
            return None
        wav = self.folder / f"{name}.wav"
        text = wav.with_suffix(".txt")
        if not (wav.is_file() and text.is_file()):
            return None
        return wav, text.read_text("utf-8")

    def put(self, name: str, wav: bytes, transcript: str) -> None:
        if not NAME.match(name):
            raise ValueError(f"{name!r} is not a valid voice name")
        check_clip(wav, transcript)
        self.folder.mkdir(parents=True, exist_ok=True)
        # The transcript goes first: a voice exists only when both files do, so a half-written one is never listed.
        self._write(self.folder / f"{name}.txt", transcript.strip().encode("utf-8"))
        self._write(self.folder / f"{name}.wav", wav)

    def delete(self, name: str) -> bool:
        if not NAME.match(name):
            return False
        found = False
        for suffix in (".wav", ".txt"):
            try:
                (self.folder / f"{name}{suffix}").unlink()
                found = True
            except FileNotFoundError:
                pass
        return found

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
        try:
            temp.write_bytes(data)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
