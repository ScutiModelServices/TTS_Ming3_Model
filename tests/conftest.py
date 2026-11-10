"""CPU-only fake backend; never imports torch or downloads model files."""
import wave
import io
import pytest
from engine import Ming3Engine

buffer = io.BytesIO()
with wave.open(buffer, 'wb') as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(24000)
    output.writeframes(b'\0\0' * 2400)
WAV = buffer.getvalue()


@pytest.fixture(autouse=True)
def _tmp_files(tmp_path, monkeypatch):
    monkeypatch.setenv('SERVICE_TOKENS_FILE', str(tmp_path / 'tokens.json'))
    monkeypatch.setenv('SERVICE_VOICES_DIR', str(tmp_path / 'voices'))


class FakeSpeaker:
    rate = 24000

    def __init__(self):
        self.calls = []

    def speak(self, text, seed, temperature, normalize, maximum, threads, arguments):
        self.calls.append((text, seed, dict(arguments)))
        return b'\1\0' * 2400


def make_engine(**options):
    model = FakeSpeaker()
    options.setdefault('loader', lambda *_: model)
    engine = Ming3Engine(**options)
    engine.fake = model
    return engine
