import pytest

from conftest import WAV
from voices import MAX_WAV_BYTES, VoiceStore


def test_a_voice_is_listed_only_with_its_clip_and_transcript(tmp_path):
    store = VoiceStore(tmp_path)
    store.put("anna", WAV, "  Hello.  ")
    (tmp_path / "half.wav").write_bytes(WAV)  # no transcript
    assert store.names() == ["anna"]
    path, text = store.get("anna")
    assert path.read_bytes() == WAV and text == "Hello."
    assert store.get("half") is None and store.get("nobody") is None


def test_put_replaces_and_delete_removes(tmp_path):
    store = VoiceStore(tmp_path)
    store.put("anna", WAV, "One.")
    store.put("anna", WAV + b"x", "Two.")
    assert store.get("anna")[1] == "Two." and store.names() == ["anna"]
    assert store.delete("anna") and not store.delete("anna") and store.names() == []
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize("name, wav, text", [
    ("../evil", WAV, "x"), ("a b", WAV, "x"), (".hidden", WAV, "x"),
    ("anna", b"not a wav file", "x"), ("anna", WAV, "   "), ("anna", WAV + b"\0" * MAX_WAV_BYTES, "x"),
])
def test_bad_voices_are_refused(tmp_path, name, wav, text):
    with pytest.raises(ValueError):
        VoiceStore(tmp_path).put(name, wav, text)
    assert VoiceStore(tmp_path).names() == []


def test_a_name_with_a_path_never_leaves_the_folder(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    assert store.get("../x") is None and not store.delete("../x")
