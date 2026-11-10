"""Safety and upstream boundary tests without importing the GPU runtime."""
import asyncio
import threading
from pathlib import Path
import sys
import types
import pytest
import engine as module
from engine import EngineError, MODEL_ID, task_arguments
from conftest import WAV, make_engine


def test_language_and_attribute_mapping_matches_upstream_contract():
    import tts_tasks
    reference = (Path('reference.wav'), 'Reference words')
    for features in ({'language_id': 'Japanese'}, {'emotion': '高兴'},
                     {'speed': '慢速', 'pitch': '中', 'volume': '低'}):
        arguments = task_arguments(None, reference, features)
        built = tts_tasks.build_task_generation_kwargs(text='Hello [Sigh]', **arguments)
        assert built['text'] == 'Hello [Sigh]'
        assert built['use_spk_emb'] is True


def test_snapshot_is_offline_and_has_only_dense_repo(tmp_path, monkeypatch):
    seen = []
    for name in ('config.json', 'model.safetensors', 'campplus.onnx', 'tokenizer.json'):
        (tmp_path / name).touch()
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(
        snapshot_download=lambda *args, **kwargs: seen.append((args, kwargs)) or str(tmp_path)))
    module._snapshot(MODEL_ID, 'main')
    assert seen[0][0] == (MODEL_ID,)
    assert seen[0][1]['local_files_only'] is True
    assert 'AI' in seen[0][1]['cache_dir']


def test_load_requires_gpu_before_constructing_ming(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_snapshot', lambda *_: tmp_path)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(EngineError, match='GPU') as error:
        module.load(MODEL_ID, 'main', 'cuda:0')
    assert error.value.status == 503


def test_ming_constructor_and_waveform_adapter(tmp_path, monkeypatch):
    """Verify real constructor keywords and synthesize call without GPU libraries."""
    calls = []
    monkeypatch.setattr(module, '_snapshot', lambda *_: tmp_path)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))
    fake = types.SimpleNamespace(sample_rate=24000)
    monkeypatch.setitem(sys.modules, 'ming_tts', types.SimpleNamespace(MingTTS=lambda *a, **kw: calls.append((a, kw)) or fake))
    speaker = module.load(MODEL_ID, 'revision', 'cuda:0')
    assert speaker.rate == 24000
    assert calls == [((str(tmp_path),), {'device': 'cuda:0', 'text_normalization': False, 'cache_prompts': False})]


def test_cancelled_generation_keeps_slot_until_worker_finishes():
    entered, finish = threading.Event(), threading.Event()
    async def run():
        engine = make_engine()
        def work():
            entered.set()
            assert finish.wait(5)
        task = asyncio.create_task(engine._work(work))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        second_started = threading.Event()
        second = asyncio.create_task(engine._work(second_started.set))
        await asyncio.sleep(.05)
        assert not second_started.is_set() and not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await second
        assert second_started.is_set()
    asyncio.run(run())


def test_single_load_and_replica_marker(tmp_path):
    async def run():
        marker = tmp_path / 'marker'
        engine = make_engine(marker=marker)
        loads = []
        engine._loader = lambda *_: loads.append(1) or engine.fake
        await engine.warm()
        await engine.synthesize('hello', None, 1, None)
        assert len(loads) == 1
        marker.write_text('another-replica', 'utf-8')
        await engine.sync()
        assert engine.ready is False and engine.generation == 1
    asyncio.run(run())


def _wav(seconds: float) -> bytes:
    import io
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(b'\0\0' * int(8000 * seconds))
    return buffer.getvalue()


def test_audible_needs_a_sample_above_the_silence_peak():
    assert not module.audible(b'')
    assert not module.audible(b'\0\0' * 100)
    assert not module.audible(b'\x01\0' * 100)  # about -90 dB
    assert module.audible(b'\0\x20' * 100)


def _silent_engine():
    engine = make_engine()
    engine.fake.speak = lambda *args: b'\x01\0' * 2400
    return engine


def test_silent_output_is_an_error_not_a_wav():
    with pytest.raises(EngineError, match='no audible speech') as error:
        asyncio.run(_silent_engine().synthesize('hello', None, 1, None))
    assert error.value.status == 500


def test_silent_clone_with_long_reference_names_the_length():
    with pytest.raises(EngineError, match='over about 15 seconds') as error:
        asyncio.run(_silent_engine().synthesize('hello', None, 1, None, clip=(_wav(16), 'words')))
    assert error.value.status == 422


def test_silent_clone_with_short_reference_gives_no_length_hint():
    with pytest.raises(EngineError) as error:
        asyncio.run(_silent_engine().synthesize('hello', None, 1, None, clip=(_wav(5), 'words')))
    assert 'over about 15 seconds' not in str(error.value) and error.value.status == 422


def test_text_is_spoken_sentence_by_sentence_with_a_pause_between():
    engine = make_engine()
    model = engine.fake
    result = asyncio.run(engine.synthesize("First one. Second one! Third?", None, 7, None))
    assert [call[0] for call in model.calls] == ["First one.", "Second one!", "Third?"]
    assert {call[1] for call in model.calls} == {7}  # same seed keeps one timbre
    assert result["seconds"] == pytest.approx(3 * 0.1 + 2 * module.PAUSE_SECONDS)


def test_sentences_ignores_numbers_and_empty_pieces():
    assert module.sentences("It is 3.14 at 10:30. ... Done.") == ["It is 3.14 at 10:30.", "Done."]


def test_tuning_knobs_reach_upstream_and_unset_ones_do_not():
    engine = make_engine()
    asyncio.run(engine.synthesize("Hi there.", None, 1, None, cfg=3.0, max_length=60))
    args = engine.fake.calls[0][2]
    assert args["cfg"] == 3.0 and args["max_length"] == 60 and "sigma" not in args


@pytest.mark.cloning_off
def test_cloning_is_refused_for_one_call_and_saved_references(tmp_path):
    engine = make_engine()
    with pytest.raises(EngineError, match="cloning is turned off") as error:
        asyncio.run(engine.synthesize("Hi.", None, 1, None, clip=(b"x", "text")))
    assert error.value.status == 422
    engine.store.put("mine", WAV, "Reference text")
    with pytest.raises(EngineError, match="cloning is turned off"):
        asyncio.run(engine.synthesize("Hi.", "mine", 1, None))
    assert asyncio.run(engine.synthesize("Hi.", "US_Emma", 1, None))["voice"] == "US_Emma"
