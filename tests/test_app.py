"""Public/admin HTTP contract against an injected dense model stand-in."""
import base64
import asyncio
from fastapi.testclient import TestClient
import pytest
from app import create_app
from conftest import make_engine, WAV
from tokens import TokenStore
from options import SETTINGS
import settings


def test_health_speak_and_cache():
    engine = make_engine()
    with TestClient(create_app(engine, cache_size=4)) as client:
        assert client.get('/health').json()['ready'] is False
        body = {'text': 'Hello [Laughter]', 'seed': 1}
        response = client.post('/v1/speak', json=body)
        assert response.status_code == 200
        assert response.content.startswith(b'RIFF')
        assert response.headers['x-voice'] == 'US_Emma'
        assert response.headers['x-audio-seconds'] == '0.100'
        assert client.get('/health').json()['ready'] is True
        assert client.post('/v1/speak', json=body).headers['x-cache'] == 'hit'
        assert engine.fake.calls[0][2] == {'task': 'IP', 'controls': {'ip': 'US_Emma'}, 'use_zero_spk_emb': True}


@pytest.mark.parametrize('body', [{'text': ''}, {'text': 'x', 'oops': 1}, {'text': 'x', 'speed': 1.2},
    {'text': 'x', 'emotion': 'happy'}, {'text': 'x', 'speed': '快速'}, {'text': 'x', 'reference_text': 'missing clip'},
    {'text': 'x', 'voice_design': 'warm', 'emotion': '高兴'}, {'text': 'x', 'preset': 'stable'},
    {'text': 'x', 'voice_design': 'warm', 'voice': 'US_Emma'}])
def test_validation(body):
    assert TestClient(create_app(make_engine())).post('/v1/speak', json=body).status_code == 422


def test_clone_multilingual_control_and_temp_cleanup():
    from pathlib import Path
    engine = make_engine()
    client = TestClient(create_app(engine, cache_size=4))
    clip = {'reference_audio_base64': base64.b64encode(WAV).decode(), 'reference_text': 'Reference text'}
    for extra, task in [({'language_id': 'Japanese'}, 'MULTILINGUAL'),
                         ({'speed': '快速', 'pitch': '高', 'volume': '中'}, 'BASIC'), ({'emotion': '高兴'}, 'EMOTION')]:
        response = client.post('/v1/speak', json={'text': 'Hello', 'seed': 1, **clip, **extra})
        assert response.status_code == 200
        args = engine.fake.calls[-1][2]
        assert args['task'] == task
        assert not Path(args['prompt_wav_path']).exists()
        assert response.headers['x-cache'] == 'miss'
    assert len(engine.fake.calls) == 3


def test_design_stream_and_chinese_voice():
    engine = make_engine()
    client = TestClient(create_app(engine, cache_size=4))
    body = {'text': 'Hello', 'voice_design': 'Warm narrator', 'stream': True, 'seed': 42}
    response = client.post('/v1/speak', json=body)
    assert response.status_code == 200 and response.content.startswith(b'RIFF')
    assert response.headers['x-voice'] == 'designed'
    assert engine.fake.calls[-1][2]['task'] == 'VOICE_DESIGN'
    cached = client.post('/v1/speak', json=body)
    assert cached.headers['x-cache'] == 'hit' and cached.headers['x-voice'] == 'designed'
    assert client.post('/v1/speak', json={'text': '你好', 'voice': '灵楷'}).status_code == 200


def test_auth_scopes_admin_and_error_format(tmp_path):
    store = TokenStore(tmp_path / 't.json')
    config = settings.Config(store.settings_path, SETTINGS)
    config.set({'warm': False, 'update_check_hours': 0})
    with TestClient(create_app(make_engine(), 'admin', tokens=store, config=config)) as client:
        assert client.get('/v1/voices').status_code == 401
        admin = {'X-Internal-Token': 'admin'}
        result = client.post('/admin/tokens', json={'name': 'test', 'projects': ['US_Emma']}, headers=admin)
        token = {'X-Internal-Token': result.json()['token']}
        assert client.post('/v1/speak', json={'text': 'x', 'voice': 'UK_George'}, headers=token).status_code == 403
        assert client.post('/v1/speak', json={'text': 'x', 'voice_design': 'warm'}, headers=token).status_code == 403
        for route in ['/admin/tokens', '/admin/settings', '/admin/voices', '/admin/workers', '/admin/updates', '/metrics']:
            assert client.get(route, headers=admin).status_code == 200
        assert client.get('/health').status_code == 200
        assert client.post('/v1/speak', json={'text': 'x'}).json() == {'detail': 'Invalid token'}


def test_missing_weights_503():
    def missing(*_):
        raise OSError('missing')
    response = TestClient(create_app(make_engine(loader=missing))).post('/v1/speak', json={'text': 'x'})
    assert response.status_code == 503 and 'Download' in response.json()['detail']
