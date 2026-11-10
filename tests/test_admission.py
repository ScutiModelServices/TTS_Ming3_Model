"""Saturation must preserve responsive health and single-model ownership."""
import asyncio
import threading
import httpx
from app import create_app
from conftest import make_engine


def test_admission_is_bounded_and_health_does_not_wait_for_inference():
    async def run():
        entered, finish = threading.Event(), threading.Event()
        engine = make_engine()
        original = engine.fake.speak
        def slow(*args):
            entered.set()
            assert finish.wait(5)
            return original(*args)
        engine.fake.speak = slow
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine, max_inflight=1)), base_url='http://test') as client:
            active = asyncio.create_task(client.post('/v1/speak', json={'text': 'hello'}))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                health = await asyncio.wait_for(client.get('/health'), 1)
                assert health.status_code == 200
                rejected = await client.post('/v1/speak', json={'text': 'waiting'})
                assert rejected.status_code == 429
                assert rejected.headers['retry-after'] == '5'
            finally:
                finish.set()
            assert (await active).status_code == 200
    asyncio.run(run())
