# CLI and asynchronous client

Start the server with the [native guide](guide-no-docker-same-device.md); explicitly download weights in admin Updates before synthesis. Port 29324.

Save this complete example as client.py and run `uv run --with httpx python client.py`. Set MING_TOKEN in the process environment to a public token. Network/file I/O runs without blocking the event loop.

```python
"""Generate speech through the running Ming service."""
import asyncio
import os
from pathlib import Path
import httpx

class SpeechClient:
    """One reusable asynchronous HTTP session per client."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def speak(self, text: str, destination: Path) -> None:
        response = await self.client.post('/v1/speak', json={'text': text, 'voice': 'US_Emma'})
        response.raise_for_status()
        await asyncio.to_thread(destination.write_bytes, response.content)

async def main() -> None:
    async with httpx.AsyncClient(base_url='http://127.0.0.1:29324', timeout=300,
                                headers={'X-Internal-Token': os.environ['MING_TOKEN']}) as client:
        await SpeechClient(client).speak('Hello from Ming.', Path('speech.wav'))

if __name__ == '__main__':
    asyncio.run(main())
```
