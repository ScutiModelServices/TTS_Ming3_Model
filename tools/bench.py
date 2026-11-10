"""Times one TTS service: py tools/bench.py http://127.0.0.1:29324 [--token T] [--voice V] [--runs 3] [--stream] [--save DIR]

Sends the same sentences to /v1/speak and prints, per sentence, the wait, the audio length and how many times faster
than real time it was. Calls carry no seed, so the answer cache never serves them.
With --stream it asks for the audio sentence by sentence and also prints the wait for the first audio. Run it against TTS_Piper_Model, TTS_Paradee_Model and TTS_Ming3_Model with the same sentences to compare them.
Uses only the standard library."""
import argparse
import json
import pathlib
import time
import urllib.request

SENTENCES = [
    "Hello there.",
    "In the wild, parakeets fly in loud, colorful flocks across the open grasslands of Australia.",
    "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore. Tomatoes and avocados were on the schedule.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--token", default="")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--save", default="")
    args = parser.parse_args()
    for number, text in enumerate(SENTENCES, 1):
        for run in range(1, args.runs + 1):
            body = {"text": text, **({"voice": args.voice} if args.voice else {}),
                    **({"stream": True} if args.stream else {})}
            request = urllib.request.Request(args.url.rstrip("/") + "/v1/speak", json.dumps(body).encode(),
                                             {"Content-Type": "application/json", "X-Internal-Token": args.token})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=300) as reply:
                first = reply.read(1024)
                first_wait = time.perf_counter() - started
                audio, headers = first + reply.read(), reply.headers
            wait = time.perf_counter() - started
            rate = int.from_bytes(audio[24:28], "little")
            seconds = (len(audio) - 44) / 2 / rate
            print(f"sentence {number} run {run}: wait {wait:.3f}s  audio {seconds:.2f}s  {seconds / wait:5.1f}x real time  voice {headers['X-Voice']}"
                  + (f"  first audio after {first_wait:.3f}s" if args.stream else ""))
            if args.save and run == 1:
                folder = pathlib.Path(args.save)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{number}-{headers['X-Voice']}.wav").write_bytes(audio)


if __name__ == "__main__":
    main()
