"""Minimal unofficial Talkie web API example.

This script calls the same streaming endpoint used by the Talkie web chat UI:

    POST https://api.talkie-lm.com/api/chat/stream

It is not an official public API. Keep request volume low, add delays for
experiments, and be prepared for the endpoint schema to change.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator

import requests


TALKIE_STREAM_URL = "https://api.talkie-lm.com/api/chat/stream"


def iter_talkie_events(
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 256,
    timeout: int = 120,
) -> Iterator[tuple[str, str]]:
    """Yield ``(event_name, data)`` pairs from Talkie's SSE stream."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "accept": "text/event-stream",
        "content-type": "application/json",
        "origin": "https://talkie-lm.com",
        "referer": "https://talkie-lm.com/",
        "user-agent": "talkie-project-example/0.1",
    }

    with requests.post(
        TALKIE_STREAM_URL,
        json=payload,
        headers=headers,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()

        event_name = "message"
        data_lines: list[str] = []

        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            line = raw_line.rstrip("\r")
            if not line:
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                    event_name = "message"
                    data_lines = []
                continue

            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())

        if data_lines:
            yield event_name, "\n".join(data_lines)


def ask_talkie(
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 256,
    stream_to_stdout: bool = False,
) -> str:
    """Return the full assistant answer from the unofficial web endpoint."""
    chunks: list[str] = []

    for event_name, data in iter_talkie_events(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        if event_name == "token":
            chunks.append(data)
            if stream_to_stdout:
                print(data, end="", flush=True)
        elif event_name == "moderation":
            moderation = _try_parse_json(data)
            if moderation and moderation.get("is_unsafe"):
                print("\n[moderation] response marked unsafe")
        elif event_name == "done":
            done = _try_parse_json(data)
            if done and stream_to_stdout:
                print(f"\n[done] finish_reason={done.get('finish_reason')}")

    if stream_to_stdout:
        print()
    return "".join(chunks)


def _try_parse_json(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call the unofficial Talkie web chat streaming endpoint."
    )
    parser.add_argument("prompt", nargs="?", default="What is radio?")
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("-n", "--max-tokens", type=int, default=256)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print the answer after the request finishes.",
    )
    args = parser.parse_args()

    answer = ask_talkie(
        args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream_to_stdout=not args.no_stream,
    )

    if args.no_stream:
        print(answer)


if __name__ == "__main__":
    main()
