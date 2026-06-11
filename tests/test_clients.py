from __future__ import annotations

import sys
import types
from pathlib import Path

from talkie_bridge.clients import UnofficialTalkieApiClient


class _FakeResponse:
    def __init__(self, status_code: int, lines: list[str] | None = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._lines = lines or []
        self.headers = headers or {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, *, decode_unicode: bool) -> list[str]:
        assert decode_unicode is True
        return self._lines


def test_unofficial_api_retries_after_rate_limit(monkeypatch, tmp_path: Path) -> None:
    calls = []
    responses = [
        _FakeResponse(429, headers={"Retry-After": "0"}),
        _FakeResponse(200, ["event: token", "data: C", ""]),
    ]

    fake_requests = types.SimpleNamespace(
        RequestException=RuntimeError,
        post=lambda *args, **kwargs: calls.append((args, kwargs)) or responses.pop(0),
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    sleeps: list[float] = []
    events: list[dict[str, object]] = []
    client = UnofficialTalkieApiClient(
        cache_path=tmp_path / "cache.jsonl",
        max_retries=1,
        retry_base_delay=0,
        request_delay=0,
        sleep_fn=sleeps.append,
        event_callback=events.append,
    )

    answer = client.ask("Choose one: A/B/C/D")

    assert answer["raw_response"] == "C"
    assert len(calls) == 2
    assert sleeps == [0.0]
    assert any(event["event"] == "retry_wait" and event["reason"] == "HTTP 429" for event in events)
    assert any(event["event"] == "response_complete" for event in events)
