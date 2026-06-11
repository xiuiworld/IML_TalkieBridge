"""Response collection clients for manual CSV and optional Talkie SSE calls."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from talkie_bridge.data_schema import read_csv_dicts, read_jsonl


TALKIE_STREAM_URL = "https://api.talkie-lm.com/api/chat/stream"


def load_manual_responses(path: Path) -> dict[tuple[str, str], str]:
    return {
        key: str(record.get("raw_response", ""))
        for key, record in load_manual_response_records(path).items()
    }


def load_manual_response_records(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    responses: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv_dicts(path):
        item_id = row.get("item_id") or row.get("id")
        condition = row.get("condition")
        raw = row.get("raw_response_manual") or row.get("raw_response") or ""
        if item_id and condition:
            responses[(item_id, condition)] = {
                "raw_response": raw,
                "prompt_hash": row.get("prompt_hash", ""),
            }
    return responses


class JsonlCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(path):
            key = str(row.get("cache_key", ""))
            if key:
                self.rows[key] = row

    def get(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(key)

    def put(self, row: dict[str, Any]) -> None:
        key = str(row["cache_key"])
        self.rows[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        last_error: OSError | None = None
        for attempt in range(5):
            try:
                with self.path.open("a", encoding="utf-8", newline="") as handle:
                    handle.write(line)
                    handle.write("\n")
                return
            except OSError as exc:
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(0.5 * (attempt + 1))
        if last_error is not None:
            raise last_error


class UnofficialTalkieApiClient:
    def __init__(
        self,
        *,
        cache_path: Path,
        temperature: float = 0.0,
        max_tokens: int = 8,
        timeout: int = 120,
        request_delay: float = 1.0,
        max_retries: int = 6,
        retry_base_delay: float = 30.0,
        retry_max_delay: float = 300.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.cache = JsonlCache(cache_path)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.sleep_fn = sleep_fn
        self.event_callback = event_callback

    def ask(self, prompt: str) -> dict[str, Any]:
        cache_key = _cache_key(prompt, self.temperature, self.max_tokens)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._emit({"event": "cache_hit", "cache_key": cache_key})
            row = dict(cached)
            row["cache_hit"] = True
            return row
        self._emit({"event": "request_start", "cache_key": cache_key})
        answer = "".join(
            chunk
            for event, chunk in self._iter_events(prompt)
            if event == "token"
        )
        row = {
            "cache_key": cache_key,
            "raw_response": answer,
            "provider": "unofficial_talkie_api",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_hit": False,
        }
        self.cache.put(row)
        self._emit({"event": "response_complete", "cache_key": cache_key, "response_chars": len(answer)})
        if self.request_delay > 0:
            self._emit({"event": "request_delay", "delay_seconds": self.request_delay})
            self.sleep_fn(self.request_delay)
        return row

    def _iter_events(self, prompt: str) -> Iterator[tuple[str, str]]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("The unofficial API provider requires the optional 'requests' package.") from exc

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "origin": "https://talkie-lm.com",
            "referer": "https://talkie-lm.com/",
            "user-agent": "talkie-bridge/0.1",
        }
        for attempt in range(self.max_retries + 1):
            try:
                with requests.post(
                    TALKIE_STREAM_URL,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout,
                ) as response:
                    status_code = getattr(response, "status_code", 0)
                    if _is_retryable_status(status_code) and attempt < self.max_retries:
                        delay = self._retry_delay(attempt, response)
                        self._emit(
                            {
                                "event": "retry_wait",
                                "attempt": attempt + 1,
                                "max_retries": self.max_retries,
                                "delay_seconds": delay,
                                "status_code": status_code,
                                "reason": f"HTTP {status_code}",
                            }
                        )
                        self.sleep_fn(delay)
                        continue
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
                    return
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    self._emit(
                        {
                            "event": "request_failed",
                            "attempt": attempt + 1,
                            "max_retries": self.max_retries,
                            "reason": str(exc),
                        }
                    )
                    raise
                delay = self._retry_delay(attempt, None)
                self._emit(
                    {
                        "event": "retry_wait",
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "delay_seconds": delay,
                        "status_code": "",
                        "reason": str(exc),
                    }
                )
                self.sleep_fn(delay)

    def _retry_delay(self, attempt: int, response: Any | None) -> float:
        retry_after = _retry_after_seconds(response)
        if retry_after is not None:
            return min(retry_after, self.retry_max_delay)
        delay = self.retry_base_delay * (2**attempt)
        return min(delay, self.retry_max_delay)

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event)


def _cache_key(prompt: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps(
        {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def _retry_after_seconds(response: Any | None) -> float | None:
    if response is None:
        return None
    headers = getattr(response, "headers", {}) or {}
    raw_value = headers.get("retry-after") or headers.get("Retry-After")
    if raw_value is None:
        return None
    try:
        return max(float(raw_value), 0.0)
    except (TypeError, ValueError):
        return None
