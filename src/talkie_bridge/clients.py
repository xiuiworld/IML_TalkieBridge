"""Response collection clients for manual CSV and optional Talkie SSE calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from talkie_bridge.data_schema import read_csv_dicts, read_jsonl, write_jsonl


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
        write_jsonl(self.path, self.rows.values())


class UnofficialTalkieApiClient:
    def __init__(
        self,
        *,
        cache_path: Path,
        temperature: float = 0.0,
        max_tokens: int = 8,
        timeout: int = 120,
    ) -> None:
        self.cache = JsonlCache(cache_path)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def ask(self, prompt: str) -> dict[str, Any]:
        cache_key = _cache_key(prompt, self.temperature, self.max_tokens)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
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
        }
        self.cache.put(row)
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
        with requests.post(
            TALKIE_STREAM_URL,
            json=payload,
            headers=headers,
            stream=True,
            timeout=self.timeout,
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


def _cache_key(prompt: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps(
        {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
