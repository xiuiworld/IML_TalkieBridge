"""Fill TalkieBridge judge outputs with the OpenAI API.

The API key is read only from OPENAI_API_KEY. Do not hard-code it in this file.
The script is resumable: rows with nonblank judge_raw_output are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


RESPONSES_API_URL = "https://api.openai.com/v1/responses"
CHAT_COMPLETIONS_API_URL = "https://api.openai.com/v1/chat/completions"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", default="input_data/open_ended_judge_input_sheet.csv")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api", choices=["responses", "chat"], default="responses")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--limit", type=int, default=0, help="Maximum blank rows to fill in this invocation.")
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    path = Path(args.sheet)
    rows = read_rows(path)
    total = len(rows)
    blanks = [idx for idx, row in enumerate(rows) if not row.get("judge_raw_output", "").strip()]
    to_fill = blanks[: args.limit] if args.limit and args.limit > 0 else blanks
    print(f"[judge] sheet={path} model={args.model} api={args.api} total={total} blank={len(blanks)} filling={len(to_fill)}", flush=True)

    started_at = time.monotonic()
    completed = 0
    for idx in to_fill:
        row = rows[idx]
        pair_id = row.get("pair_id", "")
        prompt = row.get("judge_prompt", "")
        item_start = time.monotonic()
        print(f"[judge:start] {completed + 1}/{len(to_fill)} row={idx + 1}/{total} {pair_id}", flush=True)
        output = call_openai(
            api_key=api_key,
            model=args.model,
            api=args.api,
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            reasoning_effort=args.reasoning_effort,
        )
        row["judge_raw_output"] = output
        write_rows(path, rows)
        completed += 1
        elapsed = time.monotonic() - started_at
        avg = elapsed / completed if completed else 0.0
        remaining = len(to_fill) - completed
        eta = avg * remaining
        print(
            f"[judge:done] {completed}/{len(to_fill)} {pair_id}; "
            f"item={format_duration(time.monotonic() - item_start)}; "
            f"elapsed={format_duration(elapsed)}; eta={format_duration(eta)}; chars={len(output)}",
            flush=True,
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    remaining_blanks = sum(1 for row in rows if not row.get("judge_raw_output", "").strip())
    print(f"[judge:complete] filled={completed} remaining_blank={remaining_blanks}", flush=True)
    return 0


def call_openai(
    *,
    api_key: str,
    model: str,
    api: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str,
) -> str:
    if api == "chat":
        payload = chat_payload(model=model, prompt=prompt, max_tokens=max_tokens)
        url = CHAT_COMPLETIONS_API_URL
    else:
        payload = responses_payload(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        url = RESPONSES_API_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                delay = retry_delay(attempt, response)
                print(
                    f"[judge:retry] HTTP {response.status_code}; retry {attempt + 1}/5 after {format_duration(delay)}",
                    flush=True,
                )
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
            data = response.json()
            content = extract_output_text(data, api=api)
            validate_json(content)
            return content
        except (requests.RequestException, KeyError, ValueError, RuntimeError) as exc:
            last_error = exc
            if isinstance(exc, RuntimeError) and str(exc).startswith("HTTP 4"):
                break
            if attempt >= 5:
                break
            delay = retry_delay(attempt, None)
            print(f"[judge:retry] {exc}; retry {attempt + 1}/5 after {format_duration(delay)}", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"OpenAI judge call failed after retries: {last_error}")


def responses_payload(*, model: str, prompt: str, max_tokens: int, reasoning_effort: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "instructions": "You are a strict evaluation judge. Return only valid JSON.",
        "input": prompt,
        "max_output_tokens": max_tokens,
        "text": {"format": {"type": "json_object"}},
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def chat_payload(*, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluation judge. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    return payload


def extract_output_text(data: dict[str, Any], *, api: str) -> str:
    if api == "chat":
        return data["choices"][0]["message"]["content"]
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    if parts:
        return "".join(parts)
    raise KeyError("Could not find output_text in Responses API result.")


def validate_json(text: str) -> None:
    payload = json.loads(text)
    winner = str(payload.get("winner", ""))
    if winner not in {"A", "B", "Tie"}:
        raise ValueError(f"Invalid winner: {winner!r}")


def retry_delay(attempt: int, response: requests.Response | None) -> float:
    if response is not None:
        raw = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if raw:
            try:
                return min(max(float(raw), 0.0), 120.0)
            except ValueError:
                pass
    return min(2.0 * (2**attempt), 60.0)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys())
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {rest}s"
    return f"{minutes // 60}h {minutes % 60}m {rest}s"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[judge:error] {exc}", file=sys.stderr)
        raise
