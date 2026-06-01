"""Dataset quality, qualitative examples, and demo helpers."""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from talkie_bridge.data_schema import CONDITIONS, INVALID_LABEL, json_loads, write_csv_dicts
from talkie_bridge.detector import tokenize


def compute_dataset_quality(prompt_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_rows = [row for row in prompt_rows if row.get("condition") == "raw"]
    rewrite_rows = [row for row in prompt_rows if row.get("condition") != "raw"]
    rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any) -> None:
        rows.append({"section": section, "metric": metric, "value": value})

    add("dataset", "n_items", len(raw_rows))
    add("dataset", "human_validated_count", sum(1 for row in raw_rows if _as_bool(row.get("human_validated"))))
    add("length", "mean_original_question_tokens", _mean(len(tokenize(str(row.get("original_question", "")))) for row in raw_rows))
    add("length", "mean_choice_tokens", _mean(_choice_length(row) for row in raw_rows))

    for domain, count in sorted(Counter(str(row.get("domain", "unknown")) for row in raw_rows).items()):
        add("domain_count", domain, count)
    for split, count in sorted(Counter(str(row.get("split", "unknown")) for row in raw_rows).items()):
        add("split_count", split, count)
    for answer, count in sorted(Counter(str(row.get("gold_answer", "")) for row in raw_rows).items()):
        add("answer_distribution", answer, count)

    for condition in CONDITIONS:
        condition_rows = [row for row in rewrite_rows if row.get("condition") == condition]
        if not condition_rows:
            continue
        validations = [json_loads(row.get("rewrite_validation", {}), {}) for row in condition_rows]
        add("rewrite_length", f"{condition}_mean_rewritten_tokens", _mean(len(tokenize(str(row.get("rewritten_question", "")))) for row in condition_rows))
        add("leakage", f"{condition}_leakage_risk_count", sum(1 for report in validations if report.get("leakage_risk")))
        add("choice_copying", f"{condition}_mean_choice_copying_score", _mean(float(report.get("choice_copying_score", 0.0)) for report in validations))
        add("choice_copying", f"{condition}_max_choice_copying_score", max([float(report.get("choice_copying_score", 0.0)) for report in validations] or [0.0]))
        add("answer_hint", f"{condition}_mean_choice_keyword_overlap_score", _mean(float(report.get("choice_keyword_overlap_score", 0.0)) for report in validations))
        add("answer_hint", f"{condition}_max_choice_keyword_overlap_score", max([float(report.get("choice_keyword_overlap_score", 0.0)) for report in validations] or [0.0]))
    return rows


def add_length_control_diagnostics(prompt_rows: Sequence[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> None:
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in prompt_rows:
        by_item[str(row.get("item_id"))][str(row.get("condition"))] = row
    deltas: list[int] = []
    ratios: list[float] = []
    for item_rows in by_item.values():
        length_row = item_rows.get("length_controlled")
        proposed_row = item_rows.get("proposed")
        if not length_row or not proposed_row:
            continue
        length_tokens = len(tokenize(str(length_row.get("rewritten_question", ""))))
        proposed_tokens = len(tokenize(str(proposed_row.get("rewritten_question", ""))))
        deltas.append(length_tokens - proposed_tokens)
        ratios.append(length_tokens / proposed_tokens if proposed_tokens else 0.0)
    if deltas:
        quality_rows.extend(
            [
                {"section": "length_control", "metric": "mean_delta_vs_proposed_tokens", "value": _mean(deltas)},
                {"section": "length_control", "metric": "max_abs_delta_vs_proposed_tokens", "value": max(abs(value) for value in deltas)},
                {"section": "length_control", "metric": "mean_ratio_to_proposed", "value": _mean(ratios)},
            ]
        )


def add_proposed_difference_diagnostics(prompt_rows: Sequence[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> None:
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in prompt_rows:
        by_item[str(row.get("item_id"))][str(row.get("condition"))] = row
    rewrite_diff = 0
    primitive_diff = 0
    added_counts: list[int] = []
    n = 0
    for item_rows in by_item.values():
        rule = item_rows.get("rule_only")
        proposed = item_rows.get("proposed")
        if not rule or not proposed:
            continue
        n += 1
        if str(rule.get("rewritten_question", "")) != str(proposed.get("rewritten_question", "")):
            rewrite_diff += 1
        rule_primitives = set(json_loads(rule.get("mapped_primitives", []), []))
        proposed_primitives = set(json_loads(proposed.get("mapped_primitives", []), []))
        added = proposed_primitives - rule_primitives
        if proposed_primitives != rule_primitives:
            primitive_diff += 1
        added_counts.append(len(added))
    if n:
        quality_rows.extend(
            [
                {"section": "proposed_difference", "metric": "rewrite_differs_from_rule_only_rate", "value": rewrite_diff / n},
                {"section": "proposed_difference", "metric": "primitive_set_differs_from_rule_only_rate", "value": primitive_diff / n},
                {"section": "proposed_difference", "metric": "mean_autoencoder_added_primitives", "value": _mean(added_counts)},
            ]
        )


def write_dataset_quality_markdown(path: Path, quality_rows: Sequence[dict[str, Any]]) -> None:
    lines = ["# Dataset Quality Report", ""]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        grouped[str(row["section"])].append(row)
    for section, rows in grouped.items():
        lines.extend([f"## {section.replace('_', ' ').title()}", "", "| Metric | Value |", "| --- | --- |"])
        for row in rows:
            lines.append(f"| {row['metric']} | {_fmt(row['value'])} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_qualitative_examples(path: Path, prompt_rows: Sequence[dict[str, Any]], result_rows: Sequence[dict[str, Any]]) -> None:
    prompts_by_key = {(str(row["item_id"]), str(row["condition"])): row for row in prompt_rows}
    results_by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in result_rows:
        results_by_item[str(row["item_id"])][str(row["condition"])] = row

    sections = {
        "Raw wrong, proposed right": [],
        "Raw right, proposed wrong": [],
        "Invalid responses": [],
        "Repaired rewrites": [],
        "High leakage or choice-copying": [],
    }
    for item_id, condition_rows in results_by_item.items():
        raw = condition_rows.get("raw")
        proposed = condition_rows.get("proposed")
        if raw and proposed and not _as_bool(raw.get("correct")) and _as_bool(proposed.get("correct")):
            sections["Raw wrong, proposed right"].append((item_id, raw, proposed))
        if raw and proposed and _as_bool(raw.get("correct")) and not _as_bool(proposed.get("correct")):
            sections["Raw right, proposed wrong"].append((item_id, raw, proposed))
        for condition, result in condition_rows.items():
            if result.get("parsed_answer") == INVALID_LABEL:
                sections["Invalid responses"].append((item_id, result, result))
            prompt = prompts_by_key.get((item_id, condition), {})
            if int(prompt.get("repair_attempts") or 0) > 0:
                sections["Repaired rewrites"].append((item_id, result, result))
            validation = json_loads(prompt.get("rewrite_validation", {}), {})
            if validation.get("leakage_risk") or float(validation.get("choice_copying_score", 0.0)) >= 0.48:
                sections["High leakage or choice-copying"].append((item_id, result, result))

    lines = ["# Qualitative Examples", ""]
    for title, examples in sections.items():
        lines.extend([f"## {title}", ""])
        if not examples:
            lines.extend(["No examples.", ""])
            continue
        for item_id, left, right in examples[:3]:
            condition = right.get("condition", "")
            prompt = prompts_by_key.get((item_id, condition), {})
            lines.extend(
                [
                    f"### {item_id} / {condition}",
                    "",
                    f"- Gold: `{right.get('gold_answer', '')}`",
                    f"- Parsed: `{right.get('parsed_answer', '')}`",
                    f"- Raw response: `{str(right.get('raw_response', ''))[:160]}`",
                    f"- Rewritten question: {prompt.get('rewritten_question', '')}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_demo_html(path: Path, prompt_rows: Sequence[dict[str, Any]], item_id: str) -> None:
    rows = [row for row in prompt_rows if str(row.get("item_id")) == item_id]
    if not rows:
        raise ValueError(f"No prompt rows found for item_id={item_id}")
    raw = next((row for row in rows if row.get("condition") == "raw"), rows[0])
    body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>TalkieBridge Demo</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45}section{border:1px solid #ccc;padding:16px;margin:12px 0}code{white-space:pre-wrap}</style>",
        "</head><body>",
        f"<h1>TalkieBridge Demo: {html.escape(item_id)}</h1>",
        f"<p><strong>Domain:</strong> {html.escape(str(raw.get('domain','')))} | <strong>Split:</strong> {html.escape(str(raw.get('split','')))}</p>",
        f"<h2>Original</h2><p>{html.escape(str(raw.get('original_question','')))}</p>",
        f"<p><strong>Gold answer:</strong> {html.escape(str(raw.get('gold_answer','')))}</p>",
    ]
    for row in rows:
        validation = json_loads(row.get("rewrite_validation", {}), {})
        body.extend(
            [
                f"<section><h2>{html.escape(str(row.get('condition')))}</h2>",
                f"<p><strong>Detected:</strong> {html.escape(str(row.get('detected_terms', [])))}</p>",
                f"<p><strong>Primitive candidates:</strong> {html.escape(str(row.get('mapped_primitives', [])))}</p>",
                f"<p><strong>Validation:</strong> {html.escape(str(validation))}</p>",
                f"<h3>Question</h3><code>{html.escape(str(row.get('rewritten_question','')))}</code>",
                "</section>",
            ]
        )
    body.extend(["</body></html>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def _choice_length(row: dict[str, Any]) -> float:
    return _mean(len(tokenize(str(row.get(f"choice_{label}", "")))) for label in ("A", "B", "C", "D"))


def _mean(values: Any) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
