"""Open-ended response-quality workflow helpers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Sequence

from talkie_bridge.clients import clean_talkie_response_text
from talkie_bridge.data_schema import CONDITIONS, DatasetItem, json_loads, stable_hash
from talkie_bridge.prompting import build_judge_prompt, build_open_ended_prompt, prompt_hash


PRIMARY_JUDGE_COMPARISONS = (
    ("raw", "proposed"),
    ("rule_only", "proposed"),
    ("length_controlled", "proposed"),
    ("proposed_no_validator", "proposed"),
)


def open_question_for_item(item: DatasetItem) -> str:
    if item.open_question.strip():
        return item.open_question.strip()
    if item.concept_term.strip() and item.task.strip():
        return f"Why might {item.concept_term.strip()} be useful for {item.task.strip()}?"
    return _mcq_stem_to_open_question(item.original_question)


def expected_mechanism_for_item(item: DatasetItem) -> str:
    return item.expected_mechanism.strip() or item.primitive_phrase.strip()


def judge_reference_points_for_item(item: DatasetItem) -> list[str]:
    if item.judge_reference_points:
        return list(item.judge_reference_points)
    if item.primitive_phrase.strip():
        return [item.primitive_phrase.strip()]
    return []


def leakage_sensitive_terms_for_item(item: DatasetItem) -> list[str]:
    if item.leakage_sensitive_terms:
        return list(item.leakage_sensitive_terms)
    return list(item.answer_leakage_terms)


def item_with_open_question(item: DatasetItem) -> DatasetItem:
    return replace(item, original_question=open_question_for_item(item))


def open_ended_prompt_row_for_item(item: DatasetItem, *, condition: str, question: str, artifact: Any | None) -> dict[str, Any]:
    validation = artifact.rewrite_validation if artifact else {"pass": True}
    detected_terms = artifact.detected_terms if artifact else []
    mapped_primitives = artifact.mapped_primitives if artifact else []
    repair_attempts = artifact.repair_attempts if artifact else 0
    pass_validation = artifact.pass_validation if artifact else True
    detector_token_scores = artifact.detector_token_scores if artifact else []
    model_info = artifact.model_info if artifact else {}
    prompt = build_open_ended_prompt(question)
    open_question = open_question_for_item(item)
    return {
        "item_id": item.id,
        "id": item.id,
        "domain": item.domain,
        "split": item.split,
        "condition": condition,
        "prompt_task": "open_ended_response_quality",
        "prompt": prompt,
        "prompt_hash": prompt_hash(prompt),
        "original_question": item.original_question,
        "open_question": open_question,
        "rewritten_question": question,
        "expected_mechanism": expected_mechanism_for_item(item),
        "judge_reference_points": judge_reference_points_for_item(item),
        "leakage_sensitive_terms": leakage_sensitive_terms_for_item(item),
        "choices": item.choices,
        "choice_A": item.choices["A"],
        "choice_B": item.choices["B"],
        "choice_C": item.choices["C"],
        "choice_D": item.choices["D"],
        "gold_answer": item.gold_answer,
        "gold_anachronism_terms": item.gold_anachronism_terms,
        "forbidden_terms": item.forbidden_terms,
        "required_primitives": item.required_primitives,
        "primitive_phrase": item.primitive_phrase,
        "concept_term": item.concept_term,
        "task": item.task,
        "human_validated": item.human_validated,
        "detected_terms": detected_terms,
        "detector_token_scores": detector_token_scores,
        "mapped_primitives": mapped_primitives,
        "model_info": model_info,
        "rewrite_validation": validation,
        "repair_attempts": repair_attempts,
        "pass_validation": pass_validation,
    }


def build_open_ended_response_rows(
    prompt_rows: Sequence[dict[str, Any]],
    responses: dict[tuple[str, str], dict[str, str] | str],
    *,
    provider: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in prompt_rows:
        response_record = responses.get((str(row["item_id"]), str(row["condition"])), {})
        if isinstance(response_record, str):
            raw_response = clean_talkie_response_text(response_record)
            response_prompt_hash = ""
        else:
            raw_response = clean_talkie_response_text(response_record.get("raw_response", ""))
            response_prompt_hash = response_record.get("prompt_hash", "")
        current_prompt_hash = str(row.get("prompt_hash", ""))
        rows.append(
            {
                "item_id": row["item_id"],
                "domain": row["domain"],
                "split": row.get("split", ""),
                "condition": row["condition"],
                "prompt_task": "open_ended_response_quality",
                "prompt": row["prompt"],
                "prompt_hash": current_prompt_hash,
                "response_prompt_hash": response_prompt_hash,
                "prompt_hash_match": bool(response_prompt_hash) and response_prompt_hash == current_prompt_hash,
                "raw_response": raw_response,
                "provider": provider,
                "open_question": row.get("open_question", ""),
                "rewritten_question": row.get("rewritten_question", ""),
                "expected_mechanism": row.get("expected_mechanism", ""),
                "judge_reference_points": row.get("judge_reference_points", []),
                "leakage_sensitive_terms": row.get("leakage_sensitive_terms", []),
                "rewrite_validation": row.get("rewrite_validation", {}),
                "repair_attempts": row.get("repair_attempts", 0),
            }
        )
    return rows


def build_judge_pair_rows(
    response_rows: Sequence[dict[str, Any]],
    *,
    seed: int = 13,
    comparisons: Sequence[tuple[str, str]] = PRIMARY_JUDGE_COMPARISONS,
) -> list[dict[str, Any]]:
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in response_rows:
        by_item[str(row["item_id"])][str(row["condition"])] = row

    output: list[dict[str, Any]] = []
    for item_id, condition_rows in sorted(by_item.items()):
        for baseline_condition, condition in comparisons:
            if baseline_condition not in condition_rows or condition not in condition_rows:
                continue
            baseline = condition_rows[baseline_condition]
            candidate = condition_rows[condition]
            pair_id = f"{item_id}::{condition}_vs_{baseline_condition}"
            swap = stable_hash(f"{seed}:{pair_id}") % 2 == 1
            first = candidate if swap else baseline
            second = baseline if swap else candidate
            response_a = clean_talkie_response_text(str(first.get("raw_response", "")))
            response_b = clean_talkie_response_text(str(second.get("raw_response", "")))
            question = str(candidate.get("open_question") or baseline.get("open_question") or "")
            expected = str(candidate.get("expected_mechanism") or baseline.get("expected_mechanism") or "")
            reference_points = _list_value(candidate.get("judge_reference_points") or baseline.get("judge_reference_points"))
            judge_prompt = build_judge_prompt(
                question=question,
                response_a=response_a,
                response_b=response_b,
                expected_mechanism=expected,
                reference_points=reference_points,
            )
            output.append(
                {
                    "pair_id": pair_id,
                    "item_id": item_id,
                    "domain": candidate.get("domain", baseline.get("domain", "")),
                    "split": candidate.get("split", baseline.get("split", "")),
                    "comparison": f"{condition}_vs_{baseline_condition}",
                    "baseline_condition": baseline_condition,
                    "condition": condition,
                    "condition_a": first["condition"],
                    "condition_b": second["condition"],
                    "question": question,
                    "expected_mechanism": expected,
                    "judge_reference_points": reference_points,
                    "response_a": response_a,
                    "response_b": response_b,
                    "judge_prompt": judge_prompt,
                    "judge_prompt_hash": prompt_hash(judge_prompt),
                    "randomization_seed": seed,
                }
            )
    return output


def judge_input_rows(pair_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": row["pair_id"],
            "judge_prompt_hash": row["judge_prompt_hash"],
            "judge_prompt": row["judge_prompt"],
            "judge_raw_output": "",
        }
        for row in pair_rows
    ]


def parse_judge_rows(
    pair_rows: Sequence[dict[str, Any]],
    judge_records: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    by_pair = {str(row["pair_id"]): row for row in pair_rows}
    parsed_rows: list[dict[str, Any]] = []
    for pair_id, pair in by_pair.items():
        record = judge_records.get(pair_id, {})
        raw_output = str(record.get("judge_raw_output", ""))
        winner, scores, rationale = parse_judge_output(raw_output)
        winner_condition = "Tie"
        if winner == "A":
            winner_condition = str(pair["condition_a"])
        elif winner == "B":
            winner_condition = str(pair["condition_b"])
        parsed: dict[str, Any] = {
            **pair,
            "response_a": pair.get("response_a", ""),
            "response_b": pair.get("response_b", ""),
            "judge_raw_output": raw_output,
            "response_judge_prompt_hash": record.get("judge_prompt_hash", ""),
            "judge_prompt_hash_match": bool(record.get("judge_prompt_hash")) and record.get("judge_prompt_hash") == pair.get("judge_prompt_hash"),
            "winner": winner,
            "winner_condition": winner_condition,
            "rationale": rationale,
        }
        for metric, values in scores.items():
            parsed[f"{metric}_a"] = values.get("A", "")
            parsed[f"{metric}_b"] = values.get("B", "")
        parsed_rows.append(parsed)
    return parsed_rows


def parse_judge_output(raw_output: str) -> tuple[str, dict[str, dict[str, int]], str]:
    text = raw_output.strip()
    payload = _load_json_object(text)
    if payload:
        winner = _normalize_winner(str(payload.get("winner", "")))
        scores = {
            metric: _score_pair(payload.get(metric, {}))
            for metric in (
                "task_relevance",
                "functional_correctness",
                "era_neutrality",
                "anachronism_handling",
                "usefulness",
                "leakage_risk",
            )
        }
        return winner, scores, str(payload.get("rationale", ""))

    winner = _normalize_winner(text)
    scores = {
        metric: {"A": 0, "B": 0}
        for metric in (
            "task_relevance",
            "functional_correctness",
            "era_neutrality",
            "anachronism_handling",
            "usefulness",
            "leakage_risk",
        )
    }
    return winner, scores, text[:500]


def compute_judge_pairwise_metrics(parsed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed_rows:
        grouped[str(row.get("comparison", ""))].append(row)

    output: list[dict[str, Any]] = []
    for comparison, rows in sorted(grouped.items()):
        if not comparison:
            continue
        n = len(rows)
        condition = str(rows[0].get("condition", ""))
        baseline = str(rows[0].get("baseline_condition", ""))
        wins = sum(1 for row in rows if row.get("winner_condition") == condition)
        baseline_wins = sum(1 for row in rows if row.get("winner_condition") == baseline)
        ties = sum(1 for row in rows if row.get("winner_condition") == "Tie")
        judged = wins + baseline_wins + ties
        output.append(
            {
                "comparison": comparison,
                "condition": condition,
                "baseline_condition": baseline,
                "n_pairs": n,
                "judged_pairs": judged,
                "condition_wins": wins,
                "baseline_wins": baseline_wins,
                "ties": ties,
                "condition_win_rate_all": wins / n if n else 0.0,
                "condition_win_rate_excluding_ties": wins / (wins + baseline_wins) if wins + baseline_wins else 0.0,
                "tie_rate": ties / n if n else 0.0,
                **_score_deltas(rows, condition=condition, baseline=baseline),
            }
        )
    return output


def judge_integrity_rows(parsed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(parsed_rows)
    hash_matches = sum(1 for row in parsed_rows if row.get("judge_prompt_hash_match"))
    blank = sum(1 for row in parsed_rows if not str(row.get("judge_raw_output", "")).strip())
    parsed = sum(1 for row in parsed_rows if row.get("winner") in {"A", "B", "Tie"})
    return [
        {"metric": "judge_prompt_hash_match_rate", "value": hash_matches / n if n else 0.0},
        {"metric": "blank_judge_output_count", "value": blank},
        {"metric": "judge_parse_rate", "value": parsed / n if n else 0.0},
        {"metric": "n_pairs", "value": n},
    ]


def write_response_quality_report(
    *,
    metrics: Sequence[dict[str, Any]],
    integrity: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Open-Ended Response Quality Report",
        "",
        "This report summarizes blind pairwise LLM Judge results for TalkieBridge open-ended responses.",
        "Condition labels are hidden in the judge prompt and restored only for analysis.",
        "",
        "## Summary",
        "",
        *_response_quality_summary(metrics, integrity),
        "",
        "## Pairwise Metrics",
        "",
        *_markdown_table(
            metrics,
            [
                "comparison",
                "n_pairs",
                "condition_wins",
                "baseline_wins",
                "ties",
                "condition_win_rate_all",
                "condition_win_rate_excluding_ties",
                "tie_rate",
            ],
        ),
        "",
        "## Judge Integrity",
        "",
        *_markdown_table(integrity, ["metric", "value"]),
        "",
        "## Claim Boundary",
        "",
        "- These results support claims about the preprocessing layer's effect on Talkie open-ended response quality.",
        "- They do not support a claim that Talkie itself was modified or improved.",
        "- They do not overturn the separate MCQ diagnostic result, where proposed preprocessing did not improve four-choice accuracy.",
        "- The `proposed_no_validator` ablation should be used to state whether the validator had a measurable downstream effect.",
        "",
    ]
    return "\n".join(lines)


def _mcq_stem_to_open_question(question: str) -> str:
    text = question.strip()
    text = re.sub(r"\bWhich explanation best supports using\b", "Why might using", text, flags=re.IGNORECASE)
    text = re.sub(r"\bWhich mechanism explains why\b", "Why", text, flags=re.IGNORECASE)
    text = re.sub(r"\bWhat is its most relevant advantage\?", "Why might it be useful?", text, flags=re.IGNORECASE)
    if not text.endswith("?"):
        text += "?"
    return text


def _list_value(value: Any) -> list[str]:
    loaded = json_loads(value, value)
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    if isinstance(loaded, str) and loaded.strip():
        return [loaded.strip()]
    return []


def _load_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}


def _normalize_winner(value: str) -> str:
    text = value.strip().upper()
    if text in {"A", "ANSWER A", "WINNER A"}:
        return "A"
    if text in {"B", "ANSWER B", "WINNER B"}:
        return "B"
    if text in {"TIE", "DRAW", "EQUAL"}:
        return "Tie"
    match = re.search(r'"?winner"?\s*[:\-]\s*"?([AB])"?', value, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if re.search(r'"?winner"?\s*[:\-]\s*"?tie"?', value, re.IGNORECASE):
        return "Tie"
    first = re.search(r"\b(A|B|TIE)\b", text)
    if first:
        return "Tie" if first.group(1) == "TIE" else first.group(1)
    return ""


def _score_pair(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {"A": 0, "B": 0}
    return {"A": _score_value(value.get("A")), "B": _score_value(value.get("B"))}


def _score_value(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(score, 5))


def _score_deltas(rows: Sequence[dict[str, Any]], *, condition: str, baseline: str) -> dict[str, Any]:
    metrics = (
        "task_relevance",
        "functional_correctness",
        "era_neutrality",
        "anachronism_handling",
        "usefulness",
        "leakage_risk",
    )
    deltas: dict[str, Any] = {}
    for metric in metrics:
        values: list[float] = []
        for row in rows:
            condition_score, baseline_score = _condition_baseline_scores(row, metric, condition=condition, baseline=baseline)
            if condition_score or baseline_score:
                values.append(condition_score - baseline_score)
        deltas[f"mean_{metric}_delta"] = sum(values) / len(values) if values else 0.0
    return deltas


def _condition_baseline_scores(row: dict[str, Any], metric: str, *, condition: str, baseline: str) -> tuple[float, float]:
    score_a = float(row.get(f"{metric}_a") or 0)
    score_b = float(row.get(f"{metric}_b") or 0)
    condition_score = 0.0
    baseline_score = 0.0
    if row.get("condition_a") == condition:
        condition_score = score_a
    elif row.get("condition_b") == condition:
        condition_score = score_b
    if row.get("condition_a") == baseline:
        baseline_score = score_a
    elif row.get("condition_b") == baseline:
        baseline_score = score_b
    return condition_score, baseline_score


def _response_quality_summary(metrics: Sequence[dict[str, Any]], integrity: Sequence[dict[str, Any]]) -> list[str]:
    by_comparison = {str(row.get("comparison", "")): row for row in metrics}
    lines: list[str] = []
    n_pairs = _integrity_value(integrity, "n_pairs")
    parse_rate = _integrity_value(integrity, "judge_parse_rate")
    hash_rate = _integrity_value(integrity, "judge_prompt_hash_match_rate")
    blank_count = _integrity_value(integrity, "blank_judge_output_count")
    if n_pairs:
        lines.append(
            f"- Judge coverage: {int(n_pairs)} pairs, parse rate {_format_rate(parse_rate)}, "
            f"prompt hash match rate {_format_rate(hash_rate)}, blank outputs {int(blank_count)}."
        )
    for comparison in (
        "proposed_vs_raw",
        "proposed_vs_rule_only",
        "proposed_vs_length_controlled",
        "proposed_vs_proposed_no_validator",
    ):
        row = by_comparison.get(comparison)
        if row:
            lines.append(f"- {_comparison_sentence(row)}")
    if not lines:
        return ["No judge metrics were available."]
    return lines


def _comparison_sentence(row: dict[str, Any]) -> str:
    comparison = str(row.get("comparison", ""))
    condition = str(row.get("condition", "condition"))
    baseline = str(row.get("baseline_condition", "baseline"))
    wins = int(row.get("condition_wins") or 0)
    losses = int(row.get("baseline_wins") or 0)
    ties = int(row.get("ties") or 0)
    n_pairs = int(row.get("n_pairs") or 0)
    win_rate = float(row.get("condition_win_rate_excluding_ties") or 0.0)
    all_rate = float(row.get("condition_win_rate_all") or 0.0)
    tie_word = "tie" if ties == 1 else "ties"
    non_tie_total = wins + losses
    non_tie_rate = _format_rate(win_rate) if non_tie_total else "not defined because all pairs tied"
    if wins > losses:
        direction = "beat"
    elif losses > wins:
        direction = "did not beat"
    else:
        direction = "tied"
    return (
        f"{comparison}: `{condition}` {direction} `{baseline}` "
        f"({wins} wins, {losses} losses, {ties} {tie_word} over {n_pairs} pairs; "
        f"win rate excluding ties {non_tie_rate}, all-pair win rate {_format_rate(all_rate)})."
    )


def _integrity_value(rows: Sequence[dict[str, Any]], metric: str) -> float:
    for row in rows:
        if row.get("metric") == metric:
            try:
                return float(row.get("value") or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _format_rate(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines
