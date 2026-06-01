"""Evaluation metrics for paired multiple-choice TalkieBridge runs."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Sequence

from talkie_bridge.data_schema import CONDITIONS, INVALID_LABEL, LABELS, json_loads


def compute_condition_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    metrics: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_rows = grouped.get(condition, [])
        if condition_rows:
            metrics.append({"condition": condition, **_class_metrics(condition_rows)})
    for condition, condition_rows in grouped.items():
        if condition not in CONDITIONS:
            metrics.append({"condition": condition, **_class_metrics(condition_rows)})
    return metrics


def compute_domain_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("domain", "unknown")), str(row["condition"]))].append(row)
    output = []
    for (domain, condition), domain_rows in sorted(grouped.items()):
        output.append({"domain": domain, "condition": condition, **_class_metrics(domain_rows)})
    return output


def compute_component_metrics(prompt_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rewrite_rows = [row for row in prompt_rows if row.get("condition") != "raw"]
    if not rewrite_rows:
        return []

    detection_tp = detection_fp = detection_fn = 0
    removal_pass = primitive_scores = validator_pass = leakage_count = repair_total = 0.0
    choice_scores: list[float] = []
    hint_scores: list[float] = []
    for row in rewrite_rows:
        detected = set(str(term).lower() for term in json_loads(row.get("detected_terms", []), []))
        gold = set(str(term).lower() for term in json_loads(row.get("gold_anachronism_terms", []), []))
        detection_tp += len(detected & gold)
        detection_fp += len(detected - gold)
        detection_fn += len(gold - detected)

        validation = json_loads(row.get("rewrite_validation", {}), {})
        removal_pass += 0 if validation.get("forbidden_terms_remaining") else 1
        primitive_scores += float(validation.get("primitive_recall", 0.0))
        choice_scores.append(float(validation.get("choice_copying_score", 0.0)))
        hint_scores.append(float(validation.get("choice_keyword_overlap_score", 0.0)))
        validator_pass += 1 if validation.get("pass") else 0
        leakage_count += 1 if validation.get("leakage_risk") else 0
        repair_total += int(row.get("repair_attempts") or 0)

    n = len(rewrite_rows)
    precision = detection_tp / (detection_tp + detection_fp) if detection_tp + detection_fp else 0.0
    recall = detection_tp / (detection_tp + detection_fn) if detection_tp + detection_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return [
        {"component": "detector", "metric": "precision", "value": precision},
        {"component": "detector", "metric": "recall", "value": recall},
        {"component": "detector", "metric": "f1", "value": f1},
        {"component": "rewriter", "metric": "anachronism_removal_rate", "value": removal_pass / n},
        {"component": "rewriter", "metric": "required_primitive_recall", "value": primitive_scores / n},
        {"component": "validator", "metric": "rewrite_pass_rate", "value": validator_pass / n},
        {"component": "validator", "metric": "leakage_risk_rate", "value": leakage_count / n},
        {"component": "validator", "metric": "mean_choice_copying_score", "value": sum(choice_scores) / len(choice_scores) if choice_scores else 0.0},
        {"component": "validator", "metric": "max_choice_copying_score", "value": max(choice_scores) if choice_scores else 0.0},
        {"component": "validator", "metric": "mean_choice_keyword_overlap_score", "value": sum(hint_scores) / len(hint_scores) if hint_scores else 0.0},
        {"component": "validator", "metric": "max_choice_keyword_overlap_score", "value": max(hint_scores) if hint_scores else 0.0},
        {"component": "repair", "metric": "mean_repair_attempts", "value": repair_total / n},
    ]


def compute_paired_tests(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_condition: str = "raw",
    seed: int = 13,
    n_bootstrap: int = 1000,
) -> list[dict[str, Any]]:
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_item[str(row["item_id"])][str(row["condition"])] = row

    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        if condition == baseline_condition:
            continue
        pairs = [
            (item_rows[baseline_condition], item_rows[condition])
            for item_rows in by_item.values()
            if baseline_condition in item_rows and condition in item_rows
        ]
        if not pairs:
            continue
        deltas = [_correct(proposed) - _correct(base) for base, proposed in pairs]
        low, high = bootstrap_ci(deltas, seed=seed, n_bootstrap=n_bootstrap)
        mcnemar = exact_mcnemar_from_pairs(pairs)
        output.append(
            {
                "comparison": f"{condition}_vs_{baseline_condition}",
                "condition": condition,
                "baseline_condition": baseline_condition,
                "n_items": len(pairs),
                "accuracy_delta": sum(deltas) / len(deltas),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                **mcnemar,
            }
        )
    return output


def compute_key_comparisons(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = 13,
    n_bootstrap: int = 1000,
) -> list[dict[str, Any]]:
    comparisons = [
        ("raw", "rule_only"),
        ("raw", "length_controlled"),
        ("raw", "proposed"),
        ("raw", "proposed_no_validator"),
        ("rule_only", "proposed"),
        ("length_controlled", "proposed"),
        ("proposed_no_validator", "proposed"),
    ]
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_item[str(row["item_id"])][str(row["condition"])] = row

    output: list[dict[str, Any]] = []
    for baseline_condition, condition in comparisons:
        pairs = [
            (item_rows[baseline_condition], item_rows[condition])
            for item_rows in by_item.values()
            if baseline_condition in item_rows and condition in item_rows
        ]
        if not pairs:
            continue
        deltas = [_correct(proposed) - _correct(base) for base, proposed in pairs]
        low, high = bootstrap_ci(deltas, seed=seed, n_bootstrap=n_bootstrap)
        output.append(
            {
                "comparison": f"{condition}_vs_{baseline_condition}",
                "condition": condition,
                "baseline_condition": baseline_condition,
                "n_items": len(pairs),
                "accuracy_delta": sum(deltas) / len(deltas),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                **exact_mcnemar_from_pairs(pairs),
            }
        )
    return output


def bootstrap_ci(values: Sequence[float], *, seed: int = 13, n_bootstrap: int = 1000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates: list[float] = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _idx in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    low_index = int(0.025 * (len(estimates) - 1))
    high_index = int(0.975 * (len(estimates) - 1))
    return estimates[low_index], estimates[high_index]


def exact_mcnemar_from_pairs(pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    b = 0
    c = 0
    for baseline, proposed in pairs:
        base_correct = bool(_correct(baseline))
        proposed_correct = bool(_correct(proposed))
        if not base_correct and proposed_correct:
            b += 1
        elif base_correct and not proposed_correct:
            c += 1
    n = b + c
    p_value = 1.0 if n == 0 else min(1.0, 2.0 * _binomial_cdf(min(b, c), n, 0.5))
    return {"mcnemar_b": b, "mcnemar_c": c, "exact_mcnemar_p": p_value}


def _class_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if _correct(row))
    invalid = sum(1 for row in rows if row.get("parsed_answer") == INVALID_LABEL)
    f1s: list[float] = []
    for label in LABELS:
        tp = sum(1 for row in rows if row.get("parsed_answer") == label and row.get("gold_answer") == label)
        fp = sum(1 for row in rows if row.get("parsed_answer") == label and row.get("gold_answer") != label)
        fn = sum(1 for row in rows if row.get("parsed_answer") != label and row.get("gold_answer") == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "invalid_rate": invalid / n if n else 0.0,
        "n_items": n,
    }


def _correct(row: dict[str, Any]) -> int:
    value = row.get("correct", False)
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes"} else 0
    return 1 if value else 0


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k + 1))
