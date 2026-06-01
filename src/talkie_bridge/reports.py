"""Markdown report generation for TalkieBridge runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from talkie_bridge.data_schema import write_json


def write_report(
    path: Path,
    *,
    provider: str,
    n_items: int,
    final_metrics: Sequence[dict[str, Any]],
    test_metrics: Sequence[dict[str, Any]] = (),
    component_metrics: Sequence[dict[str, Any]] = (),
    dataset_quality: Sequence[dict[str, Any]] = (),
    domain_metrics: Sequence[dict[str, Any]] = (),
    response_integrity: Sequence[dict[str, Any]] = (),
    resource_source: dict[str, Any] | None = None,
    paired_tests: Sequence[dict[str, Any]] = (),
    test_paired_tests: Sequence[dict[str, Any]] = (),
    key_comparisons: Sequence[dict[str, Any]] = (),
    test_key_comparisons: Sequence[dict[str, Any]] = (),
) -> None:
    lines = [
        "# TalkieBridge Experiment Report",
        "",
        "This run evaluates the Era-Neutral Prompt Generator in front of a fixed Talkie-1930 downstream evaluator.",
        "Talkie is not trained or modified by this pipeline.",
        "",
        "## Run Metadata",
        "",
        f"- Provider: `{provider}`",
        f"- Items: `{n_items}`",
    ]
    if resource_source:
        lines.extend(
            [
                f"- Generation resource source: `{resource_source.get('source', '')}`",
                f"- Concept dictionary path: `{resource_source.get('concept_dictionary_json', '')}`",
                f"- Primitive dictionary path: `{resource_source.get('primitive_dictionary_json', '')}`",
            ]
        )
    if provider == "unofficial_talkie_api":
        lines.extend(
            [
                "- Provider note: this uses the unofficial Talkie web SSE endpoint and should be treated as a convenience path, not the primary reproducibility source.",
            ]
        )
    warnings = _report_warnings(dataset_quality, response_integrity, resource_source or {})
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    if test_metrics:
        lines.extend(["", "## Primary Test Split Metrics", ""])
        lines.extend(_markdown_table(test_metrics, ["condition", "accuracy", "macro_f1", "invalid_rate", "n_items"]))

    lines.extend(["", "## All Selected Rows Metrics", ""])
    lines.extend(_markdown_table(final_metrics, ["condition", "accuracy", "macro_f1", "invalid_rate", "n_items"]))

    if test_key_comparisons:
        lines.extend(["", "## Primary Test Split Key Comparisons", ""])
        lines.extend(_comparison_table(test_key_comparisons))

    lines.extend(["", "## Key Comparisons", ""])
    lines.extend(_comparison_table(key_comparisons))

    lines.extend(["", "## Paired Tests Against Raw", ""])
    lines.extend(
        _markdown_table(
            paired_tests,
            [
                "comparison",
                "accuracy_delta",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "mcnemar_b",
                "mcnemar_c",
                "exact_mcnemar_p",
            ],
        )
    )

    if test_paired_tests:
        lines.extend(["", "## Test Split Paired Tests Against Raw", ""])
        lines.extend(
            _markdown_table(
                test_paired_tests,
                [
                    "comparison",
                    "accuracy_delta",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "mcnemar_b",
                    "mcnemar_c",
                    "exact_mcnemar_p",
                ],
            )
        )

    lines.extend(["", "## Component Metrics", ""])
    lines.extend(_markdown_table(component_metrics, ["component", "metric", "value"]))

    lines.extend(["", "## Dataset Quality", ""])
    lines.extend(_markdown_table(dataset_quality, ["section", "metric", "value"]))

    lines.extend(["", "## Domain Metrics", ""])
    lines.extend(_markdown_table(domain_metrics, ["domain", "condition", "accuracy", "macro_f1", "invalid_rate", "n_items"]))

    lines.extend(["", "## Response Integrity", ""])
    lines.extend(_markdown_table(response_integrity, ["metric", "value"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(path.with_name("paired_tests.json"), list(paired_tests))


def _markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for row in rows:
        cells = [_format_cell(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _comparison_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    return _markdown_table(
        rows,
        [
            "comparison",
            "accuracy_delta",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "mcnemar_b",
            "mcnemar_c",
            "exact_mcnemar_p",
        ],
    )


def _report_warnings(
    dataset_quality: Sequence[dict[str, Any]],
    response_integrity: Sequence[dict[str, Any]],
    resource_source: dict[str, Any],
) -> list[str]:
    metrics = {str(row.get("metric")): row.get("value") for row in dataset_quality}
    integrity = {str(row.get("metric")): row.get("value") for row in response_integrity}
    warnings: list[str] = []
    n_items = float(metrics.get("n_items", 0) or 0)
    if n_items and n_items < 100:
        warnings.append("Dataset has fewer than 100 items; do not use this run for final proposal-level claims.")
    validated = float(metrics.get("human_validated_count", 0) or 0)
    if n_items and validated < n_items:
        warnings.append("Not all dataset items are human_validated=true.")
    hash_rate = float(integrity.get("prompt_hash_match_rate", 1.0) or 0.0)
    completion_rate = float(integrity.get("response_completion_rate", 1.0) or 0.0)
    if response_integrity and hash_rate < 1.0:
        warnings.append("Some manual responses do not have matching prompt hashes; verify responses were collected from the current prompt sheet.")
    if response_integrity and completion_rate < 1.0:
        warnings.append("Some manual responses are blank; downstream metrics from this run are incomplete.")
    if resource_source.get("source") == "predeclared_dictionary":
        warnings.append("Predeclared dictionaries are trusted prior resources; verify they were authored independently of dev/test item annotations.")
    primitive_diff_rate = float(metrics.get("primitive_set_differs_from_rule_only_rate", 1.0) or 0.0)
    if primitive_diff_rate == 0.0:
        warnings.append("Proposed selected the same primitive sets as rule_only in this run; do not claim learned selector impact from these outputs alone.")
    return warnings
