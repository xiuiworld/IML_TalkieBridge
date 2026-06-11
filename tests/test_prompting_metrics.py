from __future__ import annotations

from talkie_bridge.data_schema import INVALID_LABEL
from talkie_bridge.metrics import compute_condition_metrics, compute_paired_tests
from talkie_bridge.prompting import normalize_choice


def test_normalize_choice_is_strict_and_handles_common_answer_forms() -> None:
    assert normalize_choice("A") == "A"
    assert normalize_choice("Answer: C") == "C"
    assert normalize_choice("option b") == "B"
    assert normalize_choice("I think A or B") == INVALID_LABEL


def test_normalize_choice_can_recover_option_text_answers() -> None:
    choices = {
        "A": "To store pictures as data that can be copied immediately.",
        "B": "To read a printed address without manual typing.",
        "C": "To allow playback while later parts are still arriving.",
        "D": "To let many distant people answer the same topic over time.",
    }

    assert normalize_choice('"It reads a printed address without manual typing"', choices) == "B"
    assert normalize_choice('"To allow playback while later parts are still arriving."', choices) == "C"
    assert normalize_choice("manual topic over time", choices) == INVALID_LABEL


def test_metrics_include_invalid_rate_and_paired_exact_mcnemar() -> None:
    rows = [
        {"item_id": "1", "condition": "raw", "gold_answer": "A", "parsed_answer": "A", "correct": True},
        {"item_id": "1", "condition": "proposed", "gold_answer": "A", "parsed_answer": "A", "correct": True},
        {"item_id": "2", "condition": "raw", "gold_answer": "B", "parsed_answer": INVALID_LABEL, "correct": False},
        {"item_id": "2", "condition": "proposed", "gold_answer": "B", "parsed_answer": "B", "correct": True},
    ]

    metrics = {row["condition"]: row for row in compute_condition_metrics(rows)}
    paired = compute_paired_tests(rows, n_bootstrap=20)

    assert metrics["raw"]["accuracy"] == 0.5
    assert metrics["raw"]["invalid_rate"] == 0.5
    proposed_pair = next(row for row in paired if row["condition"] == "proposed")
    assert proposed_pair["accuracy_delta"] == 0.5
    assert proposed_pair["mcnemar_b"] == 1
    assert proposed_pair["mcnemar_c"] == 0
