from __future__ import annotations

import csv
import json
from pathlib import Path

from talkie_bridge.cli import main
from talkie_bridge.data_schema import read_csv_dicts


def _common_args(tmp_path: Path) -> list[str]:
    return [
        "--data-dir",
        str(tmp_path / "data"),
        "--dataset-jsonl",
        str(tmp_path / "data" / "generated_questions.jsonl"),
        "--input-dir",
        str(tmp_path / "input_data"),
        "--out-dir",
        str(tmp_path / "results"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]


def test_prepare_open_ended_writes_talkie_sheet_without_mcq_choices(tmp_path: Path) -> None:
    assert main(["init-mock-data", *_common_args(tmp_path), "--n-items", "3", "--force"]) == 0

    assert main(["prepare-open-ended", *_common_args(tmp_path), "--eval-split", "all"]) == 0

    prepared = read_csv_dicts(tmp_path / "results" / "open_ended_prepared_prompts.csv")
    manual = read_csv_dicts(tmp_path / "input_data" / "open_ended_talkie_input_sheet.csv")

    assert len(prepared) == 15
    assert len(manual) == 15
    assert {row["condition"] for row in prepared} == {
        "raw",
        "rule_only",
        "length_controlled",
        "proposed",
        "proposed_no_validator",
    }
    assert all("Choices:" not in row["prompt"] for row in prepared)
    assert all("Choose the option" not in row["rewritten_question"] for row in prepared)


def test_open_ended_manual_responses_prepare_and_evaluate_judge_sheet(tmp_path: Path) -> None:
    assert main(["init-mock-data", *_common_args(tmp_path), "--n-items", "3", "--force"]) == 0
    assert main(["prepare-open-ended", *_common_args(tmp_path), "--eval-split", "all"]) == 0

    prepared = read_csv_dicts(tmp_path / "results" / "open_ended_prepared_prompts.csv")
    manual_path = tmp_path / "open_responses.csv"
    with manual_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "condition", "prompt_hash", "prompt", "raw_response_manual"])
        writer.writeheader()
        for row in prepared:
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "condition": row["condition"],
                    "prompt_hash": row["prompt_hash"],
                    "prompt": row["prompt"],
                    "raw_response_manual": f"{row['condition']} explains the mechanism in ordinary terms.",
                }
            )

    assert main(["evaluate-open-ended-manual", *_common_args(tmp_path), "--eval-split", "all", "--manual-response-csv", str(manual_path)]) == 0
    pair_rows = read_csv_dicts(tmp_path / "results" / "open_ended_judge_pairs_unblinded.csv")
    judge_input = read_csv_dicts(tmp_path / "input_data" / "open_ended_judge_input_sheet.csv")

    assert len(pair_rows) == 12
    assert len(judge_input) == 12
    assert "condition_a" not in judge_input[0]
    assert "Answer A:" in judge_input[0]["judge_prompt"]

    judge_path = tmp_path / "judge_outputs.csv"
    judge_payload = {
        "winner": "A",
        "task_relevance": {"A": 4, "B": 3},
        "functional_correctness": {"A": 4, "B": 3},
        "era_neutrality": {"A": 4, "B": 3},
        "anachronism_handling": {"A": 4, "B": 3},
        "usefulness": {"A": 4, "B": 3},
        "leakage_risk": {"A": 1, "B": 1},
        "rationale": "A is slightly clearer.",
    }
    with judge_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "judge_prompt_hash", "judge_prompt", "judge_raw_output"])
        writer.writeheader()
        for row in judge_input:
            writer.writerow({**row, "judge_raw_output": json.dumps(judge_payload)})

    assert main(["evaluate-open-ended-judge", *_common_args(tmp_path), "--judge-response-csv", str(judge_path)]) == 0

    assert (tmp_path / "results" / "open_ended_pairwise_metrics.csv").exists()
    assert (tmp_path / "results" / "open_ended_response_quality.md").exists()
