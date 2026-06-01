from __future__ import annotations

import csv
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


def test_rewrite_only_and_prepare_manual_create_expected_files(tmp_path: Path) -> None:
    assert main(["init-mock-data", *_common_args(tmp_path), "--n-items", "3", "--force"]) == 0
    assert main(["rewrite-only", *_common_args(tmp_path), "--eval-split", "all"]) == 0
    assert (tmp_path / "results" / "prepared_prompts.csv").exists()
    prepared = read_csv_dicts(tmp_path / "results" / "prepared_prompts.csv")
    assert len(prepared) == 15
    assert {row["condition"] for row in prepared} == {
        "raw",
        "rule_only",
        "length_controlled",
        "proposed",
        "proposed_no_validator",
    }

    assert main(["prepare-manual", *_common_args(tmp_path), "--eval-split", "all"]) == 0
    assert (tmp_path / "input_data" / "manual_talkie_input_sheet.csv").exists()


def test_evaluate_manual_writes_metrics_and_report(tmp_path: Path) -> None:
    assert main(["init-mock-data", *_common_args(tmp_path), "--n-items", "3", "--force"]) == 0
    assert main(["prepare-manual", *_common_args(tmp_path), "--eval-split", "all"]) == 0
    prepared = read_csv_dicts(tmp_path / "results" / "prepared_prompts.csv")
    manual_path = tmp_path / "manual_responses.csv"
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
                    "raw_response_manual": row["gold_answer"],
                }
            )

    assert main(["evaluate-manual", *_common_args(tmp_path), "--eval-split", "all", "--manual-response-csv", str(manual_path)]) == 0
    result_rows = read_csv_dicts(tmp_path / "results" / "per_item_results.csv")
    final_metrics = read_csv_dicts(tmp_path / "results" / "final_metrics.csv")

    assert len(result_rows) == 15
    assert {row["condition"] for row in final_metrics} >= {"raw", "proposed"}
    assert (tmp_path / "results" / "report.md").exists()
    assert (tmp_path / "results" / "dataset_quality.csv").exists()
    assert (tmp_path / "results" / "qualitative_examples.md").exists()


def test_demo_command_writes_static_html(tmp_path: Path) -> None:
    assert main(["init-mock-data", *_common_args(tmp_path), "--n-items", "3", "--force"]) == 0
    out = tmp_path / "demo.html"

    assert main(["demo", *_common_args(tmp_path), "--eval-split", "all", "--item-id", "q001", "--out", str(out)]) == 0

    assert out.exists()
    assert "TalkieBridge Demo" in out.read_text(encoding="utf-8")


def test_predeclared_dictionary_allows_test_concept_rewrite(tmp_path: Path) -> None:
    assert main(["init-mock-data", *_common_args(tmp_path), "--n-items", "3", "--force"]) == 0
    assert main(
        [
            "prepare-manual",
            *_common_args(tmp_path),
            "--concept-dictionary-json",
            str(tmp_path / "data" / "mock_modern_terms_dictionary.json"),
            "--primitive-dictionary-json",
            str(tmp_path / "data" / "mock_primitive_dictionary.json"),
        ]
    ) == 0

    prepared = read_csv_dicts(tmp_path / "results" / "prepared_prompts.csv")
    proposed_test = next(row for row in prepared if row["item_id"] == "q003" and row["condition"] == "proposed")
    assert "API" not in proposed_test["rewritten_question"]


def test_default_does_not_auto_load_generated_dictionary_files(tmp_path: Path) -> None:
    assert main(["init-mock-data", *_common_args(tmp_path), "--n-items", "3", "--force"]) == 0
    assert main(["prepare-manual", *_common_args(tmp_path)]) == 0

    prepared = read_csv_dicts(tmp_path / "results" / "prepared_prompts.csv")
    proposed_test = next(row for row in prepared if row["item_id"] == "q003" and row["condition"] == "proposed")
    source = (tmp_path / "cache" / "generation_resource_source.json").read_text(encoding="utf-8")

    assert "API" in proposed_test["rewritten_question"]
    assert "train_split_annotations" in source


def test_evaluate_manual_fails_on_blank_responses_by_default(tmp_path: Path) -> None:
    assert main(["init-mock-data", *_common_args(tmp_path), "--n-items", "3", "--force"]) == 0
    assert main(["prepare-manual", *_common_args(tmp_path), "--eval-split", "all"]) == 0

    try:
        main(["evaluate-manual", *_common_args(tmp_path), "--eval-split", "all"])
    except ValueError as exc:
        assert "blank responses" in str(exc) or "prompt_hash" in str(exc)
    else:
        raise AssertionError("Expected blank manual responses to fail by default.")
