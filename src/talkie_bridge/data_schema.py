"""Shared schemas and file helpers for the TalkieBridge pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


LABELS = ("A", "B", "C", "D")
CONDITIONS = ("raw", "rule_only", "length_controlled", "proposed", "proposed_no_validator")
INVALID_LABEL = "INVALID"
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class DatasetItem:
    id: str
    domain: str
    original_question: str
    choices: dict[str, str]
    gold_answer: str
    gold_anachronism_terms: list[str]
    forbidden_terms: list[str]
    required_primitives: list[str]
    primitive_phrase: str
    human_validated: bool = False
    target_year: int = 1930
    split: str = ""
    concept_term: str = ""
    task: str = ""
    open_question: str = ""
    expected_mechanism: str = ""
    judge_reference_points: list[str] = field(default_factory=list)
    leakage_sensitive_terms: list[str] = field(default_factory=list)
    answer_leakage_terms: list[str] = field(default_factory=list)
    dataset_version: str = ""
    review_status: str = ""
    review_notes: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DatasetItem":
        choices = row.get("choices")
        if isinstance(choices, str):
            choices = json.loads(choices)
        if not choices:
            choices = {
                "A": row.get("choice_A", ""),
                "B": row.get("choice_B", ""),
                "C": row.get("choice_C", ""),
                "D": row.get("choice_D", ""),
            }

        def list_field(name: str) -> list[str]:
            value = row.get(name, [])
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return []
                if value.startswith("["):
                    loaded = json.loads(value)
                    return [str(item) for item in loaded]
                return [part.strip() for part in value.split(";") if part.strip()]
            return [str(item) for item in value]

        return cls(
            id=str(row["id"]),
            domain=str(row["domain"]),
            original_question=str(row["original_question"]),
            choices={label: str(choices[label]) for label in LABELS},
            gold_answer=str(row["gold_answer"]).strip().upper(),
            gold_anachronism_terms=list_field("gold_anachronism_terms"),
            forbidden_terms=list_field("forbidden_terms"),
            required_primitives=list_field("required_primitives"),
            primitive_phrase=str(row.get("primitive_phrase", "")),
            human_validated=as_bool(row.get("human_validated", False)),
            target_year=int(row.get("target_year", 1930)),
            split=str(row.get("split", "")).strip().lower(),
            concept_term=str(row.get("concept_term", "")),
            task=str(row.get("task", "")),
            open_question=str(row.get("open_question", "")),
            expected_mechanism=str(row.get("expected_mechanism", "")),
            judge_reference_points=list_field("judge_reference_points"),
            leakage_sensitive_terms=list_field("leakage_sensitive_terms"),
            answer_leakage_terms=list_field("answer_leakage_terms"),
            dataset_version=str(row.get("dataset_version", "")),
            review_status=str(row.get("review_status", "")),
            review_notes=str(row.get("review_notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewriteArtifact:
    item_id: str
    condition: str
    original_question: str
    rewritten_question: str
    detected_terms: list[str] = field(default_factory=list)
    mapped_primitives: list[str] = field(default_factory=list)
    rewrite_validation: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 0
    pass_validation: bool = True
    detector_token_scores: list[dict[str, Any]] = field(default_factory=list)
    model_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    data_dir: Path
    input_dir: Path
    out_dir: Path
    cache_dir: Path

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        data_dir: str | Path = "data",
        input_dir: str | Path = "input_data",
        out_dir: str | Path = "results",
        cache_dir: str | Path = "cache",
    ) -> "RunPaths":
        return cls(
            root=root,
            data_dir=resolve_under(root, data_dir),
            input_dir=resolve_under(root, input_dir),
            out_dir=resolve_under(root, out_dir),
            cache_dir=resolve_under(root, cache_dir),
        )

    def ensure(self) -> None:
        for path in (self.data_dir, self.input_dir, self.out_dir, self.cache_dir):
            ensure_dir(path)


def resolve_under(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def infer_split(item_id: str, seed: int = 13) -> str:
    bucket = stable_hash(f"{seed}:{item_id}") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "dev"
    return "test"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_loads(value: str | Any, default: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json_dumps(row))
            handle.write("\n")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json_dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value
