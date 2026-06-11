"""Prompt building and strict multiple-choice answer parsing."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

from talkie_bridge.data_schema import INVALID_LABEL, LABELS


_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "allows",
    "among",
    "answer",
    "because",
    "before",
    "being",
    "between",
    "choice",
    "could",
    "does",
    "during",
    "each",
    "from",
    "have",
    "into",
    "its",
    "later",
    "more",
    "most",
    "only",
    "option",
    "other",
    "over",
    "same",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "without",
    "would",
}


def build_mcq_prompt(question: str, choices: dict[str, str]) -> str:
    return (
        "You must answer the following multiple-choice question.\n"
        "Choose exactly one option among A, B, C, and D.\n"
        "Return only the letter.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Choices:\n"
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}\n"
    )


def normalize_choice(raw_text: str, choices: Mapping[str, str] | None = None) -> str:
    text = raw_text.strip().upper()
    if text in LABELS:
        return text

    patterns = (
        r"^\s*([ABCD])[\).:\s]",
        r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([ABCD])\b",
        r"\b(?:CHOOSE|CHOSE|SELECT|SELECTED|PICK|PICKED)\s*(?:OPTION|CHOICE)?\s*([ABCD])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        unique = sorted(set(matches))
        if len(unique) == 1:
            return unique[0]
    if choices:
        return _normalize_choice_from_text(raw_text, choices)
    matches = re.findall(r"\b([ABCD])\b", text)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return INVALID_LABEL


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _normalize_choice_from_text(raw_text: str, choices: Mapping[str, str]) -> str:
    response = _clean_text(raw_text)
    if not response:
        return INVALID_LABEL

    response_tokens = _content_tokens(response)
    scored: list[tuple[float, str]] = []
    for label in LABELS:
        choice = _clean_text(str(choices.get(label, "")))
        if not choice:
            continue
        score = _choice_text_score(response, choice, response_tokens)
        scored.append((score, label))

    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.6:
        return INVALID_LABEL
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.2:
        return INVALID_LABEL
    return scored[0][1]


def _choice_text_score(response: str, choice: str, response_tokens: set[str]) -> float:
    if len(response) >= 12 and (response in choice or choice in response):
        return 1.0
    choice_tokens = _content_tokens(choice)
    if not response_tokens or not choice_tokens:
        return 0.0
    overlap = response_tokens & choice_tokens
    if len(overlap) < 3:
        return 0.0
    response_coverage = len(overlap) / len(response_tokens)
    choice_coverage = len(overlap) / len(choice_tokens)
    return max(response_coverage, (response_coverage + choice_coverage) / 2)


def _clean_text(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"^\s*(?:answer|option|choice)?\s*[abcd]\s*[\).:\-]\s*", "", text)
    text = text.replace('"', "").replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(value: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return {token for token in tokens if len(token) >= 3 and token not in _STOPWORDS}
