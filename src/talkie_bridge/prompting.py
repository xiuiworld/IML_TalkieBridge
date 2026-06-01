"""Prompt building and strict multiple-choice answer parsing."""

from __future__ import annotations

import hashlib
import re

from talkie_bridge.data_schema import INVALID_LABEL, LABELS


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


def normalize_choice(raw_text: str) -> str:
    text = raw_text.strip().upper()
    if text in LABELS:
        return text

    patterns = (
        r"^\s*([ABCD])[\).:\s]",
        r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([ABCD])\b",
        r"\b([ABCD])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        unique = sorted(set(matches))
        if len(unique) == 1:
            return unique[0]
    return INVALID_LABEL


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
