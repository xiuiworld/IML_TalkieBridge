"""Validation and deterministic repair for rewritten prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from talkie_bridge.detector import contains_term, replace_term, tokenize
from talkie_bridge.primitive import ConceptPrimitiveMapper


LEAKAGE_PATTERNS = (
    r"\bcorrect\s+answer\b",
    r"\banswer\s+is\b",
    r"\boption\s+[ABCD]\b",
    r"\bchoice\s+[ABCD]\b",
    r"\breturn\s+[ABCD]\b",
)


@dataclass(frozen=True)
class ValidationConfig:
    min_primitive_keyword_recall: float = 0.34
    max_choice_copying_score: float = 0.48
    max_choice_keyword_overlap_score: float = 0.58
    min_length_ratio: float = 0.35
    max_length_ratio: float = 4.8
    max_repair_attempts: int = 2


class RuleBasedValidator:
    def __init__(self, primitive_dictionary: dict[str, dict[str, Any]], config: ValidationConfig | None = None) -> None:
        self.primitive_dictionary = primitive_dictionary
        self.config = config or ValidationConfig()

    def validate(
        self,
        *,
        original_question: str,
        rewritten_question: str,
        choices: dict[str, str],
        forbidden_terms: Sequence[str],
        required_primitives: Sequence[str],
    ) -> dict[str, Any]:
        fail_reasons: list[str] = []
        forbidden_remaining = [term for term in forbidden_terms if contains_term(rewritten_question, term)]
        if forbidden_remaining:
            fail_reasons.append("forbidden_terms_remaining")

        primitive_recall, primitive_detail = self._primitive_recall(rewritten_question, required_primitives)
        if primitive_recall < self.config.min_primitive_keyword_recall:
            fail_reasons.append("required_primitives_missing")

        choice_copying_score = self._choice_copying_score(rewritten_question, choices)
        if choice_copying_score > self.config.max_choice_copying_score:
            fail_reasons.append("choice_copying_risk")
        choice_keyword_overlap_score = self._choice_keyword_overlap_score(rewritten_question, choices)
        if choice_keyword_overlap_score > self.config.max_choice_keyword_overlap_score:
            fail_reasons.append("answer_hint_risk")

        leakage_risk = any(re.search(pattern, rewritten_question, re.IGNORECASE) for pattern in LEAKAGE_PATTERNS)
        if leakage_risk:
            fail_reasons.append("answer_leakage_risk")

        original_len = max(1, len(tokenize(original_question)))
        rewritten_len = len(tokenize(rewritten_question))
        length_ratio = rewritten_len / original_len
        if length_ratio < self.config.min_length_ratio or length_ratio > self.config.max_length_ratio:
            fail_reasons.append("length_ratio_out_of_bounds")

        return {
            "pass": not fail_reasons,
            "fail_reasons": fail_reasons,
            "forbidden_terms_remaining": forbidden_remaining,
            "primitive_recall": primitive_recall,
            "primitive_detail": primitive_detail,
            "choice_copying_score": choice_copying_score,
            "choice_keyword_overlap_score": choice_keyword_overlap_score,
            "leakage_risk": leakage_risk,
            "length_ratio": length_ratio,
        }

    def _primitive_recall(self, rewritten_question: str, required_primitives: Sequence[str]) -> tuple[float, dict[str, float]]:
        if not required_primitives:
            return 1.0, {}
        rewritten_tokens = set(tokenize(rewritten_question))
        detail: dict[str, float] = {}
        scores: list[float] = []
        for primitive_id in required_primitives:
            phrase = str(self.primitive_dictionary.get(primitive_id, {}).get("primitive_phrase", ""))
            keywords = [token for token in tokenize(phrase) if len(token) > 3]
            if not keywords:
                score = 1.0
            else:
                score = sum(1 for token in keywords if token in rewritten_tokens) / len(keywords)
            detail[primitive_id] = score
            scores.append(score)
        return sum(scores) / len(scores), detail

    def _choice_copying_score(self, rewritten_question: str, choices: dict[str, str]) -> float:
        rewrite_ngrams = _ngrams(tokenize(rewritten_question), 4)
        if not rewrite_ngrams:
            return 0.0
        scores: list[float] = []
        for choice in choices.values():
            choice_ngrams = _ngrams(tokenize(choice), 4)
            if choice_ngrams:
                scores.append(len(rewrite_ngrams & choice_ngrams) / len(choice_ngrams))
        return max(scores) if scores else 0.0

    def _choice_keyword_overlap_score(self, rewritten_question: str, choices: dict[str, str]) -> float:
        rewrite_tokens = _content_tokens(rewritten_question)
        if not rewrite_tokens:
            return 0.0
        scores: list[float] = []
        for choice in choices.values():
            choice_tokens = _content_tokens(choice)
            if choice_tokens:
                scores.append(len(rewrite_tokens & choice_tokens) / len(choice_tokens))
        return max(scores) if scores else 0.0


class DeterministicRepairLoop:
    def __init__(
        self,
        mapper: ConceptPrimitiveMapper,
        validator: RuleBasedValidator,
        config: ValidationConfig | None = None,
    ) -> None:
        self.mapper = mapper
        self.validator = validator
        self.config = config or validator.config

    def repair(
        self,
        *,
        original_question: str,
        rewritten_question: str,
        choices: dict[str, str],
        forbidden_terms: Sequence[str],
        validation_primitives: Sequence[str],
        repair_primitives: Sequence[str],
        removable_terms: Sequence[str],
    ) -> tuple[str, dict[str, Any], int]:
        current = rewritten_question
        report = self.validator.validate(
            original_question=original_question,
            rewritten_question=current,
            choices=choices,
            forbidden_terms=forbidden_terms,
            required_primitives=validation_primitives,
        )
        attempts = 0
        while not report["pass"] and attempts < self.config.max_repair_attempts:
            attempts += 1
            current = self._single_repair(current, report, removable_terms, repair_primitives)
            report = self.validator.validate(
                original_question=original_question,
                rewritten_question=current,
                choices=choices,
                forbidden_terms=forbidden_terms,
                required_primitives=validation_primitives,
            )
        return current, report, attempts

    def _single_repair(
        self,
        rewritten_question: str,
        report: dict[str, Any],
        removable_terms: Sequence[str],
        required_primitives: Sequence[str],
    ) -> str:
        repaired = rewritten_question
        for term in removable_terms:
            repaired = replace_term(repaired, term, "this method")

        if "answer_leakage_risk" in report.get("fail_reasons", []):
            repaired = re.sub(r"\b(correct\s+answer|answer\s+is|option\s+[ABCD]|choice\s+[ABCD]|return\s+[ABCD])\b", "", repaired, flags=re.IGNORECASE)

        if "required_primitives_missing" in report.get("fail_reasons", []):
            phrases = self.mapper.phrases_for(required_primitives)
            if phrases:
                repaired = f"Consider a method that {phrases[0]}. {repaired}"

        if "choice_copying_risk" in report.get("fail_reasons", []):
            repaired = repaired.replace("It ", "The method ")
        if "answer_hint_risk" in report.get("fail_reasons", []):
            repaired = repaired.replace("relevant", "available").replace("records", "materials")

        return normalize_space(repaired)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "before",
    "best",
    "by",
    "can",
    "choice",
    "choose",
    "for",
    "from",
    "in",
    "is",
    "it",
    "method",
    "of",
    "on",
    "one",
    "only",
    "option",
    "or",
    "question",
    "same",
    "that",
    "the",
    "this",
    "to",
    "use",
    "uses",
    "why",
    "with",
}


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 3 and token not in STOPWORDS}
