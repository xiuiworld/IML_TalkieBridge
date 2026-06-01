"""Modern-term detection used before era-neutral rewriting."""

from __future__ import annotations

import re
from dataclasses import dataclass
import math
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class DetectedTerm:
    term: str
    canonical_term: str
    primitive_id: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "canonical_term": self.canonical_term,
            "primitive_id": self.primitive_id,
            "start": self.start,
            "end": self.end,
        }


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def normalize_term(text: str) -> str:
    return " ".join(tokenize(text))


def contains_term(text: str, term: str) -> bool:
    return find_term_span(text, term) is not None


def replace_term(text: str, term: str, replacement: str) -> str:
    pattern = _term_pattern(term)
    return pattern.sub(replacement, text)


def find_term_span(text: str, term: str) -> tuple[int, int] | None:
    match = _term_pattern(term).search(text)
    return (match.start(), match.end()) if match else None


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


class AnachronismDetector:
    """Dictionary-backed detector for concepts unlikely to be known in 1930."""

    def __init__(
        self,
        modern_terms_dictionary: dict[str, dict[str, Any]],
        token_classifier: "BernoulliModernTermClassifier | None" = None,
    ) -> None:
        self.modern_terms_dictionary = modern_terms_dictionary
        self.token_classifier = token_classifier
        aliases = []
        for alias, meta in modern_terms_dictionary.items():
            display = str(meta.get("alias", alias))
            aliases.append((len(display), display, meta))
        self._aliases = sorted(aliases, reverse=True)

    def detect_details(self, text: str) -> list[DetectedTerm]:
        found: list[DetectedTerm] = []
        occupied: list[tuple[int, int]] = []
        for _length, alias, meta in self._aliases:
            for match in _term_pattern(alias).finditer(text):
                span = (match.start(), match.end())
                if any(_overlaps(span, previous) for previous in occupied):
                    continue
                occupied.append(span)
                found.append(
                    DetectedTerm(
                        term=match.group(0),
                        canonical_term=str(meta["canonical_term"]),
                        primitive_id=str(meta["primitive_id"]),
                        start=span[0],
                        end=span[1],
                    )
                )
        return sorted(found, key=lambda item: (item.start, item.end))

    def detect(self, text: str) -> list[str]:
        return [item.term for item in self.detect_details(text)]

    def score_tokens(self, text: str) -> list[dict[str, Any]]:
        if self.token_classifier is None:
            return []
        return [
            {
                "token": token,
                "modern_score": score,
                "predicted_modern": score >= self.token_classifier.threshold,
            }
            for token, score in self.token_classifier.score_tokens(text)
        ]


class BernoulliModernTermClassifier:
    """Tiny token-level Naive Bayes classifier for modern-term evidence."""

    def __init__(self, *, alpha: float = 1.0, threshold: float = 0.65) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self.modern_counts: dict[str, int] = {}
        self.background_counts: dict[str, int] = {}
        self.n_modern = 0
        self.n_background = 0
        self.vocab: set[str] = set()

    def fit(self, examples: list[tuple[str, bool]]) -> None:
        for token, is_modern in examples:
            normalized = normalize_term(token)
            if not normalized:
                continue
            for part in normalized.split():
                self.vocab.add(part)
                if is_modern:
                    self.modern_counts[part] = self.modern_counts.get(part, 0) + 1
                    self.n_modern += 1
                else:
                    self.background_counts[part] = self.background_counts.get(part, 0) + 1
                    self.n_background += 1

    def score_tokens(self, text: str) -> list[tuple[str, float]]:
        return [(token, self.score(token)) for token in tokenize(text)]

    def score(self, token: str) -> float:
        token = normalize_term(token)
        if not token:
            return 0.0
        vocab_size = max(1, len(self.vocab))
        modern_total = self.n_modern + self.alpha * vocab_size
        background_total = self.n_background + self.alpha * vocab_size
        modern_prob = (self.modern_counts.get(token, 0) + self.alpha) / modern_total
        background_prob = (self.background_counts.get(token, 0) + self.alpha) / background_total
        prior_modern = (self.n_modern + self.alpha) / (self.n_modern + self.n_background + 2 * self.alpha)
        prior_background = 1.0 - prior_modern
        log_modern = math.log(prior_modern) + math.log(modern_prob)
        log_background = math.log(prior_background) + math.log(background_prob)
        max_log = max(log_modern, log_background)
        modern_exp = math.exp(log_modern - max_log)
        background_exp = math.exp(log_background - max_log)
        return modern_exp / (modern_exp + background_exp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "n_modern": self.n_modern,
            "n_background": self.n_background,
            "vocab_size": len(self.vocab),
        }


def build_modern_token_examples(
    questions: list[str],
    modern_terms: list[list[str]],
) -> list[tuple[str, bool]]:
    examples: list[tuple[str, bool]] = []
    for question, terms in zip(questions, modern_terms):
        modern_tokens = {token for term in terms for token in tokenize(term)}
        for token in tokenize(question):
            examples.append((token, token in modern_tokens))
    return examples


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
