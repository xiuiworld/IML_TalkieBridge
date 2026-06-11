"""Era-neutral rewriting conditions used by the experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from talkie_bridge.autoencoder import DenoisingPrimitiveAutoencoder
from talkie_bridge.data_schema import DatasetItem, RewriteArtifact
from talkie_bridge.detector import AnachronismDetector, DetectedTerm, replace_term, tokenize
from talkie_bridge.primitive import ConceptPrimitiveMapper
from talkie_bridge.validator import DeterministicRepairLoop, RuleBasedValidator


@dataclass(frozen=True)
class PrimitiveBottleneck:
    primitive_ids: tuple[str, ...]
    phrases: tuple[str, ...]

    @property
    def primary_phrase(self) -> str:
        return self.phrases[0] if self.phrases else "uses a function that can be described without its modern name"


class RuleOnlyRewriter:
    def rewrite(self, item: DatasetItem, detected_terms: Sequence[DetectedTerm], mapper: ConceptPrimitiveMapper, *, task: str = "mcq") -> str:
        text = replace_detected_terms(item.original_question, detected_terms)
        primitive_ids = mapper.map_terms(detected_terms)
        phrases = mapper.phrases_for(primitive_ids)
        phrase = phrases[0] if phrases else "can be described by its practical function"
        if task == "open_ended":
            return (
                f"Consider a method that {phrase}. "
                f"{text} Focus on the mechanism, not on the modern name."
            )
        return (
            f"Consider a method that {phrase}. "
            f"{text} Focus on the mechanism, not on the modern name."
        )


class LengthControlledRewriter:
    def rewrite(
        self,
        item: DatasetItem,
        detected_terms: Sequence[DetectedTerm],
        mapper: ConceptPrimitiveMapper,
        *,
        target_tokens: int,
        task: str = "mcq",
    ) -> str:
        text = replace_detected_terms(item.original_question, detected_terms)
        if task == "open_ended":
            base = (
                "Consider the same practical situation described in plain terms. "
                f"{text} Focus on the practical mechanism."
            )
            return match_token_length(base, target_tokens)
        base = (
            "Consider the same practical situation described in plain terms. "
            "Use only the information in the question and choices. "
            f"{text} Choose the option that best explains the mechanism."
        )
        return match_token_length(base, target_tokens)


class DenoisingPrimitiveBottleneckRewriter:
    """Primitive bottleneck rewriter backed by an optional learned autoencoder."""

    def __init__(
        self,
        primitive_dictionary: dict[str, dict[str, Any]],
        *,
        autoencoder: DenoisingPrimitiveAutoencoder | None = None,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        self.primitive_dictionary = primitive_dictionary
        self.autoencoder = autoencoder
        self.model_info = model_info or {"enabled": False, "fallback_reason": "not_configured"}

    def compress(self, item: DatasetItem, detected_terms: Sequence[DetectedTerm], mapper: ConceptPrimitiveMapper) -> PrimitiveBottleneck:
        primitive_ids = mapper.map_terms(detected_terms)
        if self.autoencoder is not None and self.autoencoder.enabled:
            for primitive_id, score in self.autoencoder.primitive_scores(item.original_question, self.primitive_dictionary):
                if score < self.autoencoder.config.decode_threshold and primitive_ids:
                    continue
                if primitive_id not in primitive_ids:
                    primitive_ids.append(primitive_id)
                if len(primitive_ids) >= self.autoencoder.config.top_k_primitives:
                    break
        phrases = tuple(mapper.phrases_for(primitive_ids))
        return PrimitiveBottleneck(tuple(primitive_ids), phrases)

    def rewrite(
        self,
        item: DatasetItem,
        detected_terms: Sequence[DetectedTerm],
        mapper: ConceptPrimitiveMapper,
        *,
        task: str = "mcq",
    ) -> tuple[str, PrimitiveBottleneck]:
        bottleneck = self.compress(item, detected_terms, mapper)
        neutral_question = replace_detected_terms(item.original_question, detected_terms)
        if task == "open_ended":
            rewritten = (
                f"Consider a method that {bottleneck.primary_phrase}. "
                f"{neutral_question} "
                "Focus on the functional reason in ordinary terms."
            )
            return rewritten, bottleneck
        rewritten = (
            f"Consider a method that {bottleneck.primary_phrase}. "
            f"{neutral_question} "
            "Explain the functional reason in ordinary terms and choose the best option."
        )
        return rewritten, bottleneck


class EraNeutralPromptGenerator:
    def __init__(
        self,
        detector: AnachronismDetector,
        mapper: ConceptPrimitiveMapper,
        validator: RuleBasedValidator,
        repair_loop: DeterministicRepairLoop,
        proposed_rewriter: DenoisingPrimitiveBottleneckRewriter,
    ) -> None:
        self.detector = detector
        self.mapper = mapper
        self.validator = validator
        self.repair_loop = repair_loop
        self.rule_only = RuleOnlyRewriter()
        self.length_controlled = LengthControlledRewriter()
        self.proposed = proposed_rewriter

    def rewrite_item(self, item: DatasetItem) -> list[RewriteArtifact]:
        return self._rewrite_item(item, task="mcq")

    def rewrite_open_item(self, item: DatasetItem) -> list[RewriteArtifact]:
        return self._rewrite_item(item, task="open_ended")

    def _rewrite_item(self, item: DatasetItem, *, task: str) -> list[RewriteArtifact]:
        detected = self.detector.detect_details(item.original_question)
        detected_terms = [term.term for term in detected]
        mapped = self.mapper.map_terms(detected)
        detector_token_scores = self.detector.score_tokens(item.original_question)

        artifacts: list[RewriteArtifact] = []
        artifacts.append(
            self._artifact(
                item=item,
                condition="rule_only",
                rewritten_question=self.rule_only.rewrite(item, detected, self.mapper, task=task),
                detected_terms=detected_terms,
                mapped_primitives=mapped,
                detector_token_scores=detector_token_scores,
                model_info={},
                repair=True,
                repair_primitives=mapped,
            )
        )
        proposed_question, bottleneck = self.proposed.rewrite(item, detected, self.mapper, task=task)
        proposed_mapped = list(bottleneck.primitive_ids)
        artifacts.append(
            self._artifact(
                item=item,
                condition="length_controlled",
                rewritten_question=self.length_controlled.rewrite(
                    item,
                    detected,
                    self.mapper,
                    target_tokens=len(tokenize(proposed_question)),
                    task=task,
                ),
                detected_terms=detected_terms,
                mapped_primitives=mapped,
                detector_token_scores=detector_token_scores,
                model_info={"target_condition": "proposed", "target_tokens": len(tokenize(proposed_question))},
                repair=False,
                repair_primitives=mapped,
            )
        )
        artifacts.append(
            self._artifact(
                item=item,
                condition="proposed_no_validator",
                rewritten_question=proposed_question,
                detected_terms=detected_terms,
                mapped_primitives=proposed_mapped,
                detector_token_scores=detector_token_scores,
                model_info=self.proposed.model_info,
                repair=False,
                repair_primitives=proposed_mapped,
            )
        )
        artifacts.append(
            self._artifact(
                item=item,
                condition="proposed",
                rewritten_question=proposed_question,
                detected_terms=detected_terms,
                mapped_primitives=proposed_mapped,
                detector_token_scores=detector_token_scores,
                model_info=self.proposed.model_info,
                repair=True,
                repair_primitives=proposed_mapped,
            )
        )
        return artifacts

    def _artifact(
        self,
        *,
        item: DatasetItem,
        condition: str,
        rewritten_question: str,
        detected_terms: list[str],
        mapped_primitives: list[str],
        detector_token_scores: list[dict[str, Any]],
        model_info: dict[str, Any],
        repair: bool,
        repair_primitives: Sequence[str],
    ) -> RewriteArtifact:
        generation_validation = self.validator.validate(
            original_question=item.original_question,
            rewritten_question=rewritten_question,
            choices=item.choices,
            forbidden_terms=detected_terms,
            required_primitives=repair_primitives,
        )
        repair_attempts = 0
        final_question = rewritten_question
        if repair:
            final_question, generation_validation, repair_attempts = self.repair_loop.repair(
                original_question=item.original_question,
                rewritten_question=rewritten_question,
                choices=item.choices,
                forbidden_terms=detected_terms,
                validation_primitives=repair_primitives,
                repair_primitives=repair_primitives,
                removable_terms=detected_terms,
            )
        gold_validation = self.validator.validate(
            original_question=item.original_question,
            rewritten_question=final_question,
            choices=item.choices,
            forbidden_terms=item.forbidden_terms,
            required_primitives=item.required_primitives,
        )
        gold_validation["generation_validation"] = generation_validation
        return RewriteArtifact(
            item_id=item.id,
            condition=condition,
            original_question=item.original_question,
            rewritten_question=final_question,
            detected_terms=detected_terms,
            mapped_primitives=mapped_primitives,
            rewrite_validation=gold_validation,
            repair_attempts=repair_attempts,
            pass_validation=bool(gold_validation.get("pass")),
            detector_token_scores=detector_token_scores,
            model_info=model_info,
        )


def build_default_generator() -> EraNeutralPromptGenerator:
    from talkie_bridge.primitive import build_modern_terms_dictionary, build_primitive_dictionary

    modern = build_modern_terms_dictionary()
    primitives = build_primitive_dictionary()
    return build_generator(modern, primitives)


def build_generator(
    modern_terms_dictionary: dict[str, dict[str, Any]],
    primitive_dictionary: dict[str, dict[str, Any]],
    *,
    validation_primitive_dictionary: dict[str, dict[str, Any]] | None = None,
    token_classifier: Any | None = None,
    autoencoder: DenoisingPrimitiveAutoencoder | None = None,
    autoencoder_info: dict[str, Any] | None = None,
) -> EraNeutralPromptGenerator:
    detector = AnachronismDetector(modern_terms_dictionary, token_classifier=token_classifier)
    mapper = ConceptPrimitiveMapper(primitive_dictionary, modern_terms_dictionary)
    validator = RuleBasedValidator(validation_primitive_dictionary or primitive_dictionary)
    repair = DeterministicRepairLoop(mapper, validator)
    proposed = DenoisingPrimitiveBottleneckRewriter(
        primitive_dictionary,
        autoencoder=autoencoder,
        model_info=autoencoder_info,
    )
    return EraNeutralPromptGenerator(detector, mapper, validator, repair, proposed)


def replace_detected_terms(text: str, detected_terms: Sequence[DetectedTerm]) -> str:
    rewritten = text
    for term in sorted({item.term for item in detected_terms}, key=len, reverse=True):
        rewritten = replace_term(rewritten, term, "this method")
    return rewritten


def match_token_length(text: str, target_tokens: int) -> str:
    current = len(tokenize(text))
    if target_tokens <= 0:
        return text
    matched = text
    filler_tokens = ["neutral", "plain", "general", "ordinary", "brief", "careful"]
    index = 0
    while current < target_tokens:
        matched += " " + filler_tokens[index % len(filler_tokens)]
        index += 1
        current = len(tokenize(matched))
    return matched
