"""Concept primitives and a small proposal-aligned seed dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from talkie_bridge.data_schema import DatasetItem, LABELS
from talkie_bridge.detector import DetectedTerm, normalize_term


@dataclass(frozen=True)
class ModernConcept:
    domain: str
    term: str
    aliases: tuple[str, ...]
    task: str
    primitive_id: str
    primitive_phrase: str
    correct_mechanism: str


DEFAULT_CONCEPTS: tuple[ModernConcept, ...] = (
    ModernConcept(
        "AI_Computing",
        "RAG",
        ("RAG", "retrieval augmented generation"),
        "reducing unsupported answers from an automatic writing system",
        "record_lookup_before_reply",
        "first searches a store of relevant records before composing a reply",
        "It checks relevant records before writing, so the reply is less likely to rest only on memory.",
    ),
    ModernConcept(
        "AI_Computing",
        "LLM hallucination",
        ("LLM hallucination", "hallucination", "large language model"),
        "avoiding unsupported statements in generated text",
        "unsupported_automatic_text",
        "an automatic writing system may produce unsupported statements unless constrained by evidence",
        "It reduces unsupported output by tying the answer to evidence rather than fluent wording alone.",
    ),
    ModernConcept(
        "AI_Computing",
        "API",
        ("API", "application programming interface"),
        "letting two software services cooperate",
        "published_machine_commands",
        "offers a published set of commands one machine can use to request work from another",
        "It gives both systems a stable contract for requests and replies.",
    ),
    ModernConcept(
        "AI_Computing",
        "GPU",
        ("GPU", "graphics processing unit"),
        "speeding up repeated numerical work",
        "parallel_small_calculations",
        "performs many small calculations at the same time",
        "It finishes repeated similar calculations faster by doing many of them in parallel.",
    ),
    ModernConcept(
        "Medicine_Biology",
        "PCR",
        ("PCR", "polymerase chain reaction"),
        "finding a tiny trace of biological material",
        "copy_small_biological_trace",
        "repeatedly copies a selected biological trace until it is easier to detect",
        "It makes a small trace measurable by copying the selected material many times.",
    ),
    ModernConcept(
        "Medicine_Biology",
        "mRNA vaccine",
        ("mRNA vaccine", "messenger RNA vaccine"),
        "training the body to recognize a disease marker",
        "temporary_body_instruction",
        "delivers a temporary instruction that causes the body to make a harmless marker for practice",
        "It teaches recognition of a marker without requiring the full disease agent.",
    ),
    ModernConcept(
        "Medicine_Biology",
        "MRI",
        ("MRI", "magnetic resonance imaging"),
        "seeing soft tissues inside the body",
        "magnet_radio_body_picture",
        "uses strong magnetism and radio signals to form pictures of internal soft tissue",
        "It distinguishes soft tissues without cutting into the body.",
    ),
    ModernConcept(
        "Communication_Media",
        "smartphone",
        ("smartphone", "smart phone"),
        "checking messages and records while away from a desk",
        "pocket_wireless_record_device",
        "a pocket device combines wireless communication with stored records and small programs",
        "It lets a person communicate and retrieve stored information while moving around.",
    ),
    ModernConcept(
        "Communication_Media",
        "social media",
        ("social media", "social network"),
        "spreading short public messages among many people",
        "public_message_network",
        "a shared communication network lets many people publish and respond to short messages",
        "It spreads messages quickly because each participant can pass them to many others.",
    ),
    ModernConcept(
        "Communication_Media",
        "QR code",
        ("QR code", "quick response code"),
        "opening stored information from a printed sign",
        "camera_read_square_symbol",
        "stores coded information in a square pattern that a camera can read",
        "It lets a machine read the stored address or instruction without manual typing.",
    ),
    ModernConcept(
        "Transportation_Engineering",
        "GPS",
        ("GPS", "global positioning system"),
        "finding a route through an unfamiliar city",
        "radio_position_signals",
        "uses timed radio signals from known points to estimate position",
        "It estimates location from signals and can compare that location with a stored map.",
    ),
    ModernConcept(
        "Transportation_Engineering",
        "electric vehicle",
        ("electric vehicle", "EV"),
        "moving a car without burning fuel in the engine",
        "stored_electric_motor_drive",
        "uses stored electric charge to turn a motor that moves the vehicle",
        "It moves by using stored electrical energy instead of explosions in an engine.",
    ),
    ModernConcept(
        "Transportation_Engineering",
        "autopilot",
        ("autopilot",),
        "keeping an aircraft steady on a long trip",
        "instrument_feedback_steering",
        "uses instruments and control rules to adjust steering automatically",
        "It makes repeated small corrections without constant manual steering.",
    ),
    ModernConcept(
        "Environment_Energy",
        "climate model",
        ("climate model",),
        "estimating future temperature patterns",
        "mathematical_weather_system",
        "uses mathematical rules to simulate connected air, water, and energy flows",
        "It tests how changes may influence a large connected physical system.",
    ),
    ModernConcept(
        "Environment_Energy",
        "solar panel",
        ("solar panel", "photovoltaic panel"),
        "making electric power from sunlight",
        "light_to_electric_current",
        "turns light falling on a prepared surface into electric current",
        "It produces power when light strikes the prepared material.",
    ),
    ModernConcept(
        "Environment_Energy",
        "lithium-ion battery",
        ("lithium-ion battery", "lithium ion battery"),
        "storing energy for a portable machine",
        "reversible_chemical_charge_store",
        "stores electric charge through a reversible chemical process",
        "It can be charged and discharged many times to supply a portable device.",
    ),
    ModernConcept(
        "Daily_Tech_Society",
        "contactless payment",
        ("contactless payment",),
        "paying quickly at a shop counter",
        "short_range_wireless_payment",
        "sends payment information over a very short wireless distance",
        "It avoids handling coins or writing account details by exchanging information at close range.",
    ),
    ModernConcept(
        "Daily_Tech_Society",
        "recommendation algorithm",
        ("recommendation algorithm", "recommender system"),
        "choosing a film or article a person may like",
        "rank_by_past_preference",
        "ranks options using patterns in earlier choices and similarities",
        "It compares preference patterns to put likely useful choices first.",
    ),
)


DISTRACTORS: tuple[str, ...] = (
    "It works mainly by changing the names of the answer choices.",
    "It succeeds because it hides the important information from the user.",
    "It improves the result only by making the message longer.",
    "It removes the need for any stored information or measurement.",
    "It guarantees a correct answer without checking evidence.",
    "It depends on guessing randomly among several alternatives.",
)


QUESTION_TEMPLATES: tuple[str, ...] = (
    "Why is {term} useful for {task}?",
    "Which reason best explains why {term} helps with {task}?",
    "How does {term} mainly support {task}?",
)


class ConceptPrimitiveMapper:
    def __init__(self, primitive_dictionary: dict[str, dict[str, Any]], modern_terms_dictionary: dict[str, dict[str, Any]]) -> None:
        self.primitive_dictionary = primitive_dictionary
        self.modern_terms_dictionary = modern_terms_dictionary

    def map_terms(self, detected_terms: Sequence[str | DetectedTerm]) -> list[str]:
        primitive_ids: list[str] = []
        for term in detected_terms:
            if isinstance(term, DetectedTerm):
                primitive_id = term.primitive_id
            else:
                meta = self.modern_terms_dictionary.get(normalize_term(str(term)))
                primitive_id = str(meta["primitive_id"]) if meta else ""
            if primitive_id and primitive_id not in primitive_ids:
                primitive_ids.append(primitive_id)
        return primitive_ids

    def phrases_for(self, primitive_ids: Sequence[str]) -> list[str]:
        phrases: list[str] = []
        for primitive_id in primitive_ids:
            meta = self.primitive_dictionary.get(primitive_id)
            if meta:
                phrases.append(str(meta["primitive_phrase"]))
        return phrases

    def phrase_for(self, primitive_id: str) -> str:
        meta = self.primitive_dictionary.get(primitive_id, {})
        return str(meta.get("primitive_phrase", ""))


def build_modern_terms_dictionary(concepts: Sequence[ModernConcept] = DEFAULT_CONCEPTS) -> dict[str, dict[str, Any]]:
    dictionary: dict[str, dict[str, Any]] = {}
    for concept in concepts:
        for alias in concept.aliases:
            dictionary[normalize_term(alias)] = {
                "alias": alias,
                "canonical_term": concept.term,
                "primitive_id": concept.primitive_id,
                "domain": concept.domain,
            }
    return dictionary


def build_primitive_dictionary(concepts: Sequence[ModernConcept] = DEFAULT_CONCEPTS) -> dict[str, dict[str, Any]]:
    return {
        concept.primitive_id: {
            "concept_term": concept.term,
            "domain": concept.domain,
            "primitive_phrase": concept.primitive_phrase,
        }
        for concept in concepts
    }


def build_dictionaries_from_dataset(items: Sequence[DatasetItem]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build detector and primitive dictionaries from user-provided items."""
    modern_terms: dict[str, dict[str, Any]] = {}
    primitives: dict[str, dict[str, Any]] = {}

    for item in items:
        primary_term = item.gold_anachronism_terms[0] if item.gold_anachronism_terms else ""
        primary_primitive = item.required_primitives[0] if item.required_primitives else normalize_term(primary_term)
        for primitive_id in item.required_primitives:
            primitives[primitive_id] = {
                "concept_term": primary_term or primitive_id,
                "domain": item.domain,
                "primitive_phrase": item.primitive_phrase,
            }

        aliases = [*item.gold_anachronism_terms, *item.forbidden_terms]
        normalized_gold = [normalize_term(term) for term in item.gold_anachronism_terms]
        for alias in aliases:
            primitive_id = primary_primitive
            normalized_alias = normalize_term(alias)
            for gold_index, normalized_term in enumerate(normalized_gold):
                if not normalized_term:
                    continue
                if normalized_alias in normalized_term or normalized_term in normalized_alias:
                    primitive_id = item.required_primitives[min(gold_index, len(item.required_primitives) - 1)] if item.required_primitives else primary_primitive
                    break
            key = normalize_term(alias)
            if not key:
                continue
            modern_terms[key] = {
                "alias": alias,
                "canonical_term": primary_term or alias,
                "primitive_id": primitive_id,
                "domain": item.domain,
            }

    return modern_terms, primitives


def generate_dataset(n_items: int = 30, target_year: int = 1930, seed: int = 13) -> list[DatasetItem]:
    rng = random.Random(seed)
    rows: list[DatasetItem] = []
    concepts = list(DEFAULT_CONCEPTS)
    for index in range(n_items):
        concept = concepts[index % len(concepts)]
        template = QUESTION_TEMPLATES[(index // len(concepts)) % len(QUESTION_TEMPLATES)]
        question = template.format(term=concept.term, task=concept.task)
        choices, gold = _choices_for(concept, index, rng)
        rows.append(
            DatasetItem(
                id=f"q{index + 1:03d}",
                domain=concept.domain,
                original_question=question,
                choices=choices,
                gold_answer=gold,
                gold_anachronism_terms=[concept.term],
                forbidden_terms=list(concept.aliases),
                required_primitives=[concept.primitive_id],
                primitive_phrase=concept.primitive_phrase,
                human_validated=False,
                target_year=target_year,
            )
        )
    return rows


def _choices_for(concept: ModernConcept, index: int, rng: random.Random) -> tuple[dict[str, str], str]:
    distractors = list(DISTRACTORS)
    rng.shuffle(distractors)
    ordered = [concept.correct_mechanism, *distractors[:3]]
    shift = index % len(LABELS)
    rotated = ordered[-shift:] + ordered[:-shift] if shift else ordered
    choices = {label: rotated[pos] for pos, label in enumerate(LABELS)}
    gold = next(label for label, text in choices.items() if text == concept.correct_mechanism)
    return choices, gold
