from __future__ import annotations

from talkie_bridge.detector import AnachronismDetector
from talkie_bridge.primitive import (
    ConceptPrimitiveMapper,
    build_modern_terms_dictionary,
    build_primitive_dictionary,
)


def test_detector_matches_aliases_and_mapper_returns_primitives() -> None:
    modern = build_modern_terms_dictionary()
    primitives = build_primitive_dictionary()
    detector = AnachronismDetector(modern)
    mapper = ConceptPrimitiveMapper(primitives, modern)

    detected = detector.detect_details("Why does RAG help an LLM avoid hallucination?")
    detected_terms = [item.term for item in detected]
    primitive_ids = mapper.map_terms(detected)

    assert "RAG" in detected_terms
    assert any(term.lower() == "llm" or "hallucination" in term.lower() for term in detected_terms)
    assert "record_lookup_before_reply" in primitive_ids
    assert "unsupported_automatic_text" in primitive_ids

