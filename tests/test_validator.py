from __future__ import annotations

from talkie_bridge.primitive import ConceptPrimitiveMapper, build_modern_terms_dictionary, build_primitive_dictionary
from talkie_bridge.validator import DeterministicRepairLoop, RuleBasedValidator


def test_validator_flags_forbidden_terms_and_repair_removes_them() -> None:
    modern = build_modern_terms_dictionary()
    primitives = build_primitive_dictionary()
    mapper = ConceptPrimitiveMapper(primitives, modern)
    validator = RuleBasedValidator(primitives)
    repair = DeterministicRepairLoop(mapper, validator)

    original = "Why is RAG useful for reducing unsupported answers?"
    rewritten = "RAG is useful because the answer is option A."
    choices = {
        "A": "It checks records before writing.",
        "B": "It guesses randomly.",
        "C": "It hides information.",
        "D": "It changes names.",
    }
    report = validator.validate(
        original_question=original,
        rewritten_question=rewritten,
        choices=choices,
        forbidden_terms=["RAG"],
        required_primitives=["record_lookup_before_reply"],
    )

    fixed, fixed_report, attempts = repair.repair(
        original_question=original,
        rewritten_question=rewritten,
        choices=choices,
        forbidden_terms=["RAG"],
        validation_primitives=["record_lookup_before_reply"],
        repair_primitives=["record_lookup_before_reply"],
        removable_terms=["RAG"],
    )

    assert "forbidden_terms_remaining" in report["fail_reasons"]
    assert "answer_leakage_risk" in report["fail_reasons"]
    assert "RAG" not in fixed
    assert attempts > 0
    assert fixed_report["forbidden_terms_remaining"] == []
