from __future__ import annotations

from talkie_bridge.autoencoder import train_select_autoencoder
from talkie_bridge.data_schema import DatasetItem
from talkie_bridge.primitive import build_dictionaries_from_dataset
from talkie_bridge.rewriter import build_generator


def _item(index: int, split: str, term: str, primitive_id: str, phrase: str) -> DatasetItem:
    return DatasetItem(
        id=f"q{index:03d}",
        domain="AI_Computing",
        original_question=f"Why is {term} useful for a machine task {index}?",
        choices={"A": "correct", "B": "wrong", "C": "wrong", "D": "wrong"},
        gold_answer="A",
        gold_anachronism_terms=[term],
        forbidden_terms=[term],
        required_primitives=[primitive_id],
        primitive_phrase=phrase,
        human_validated=True,
        split=split,
    )


def test_autoencoder_trains_on_train_split_and_rewriter_does_not_need_gold_primitives() -> None:
    items = [
        _item(1, "train", "RAG", "record_lookup_before_reply", "searches records before composing a reply"),
        _item(2, "train", "search API", "published_machine_commands", "uses published commands between machines"),
        _item(3, "train", "GPU", "parallel_small_calculations", "performs many small calculations at the same time"),
        _item(4, "train", "cache", "nearby_result_store", "keeps recent results in a nearby store"),
        _item(5, "dev", "QR code", "camera_read_square_symbol", "stores coded information in a square pattern"),
    ]
    modern, primitives = build_dictionaries_from_dataset(items)
    model, selection = train_select_autoencoder(items, primitives, seed=7)

    generator = build_generator(modern, primitives, autoencoder=model, autoencoder_info=selection.__dict__)
    test_item = _item(6, "test", "RAG", "deliberately_wrong_gold", "this gold phrase must not drive generation")
    artifacts = generator.rewrite_item(test_item)
    proposed = next(artifact for artifact in artifacts if artifact.condition == "proposed")

    assert selection.train_items == 4
    assert selection.dev_items == 1
    assert "deliberately_wrong_gold" not in proposed.mapped_primitives
    assert "record_lookup_before_reply" in proposed.mapped_primitives


def test_eval_forbidden_terms_are_not_used_for_generation() -> None:
    train = _item(1, "train", "RAG", "record_lookup_before_reply", "searches records before composing a reply")
    eval_item = _item(2, "test", "never-seen-modern-term", "secret_gold_primitive", "secret gold phrase")
    modern, primitives = build_dictionaries_from_dataset([train])
    generator = build_generator(modern, primitives)

    proposed = next(artifact for artifact in generator.rewrite_item(eval_item) if artifact.condition == "proposed")

    assert "never-seen-modern-term" in proposed.rewritten_question
    assert proposed.mapped_primitives == []
    assert "secret gold phrase" not in proposed.rewritten_question


def test_eval_required_primitives_do_not_drive_repair() -> None:
    train = _item(1, "train", "RAG", "record_lookup_before_reply", "searches records before composing a reply")
    eval_item = _item(2, "test", "RAG", "secret_gold_primitive", "secret gold phrase")
    modern, primitives = build_dictionaries_from_dataset([train])
    validation_primitives = {
        **primitives,
        "secret_gold_primitive": {
            "concept_term": "secret",
            "domain": "AI_Computing",
            "primitive_phrase": "secret gold phrase",
        },
    }
    generator = build_generator(modern, primitives, validation_primitive_dictionary=validation_primitives)

    proposed = next(artifact for artifact in generator.rewrite_item(eval_item) if artifact.condition == "proposed")

    assert proposed.repair_attempts == 0
    assert "secret gold phrase" not in proposed.rewritten_question
    assert "required_primitives_missing" in proposed.rewrite_validation["fail_reasons"]
