# TalkieBridge Dataset Schema

Put the final experiment dataset at `data/generated_questions.jsonl` by default.
Each line must be one JSON object for one four-choice question.

Use `data/generated_questions_example.jsonl` only as a filename/schema example.
Copy it to `data/generated_questions.jsonl` before editing real items.

Required fields:

| Field | Fill with |
|---|---|
| `id` | Stable question id, for example `q001` |
| `domain` | One domain label, for example `AI_Computing` or `Medicine_Biology` |
| `original_question` | The modern-language multiple-choice question stem |
| `choices` | JSON object with exactly `A`, `B`, `C`, `D` |
| `gold_answer` | Correct label: `A`, `B`, `C`, or `D` |
| `gold_anachronism_terms` | Modern terms that should be detected |
| `forbidden_terms` | Terms that must not remain in rewritten prompts |
| `required_primitives` | Primitive ids that should be preserved |
| `primitive_phrase` | Era-neutral functional description of the modern concept |
| `human_validated` | `true` only after manual review |

Optional field:

| Field | Fill with |
|---|---|
| `target_year` | Default is `1930` |
| `split` | `train`, `dev`, or `test`; if omitted, the CLI assigns a deterministic 70/15/15 split from `id` |

Example JSONL row:

```json
{"id":"q001","domain":"AI_Computing","original_question":"Why is RAG useful for reducing LLM hallucination?","choices":{"A":"It checks relevant records before writing, so the reply is less likely to rest only on memory.","B":"It improves the result only by making the message longer.","C":"It works mainly by changing the names of answer choices.","D":"It guarantees a correct answer without checking evidence."},"gold_answer":"A","gold_anachronism_terms":["RAG","LLM hallucination"],"forbidden_terms":["RAG","retrieval augmented generation","LLM","large language model","hallucination"],"required_primitives":["record_lookup_before_reply","unsupported_automatic_text"],"primitive_phrase":"first searches a store of relevant records before composing a reply, so unsupported statements from an automatic writing system can be reduced","human_validated":false,"target_year":1930,"split":"test"}
```

Important:

- `required_primitives` is treated as gold annotation for validation and metrics.
- `forbidden_terms`, `required_primitives`, and `primitive_phrase` are treated as annotation fields.
- Rewriting and deterministic repair do not read the current eval item's annotation
  fields for generation-time control.
- Gold annotation validation is computed after the final rewritten prompt is
  produced and is stored as metrics/report evidence.
- Generation dictionaries and the detector are built from predeclared dictionary files
  when provided; otherwise they fall back to `split=train` rows only.
- The learned primitive bottleneck trains on `split=train`, selects on `split=dev`,
  and reports final evidence on `split=test` when test rows exist.
- Very small mock datasets fall back to deterministic primitive rewriting and write
  the reason to `cache/primitive_autoencoder_selection.json`.

Use this to create a local mock file:

```powershell
$env:PYTHONPATH='src'
python -m talkie_bridge.cli init-mock-data --force
```

Then replace `data/generated_questions.jsonl` with your manually authored and
reviewed dataset before running `prepare-manual`.

Predeclared generation dictionaries:

- `modern_terms_dictionary.json` maps normalized aliases to `{alias, canonical_term, primitive_id, domain}`.
- `primitive_dictionary.json` maps primitive ids to `{concept_term, domain, primitive_phrase}`.
- `modern_terms_dictionary_example.json` and `primitive_dictionary_example.json`
  are template files only; copy them to the non-example filenames before using
  them in CLI commands.
- These files are allowed to be prior project resources. They should not be
  generated from dev/test item annotations during final evaluation.
- They are used only when passed explicitly with `--concept-dictionary-json`
  and `--primitive-dictionary-json`; otherwise generation resources come from
  the train split.
- Mock dictionary files are for diagnostics only and are blocked during
  evaluation unless `--allow-mock-dictionary` is supplied.
