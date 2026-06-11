# Archived MCQ Diagnostic Run - 2026-06-11

This folder is a frozen copy of the previous TalkieBridge multiple-choice
diagnostic experiment, saved before implementing the revised open-ended response
quality evaluation.

The purpose of this archive is to preserve materials that may be useful for the
final report, presentation, or negative-result discussion.

## What This Run Shows

The 100-item four-choice experiment completed end to end, but it did not show
that the proposed preprocessing method improves Talkie multiple-choice
accuracy.

All-row accuracy:

| Condition | Correct / 100 | Accuracy |
|---|---:|---:|
| `raw` | 21 | 0.21 |
| `rule_only` | 19 | 0.19 |
| `length_controlled` | 24 | 0.24 |
| `proposed` | 19 | 0.19 |
| `proposed_no_validator` | 19 | 0.19 |

Main interpretation:

- The MCQ setup is near the 25% random baseline.
- `proposed` does not beat `raw`.
- `length_controlled` is the best condition, but only by a small margin.
- `proposed` and `proposed_no_validator` are identical on downstream accuracy.
- The run is still useful as an integrity-checked diagnostic result.

## Folder Contents

| Folder/File | Contents |
|---|---|
| `progress_summary_2026-06-11.md` | Human-readable progress summary of the previous experiment |
| `results/` | Metrics, paired comparisons, reports, prepared prompts, and item-level outputs |
| `data/` | Dataset and dictionary files used for the run |
| `input_data/` | Generated prompt sheets and manual-response sheet |
| `cache/` | Generation resource metadata, primitive selector cache, and unofficial API cache |

## Most Useful Files

For report writing:

- `results/report.md`
- `results/final_metrics.csv`
- `results/key_comparisons.csv`
- `results/paired_tests.csv`
- `results/component_metrics.csv`
- `results/dataset_quality.md`
- `results/qualitative_examples.md`

For reproducibility/auditing:

- `data/generated_questions.jsonl`
- `data/modern_terms_dictionary.json`
- `data/primitive_dictionary.json`
- `results/prepared_prompts.csv`
- `results/per_item_results.csv`
- `results/response_integrity.csv`
- `cache/generation_resource_source.json`
- `cache/talkie_unofficial_api_cache.jsonl`

## How To Use This Archive

Use this archive as evidence for the initial diagnostic stage:

> The original MCQ evaluation was completed with valid prompt hashes and full
> response coverage, but it did not support the claim that the proposed
> preprocessor improves multiple-choice accuracy. This motivated the revised
> primary evaluation based on open-ended response quality.

Do not use this archive to claim that Talkie itself was improved or that the
proposed method improved MCQ accuracy.
