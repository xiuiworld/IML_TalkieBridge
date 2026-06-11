# IML_TalkieBridge

TalkieBridge is an Introduction to Machine Learning term project about
anachronism-aware prompt rewriting for Talkie-1930.

The project does **not** modify Talkie-1930. Talkie is treated as a fixed
downstream evaluator. The code in this repository builds and evaluates a
preprocessing layer that rewrites modern/anachronistic prompts into
1930-era-neutral primitive descriptions.

## Current Research Direction

The current evidence does not support the claim that the proposed preprocessing
improves Talkie multiple-choice accuracy. A completed 100-item diagnostic run
produced:

| Condition | Accuracy |
|---|---:|
| `raw` | 0.21 |
| `rule_only` | 0.19 |
| `length_controlled` | 0.24 |
| `proposed` | 0.19 |
| `proposed_no_validator` | 0.19 |

Because four-choice accuracy is near the 25% random baseline, the most
defensible proposal is to treat MCQ accuracy as a diagnostic/secondary
evaluation and use open-ended response quality as the primary evaluation.

The updated research question is:

> Does era-neutral prompt rewriting improve the quality of Talkie's downstream
> open-ended responses to modern/anachronistic prompts?

See `PROJECT_PROPOSAL.md` for the full revised proposal.

## Repository Layout

```text
.
├─ PROJECT_PROPOSAL.md          # Revised proposal and claims boundary
├─ DATASET_SCHEMA.md            # Current MCQ schema and open-ended judge extension
├─ pyproject.toml               # Package metadata and console script
├─ src/talkie_bridge/           # Main implementation
├─ tests/                       # Unit and pipeline tests
├─ data/                        # Dataset and dictionary files
├─ input_data/                  # Generated prompt/response collection sheets
├─ results/                     # Metrics, reports, and generated artifacts
└─ cache/                       # Model/resource/cache artifacts
```

`prototypes/`, `docs/`, and `Teammate/` are not the final execution path.
Prototype single-file scripts are historical references only. Use
`src/talkie_bridge/` and the CLI below for current runs.

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

The package has no required third-party runtime dependencies.

After editable install, either command style works:

```powershell
python -m talkie_bridge.cli --help
talkie-bridge --help
```

## Data Files

The main dataset path is:

```text
data/generated_questions.jsonl
```

Dictionary paths:

```text
data/modern_terms_dictionary.json
data/primitive_dictionary.json
```

See `DATASET_SCHEMA.md` before editing these files. The current CLI expects
four-choice items, but the proposal recommends extending the data with
open-ended fields for the primary response-quality evaluation.

## Common Commands

Generate a small mock dataset for smoke testing only:

```powershell
python -m talkie_bridge.cli init-mock-data --force
```

Generate rewrites and prompts without Talkie responses:

```powershell
python -m talkie_bridge.cli rewrite-only --eval-split all --concept-dictionary-json data/modern_terms_dictionary.json --primitive-dictionary-json data/primitive_dictionary.json
```

Prepare a manual Talkie response sheet:

```powershell
python -m talkie_bridge.cli prepare-manual --eval-split all --concept-dictionary-json data/modern_terms_dictionary.json --primitive-dictionary-json data/primitive_dictionary.json
```

Paste Talkie responses into:

```text
input_data/manual_talkie_input_sheet.csv
```

Then evaluate the manual responses:

```powershell
python -m talkie_bridge.cli evaluate-manual --eval-split all --manual-response-csv input_data/manual_talkie_input_sheet.csv --concept-dictionary-json data/modern_terms_dictionary.json --primitive-dictionary-json data/primitive_dictionary.json
```

## Open-Ended Evaluation Commands

The revised proposal's primary path is open-ended response quality. The local
input files have already been generated with:

```powershell
python -m talkie_bridge.cli prepare-open-ended --eval-split all --concept-dictionary-json data\modern_terms_dictionary.json --primitive-dictionary-json data\primitive_dictionary.json
```

This writes the Talkie input sheet:

```text
input_data/open_ended_talkie_input_sheet.csv
```

It contains 500 rows: 100 items x 5 conditions. These prompts do not include
MCQ choices and ask Talkie for 1-2 sentence mechanism explanations.

To run the long unofficial API collection step later:

```powershell
python -m talkie_bridge.cli run-open-ended-api --provider unofficial-api --eval-split all --max-tokens 96 --concept-dictionary-json data\modern_terms_dictionary.json --primitive-dictionary-json data\primitive_dictionary.json
```

During this long run, the CLI prints progress and retry status:

```text
[start] Open-ended Talkie API: 500 prompts
[start] 37/500 (7.2%) fixed_q008/proposed
[retry] fixed_q008/proposed: HTTP 429; retry 1/6 after 2m 0s
[done] 37/500 (7.4%) fixed_q008/proposed; item=2m 7s; elapsed=9m 14s; eta=1h 55m 42s; cache=no; chars=312
```

Cached responses are marked as `cache=yes`, so reruns should show faster
progress for already collected prompts.

That command writes:

```text
results/open_ended_responses.csv
input_data/open_ended_judge_input_sheet.csv
results/open_ended_judge_pairs_unblinded.csv
```

Manual collection is also supported. Fill `raw_response_manual` in
`input_data/open_ended_talkie_input_sheet.csv`, then run:

```powershell
python -m talkie_bridge.cli evaluate-open-ended-manual --eval-split all --manual-response-csv input_data/open_ended_talkie_input_sheet.csv --concept-dictionary-json data\modern_terms_dictionary.json --primitive-dictionary-json data\primitive_dictionary.json
```

After the judge sheet is filled with LLM Judge outputs in `judge_raw_output`,
compute pairwise response-quality metrics with:

```powershell
python -m talkie_bridge.cli evaluate-open-ended-judge --judge-response-csv input_data/open_ended_judge_input_sheet.csv
```

The final judge-side outputs are:

```text
results/open_ended_judge_scores.csv
results/open_ended_pairwise_metrics.csv
results/open_ended_judge_integrity.csv
results/open_ended_response_quality.md
```

Optionally collect responses through the unofficial Talkie API:

```powershell
python -m talkie_bridge.cli run-api --provider unofficial-api --eval-split all --concept-dictionary-json data/modern_terms_dictionary.json --primitive-dictionary-json data/primitive_dictionary.json
```

Manual CSV mode is the preferred reproducibility path. The unofficial API mode
is a convenience path and should be labeled as such in reports.

Generate a static demo page for one item:

```powershell
python -m talkie_bridge.cli demo --item-id q001 --out results/demo.html
```

## Generated Outputs

Important output files:

| Path | Purpose |
|---|---|
| `results/prepared_prompts.csv` | Generated prompts for all conditions |
| `input_data/manual_talkie_input_sheet.csv` | Manual response collection sheet |
| `results/per_item_results.csv` | Item-level responses, parsed answers, correctness |
| `results/final_metrics.csv` | Accuracy, macro-F1, invalid rate |
| `results/key_comparisons.csv` | Paired condition comparisons |
| `results/paired_tests.csv` | Bootstrap intervals and exact McNemar tests |
| `results/component_metrics.csv` | Detector, rewrite, validator, repair metrics |
| `results/dataset_quality.csv` | Dataset and rewrite diagnostics |
| `results/response_integrity.csv` | Prompt hash and response completion checks |
| `results/report.md` | Generated experiment report |
| `input_data/open_ended_talkie_input_sheet.csv` | Open-ended Talkie input sheet ready for collection |
| `results/open_ended_prepared_prompts.csv` | Open-ended prompts for all conditions |
| `results/open_ended_responses.csv` | Collected open-ended Talkie responses |
| `input_data/open_ended_judge_input_sheet.csv` | Blind LLM Judge input sheet |
| `results/open_ended_pairwise_metrics.csv` | Pairwise judge win/tie metrics |
| `results/open_ended_response_quality.md` | Open-ended response-quality report |

The existing `results/report.md` is a diagnostic MCQ report, not proof of MCQ
improvement.

## Current Diagnostic Result

The latest 100-item run used the unofficial Talkie API and collected 500
responses across five conditions. Integrity checks passed:

- prompt hash match rate: 1.0
- response completion rate: 1.0
- blank responses: 0

All-row MCQ accuracy:

```text
raw               21/100
rule_only         19/100
length_controlled 24/100
proposed          19/100
proposed_no_validator 19/100
```

The honest interpretation is that the current proposed method does not improve
four-choice accuracy. The result motivates a better primary evaluation:
blind pairwise judging of open-ended Talkie response quality.

## Planned Primary Evaluation

For the revised proposal, each item should be converted to an open-ended task:

```text
Answer in 1-2 sentences. Explain the practical mechanism.
```

Talkie responses should be collected for `raw`, `rule_only`,
`length_controlled`, and `proposed`. A blind LLM Judge should compare response
pairs with randomized A/B order and a fixed rubric:

- task relevance,
- functional correctness,
- era-neutrality,
- anachronism handling,
- answer usefulness,
- leakage risk.

Store judge prompts, raw judge outputs, parsed scores, condition order,
randomization seed, and model/version metadata. Suggested output files are
documented in `DATASET_SCHEMA.md`.

## Claims Boundary

Valid claims:

- The repository implements a prompt rewriting and evaluation pipeline in front
  of a fixed Talkie evaluator.
- The current MCQ diagnostic run does not show proposed accuracy improvement.
- Component metrics can evaluate anachronism removal, primitive recall,
  validation pass rate, and leakage risk.
- The revised primary question is downstream open-ended response quality.

Avoid these claims:

- "We improved Talkie."
- "The proposed method improves multiple-choice accuracy."
- "The current primitive bottleneck selector is a full natural-language DAE."
- "The validator improves performance" unless a future ablation supports it.

## Running Tests

```powershell
python -m pytest
```

The tests cover schema handling, detector/mapper behavior, validator checks,
prompting/metrics, clients, autoencoder selection, and the CLI pipeline.
