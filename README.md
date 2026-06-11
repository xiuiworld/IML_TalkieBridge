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

The open-ended evaluation has now been run on 100 items x 5 conditions, with
400 blind pairwise LLM Judge comparisons. The strongest result is that
`proposed` beats `raw` on 79/100 pairs and `length_controlled` on 88/100 pairs.
The important caveat is that `proposed` does not beat `rule_only` in this run
(45 wins, 51 losses, 4 ties), and `proposed` ties `proposed_no_validator` on all
100 pairs.

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

Optional long-run collection helpers need `requests`:

```powershell
python -m pip install requests
```

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

The repository also includes a resumable OpenAI judge helper. It reads the API
key only from `OPENAI_API_KEY`; do not hard-code keys in files:

```powershell
$env:OPENAI_API_KEY = "<your key>"
python tools\fill_openai_judge.py --api responses --model gpt-5.4-mini --reasoning-effort none
Remove-Item Env:\OPENAI_API_KEY
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

Generate the presentation-safe cached live demo for the completed open-ended
run:

```powershell
python -m talkie_bridge.cli demo-open-ended --out results/open_ended_live_demo.html
```

This HTML uses the stored Talkie responses and stored LLM Judge outputs. It
does not call Talkie, OpenAI, or any other external API during the presentation.
The default page is a card-based presentation demo designed for a short live
walkthrough. Click the next highlighted card to show detection, the
era-neutral rewrite, Talkie A/B responses, and the blind judge result.

For GitHub Pages, copy the generated demo to `docs/index.html` and configure
Pages to serve from `main` / `/docs`:

```powershell
Copy-Item results\open_ended_live_demo.html docs\index.html -Force
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

## Completed Primary Open-Ended Evaluation

The revised proposal's primary evaluation was run as an open-ended task:

```text
Answer in 1-2 sentences. Explain the practical mechanism.
```

Talkie responses were collected for `raw`, `rule_only`, `length_controlled`,
`proposed`, and `proposed_no_validator`. A blind LLM Judge compared response
pairs with randomized A/B order and a fixed rubric:

- task relevance,
- functional correctness,
- era-neutrality,
- anachronism handling,
- answer usefulness,
- leakage risk.

The 400-pair judge run completed with full integrity:

- judge prompt hash match rate: 1.0
- judge parse rate: 1.0
- blank judge outputs: 0

Pairwise results:

| Comparison | Proposed Wins | Baseline Wins | Ties | Proposed Win Rate, Excluding Ties |
|---|---:|---:|---:|---:|
| `proposed` vs `raw` | 79 | 18 | 3 | 0.8144 |
| `proposed` vs `rule_only` | 45 | 51 | 4 | 0.4688 |
| `proposed` vs `length_controlled` | 88 | 11 | 1 | 0.8889 |
| `proposed` vs `proposed_no_validator` | 0 | 0 | 100 | 0.0000 |

Interpretation:

- The open-ended judge result supports the claim that `proposed` improves
  response quality over raw modern prompts.
- `proposed` also strongly beats the length-controlled condition, so the gain
  is not explained only by prompt length.
- `rule_only` is competitive and slightly better than `proposed` by pairwise
  wins in this run, so the final report should not claim that the full pipeline
  dominates every simpler baseline.
- The validator did not show downstream effect because `proposed` and
  `proposed_no_validator` tied on every judged pair.

Store judge prompts, raw judge outputs, parsed scores, condition order,
randomization seed, and model/version metadata. The concrete output files are
documented in `DATASET_SCHEMA.md`.

## Claims Boundary

Valid claims:

- The repository implements a prompt rewriting and evaluation pipeline in front
  of a fixed Talkie evaluator.
- The current MCQ diagnostic run does not show proposed accuracy improvement.
- The completed open-ended judge run shows that `proposed` beats `raw` and
  `length_controlled` on pairwise response quality, while not beating
  `rule_only`.
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
