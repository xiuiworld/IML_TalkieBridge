# TalkieBridge Project Proposal

## Title

**TalkieBridge: Era-Neutral Prompt Rewriting for Vintage LLM Evaluation**

## Summary

TalkieBridge studies whether a preprocessing layer can help a fixed vintage
language model respond more usefully to modern or anachronistic prompts. The
project does not train, fine-tune, patch, or vendor Talkie-1930. Instead, it
builds and evaluates a front-end prompt rewriter that converts modern surface
terms into 1930-era-neutral functional descriptions.

The initial implementation has already produced an end-to-end diagnostic
multiple-choice experiment over 100 items and five prompt conditions. That
experiment is useful, but it does not support the claim that the proposed
preprocessor improves multiple-choice accuracy. The strongest next proposal is
therefore to keep the MCQ experiment as a diagnostic/secondary result and make
LLM-judged open-ended response quality the primary evaluation.

## 1. Problem Statement

Modern prompts often contain terms that a 1930-era language model should not be
expected to understand directly: QR code, smartphone, Wi-Fi, cloud storage,
LLM, RAG, mRNA vaccine, GPS, and similar concepts. If Talkie-1930 gives a poor
answer to such a prompt, the failure may come from at least two different
causes:

- the model cannot perform the requested reasoning, or
- the input contains modern surface terms that are outside the model's intended
  historical knowledge boundary.

TalkieBridge addresses the second problem. It asks:

> Can a front-end preprocessor improve Talkie-1930's downstream response
> quality by rewriting modern/anachronistic expressions into era-neutral
> primitive descriptions?

This is deliberately not a claim about improving Talkie itself. Talkie remains
a fixed downstream evaluator. The proposed system is the prompt rewriting and
evaluation pipeline in front of Talkie.

## 2. Motivation

Vintage language models are useful for studying temporal generalization, but
their evaluation becomes ambiguous when modern concepts appear in the input.
Filtering out anachronistic questions is clean for benchmark analysis, but it
does not answer a practical question: can a user still ask about modern objects
if the query is translated into older, functional language?

For example:

```text
Raw:
Why might a QR code be selected for opening a stored address from a printed sign?

Era-neutral rewrite:
Why might a square printed pattern that stores coded information and can be read
by a camera be selected for opening a stored address from a printed sign?
```

The rewrite does not require Talkie to know the term "QR code". It exposes the
mechanism using more primitive concepts: printed pattern, coded information,
camera reading, and stored address. If this improves the resulting answer, the
project provides evidence that era-neutral rewriting can bridge part of the
temporal vocabulary gap.

## 3. Current Implementation

The implementation that should be treated as the project source of truth lives
in:

```text
src/talkie_bridge/
```

Important scope constraints:

- The original Talkie code is not included in this repository.
- `prototypes/`, `docs/`, and `Teammate/` are not the final execution path.
- Prototype single-file scripts such as `TermProject_team2.py` are not the
  current repository execution structure.
- The system should not be described as a full natural-language Denoising Text
  Autoencoder. The current implementation is closer to a dependency-free
  primitive bottleneck selector with deterministic rewrite and validation
  logic.

The current pipeline supports:

- a 100-item dataset,
- modern term and primitive dictionaries,
- prompt generation for five conditions,
- manual CSV evaluation and optional unofficial Talkie API collection,
- prompt hash integrity checks,
- response parsing for letter and sentence-style answers,
- condition-level, paired, component, dataset-quality, and integrity reports.

Implemented prompt conditions:

| Condition | Role |
|---|---|
| `raw` | Original modern-language prompt |
| `rule_only` | Dictionary/rule rewrite baseline |
| `length_controlled` | Rewrite with length control to separate content effects from prompt length |
| `proposed` | Primitive bottleneck rewrite with validation |
| `proposed_no_validator` | Ablation without validator effect |

## 4. Initial Diagnostic Result

The completed 100-item MCQ run produced the following all-row accuracy results:

| Condition | Correct / 100 | Accuracy |
|---|---:|---:|
| `raw` | 21 | 0.21 |
| `rule_only` | 19 | 0.19 |
| `length_controlled` | 24 | 0.24 |
| `proposed` | 19 | 0.19 |
| `proposed_no_validator` | 19 | 0.19 |

Key paired comparisons:

| Comparison | Net Change | Accuracy Delta |
|---|---:|---:|
| `length_controlled` vs `raw` | +3 | +0.03 |
| `proposed` vs `raw` | -2 | -0.02 |
| `proposed` vs `length_controlled` | -5 | -0.05 |
| `proposed` vs `proposed_no_validator` | 0 | 0.00 |

These results do not support the claim that the proposed preprocessor improves
Talkie multiple-choice accuracy. The best condition is `length_controlled`, but
its 24% accuracy remains near or below the 25% random baseline for a balanced
four-choice task. The validator also did not produce a measurable repair or
fallback effect in this run because `proposed` and `proposed_no_validator` were
identical on the downstream metric.

The diagnostic run is still valuable. It shows that the full pipeline executes,
that response collection is complete, and that prompt/response integrity is
sound. It also shows that multiple-choice letter selection may be a poor
primary task for this research question.

Current component-level evidence is more promising:

| Component Metric | Value |
|---|---:|
| Detector precision | 0.9300 |
| Detector recall | 0.6739 |
| Detector F1 | 0.7815 |
| Anachronism removal rate | 1.0000 |
| Required primitive recall | 0.7670 |
| Validator rewrite pass rate | 0.7575 |
| Leakage risk rate | 0.0000 |
| Response completion rate | 1.0000 |
| Prompt hash match rate | 1.0000 |

This suggests a better evaluation target: not "does the preprocessor make
Talkie pick more correct letters?", but "does the preprocessor make Talkie's
free-form answers more relevant, functionally correct, and era-neutral?"

## 5. Proposed Method

TalkieBridge uses a preprocessing pipeline:

```text
modern/anachronistic prompt
-> modern term detector
-> primitive mapper
-> primitive bottleneck selector
-> era-neutral rewriter
-> validator
-> Talkie prompt
-> downstream evaluation
```

### 5.1 Modern Term Detection

The detector identifies modern or anachronistic expressions in the input. It
uses predeclared dictionary resources and a lightweight token classifier built
from the training split or explicit dictionary files. Its purpose is not to
answer the task; it only decides which surface terms need rewriting.

### 5.2 Primitive Mapping

Detected terms are mapped to functional primitive descriptions. Examples:

| Modern Term | Era-Neutral Primitive Description |
|---|---|
| QR code | a square printed pattern that stores coded information and can be read by a camera |
| smartphone | a pocket-sized wireless device for communication and viewing stored messages |
| GPS | a system that determines location using signals from distant transmitting stations |
| cloud storage | keeping records on distant machines and retrieving them over a communication network |
| RAG | searching relevant records before composing a reply |

The primitive description should preserve the functional mechanism without
using the modern label itself.

### 5.3 Primitive Bottleneck Selector

The project should describe the implemented selector conservatively. It is a
small dependency-free primitive bottleneck mechanism, not a full generative DAE.
Its job is to select or recover primitive concepts that should appear in the
rewrite. The expected benefit is better semantic coverage than simple
dictionary replacement, while staying simple enough for a term project.

### 5.4 Era-Neutral Rewrite

The rewriter creates a prompt that removes modern labels and includes the
selected primitive descriptions. It should preserve the task, avoid copying
answer choices, and avoid revealing the gold answer.

### 5.5 Validator

The validator checks whether the rewrite is suitable for evaluation:

- forbidden modern terms should be removed,
- required primitives should be preserved,
- answer labels and direct answer leakage should not appear,
- choice-copying should remain low,
- rewrite length should remain controlled,
- prompt hash and response integrity should be stored.

If future runs show no difference between `proposed` and
`proposed_no_validator`, the report should state that the validator did not
demonstrate downstream impact in that setting.

## 6. Experimental Setup

### 6.1 Primary Evaluation: Open-Ended Response Quality

The primary evaluation should be open-ended rather than four-choice accuracy.
Each item asks Talkie for a short explanation:

```text
Answer in 1-2 sentences. Explain the practical mechanism.
```

For each item, collect Talkie responses under at least these conditions:

- `raw`
- `rule_only`
- `length_controlled`
- `proposed`

Then compare responses using a blind pairwise LLM Judge. The judge should not
see the condition names. Response order should be randomized. The raw judge
outputs, parsed labels, rubric version, random seed, and item IDs should be
saved.

Example pairwise judgment question:

```text
You are judging two answers to the same question for a 1930-era model setting.
Which answer is better overall?

Consider:
1. task relevance
2. functional correctness
3. era-neutrality
4. handling of modern/anachronistic terms
5. usefulness and clarity
6. whether the prompt rewrite appears to leak the answer

Return A, B, or Tie, then a short rationale.
```

Primary metrics:

- `proposed` vs `raw` pairwise win rate,
- `proposed` vs `rule_only` pairwise win rate,
- `proposed` vs `length_controlled` pairwise win rate,
- mean judge score difference by rubric dimension,
- tie rate,
- bootstrap confidence intervals,
- paired sign test or Wilcoxon signed-rank test when appropriate.

### 6.2 Secondary Evaluation: Rewrite Quality

Automatic component metrics should remain part of the evaluation:

- anachronism removal rate,
- required primitive recall,
- semantic preservation score,
- answer preservation score,
- leakage risk score,
- choice-copying score,
- validator pass rate,
- repair/fallback rate,
- prompt length statistics.

These metrics establish whether the preprocessing layer does what it claims,
even when downstream Talkie behavior is noisy.

### 6.3 Diagnostic Evaluation: Multiple-Choice Accuracy

The existing MCQ experiment should be reported as an initial diagnostic and
secondary result. It is useful because it is objective and already reproducible,
but it should not be the central success criterion.

The current diagnostic conclusion is:

> Under the existing 100-item MCQ setup, Talkie responses are near random-level
> accuracy, and the proposed preprocessor does not improve accuracy over raw
> prompts. This motivates shifting the primary evaluation to open-ended response
> quality.

## 7. Baselines and Ablations

The proposal should keep the five-condition design because it makes the claims
more defensible:

| Condition | Purpose |
|---|---|
| `raw` | Tests Talkie on the original modern wording |
| `rule_only` | Tests whether direct dictionary replacement is enough |
| `length_controlled` | Controls for prompt length and extra explanation |
| `proposed` | Tests the full primitive bottleneck rewrite pipeline |
| `proposed_no_validator` | Tests whether validation changes outcomes |

The most important comparison for the revised proposal is not only `proposed`
vs `raw`. It is also `proposed` vs `length_controlled`. If
`length_controlled` performs as well as or better than `proposed`, then the
honest conclusion is that prompt simplification or length control may matter
more than the primitive selector.

## 8. Expected Outcomes

The expected outcome is not guaranteed accuracy improvement. A defensible
success condition is:

> The proposed preprocessing pipeline improves Talkie's open-ended response
> quality on modern/anachronistic prompts, while preserving task meaning,
> removing modern terms, and avoiding answer leakage.

Possible outcomes and interpretations:

| Outcome | Interpretation |
|---|---|
| `proposed` beats `raw` and `rule_only` in blind pairwise judging | Primitive rewriting likely improves response quality |
| `length_controlled` matches or beats `proposed` | Simpler prompt control may be the main driver |
| Component metrics are strong but response quality does not improve | The rewrite works locally, but Talkie may not use the rewritten information effectively |
| MCQ remains near random but open-ended quality improves | MCQ letter selection is not a reliable primary measure for this task |
| Judge finds high leakage risk | Rewrite design must be revised before making downstream claims |

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| LLM Judge bias | Blind condition labels, randomize A/B order, use fixed rubric, store raw judge outputs |
| Rewrite leaks the answer | Add leakage rubric, choice-copying metrics, answer-overlap diagnostics, and qualitative audits |
| MCQ results contradict response-quality results | Present MCQ as diagnostic evidence and explain the task mismatch |
| Unofficial Talkie API instability | Prefer manual CSV reproducibility path; cache raw responses and prompt hashes |
| Validator has no effect | Report `proposed_no_validator` ablation honestly |
| Primitive rewrite over-explains the task | Compare against `length_controlled` and measure leakage risk |
| Dataset annotations influence test rewrites | Use predeclared dictionaries or train-split-only resources; document resource source |
| Small sample size | Use paired comparisons, bootstrap intervals, and qualitative examples rather than overclaiming p-values |

## 10. Valid Claims and Claims to Avoid

### Valid Claims

- TalkieBridge evaluates a preprocessing/prompt rewriting layer in front of a
  fixed Talkie-1930 evaluator.
- The current pipeline can generate five prompt conditions and collect/evaluate
  Talkie responses end to end.
- The current MCQ diagnostic run does not show accuracy improvement for
  `proposed`.
- Component metrics show that the current rewriter can remove annotated modern
  terms and preserve many required primitives.
- The revised primary research question is whether era-neutral rewriting
  improves open-ended downstream response quality.

### Claims to Avoid

- "We improved Talkie-1930."
- "The proposed method improves multiple-choice accuracy."
- "The project implements a full Denoising Text Autoencoder generator."
- "The validator improves downstream performance" unless a future ablation
  shows that it does.
- "Near-random MCQ accuracy proves the method fails completely."
- "LLM Judge results are objective ground truth."

## 11. Reproducibility Plan

The project should save every artifact needed to audit the run:

- dataset JSONL,
- dictionary files,
- condition name,
- original prompt,
- rewritten prompt,
- prompt hash,
- Talkie raw response,
- parsed response,
- provider name,
- response timestamp or cache record when available,
- metric outputs,
- judge prompt,
- judge model/version,
- judge raw output,
- parsed judge result,
- randomization seed.

Current implemented artifacts include:

| Path | Purpose |
|---|---|
| `data/generated_questions.jsonl` | Main 100-item MCQ dataset |
| `data/modern_terms_dictionary.json` | Modern term dictionary |
| `data/primitive_dictionary.json` | Primitive dictionary |
| `input_data/manual_talkie_input_sheet.csv` | Manual response collection sheet |
| `results/prepared_prompts.csv` | Generated prompts for all selected conditions |
| `results/response_integrity.csv` | Prompt hash and response completion checks |
| `results/final_metrics.csv` | All-row condition metrics |
| `results/key_comparisons.csv` | Paired condition comparisons |
| `results/paired_tests.csv` | Paired tests against raw |
| `results/component_metrics.csv` | Detector, rewriter, validator metrics |
| `results/per_item_results.csv` | Item-level predictions and raw responses |
| `results/report.md` | Generated diagnostic report |

For final reporting, the open-ended judge outputs should be stored in a new
results file such as:

```text
results/open_ended_judge_pairs.csv
results/open_ended_judge_scores.csv
results/open_ended_response_quality.md
```

## 12. Conclusion

The most defensible version of TalkieBridge is an evaluation of prompt
rewriting, not an attempt to improve Talkie itself. The current MCQ experiment
is a useful negative/diagnostic result: the pipeline runs, but the proposed
condition does not improve four-choice accuracy. The project should therefore
pivot its primary evaluation to blind, rubric-based open-ended response quality
while keeping MCQ accuracy as a secondary diagnostic.

This framing is honest, measurable, and aligned with the actual system:
TalkieBridge is a small era-neutral preprocessing layer that tests whether
modern prompts become more usable for a fixed 1930-era model after functional
rewrite.
