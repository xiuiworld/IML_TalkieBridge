# Open-Ended Response Quality Report

This report summarizes blind pairwise LLM Judge results for TalkieBridge open-ended responses.
Condition labels are hidden in the judge prompt and restored only for analysis.

## Summary

- Judge coverage: 400 pairs, parse rate 100.0%, prompt hash match rate 100.0%, blank outputs 0.
- proposed_vs_raw: `proposed` beat `raw` (79 wins, 18 losses, 3 ties over 100 pairs; win rate excluding ties 81.4%, all-pair win rate 79.0%).
- proposed_vs_rule_only: `proposed` did not beat `rule_only` (45 wins, 51 losses, 4 ties over 100 pairs; win rate excluding ties 46.9%, all-pair win rate 45.0%).
- proposed_vs_length_controlled: `proposed` beat `length_controlled` (88 wins, 11 losses, 1 tie over 100 pairs; win rate excluding ties 88.9%, all-pair win rate 88.0%).
- proposed_vs_proposed_no_validator: `proposed` tied `proposed_no_validator` (0 wins, 0 losses, 100 ties over 100 pairs; win rate excluding ties not defined because all pairs tied, all-pair win rate 0.0%).

## Pairwise Metrics

| comparison | n_pairs | condition_wins | baseline_wins | ties | condition_win_rate_all | condition_win_rate_excluding_ties | tie_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_vs_length_controlled | 100 | 88 | 11 | 1 | 0.8800 | 0.8889 | 0.0100 |
| proposed_vs_proposed_no_validator | 100 | 0 | 0 | 100 | 0.0000 | 0.0000 | 1.0000 |
| proposed_vs_raw | 100 | 79 | 18 | 3 | 0.7900 | 0.8144 | 0.0300 |
| proposed_vs_rule_only | 100 | 45 | 51 | 4 | 0.4500 | 0.4688 | 0.0400 |

## Judge Integrity

| metric | value |
| --- | --- |
| judge_prompt_hash_match_rate | 1.0000 |
| blank_judge_output_count | 0 |
| judge_parse_rate | 1.0000 |
| n_pairs | 400 |

## Claim Boundary

- These results support claims about the preprocessing layer's effect on Talkie open-ended response quality.
- They do not support a claim that Talkie itself was modified or improved.
- They do not overturn the separate MCQ diagnostic result, where proposed preprocessing did not improve four-choice accuracy.
- The `proposed_no_validator` ablation should be used to state whether the validator had a measurable downstream effect.
