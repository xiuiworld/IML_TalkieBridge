# Dataset Quality Report

## Dataset

| Metric | Value |
| --- | --- |
| n_items | 100 |
| human_validated_count | 0 |

## Length

| Metric | Value |
| --- | --- |
| mean_original_question_tokens | 13.0200 |
| mean_choice_tokens | 9.0350 |

## Domain Count

| Metric | Value |
| --- | --- |
| AI_Computing | 20 |
| Communication_Media | 15 |
| Daily_Tech_Society | 15 |
| Environment_Energy | 15 |
| Medicine_Biology | 20 |
| Transportation_Engineering | 15 |

## Split Count

| Metric | Value |
| --- | --- |
| dev | 10 |
| test | 15 |
| train | 75 |

## Answer Distribution

| Metric | Value |
| --- | --- |
| A | 25 |
| B | 25 |
| C | 25 |
| D | 25 |

## Rewrite Length

| Metric | Value |
| --- | --- |
| rule_only_mean_rewritten_tokens | 34.6800 |
| length_controlled_mean_rewritten_tokens | 33.6800 |
| proposed_mean_rewritten_tokens | 33.6800 |
| proposed_no_validator_mean_rewritten_tokens | 33.6800 |

## Leakage

| Metric | Value |
| --- | --- |
| rule_only_leakage_risk_count | 0 |
| length_controlled_leakage_risk_count | 0 |
| proposed_leakage_risk_count | 0 |
| proposed_no_validator_leakage_risk_count | 0 |

## Choice Copying

| Metric | Value |
| --- | --- |
| rule_only_mean_choice_copying_score | 0.0000 |
| rule_only_max_choice_copying_score | 0.0000 |
| length_controlled_mean_choice_copying_score | 0.0000 |
| length_controlled_max_choice_copying_score | 0.0000 |
| proposed_mean_choice_copying_score | 0.0000 |
| proposed_max_choice_copying_score | 0.0000 |
| proposed_no_validator_mean_choice_copying_score | 0.0000 |
| proposed_no_validator_max_choice_copying_score | 0.0000 |

## Answer Hint

| Metric | Value |
| --- | --- |
| rule_only_mean_choice_keyword_overlap_score | 0.1647 |
| rule_only_max_choice_keyword_overlap_score | 0.6667 |
| length_controlled_mean_choice_keyword_overlap_score | 0.0941 |
| length_controlled_max_choice_keyword_overlap_score | 0.5556 |
| proposed_mean_choice_keyword_overlap_score | 0.1687 |
| proposed_max_choice_keyword_overlap_score | 0.6667 |
| proposed_no_validator_mean_choice_keyword_overlap_score | 0.1687 |
| proposed_no_validator_max_choice_keyword_overlap_score | 0.6667 |

## Length Control

| Metric | Value |
| --- | --- |
| mean_delta_vs_proposed_tokens | 0.0000 |
| max_abs_delta_vs_proposed_tokens | 0 |
| mean_ratio_to_proposed | 1.0000 |

## Proposed Difference

| Metric | Value |
| --- | --- |
| rewrite_differs_from_rule_only_rate | 1.0000 |
| primitive_set_differs_from_rule_only_rate | 0.0100 |
| mean_autoencoder_added_primitives | 0.0100 |
