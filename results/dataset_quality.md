# Dataset Quality Report

## Dataset

| Metric | Value |
| --- | --- |
| n_items | 100 |
| human_validated_count | 0 |

## Length

| Metric | Value |
| --- | --- |
| mean_original_question_tokens | 17.6500 |
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
| rule_only_mean_rewritten_tokens | 39.3200 |
| length_controlled_mean_rewritten_tokens | 43.8900 |
| proposed_mean_rewritten_tokens | 42.3200 |
| proposed_no_validator_mean_rewritten_tokens | 42.3200 |

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
| rule_only_mean_choice_keyword_overlap_score | 0.1567 |
| rule_only_max_choice_keyword_overlap_score | 0.4286 |
| length_controlled_mean_choice_keyword_overlap_score | 0.0880 |
| length_controlled_max_choice_keyword_overlap_score | 0.3333 |
| proposed_mean_choice_keyword_overlap_score | 0.1607 |
| proposed_max_choice_keyword_overlap_score | 0.4286 |
| proposed_no_validator_mean_choice_keyword_overlap_score | 0.1607 |
| proposed_no_validator_max_choice_keyword_overlap_score | 0.4286 |

## Length Control

| Metric | Value |
| --- | --- |
| mean_delta_vs_proposed_tokens | 1.5700 |
| max_abs_delta_vs_proposed_tokens | 5 |
| mean_ratio_to_proposed | 1.0387 |

## Proposed Difference

| Metric | Value |
| --- | --- |
| rewrite_differs_from_rule_only_rate | 1.0000 |
| primitive_set_differs_from_rule_only_rate | 0.0200 |
| mean_autoencoder_added_primitives | 0.0200 |
