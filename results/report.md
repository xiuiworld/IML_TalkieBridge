# TalkieBridge Experiment Report

This run evaluates the Era-Neutral Prompt Generator in front of a fixed Talkie-1930 downstream evaluator.
Talkie is not trained or modified by this pipeline.

## Run Metadata

- Provider: `unofficial_talkie_api`
- Items: `100`
- Generation resource source: `predeclared_dictionary`
- Concept dictionary path: `data/modern_terms_dictionary.json`
- Primitive dictionary path: `data/primitive_dictionary.json`
- Provider note: this uses the unofficial Talkie web SSE endpoint and should be treated as a convenience path, not the primary reproducibility source.

## Warnings

- Not all dataset items are human_validated=true.
- Predeclared dictionaries are trusted prior resources; verify they were authored independently of dev/test item annotations.

## Primary Test Split Metrics

| condition | accuracy | macro_f1 | invalid_rate | n_items |
| --- | --- | --- | --- | --- |
| raw | 0.2667 | 0.1667 | 0.4667 | 15 |
| rule_only | 0.2667 | 0.2000 | 0.2000 | 15 |
| length_controlled | 0.3333 | 0.2583 | 0.2000 | 15 |
| proposed | 0.2667 | 0.2154 | 0.3333 | 15 |
| proposed_no_validator | 0.2667 | 0.2154 | 0.3333 | 15 |

## All Selected Rows Metrics

| condition | accuracy | macro_f1 | invalid_rate | n_items |
| --- | --- | --- | --- | --- |
| raw | 0.2100 | 0.1163 | 0.2000 | 100 |
| rule_only | 0.1900 | 0.1348 | 0.1800 | 100 |
| length_controlled | 0.2400 | 0.1257 | 0.1200 | 100 |
| proposed | 0.1900 | 0.1256 | 0.2200 | 100 |
| proposed_no_validator | 0.1900 | 0.1256 | 0.2200 | 100 |

## Primary Test Split Key Comparisons

| comparison | accuracy_delta | bootstrap_ci_low | bootstrap_ci_high | mcnemar_b | mcnemar_c | exact_mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- |
| rule_only_vs_raw | 0.0000 | -0.2000 | 0.2000 | 1 | 1 | 1.0000 |
| length_controlled_vs_raw | 0.0667 | 0.0000 | 0.2000 | 1 | 0 | 1.0000 |
| proposed_vs_raw | 0.0000 | -0.2000 | 0.2000 | 1 | 1 | 1.0000 |
| proposed_no_validator_vs_raw | 0.0000 | -0.2000 | 0.2000 | 1 | 1 | 1.0000 |
| proposed_vs_rule_only | 0.0000 | -0.2000 | 0.2000 | 1 | 1 | 1.0000 |
| proposed_vs_length_controlled | -0.0667 | -0.2667 | 0.1333 | 1 | 2 | 1.0000 |
| proposed_vs_proposed_no_validator | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1.0000 |

## Key Comparisons

| comparison | accuracy_delta | bootstrap_ci_low | bootstrap_ci_high | mcnemar_b | mcnemar_c | exact_mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- |
| rule_only_vs_raw | -0.0200 | -0.1000 | 0.0400 | 6 | 8 | 0.7905 |
| length_controlled_vs_raw | 0.0300 | -0.0200 | 0.0800 | 5 | 2 | 0.4531 |
| proposed_vs_raw | -0.0200 | -0.0900 | 0.0400 | 5 | 7 | 0.7744 |
| proposed_no_validator_vs_raw | -0.0200 | -0.0900 | 0.0400 | 5 | 7 | 0.7744 |
| proposed_vs_rule_only | 0.0000 | -0.0600 | 0.0600 | 5 | 5 | 1.0000 |
| proposed_vs_length_controlled | -0.0500 | -0.1200 | 0.0200 | 4 | 9 | 0.2668 |
| proposed_vs_proposed_no_validator | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1.0000 |

## Paired Tests Against Raw

| comparison | accuracy_delta | bootstrap_ci_low | bootstrap_ci_high | mcnemar_b | mcnemar_c | exact_mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- |
| rule_only_vs_raw | -0.0200 | -0.1000 | 0.0400 | 6 | 8 | 0.7905 |
| length_controlled_vs_raw | 0.0300 | -0.0200 | 0.0800 | 5 | 2 | 0.4531 |
| proposed_vs_raw | -0.0200 | -0.0900 | 0.0400 | 5 | 7 | 0.7744 |
| proposed_no_validator_vs_raw | -0.0200 | -0.0900 | 0.0400 | 5 | 7 | 0.7744 |

## Test Split Paired Tests Against Raw

| comparison | accuracy_delta | bootstrap_ci_low | bootstrap_ci_high | mcnemar_b | mcnemar_c | exact_mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- |
| rule_only_vs_raw | 0.0000 | -0.2000 | 0.2000 | 1 | 1 | 1.0000 |
| length_controlled_vs_raw | 0.0667 | 0.0000 | 0.2000 | 1 | 0 | 1.0000 |
| proposed_vs_raw | 0.0000 | -0.2000 | 0.2000 | 1 | 1 | 1.0000 |
| proposed_no_validator_vs_raw | 0.0000 | -0.2000 | 0.2000 | 1 | 1 | 1.0000 |

## Component Metrics

| component | metric | value |
| --- | --- | --- |
| detector | precision | 0.9300 |
| detector | recall | 0.6739 |
| detector | f1 | 0.7815 |
| rewriter | anachronism_removal_rate | 1.0000 |
| rewriter | required_primitive_recall | 0.7670 |
| validator | rewrite_pass_rate | 0.7575 |
| validator | leakage_risk_rate | 0.0000 |
| validator | mean_choice_copying_score | 0.0000 |
| validator | max_choice_copying_score | 0.0000 |
| validator | mean_choice_keyword_overlap_score | 0.1415 |
| validator | max_choice_keyword_overlap_score | 0.4286 |
| repair | mean_repair_attempts | 0.0000 |

## Dataset Quality

| section | metric | value |
| --- | --- | --- |
| dataset | n_items | 100 |
| dataset | human_validated_count | 0 |
| length | mean_original_question_tokens | 17.6500 |
| length | mean_choice_tokens | 9.0350 |
| domain_count | AI_Computing | 20 |
| domain_count | Communication_Media | 15 |
| domain_count | Daily_Tech_Society | 15 |
| domain_count | Environment_Energy | 15 |
| domain_count | Medicine_Biology | 20 |
| domain_count | Transportation_Engineering | 15 |
| split_count | dev | 10 |
| split_count | test | 15 |
| split_count | train | 75 |
| answer_distribution | A | 25 |
| answer_distribution | B | 25 |
| answer_distribution | C | 25 |
| answer_distribution | D | 25 |
| rewrite_length | rule_only_mean_rewritten_tokens | 39.3200 |
| leakage | rule_only_leakage_risk_count | 0 |
| choice_copying | rule_only_mean_choice_copying_score | 0.0000 |
| choice_copying | rule_only_max_choice_copying_score | 0.0000 |
| answer_hint | rule_only_mean_choice_keyword_overlap_score | 0.1567 |
| answer_hint | rule_only_max_choice_keyword_overlap_score | 0.4286 |
| rewrite_length | length_controlled_mean_rewritten_tokens | 43.8900 |
| leakage | length_controlled_leakage_risk_count | 0 |
| choice_copying | length_controlled_mean_choice_copying_score | 0.0000 |
| choice_copying | length_controlled_max_choice_copying_score | 0.0000 |
| answer_hint | length_controlled_mean_choice_keyword_overlap_score | 0.0880 |
| answer_hint | length_controlled_max_choice_keyword_overlap_score | 0.3333 |
| rewrite_length | proposed_mean_rewritten_tokens | 42.3200 |
| leakage | proposed_leakage_risk_count | 0 |
| choice_copying | proposed_mean_choice_copying_score | 0.0000 |
| choice_copying | proposed_max_choice_copying_score | 0.0000 |
| answer_hint | proposed_mean_choice_keyword_overlap_score | 0.1607 |
| answer_hint | proposed_max_choice_keyword_overlap_score | 0.4286 |
| rewrite_length | proposed_no_validator_mean_rewritten_tokens | 42.3200 |
| leakage | proposed_no_validator_leakage_risk_count | 0 |
| choice_copying | proposed_no_validator_mean_choice_copying_score | 0.0000 |
| choice_copying | proposed_no_validator_max_choice_copying_score | 0.0000 |
| answer_hint | proposed_no_validator_mean_choice_keyword_overlap_score | 0.1607 |
| answer_hint | proposed_no_validator_max_choice_keyword_overlap_score | 0.4286 |
| length_control | mean_delta_vs_proposed_tokens | 1.5700 |
| length_control | max_abs_delta_vs_proposed_tokens | 5 |
| length_control | mean_ratio_to_proposed | 1.0387 |
| proposed_difference | rewrite_differs_from_rule_only_rate | 1.0000 |
| proposed_difference | primitive_set_differs_from_rule_only_rate | 0.0200 |
| proposed_difference | mean_autoencoder_added_primitives | 0.0200 |

## Domain Metrics

| domain | condition | accuracy | macro_f1 | invalid_rate | n_items |
| --- | --- | --- | --- | --- | --- |
| AI_Computing | length_controlled | 0.2500 | 0.1087 | 0.1000 | 20 |
| AI_Computing | proposed | 0.2500 | 0.1786 | 0.1500 | 20 |
| AI_Computing | proposed_no_validator | 0.2500 | 0.1786 | 0.1500 | 20 |
| AI_Computing | raw | 0.2000 | 0.1053 | 0.3000 | 20 |
| AI_Computing | rule_only | 0.2000 | 0.1548 | 0.1500 | 20 |
| Communication_Media | length_controlled | 0.2000 | 0.1000 | 0.2667 | 15 |
| Communication_Media | proposed | 0.2667 | 0.1333 | 0.2667 | 15 |
| Communication_Media | proposed_no_validator | 0.2667 | 0.1333 | 0.2667 | 15 |
| Communication_Media | raw | 0.1333 | 0.0625 | 0.2000 | 15 |
| Communication_Media | rule_only | 0.1333 | 0.0667 | 0.2667 | 15 |
| Daily_Tech_Society | length_controlled | 0.2000 | 0.0938 | 0.1333 | 15 |
| Daily_Tech_Society | proposed | 0.1333 | 0.0769 | 0.4000 | 15 |
| Daily_Tech_Society | proposed_no_validator | 0.1333 | 0.0769 | 0.4000 | 15 |
| Daily_Tech_Society | raw | 0.3333 | 0.2426 | 0.0667 | 15 |
| Daily_Tech_Society | rule_only | 0.1333 | 0.0667 | 0.2667 | 15 |
| Environment_Energy | length_controlled | 0.2667 | 0.1250 | 0.0667 | 15 |
| Environment_Energy | proposed | 0.2000 | 0.1833 | 0.1333 | 15 |
| Environment_Energy | proposed_no_validator | 0.2000 | 0.1833 | 0.1333 | 15 |
| Environment_Energy | raw | 0.2000 | 0.1154 | 0.4000 | 15 |
| Environment_Energy | rule_only | 0.2667 | 0.2000 | 0.1333 | 15 |
| Medicine_Biology | length_controlled | 0.2500 | 0.1087 | 0.1000 | 20 |
| Medicine_Biology | proposed | 0.2000 | 0.0952 | 0.2000 | 20 |
| Medicine_Biology | proposed_no_validator | 0.2000 | 0.0952 | 0.2000 | 20 |
| Medicine_Biology | raw | 0.2500 | 0.1042 | 0.0000 | 20 |
| Medicine_Biology | rule_only | 0.2500 | 0.1833 | 0.2000 | 20 |
| Transportation_Engineering | length_controlled | 0.2667 | 0.2000 | 0.0667 | 15 |
| Transportation_Engineering | proposed | 0.0667 | 0.0417 | 0.2000 | 15 |
| Transportation_Engineering | proposed_no_validator | 0.0667 | 0.0417 | 0.2000 | 15 |
| Transportation_Engineering | raw | 0.1333 | 0.0714 | 0.2667 | 15 |
| Transportation_Engineering | rule_only | 0.1333 | 0.0667 | 0.0667 | 15 |

## Response Integrity

| metric | value |
| --- | --- |
| prompt_hash_match_rate | 1.0000 |
| missing_response_prompt_hash_count | 0 |
| blank_response_count | 0 |
| response_completion_rate | 1.0000 |
| n_rows | 500 |
