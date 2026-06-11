# IML_TalkieBridge 진행 정리

작성일: 2026-06-11  
최신 공유 커밋: `e6bd0f3 Publish TalkieBridge experiment artifacts`

## 한 줄 요약

100문항 전체 실험 파이프라인, Talkie 응답 수집, 파서 개선, 결과 산출까지는 완료됐다. 다만 현재 4지선다 accuracy 기준으로는 `proposed`가 성능 개선을 보이지 못했으므로, 다음 단계에서는 평가 목표를 "Talkie MCQ 정답률"에서 "전처리기가 downstream 응답 품질을 개선하는가"로 재설계하는 방안을 검토해야 한다.

## 프로젝트 목적

이 프로젝트는 Talkie-1930 모델 자체를 수정하는 것이 아니다. Talkie 앞단에 전처리기를 붙여 modern/anachronistic 표현이 포함된 입력을 1930-era-neutral primitive 표현으로 rewrite하고, 이 전처리가 downstream Talkie 응답 품질 또는 성능을 개선하는지 평가하는 것이 목적이다.

주의할 점:

- `src/talkie_bridge/`가 실제 실행 패키지다.
- `src/talkie/` 원본 Talkie 코드는 repo에 포함하지 않는다.
- `prototypes/`, `docs/`, `Teammate/`는 local/reference 성격이며 최종 실행 경로가 아니다.
- 현재 "Denoising Text Autoencoder"는 full natural-language generator가 아니라 dependency-free primitive bottleneck selector에 가깝다.

## 현재 완료된 작업

- 100문항 dataset 구축 완료
- `data/modern_terms_dictionary.json`, `data/primitive_dictionary.json` 구축 완료
- 5개 조건 prompt 생성 완료
  - `raw`
  - `rule_only`
  - `length_controlled`
  - `proposed`
  - `proposed_no_validator`
- unofficial Talkie API로 500개 응답 수집 완료
- `prompt_hash` 기반 integrity 검증 완료
- Talkie가 선택지 글자가 아니라 선택지 문장 일부로 답하는 문제를 처리하도록 parser 개선
- 전체 결과 파일 산출 완료
  - `results/response_integrity.csv`
  - `results/final_metrics.csv`
  - `results/key_comparisons.csv`
  - `results/paired_tests.csv`
  - `results/report.md`
  - `results/per_item_results.csv`

## 코드 변경 요약

### API 수집 안정화

`src/talkie_bridge/clients.py`와 `src/talkie_bridge/cli.py`를 수정해 unofficial API 호출에 rate limit 대응을 추가했다.

추가된 기능:

- `429 Too Many Requests` 및 일시적 `5xx` 응답에 대해 retry/backoff 적용
- `Retry-After` header가 있으면 해당 대기 시간 사용
- 요청 간 지연 옵션 추가
- 이미 수집된 응답은 `cache/talkie_unofficial_api_cache.jsonl`에서 재사용

추가된 CLI 옵션:

```text
--request-delay
--max-retries
--retry-base-delay
--retry-max-delay
```

### 응답 parser 개선

기존 parser는 Talkie 응답이 `A`, `B`, `C`, `D` 같은 글자일 때만 안정적으로 처리했다. 하지만 실제 Talkie는 선택지 문장 일부를 답하는 경우가 많았다.

예:

```text
Expected parsed answer: C
Talkie raw response: "It reduces error by using independent observations together"
```

개선 후에는 응답 텍스트를 네 선택지 문장과 비교해 명확히 매칭되는 경우 A/B/C/D로 복원한다. 단, gold answer는 사용하지 않는다. 즉 정답을 보고 맞추는 것이 아니라, Talkie가 어떤 선택지 내용을 말했는지 복원하는 방식이다.

보수적 처리:

- 문장 속 관사 `a`를 선택지 `A`로 오인하지 않도록 수정
- 짧고 애매한 키워드 조합은 여전히 `INVALID`로 유지
- 조건별 비교에 동일한 parser를 적용

### 테스트

최신 테스트 결과:

```text
python -m pytest
15 passed
```

관련 테스트:

- `tests/test_clients.py`
- `tests/test_prompting_metrics.py`

## 응답 수집 무결성

`results/response_integrity.csv` 기준:

```text
prompt_hash_match_rate = 1.0
missing_response_prompt_hash_count = 0
blank_response_count = 0
response_completion_rate = 1.0
n_rows = 500
```

해석:

- 100문항 x 5조건 = 500개 응답이 모두 수집됐다.
- prompt hash mismatch가 없다.
- blank response가 없다.
- 따라서 현재 문제는 데이터 수집 실패가 아니라 방법 성능 문제다.

## 최종 4지선다 평가 결과

`results/final_metrics.csv` 기준:

| Condition | Accuracy | Invalid rate | n |
| --- | ---: | ---: | ---: |
| raw | 0.21 | 0.20 | 100 |
| rule_only | 0.19 | 0.18 | 100 |
| length_controlled | 0.24 | 0.12 | 100 |
| proposed | 0.19 | 0.22 | 100 |
| proposed_no_validator | 0.19 | 0.22 | 100 |

핵심:

- 최고 조건은 `length_controlled`다.
- `proposed`는 `raw`보다 낮다.
- 4지선다 random baseline이 약 25%라는 점을 고려하면, 현재 accuracy는 전반적으로 near-chance 수준이다.

## Paired 비교

`results/key_comparisons.csv` 기준:

| Comparison | Accuracy delta | McNemar p |
| --- | ---: | ---: |
| rule_only vs raw | -0.02 | 0.7905 |
| length_controlled vs raw | +0.03 | 0.4531 |
| proposed vs raw | -0.02 | 0.7744 |
| proposed vs rule_only | 0.00 | 1.0000 |
| proposed vs length_controlled | -0.05 | 0.2668 |
| proposed vs proposed_no_validator | 0.00 | 1.0000 |

해석:

- `length_controlled`는 raw보다 net +3문항이지만 통계적으로 강한 증거는 아니다.
- `proposed`는 raw보다 net -2문항이다.
- `proposed`와 `proposed_no_validator`가 완전히 같아 validator가 실질적인 repair/fallback 효과를 만들지 못했다.

## 문항 단위 관찰

현재 per-item 결과를 요약하면:

```text
raw correct = 21
length_controlled correct = 24
proposed correct = 19

raw wrong, length_controlled correct = 5
raw correct, length_controlled wrong = 2
=> length_controlled는 raw보다 net +3

raw wrong, proposed correct = 5
raw correct, proposed wrong = 7
=> proposed는 raw보다 net -2

length_controlled correct, proposed wrong = 9
length_controlled wrong, proposed correct = 4
=> proposed는 length_controlled보다 net -5
```

추정 가능한 상한:

```text
raw or length_controlled 중 하나라도 맞춘 문항 = 26
모든 조건 중 하나라도 맞춘 문항 = 30
```

따라서 현재 4지선다 setup에서 단순 fallback/selector만 잘 만들어도 현실적 목표는 24-26%, 낙관적 목표는 28-30% 수준이다.

## Component metric

`results/component_metrics.csv` 기준:

| Component | Metric | Value |
| --- | --- | ---: |
| detector | precision | 0.93 |
| detector | recall | 0.674 |
| detector | f1 | 0.782 |
| rewriter | anachronism_removal_rate | 1.0 |
| rewriter | required_primitive_recall | 0.767 |
| validator | rewrite_pass_rate | 0.758 |
| validator | leakage_risk_rate | 0.0 |
| repair | mean_repair_attempts | 0.0 |

해석:

- 전처리 component 자체는 anachronism removal 측면에서 좋은 수치를 보인다.
- 그러나 이 component-level 품질이 downstream Talkie MCQ accuracy 개선으로 이어지지는 않았다.
- `mean_repair_attempts = 0.0`이라 validator가 실제 repair/fallback 역할을 수행하지 못한 것으로 보인다.

## 현재 결론

정직한 결론:

```text
The proposed primitive bottleneck rewrite did not improve Talkie-1930 multiple-choice accuracy under the current setup.
The best-performing variant was length-controlled rewriting, but its accuracy remained near chance.
```

현재 결과로 피해야 할 claim:

```text
Our proposed preprocessor improves Talkie multiple-choice accuracy.
Our method significantly improves Talkie performance.
```

현재 결과로 가능한 claim:

```text
We built an end-to-end preprocessing and evaluation pipeline.
The preprocessor reliably removes modern/anachronistic terms.
The current proposed primitive bottleneck did not translate into MCQ accuracy gains.
Length-controlled rewriting performed best among tested conditions, but gains were small and not statistically strong.
```

## 문제 진단

현재 문제는 데이터 수집이나 실행 실패가 아니라 평가 설계와 방법 성능의 문제다.

가능한 원인:

1. 4지선다 task가 Talkie-1930에게 너무 어렵다.
   - 현대 개념을 primitive로 바꿔도 선택지 간 의미 구분이 어렵다.
   - Talkie가 letter-choice 형식을 잘 따르지 않는다.

2. `proposed` rewrite가 필요한 정보를 보존하지 못하거나, 문장을 더 어렵게 만들 수 있다.
   - `length_controlled`보다 correct가 낮다.
   - invalid rate도 `proposed`가 `length_controlled`보다 높다.

3. validator가 실질적 fallback 역할을 하지 않는다.
   - `proposed`와 `proposed_no_validator` 결과가 같다.
   - `repair_attempts`가 0이다.

## 다음 설계 아이디어

4지선다 정확도만으로 프로젝트를 끌고 가기 어렵다면, 평가 목표를 다음처럼 바꾸는 것을 고려할 수 있다.

기존 약한 claim:

```text
The preprocessor improves Talkie multiple-choice accuracy.
```

대체 가능한 claim:

```text
The preprocessor improves Talkie's downstream response quality on modern/anachronistic prompts by translating them into era-neutral primitive descriptions.
```

## Open-ended response quality 평가 제안

4지선다 대신 open-ended task를 사용한다.

예시:

```text
Raw:
Why might a QR code be selected for opening a stored address from a printed sign?

Proposed:
Why might a square printed pattern that stores coded information and can be read by a camera be selected for opening a stored address from a printed sign?

Talkie prompt:
Answer in 1-2 sentences. Explain the practical mechanism.
```

그 후 raw prompt에 대한 Talkie 응답과 preprocessed prompt에 대한 Talkie 응답을 LLM Judge가 blind pairwise로 비교한다.

가능한 judge rubric:

1. Task relevance
2. Functional correctness
3. Era-neutrality
4. Anachronism handling
5. Answer usefulness
6. Answer leakage risk

가능한 metric:

- `proposed` response vs `raw` response pairwise win rate
- `proposed` response vs `rule_only` response pairwise win rate
- mean judge score improvement
- anachronism removal score
- semantic preservation score
- answer preservation score
- leakage risk score

방어 장치:

- condition 이름을 숨기고 A/B 순서 randomization
- judge prompt와 raw judge output 저장
- temperature 고정
- 가능하면 2개 이상의 judge model 또는 repeated judging 사용
- LLM Judge 결과를 Talkie accuracy evidence처럼 과장하지 않기

## 추천 다음 작업

### 단기

`proposed`를 `length_controlled + conservative fallback` 구조로 재정의한다.

개념:

```text
proposed = length_controlled rewrite
         + primitive phrase는 필요할 때만 추가
         + validation 실패/길이 과다/선택지 힌트 증가 시 length_controlled로 fallback
```

목표:

- 최소한 `proposed`가 `length_controlled`보다 나빠지지 않게 만들기
- 현실적 목표: 19% -> 24% 근처
- 낙관적 목표: 26-30%

### 중기

open-ended Talkie response quality 평가를 pilot으로 20문항만 먼저 실행한다.

pilot에서 확인할 것:

- proposed vs raw LLM Judge win rate가 60% 이상인지
- answer leakage 문제가 없는지
- judge rubric이 일관적으로 작동하는지
- Talkie 응답이 4지선다보다 더 평가 가능한 형태로 나오는지

### 장기

보고서 구조를 다음처럼 조정한다.

Primary evaluation:

- LLM-judged Talkie response quality on open-ended tasks

Secondary evaluation:

- Automatic rewrite quality metrics
- MCQ accuracy as diagnostic or secondary result

Negative/limitation result:

- Current MCQ accuracy did not show robust downstream improvement.

## 공유된 파일

GitHub에 공유된 주요 파일:

- `src/talkie_bridge/cli.py`
- `src/talkie_bridge/clients.py`
- `src/talkie_bridge/prompting.py`
- `tests/test_clients.py`
- `tests/test_prompting_metrics.py`
- `data/generated_questions.jsonl`
- `data/modern_terms_dictionary.json`
- `data/primitive_dictionary.json`
- `input_data/manual_talkie_input_sheet.csv`
- `results/final_metrics.csv`
- `results/key_comparisons.csv`
- `results/paired_tests.csv`
- `results/per_item_results.csv`
- `results/report.md`
- `cache/talkie_unofficial_api_cache.jsonl`

공유하지 않은 것:

- `Teammate/`: 팀원 smoke/intermediate folder
- `docs/`: local reference 자료
- `prototypes/`: prototype reference

## 발표/보고서에 쓸 수 있는 요약 문장

```text
We implemented an end-to-end TalkieBridge pipeline that rewrites modern/anachronistic prompts into era-neutral primitive descriptions and evaluates Talkie-1930 under multiple preprocessing conditions.
```

```text
The pipeline successfully collected 500 Talkie responses across 100 items and 5 conditions with complete prompt-hash integrity.
```

```text
Although the preprocessor achieved strong anachronism removal, the current proposed primitive bottleneck did not improve multiple-choice accuracy over the raw baseline.
```

```text
This suggests that downstream evaluation should focus not only on forced-choice accuracy but also on response quality, functional explanation, and era-neutral interpretability.
```

