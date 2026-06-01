# 기계학습개론 텀프로젝트 제안서

## 프로젝트 제목

**Anachronism-Aware Prompt Rewriting for Vintage LLM Evaluation**  
**Talkie-1930을 위한 Primitive-Bottleneck 기반 시대중립 프롬프트 전처리기**

## 1. 핵심 연구 질문

현대 개념이 포함된 질문을 1930년 지식 기반 LLM에 그대로 입력하면, 모델이 문제를 틀린 이유가 실제 추론 능력 부족인지, 현대 개념을 모르는 입력 이해 실패인지 분리하기 어렵다.

본 프로젝트의 핵심 연구 질문은 다음과 같다.

> **현대 개념이 포함된 질문을 답을 유출하지 않는 시대중립 기능 설명으로 재작성하면, 고정된 vintage LLM의 downstream multiple-choice 성능이 향상되는가?**

보조 연구 질문은 다음과 같다.

1. 전처리기가 현대 용어를 제거하면서도 원래 질문의 기능적 의미를 유지하는가?
2. 성능 향상이 단순한 힌트 추가나 prompt 길이 증가 때문이 아니라, 시대착오적 표현 제거 때문이라고 볼 수 있는가?
3. primitive bottleneck 기반 rewriter가 단순 rule-only 치환보다 유의미한 이득을 주는가?

여기서 **Talkie-1930은 우리가 학습하거나 개선하는 모델이 아니다.** Talkie-1930은 고정된 downstream evaluator로 사용한다. 우리가 제안하는 모델은 Talkie 앞단에서 동작하는 **Era-Neutral Prompt Generator**다.

## 2. 배경과 필요성

Talkie-1930은 1931년 이전 영어 텍스트를 중심으로 학습된 vintage language model이다. 이 모델은 현대 웹 지식이 거의 없는 상태에서 언어모델의 일반화 능력을 관찰하는 데 유용하다. 하지만 현대 사용자의 질문에는 다음과 같은 표현이 자연스럽게 포함된다.

- LLM, RAG, hallucination, GPU, API
- smartphone, internet, social media
- PCR, mRNA vaccine, MRI
- GPS, electric vehicle, QR code

이런 표현을 그대로 입력하면 모델 성능 저하가 “모델의 추론 실패”인지 “입력 개념 이해 실패”인지 모호해진다. 따라서 본 프로젝트는 시대착오적 질문을 버리지 않고, **1930년 이전 지식만으로도 이해 가능한 기능적 설명으로 재작성하는 전처리 layer**를 제안한다.

직관적으로 말하면, 이 프로젝트는 “1930년 지식 기반 모델이 이해할 수 있는 입력 번역기”를 만드는 것이다. 단, 일반 번역처럼 언어를 바꾸는 것이 아니라, 현대 용어를 시대중립적인 기능·메커니즘 설명으로 바꾼다.

## 3. Talkie 모델과 본 프로젝트의 범위

| 모델 | 용도 |
|---|---|
| `talkie-1930-13b-base` | 1931년 이전 영어 텍스트 중심으로 학습된 base model |
| `talkie-1930-13b-it` | 1930 base model을 instruction-following 용도로 후학습한 모델 |
| `talkie-web-13b-base` | 같은 계열 구조를 현대 웹 데이터로 학습한 비교용 모델 |

본 프로젝트의 주 실험 대상은 **Talkie-1930 계열의 raw prompt vs rewritten prompt 비교**다. `talkie-web` 또는 일반 modern LLM과의 비교는 범위를 넓히므로, 필요하면 보조 분석으로만 사용한다.

로컬에서 13B 모델을 직접 실행하려면 GPU VRAM 요구량이 커서 현실적이지 않다. 따라서 실험은 Talkie 웹 응답 수집 또는 manual response CSV 방식을 사용한다.

## 4. 기존 talkie-lm.com 실험과의 차이점

talkie-lm.com의 소개 글은 vintage LLM의 특성과 benchmark 성능 분석에 초점을 둔다. 특히 시대착오적 질문을 필터링하거나 제외한 상태에서 모델 성능을 분석한다.

본 프로젝트는 그 실험을 반복하지 않는다. 핵심 차이는 다음과 같다.

| 구분 | talkie-lm.com 소개 글 | 본 프로젝트 |
|---|---|---|
| 목적 | Talkie-1930 자체의 특성과 benchmark 성능 분석 | Talkie-1930 앞단의 입력 전처리기 구현 및 평가 |
| 시대착오 질문 처리 | 평가에서 필터링하거나 제외 | 질문을 버리지 않고 시대중립 표현으로 재작성 |
| 개선 대상 | Talkie 모델이 아니라 모델 분석 | Era-Neutral Prompt Generator |
| 평가 관점 | vintage model vs modern twin | raw prompt vs rewritten prompt paired evaluation |
| 산출물 | 모델 소개와 분석 결과 | 데이터셋, 전처리 파이프라인, ablation, 정량 평가, 서비스 데모 |

따라서 본 프로젝트의 차별점은 **anachronism filtering**이 아니라 **anachronism-aware prompt rewriting**이다. 즉, 평가 데이터를 제거하는 것이 아니라, 실제 사용자가 넣은 현대어 질문을 Talkie가 이해 가능한 입력으로 바꿔주는 서비스형 전처리 layer를 만든다.

## 5. 제안 방법

전체 시스템은 다음 단계로 구성한다.

```text
modern multiple-choice question
-> anachronism detector
-> concept primitive mapper
-> primitive-bottleneck rewriter
-> validator
-> deterministic repair loop
-> Talkie-1930 downstream evaluation
```

### 5.1 Anachronism Detector

입력 질문에서 1930년 이전 지식만으로 이해하기 어려운 현대 개념을 탐지한다.

현재 실험 skeleton은 다음 두 방식을 결합한다.

- 사전 기반 탐지: 현대 개념 library의 alias와 forbidden term 목록 사용
- 경량 ML 기반 탐지: Bernoulli Naive Bayes로 token-level modern-term 가능성 추정

이 모듈의 역할은 정답을 맞히는 것이 아니라, 어떤 개념을 시대중립 설명으로 바꿔야 하는지 찾는 것이다.

### 5.2 Concept Primitive Mapper

탐지된 현대 용어를 기능적 primitive로 연결한다.

| 현대 개념 | 시대중립 primitive |
|---|---|
| RAG | 답하기 전에 관련 기록 저장소를 먼저 찾는 방법 |
| GPU | 많은 작은 계산을 동시에 수행하는 장치 |
| smartphone | 주머니에 넣고 다니는 무선 통신 및 기록 확인 장치 |
| QR code | 카메라로 읽을 수 있는 정사각형 기호 |

primitive는 현대 용어를 직접 설명하지 않고, Talkie-1930이 이해할 수 있는 기능적 역할만 제공한다.

### 5.3 Denoising Primitive Bottleneck Rewriter

기존 제안서의 “Denoising Text Autoencoder”라는 표현은 과장되어 보일 수 있으므로, 본 프로젝트에서는 더 방어 가능한 이름인 **Denoising Primitive Bottleneck Rewriter**로 설명한다.

현재 구현은 현대 질문의 bag-of-words를 낮은 차원의 latent bottleneck으로 압축한 뒤, 시대중립 primitive token을 복원한다.

```text
modern question tokens
-> encoder
-> low-dimensional primitive bottleneck
-> decoder
-> era-neutral primitive tokens
-> rewritten question
```

이 모듈의 목적은 완전한 자연어 생성 모델을 만드는 것이 아니다. 목적은 **현대 표면어를 제거하면서 질문의 기능적 의미를 보존하는 중간 표현을 만드는 것**이다. 따라서 rule-only rewrite baseline과 비교하여, learned bottleneck이 단순 사전 치환보다 도움이 되는지 검증한다.

### 5.4 Era-Neutral Rewriter

rewriter는 primitive를 사용하여 최종 질문을 만든다.

```text
Original:
Why is RAG useful for reducing LLM hallucination?

Rewritten:
Consider a method that first searches a store of relevant records before composing a reply.
Why is this method useful for reducing unsupported answers from an automatic writing system?
Focus on the mechanism, not on the modern name.
```

단, 이런 rewrite는 정답 힌트가 될 위험이 있다. 따라서 본 프로젝트는 proposed만 평가하지 않고, rule-only baseline과 length-controlled baseline을 함께 둔다.

### 5.5 Validator와 Deterministic Repair Loop

validator는 rewrite 결과가 실험적으로 공정한지 검사한다.

검사 항목:

- forbidden modern term이 남아 있는가?
- required primitive가 보존됐는가?
- 선택지를 과도하게 복사했는가?
- `A`, `B`, `C`, `D`, `correct answer`, `option A` 같은 leakage 표현이 있는가?
- 원문 대비 길이가 지나치게 짧거나 긴가?

검증 실패 시 repair loop가 deterministic하게 수정한다.

- 남은 forbidden term 제거
- 누락된 primitive phrase 보강
- answer label 또는 leakage phrase 제거
- 지나친 choice-copying 표현 완화

이 모듈은 LLM agent가 아니라 **validator-driven deterministic repair loop**로 정의한다.

## 6. 실험 설계

### 6.1 데이터셋

직접 생성한 4지선다 문제 100~150개를 사용한다. 자유 생성 답변은 정량화가 어렵기 때문에, Talkie가 A/B/C/D 중 하나를 고르게 하는 객관식 downstream task로 평가한다.

도메인 구성:

- AI / Computing
- Medicine / Biology
- Communication / Media
- Transportation / Engineering
- Environment / Energy
- Daily Technology / Society

각 item의 주요 field:

| field | 의미 |
|---|---|
| `id` | 질문 ID |
| `domain` | 질문 도메인 |
| `original_question` | 현대어 원본 질문 |
| `choices` | A/B/C/D 선택지 |
| `gold_answer` | 정답 label |
| `gold_anachronism_terms` | 탐지되어야 하는 현대 용어 |
| `forbidden_terms` | rewrite 후 남으면 안 되는 용어 |
| `required_primitives` | rewrite에 유지되어야 하는 의미 primitive |
| `primitive_phrase` | 시대중립 설명 문구 |
| `human_validated` | 사람이 검수했는지 여부 |

데이터셋 품질을 방어하기 위해 보고서에는 다음 통계를 포함한다.

- domain별 문항 수
- 정답 A/B/C/D 분포
- 원문/rewritten prompt 평균 길이
- 선택지 평균 길이
- 사람이 검수한 문항 수
- leakage 의심 문항 수
- domain별 baseline/proposed accuracy

### 6.2 비교 조건

단순히 raw vs proposed만 비교하면 “힌트를 더 준 것 아닌가?”라는 질문을 피하기 어렵다. 따라서 다음 조건을 비교한다.

| 조건 | 목적 |
|---|---|
| Raw Prompt | 현대어 원문을 그대로 입력한 기본 baseline |
| Rule-only Rewrite | 사전/primitive 기반 치환만 사용한 baseline |
| Length-controlled Rewrite | prompt 길이 증가로 인한 힌트 효과 통제 |
| Proposed | detector + primitive bottleneck rewriter + validator + repair |
| Proposed without Validator | validator와 repair가 실제로 필요한지 확인하는 ablation |

현재 end-to-end skeleton은 raw/proposed 비교와 grid search를 제공한다. 최종 실험 전 rule-only, length-controlled, no-validator 조건을 추가하여 ablation을 완성한다.

### 6.3 Paired Evaluation

각 문항은 같은 선택지와 같은 정답을 유지한 채 여러 조건으로 평가한다.

```text
Raw:
original question + same choices -> Talkie-1930 -> predicted answer

Rule-only:
rule-only rewritten question + same choices -> Talkie-1930 -> predicted answer

Proposed:
primitive-bottleneck rewritten question + same choices -> Talkie-1930 -> predicted answer
```

같은 문항을 여러 prompt 조건으로 평가하므로, 질문 난이도와 정답 분포를 통제할 수 있다.

### 6.4 Talkie 응답 수집 방식

실험 재현성과 안전성을 위해 우선순위는 다음과 같이 둔다.

1. **Manual CSV mode**: 생성된 prompt를 Talkie 웹에 입력하고 응답을 CSV로 저장한다. 가장 재현성 설명이 쉽다.
2. **Chat UI automation**: Playwright로 `https://talkie-lm.com/chat`을 자동 조작한다. UI 변경에 취약하므로 cache와 로그를 남긴다.
3. **비공식 SSE endpoint wrapper**: HAR에서 확인한 endpoint는 보조 실험용으로만 사용한다. 최종 보고서의 주 결과로 쓰지 않는 것이 안전하다.

모든 조건은 같은 decoding 설정을 사용한다.

- temperature 고정
- max token 고정
- 같은 prompt format 사용
- 응답 파싱 규칙 고정
- raw response와 parsed label 모두 저장

## 7. 정량 평가 지표

### 7.1 Downstream Utility Metrics

Talkie의 4지선다 예측 결과를 정량 평가한다.

| 지표 | 의미 |
|---|---|
| Accuracy | 정답 label을 맞힌 비율 |
| Invalid Rate | A/B/C/D로 파싱되지 않은 응답 비율 |
| Macro-F1 | A/B/C/D 각 label의 F1 평균 |
| Paired Accuracy Improvement | 같은 문항에서 proposed accuracy - baseline accuracy |
| Bootstrap 95% CI | paired improvement의 신뢰구간 |
| Exact McNemar Test | paired binary outcome에 대한 보조 유의성 점검 |

McNemar test는 보조 지표로 사용한다. 100~150개 규모에서는 discordant pair 수가 작을 수 있으므로, p-value 하나로 강한 결론을 내리지 않는다.

### 7.2 Generator Component Metrics

전처리기가 의도한 변환을 수행했는지 평가한다.

| 지표 | 의미 |
|---|---|
| Detector Precision | 탐지한 현대 용어 중 실제 gold term 비율 |
| Detector Recall | gold modern term 중 탐지된 비율 |
| Detector F1 | 탐지 precision/recall 균형 |
| Anachronism Removal Rate | rewrite 후 forbidden term이 모두 제거된 비율 |
| Required Primitive Recall | 시대중립 primitive 보존 정도 |
| Choice-copying Score | rewrite가 선택지 문장을 얼마나 베꼈는지 |
| Leakage Risk Rate | 답 유출 위험이 감지된 비율 |
| Rewrite Pass Rate | validator를 통과한 rewrite 비율 |
| Mean Repair Attempts | 평균 repair 횟수 |

## 8. 구현 계획

현재 실험 코드는 다음 위치에 있다.

```text
experiments/team2_pipeline/TermProject_team2.py
```

현재 제공되는 mode:

| mode | 용도 |
|---|---|
| `simulate` | MockTalkieClient로 빠르게 파이프라인 검증 |
| `simulate_grid` | mock evaluator로 bottleneck hyperparameter grid search |
| `rewrite_only` | Talkie 호출 없이 전처리 결과만 생성 |
| `prepare_manual` | 수동 Talkie 입력용 CSV 생성 |
| `evaluate_manual` | 수동으로 수집한 Talkie 응답 CSV 평가 |
| `run_web` | Playwright로 Talkie 웹을 자동 조작 |
| `run_web_grid` | web Talkie + grid search + 최종 평가 |

추가 구현이 필요한 항목:

- rule-only rewrite condition
- length-controlled rewrite condition
- proposed without validator condition
- bootstrap confidence interval 계산
- exact McNemar test
- dataset quality report
- Streamlit 또는 간단한 웹 데모

권장 실행 순서:

```powershell
python experiments/team2_pipeline/TermProject_team2.py --mode simulate --n_items 20 --out_dir results_smoke
python experiments/team2_pipeline/TermProject_team2.py --mode rewrite_only --n_items 100 --out_dir results_rewrite
python experiments/team2_pipeline/TermProject_team2.py --mode prepare_manual --n_items 100 --out_dir results_manual
python experiments/team2_pipeline/TermProject_team2.py --mode evaluate_manual --manual_response_csv results_manual/manual_talkie_input_sheet.csv --out_dir results_final
```

## 9. 예상 산출물

| 경로 | 내용 |
|---|---|
| `data/generated_questions.jsonl` | 생성된 4지선다 질문 |
| `data/modern_terms_dictionary.json` | 현대 용어 사전 |
| `data/primitive_dictionary.json` | 시대중립 primitive 사전 |
| `input_data/raw_4choice_questions.csv` | 원본 질문 CSV |
| `input_data/era_neutral_preprocessed_questions.csv` | rewrite 결과 CSV |
| `results*/per_item_results.csv` | item별 조건별 결과 |
| `results*/final_metrics.csv` | accuracy, macro-F1, invalid rate |
| `results*/component_metrics.csv` | detector/rewriter/validator 지표 |
| `results*/paired_test_mcnemar.json` | paired test 결과 |
| `results*/qualitative_examples.md` | 성공/실패 정성 사례 |
| `results*/report.md` | 자동 생성 실험 리포트 |
| demo UI | 입력, rewrite, validation, Talkie 비교 결과 시연 |

## 10. 역할분담 제안

| 역할 | 담당 |
|---|---|
| 데이터셋/프롬프트 담당 | 100~150개 4지선다 질문 검수, 정답 분포 확인, leakage 의심 문항 제거 |
| 모델/전처리 파이프라인 담당 | detector, primitive mapper, bottleneck rewriter, validator, repair 개선 |
| 실험 실행/결과 분석 담당 | Talkie 응답 수집, cache 관리, metric 계산, bootstrap/McNemar 분석 |
| 보고서/발표/데모 담당 | ACL 형식 보고서, 발표자료, 서비스 데모 화면 또는 CLI 시연 구성 |

## 11. 리스크와 보완 계획

| 리스크 | 보완 |
|---|---|
| “그냥 사전 치환 아닌가?” | rule-only baseline과 primitive bottleneck proposed를 비교 |
| rewrite가 정답 힌트를 줄 수 있음 | choice-copying score, leakage check, length-controlled baseline 추가 |
| 데이터셋이 인위적으로 보일 수 있음 | 100~150개 문항 전수 검수, 정답 분포/길이/domain 통계 공개 |
| 비공식 API 사용 리스크 | manual CSV 또는 chat UI 기반 수집을 주 실험으로 사용 |
| validator 효과가 불명확함 | proposed without validator ablation 추가 |
| McNemar test power 부족 | paired improvement와 bootstrap CI를 main metric으로 사용 |
| Talkie 응답 변동성 | decoding setting 고정, raw response cache 저장 |

## 12. 최종 결론

본 프로젝트는 Talkie-1930을 개선하는 프로젝트가 아니다. Talkie-1930은 고정된 downstream evaluator이고, 우리가 제안하는 모델은 현대어 질문을 시대중립 prompt로 변환하는 **Era-Neutral Prompt Generator**다.

강한 주장을 위해서는 proposed 하나만 보여주지 않는다. Raw, rule-only, length-controlled, no-validator ablation과 비교하여, 성능 향상이 단순한 prompt 길이 증가나 정답 힌트가 아니라 **anachronism-aware primitive rewriting**에서 오는지 검증한다.
