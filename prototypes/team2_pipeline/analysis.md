# TermProject_team2.py 코드 분석

> 이 문서는 `prototypes/team2_pipeline/TermProject_team2.py`에 대한 분석이다.
> 해당 파일은 현재 `IML_TalkieBridge`의 메인 구현이 아니라 참고용 프로토타입이다.

## 한 줄 요약

`TermProject_team2.py`는 **현대어 4지선다 질문을 생성하고, 이를 시대중립 질문으로 재작성한 뒤, Talkie 1930에 baseline/proposed 두 조건으로 물어보고 성능 차이를 측정하는 end-to-end 실험 파이프라인**이다.

중요한 구분은 다음과 같다.

- **Talkie 1930**: 우리가 만든 모델이 아니다. 고정된 downstream evaluator다.
- **Era-Neutral Prompt Generator**: 우리가 제안하는 모델이다. 현대어 질문을 1930년 이전 지식으로도 이해 가능한 prompt로 재작성한다.

## 코드의 큰 구조

```text
1. 현대 개념 library 100개 정의
2. 4지선다 질문 dataset 생성
3. 현대 용어 탐지
4. primitive mapping
5. denoising text autoencoder 학습
6. 시대중립 질문 rewrite
7. validator/self-repair
8. Talkie 응답 수집
9. baseline/proposed 정량 평가
10. report.md 자동 생성
```

## 주요 구성 요소

### 1. Concept Library와 Dataset

`build_concept_library()`는 100개의 현대 개념을 만든다. 도메인은 AI/Computing, Medicine/Biology, Communication/Media, Transportation/Engineering, Environment/Energy, Daily Technology/Society로 구성되어 있다.

각 concept는 다음 정보를 가진다.

- `term`: 대표 현대 개념
- `modern_terms`: 탐지되어야 하는 현대어 alias
- `task`: 질문에 들어갈 용도 설명
- `primitive_phrase`: 시대중립 설명
- `mechanism`: 정답 선택지에 들어갈 설명

`generate_dataset()`은 이 concept library를 바탕으로 4지선다 문제를 만든다.

### 2. Detector

`AnachronismDetector`는 질문 안의 현대 용어를 탐지한다. 사전 기반 alias matching과 간단한 Naive Bayes 기반 token classifier가 결합되어 있다.

평가에는 `detection_tp`, `detection_fp`, `detection_fn`이 기록되고, 최종적으로 detector precision/recall/F1이 계산된다.

### 3. Autoencoder

`DenoisingTextAutoencoder`는 외부 ML 라이브러리 없이 직접 구현된 작은 autoencoder다.

- 입력: 현대 질문 bag-of-words
- latent: 낮은 차원의 bottleneck vector
- 출력: 시대중립 primitive/task token bag-of-words

`default_autoencoder_hyperparameter_pool()`에 후보 hyperparameter가 있고, `run_autoencoder_grid_search()`가 후보별 Talkie downstream 성능을 비교한다.

### 4. Rewriter

`AutoencoderEraNeutralRewriter`는 autoencoder가 복원한 primitive를 사용해서 질문을 다시 쓴다.

현대 용어는 `this method`, `this technical idea`처럼 치환하고, 앞부분에 기능적 설명을 붙인다.

### 5. Validator와 Repairer

`RuleBasedValidator`는 rewrite 결과가 실험에 적합한지 검사한다.

검사 항목:

- forbidden modern terms remaining
- required primitive recall
- choice copying risk
- answer format leakage
- length ratio
- explicit leakage phrase

실패하면 `SelfRepairer`가 현대 용어 제거, primitive 보강, 형식 제거 등을 수행한다.

### 6. Talkie Client

두 종류의 evaluator가 있다.

- `MockTalkieClient`: 빠른 pipeline 검증용 simulator
- `PlaywrightTalkieClient`: 실제 Talkie 웹 UI 자동화용 client

Playwright client는 브라우저를 띄우고 `https://talkie-lm.com/chat`에 prompt를 입력한 뒤, body text에서 응답을 추출한다. 이 방식은 UI 변경에 취약하므로 final 실험 전 smoke test가 필요하다.

### 7. Metric

최종 지표는 두 계층으로 나뉜다.

Downstream metric:

- Accuracy
- Macro-F1
- Invalid rate
- McNemar paired test

Component metric:

- Detector precision/recall/F1
- Anachronism removal rate
- Required primitive recall
- Rewrite pass rate
- Leakage risk rate
- Mean repair attempts

## 실행 모드 해석

| mode | 설명 |
|---|---|
| `simulate` | MockTalkieClient로 빠르게 전체 pipeline 검증 |
| `simulate_grid` | mock 기반 grid search |
| `rewrite_only` | Talkie 호출 없이 rewrite 결과만 생성 |
| `prepare_manual` | Talkie에 수동 입력할 CSV 생성 |
| `evaluate_manual` | 수동 수집 응답 CSV 평가 |
| `run_web` | Playwright로 Talkie 웹 UI 자동화 |
| `run_web_grid` | 실제 웹 Talkie 기반 grid search와 최종 평가 |

## 현재 위치

이 파일은 원본 `talkie` inference library의 일부가 아니며, 현재 프로젝트의 메인 구현도 아니다. 참고용 프로토타입으로 다음 위치에 보관한다.

```text
prototypes/team2_pipeline/TermProject_team2.py
```

직접 확인해야 할 때만 루트에서 다음처럼 실행한다. 실행 시 `data/`, `input_data/`, `cache/`, `results*/`를 루트 기준으로 생성한다.

```powershell
python prototypes/team2_pipeline/TermProject_team2.py --mode simulate --n_items 20 --out_dir results_smoke
```

## 현재 코드에 대한 판단

이 판단은 프로토타입에만 해당한다. 현재 `IML_TalkieBridge` 본 구현의 품질 판단이 아니다.

## 왜 최종 구현으로 쓰기 어려운가

현재 `TermProject_team2.py`는 최종 실험 코드라기보다, 제안서 아이디어가 end-to-end로 가능한지 확인하기 위한 초기 프로토타입이다. 이 프로토타입은 4지선다 질문 생성, modern term 탐지, primitive mapping, autoencoder 기반 rewrite, validator/self-repair, Talkie 웹 평가, metric/report 생성까지 전체 흐름을 한 번에 연결했다는 점에서 의미가 있다.

하지만 실제 최종 구현으로 넘어가기 전에 다음 문제가 확인되었다.

1. 입력 데이터 생성 방식의 한계

현재 질문 100개는 외부 LLM이나 실제 사용자 데이터로 생성되는 것이 아니라, `build_concept_library()`에 사람이 미리 작성한 concept를 `generate_dataset()`이 템플릿으로 조합해 만든 synthetic dataset이다. 따라서 데이터가 너무 규칙적이고 인위적일 수 있다.

2. 데이터 누수와 힌트 누설 위험

질문 생성, 정답 선택지, primitive rewrite가 모두 같은 concept library에서 나온다. 이 때문에 proposed prompt가 정답 선택지와 의미적으로 가까워질 수 있고, 성능 향상이 진짜 시대중립화 효과인지 정답 힌트 효과인지 구분하기 어렵다.

3. Autoencoder 학습/평가 분리 부족

`DenoisingTextAutoencoder`는 전체 dataset으로 학습되고, 같은 dataset에서 평가된다. 따라서 일반화 성능을 평가한다고 보기 어렵다. 실제 구현에서는 train/validation/test split이 필요하다.

4. Ablation 실험 부족

제안서에는 raw prompt, rule-only rewrite, length-controlled rewrite, proposed, proposed without validator 비교가 필요하지만, 현재 프로토타입은 사실상 raw vs proposed만 평가한다. 그래서 autoencoder가 실제로 rule-only보다 나은지, validator가 필요한지, prompt 길이 증가가 성능 향상의 원인인지 검증할 수 없다.

5. Autoencoder 기여도 불명확

`ConceptPrimitiveMapper`가 이미 modern term을 primitive로 직접 매핑하기 때문에, autoencoder가 실제로 얼마나 중요한 역할을 했는지 현재 구조만으로는 분리하기 어렵다.

6. Talkie web grid search의 평가 오염 문제

`run_web_grid`는 Talkie 응답을 이용해 hyperparameter를 고르고, 다시 같은 데이터에서 최종 성능을 평가한다. 최종 evaluator를 tuning에 사용한 셈이므로, 실제 구현에서는 validation set에서만 hyperparameter를 선택하고 test set은 마지막에 한 번만 평가해야 한다.

7. Mock evaluator의 한계

`MockTalkieClient`는 pipeline 검증용이며 proposed가 유리하도록 확률이 설계되어 있다. 따라서 mock 결과는 실제 성능 근거로 사용할 수 없다.

8. Talkie 연동 방식의 재현성 문제

현재 `PlaywrightTalkieClient`는 공식 API가 아니라 `https://talkie-lm.com/chat` 웹 UI를 자동화하고 body text에서 응답을 scraping한다. UI 변경이나 이전 대화 내용에 취약하다.

9. 응답 파싱의 불안정성

Talkie 응답이 A/B/C/D 한 글자가 아니라 설명형으로 나오면 `normalize_choice_with_choices()`가 token overlap으로 label을 추정한다. 이 과정에서 잘못된 label이 기록될 수 있으므로 수동 검수나 더 엄격한 parsing이 필요하다.

10. 통계 분석 부족

현재는 accuracy, macro-F1, invalid rate, McNemar chi-square approximation 정도만 제공한다. 제안서 수준으로 가려면 bootstrap confidence interval, exact McNemar test, domain별 성능, 데이터셋 품질 통계가 필요하다.

따라서 이 프로토타입은 “제안서 아이디어가 기술적으로 가능한지 확인한 proof-of-concept”로 해석해야 한다. 실제 구현에서는 데이터셋 분리, ablation 조건 추가, leakage 검수, Talkie 평가 재현성 개선, 통계 분석 보강을 먼저 수행해야 한다.

장점:

- 프로젝트의 핵심 구분, 즉 “Talkie는 downstream evaluator이고 우리 모델은 preprocessor”라는 점이 명확하다.
- 데이터 생성, rewrite, validation, downstream evaluation, report generation까지 end-to-end로 이어져 있다.
- 정량 평가 지표가 과제 요구사항에 맞게 구성되어 있다.
- 실제 Talkie 웹 실행이 실패할 경우 manual mode와 mock mode가 있다.

주의점:

- Playwright 웹 자동화는 UI 변경에 취약하다.
- 생성 데이터가 모두 코드 내부 concept library에서 나오므로, 일부 샘플은 사람이 검수해야 한다.
- autoencoder가 실제 deep learning framework 기반은 아니므로, 보고서에서는 “dependency-free denoising text autoencoder”라고 정확히 설명해야 한다.
- 웹 endpoint/API 사용 시 공식 API가 아니라는 점을 보고서에 명시해야 한다.
