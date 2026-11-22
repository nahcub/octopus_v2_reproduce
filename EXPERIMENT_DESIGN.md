# 실험 설계: negative 구성이 거절 경계에 미치는 영향

(논문에 없는 추가) 재현 이후 논문용 실험이다. 재현 결과는 README.md, BFCL 테스트셋은 BFCL_TEST.md 참고.
이 문서는 **실험을 돌리기 전에** 무엇을 고정하고, 무엇을 바꾸고, 결과를 어떻게 판정할지 정해 두는 용도다.
결과를 보고 기준을 바꾸지 않도록, 바꿀 때는 맨 아래 "변경 기록"에 이유와 함께 남긴다.

## 1. 질문

| | 질문 | 이유 |
|---|---|---|
| Q1 | negative **비율**을 바꾸면 거절 경계가 좋아지는가, 아니면 기준선(문턱값)만 옮겨 가는가? | BFCL 에서 ours 는 과잉 거절, 공개 모델은 과소 거절. 문턱값을 다시 잡으면 공개 모델이 오히려 더 잘 구분했다 (BFCL_TEST.md 결과) |
| Q2 | negative **내용**을 바꾸면 (특히 positive 와 한 칸만 다른 문장, minimal pair) 경계 자체가 좋아지는가? | 두 모델 모두 near-miss 가 가장 약하고, 공개 모델은 near-miss 에서 확신을 갖고 틀린다 (선택 확률 중앙값 0.97) → 문턱값으로 못 고친다 |

용어
- **경계**: 모델이 "부를 함수가 있다 / 없다"를 가르는 선
- **기준선이 옮겨 간다**: 구분하는 능력은 그대로인데 거절을 더 자주 / 덜 자주 하게만 되는 것
- **near-miss**: 폰에 시키는 일처럼 보이지만 20개 함수에는 없는 요청 ("알람 목록 보여 줘", "에어컨 켜 줘")
- **minimal pair**: positive 문장에서 한 칸(동작·대상·기기)만 바꿔서 지원되지 않는 요청이 된 문장
  - 예: "Set an alarm for 7am" (positive) ↔ "Delete my 7am alarm" (negative, 동작만 바뀜)

## 2. 고정하는 것

모든 비교 모델에서 아래는 같다. 하나라도 다르면 그 비교는 쓰지 않는다.

| 항목 | 값 | 근거 |
|---|---|---|
| 베이스 모델 | google/gemma-2b | 재현 설정 |
| 학습 설정 | LoRA r=16, alpha=32, AdamW lr 5e-5, warm-up 10, linear, 3 epochs, batch 16, weight decay 0 | 논문 3.4절 + README "학습" (batch·weight decay 는 논문에 없는 추가) |
| 함수 명세·프롬프트 | `src/toy_api.py` 그대로 | 공개 모델 카드 |
| **positive** | 현재 `data/train.jsonl` 의 positive 2,000개 **같은 파일** | positive 가 바뀌면 negative 효과와 섞인다 |
| 학습 seed | 0 (주요 비교는 seed 0/1/2, 6절) | |
| 평가 데이터 | 아래 3절 | |

Q1 에서 positive 를 고정하고 negative 개수를 바꾸면 전체 step 수도 같이 바뀐다 (2,105개 ~ 4,000개).
step 수를 맞추면 이번에는 positive 를 보는 횟수가 달라지므로, **positive 노출을 고정하는 쪽**을 택한다. 이 점은 논문에 한계로 적는다.

## 3. 평가 데이터 역할

| 세트 | 크기 | 역할 | 사용 규칙 |
|---|---|---|---|
| 우리 test (`data/test.jsonl`) | 800 (pos 400 / neg 400) | **dev**: 문턱값 선택, 중간 확인, 회귀 확인 | 자유롭게 본다 |
| 공식 210 (`eval_official`) | 210 (neg 10) | 보조: 공개 모델 분포에서 positive 가 무너지지 않는지 | 자유롭게 본다 |
| BFCL (`data/bfcl_test.jsonl`) | 234 (pos 117 / neg 117) | **최종 test** | 모든 모델이 확정된 뒤 **한 번만** 채점. 그 결과로 데이터를 다시 고치지 않는다 |
| MASSIVE (`data/massive_test.jsonl`) | 465 (pos 233 / neg 232) | **두 번째 최종 test** (BFCL 과 독립) | BFCL 과 같은 규칙. 어떤 모델의 결과도 보기 전에 확정함 (MASSIVE_TEST.md) |

주의 (정직하게 적어 둘 것)
- 이 설계 자체가 BFCL 1차 결과(baseline 분석)를 보고 나왔다. 그래서 BFCL 이 완전히 깨끗한 test 는 아니다.
- 완화 방법: (1) 지금 이 문서로 설계를 고정한다. (2) 생성 프롬프트에 BFCL 문장·표현을 넣지 않는다. (3) minimal pair 는 **우리 positive 에서만** 만든다.
- (4) BFCL 과 다른 출처의 두 번째 test 로 **MASSIVE** 를 둔다 (아래 3.1). 이 설계를 정한 뒤에 만들고, 어떤 모델 결과도 보기 전에 확정하므로 BFCL 보다 깨끗한 test 다.

### 3.1 MASSIVE 두 번째 test (논문에 없는 추가)
- **출처**: MASSIVE 1.1 (Amazon, `alexa/massive`), 영어 `en-US.jsonl` 만 사용. 영어 부분은 SLURP 문장. 라이선스 CC BY 4.0 (받은 `LICENSE` 파일에서 확인). 논문에 쓸 때 출처를 밝힌다.
- **규모**: 영어 16,521개 (train 11,514 / dev 2,033 / test 2,974), 의도 60개. 전부 턴 1개짜리 음성 비서 질의.
- **왜 MASSIVE 인가**
  - near-miss 를 **사람이 쓴 문장으로** 충분히 모을 수 있다. 예: `alarm_set`(positive) ↔ `alarm_query`·`alarm_remove`(near-miss), `calendar_set` ↔ `calendar_query`·`calendar_remove`, 화면 밝기 ↔ `iot_hue_*`(조명), 음악 ↔ `play_podcasts`·`play_radio`·`music_query`
  - 실험 B2 의 minimal pair 는 우리가 Gemini 로 만든다. MASSIVE 의 near-miss 는 우리 생성 과정과 무관하므로 "우리가 만든 데이터에서만 좋아진 것" 이라는 반론을 확인할 수 있다
  - BFCL 은 글로 쓴 LLM 식 질의, MASSIVE 는 말로 한 음성 비서 질의라 서로 보완된다
- **positive 가 될 수 있는 의도** (괄호는 test 파티션 개수): `weather_query`(156), `play_music`(176), `alarm_set`(41), `news_query`(124), `transport_query`(51) 일부, `calendar_set`(209) 일부, `email_sendemail`(114) 일부, `audio_volume_*` 일부
  - 카메라·전화·문자·블루투스·방해 금지·Nest 온도·연기 감지기·문·영상·YouTube 에 해당하는 의도는 없다. BFCL 처럼 함수 선택을 고르게 재는 셋이 아니다
- **정답 기준**: MASSIVE 의도 라벨을 그대로 쓰지 않는다 (예: `audio_volume_up` 에 "fast", "where is the car" 가 섞여 있음). 의도 라벨은 후보를 좁히는 데만 쓰고, 정답은 BFCL_TEST.md 3절 기준과 "라벨링하면서 추가로 정한 규칙" 으로 다시 붙인다. 새 경우가 나오면 규칙을 추가하되 모델 결과를 보기 전에 정한다
- **인자**: `annot_utt` 의 칸 표시(`[time : six thirty am]`, `[place_name : paris]`)를 인자 정답 근거로 쓴다. 글자로 적힌 시간·숫자를 함수 형식("06:30")으로 바꾸는 규칙은 만들면서 정한다
- **후보·선정**: test 파티션 **전체**를 후보로 쓴다 (우리는 MASSIVE 로 학습하지 않으므로 train 파티션도 쓸 수 있지만, 원래 나눔을 존중해 test 만 쓴다). 관련 의도만 고르지 않는 이유: 의도 라벨이 틀린 문장 속에도 positive 가 있고, 잡담·계산 negative 는 `qa_*`·`general_*` 의도에서 나온다. positive 상한·negative 유형별 같은 수 추출은 BFCL_TEST.md 5절과 같은 방식, seed 0
- **후보 숫자** (`python -m src.massive_bench candidates`): test 2,974 → 같은 문장 4개 제거 → **2,970**. 의도로 본 참고 표시(hint)가 붙은 것 945 (음악 176, 일정 209, 날씨 156, 뉴스 124, 이메일 114, 볼륨 62, 경로 51, 알람 41, 연락처 12). 호출어 포함 114. 우리 train·test 와 겹치는 문장 0, BFCL 후보와 겹치는 문장 2 ("what is your name", "what time is it")
- **결과** (2026-11-18 확정, 자세한 규칙·숫자는 MASSIVE_TEST.md): 라벨 positive 615 / negative 2,036 / 제외 319 → 최종 **465 (positive 233 / negative 232)**. near-miss 68개 (BFCL 23개). 검토는 Claude 가 했고 사람 검토는 아직 안 함
- **파일**: `src/massive_bench.py`, `src/massive_labels.py`, `data/massive_test.jsonl`, `notebooks/08_eval_massive.ipynb` (git O), `data/raw/massive/` (git X). 선정·확정은 `src/bfcl_bench.py` 의 같은 코드를 쓴다
- **지표 스크립트**: `src/refusal_metrics.py` (7절 0단계의 짝지은 bootstrap·matched-recall 스크립트). BFCL 에도 `--test data/bfcl_test.jsonl` 로 같이 쓴다
- **주의 (한계로 적을 것)**
  - 음성 받아쓰기라 소문자, 숫자를 글자로 적음, 호출어("alexa", "olly")가 섞인다. positive 가 떨어지면 거절 경계 때문인지 말투가 낯설어서인지 섞일 수 있다. dev 에서 정한 문턱값을 그대로 옮겨 쓰므로 이 영향이 커질 수 있다
  - 공개 모델이 MASSIVE·SLURP 를 학습에 썼는지는 모른다 (확인 불가)

## 4. 실험 A: 비율 (Q1)

| 모델 | positive | negative | negative 비율 | 새로 필요한 것 |
|---|---|---|---|---|
| A05 | 2,000 | 105 | 5% | 학습 1회 |
| A20 | 2,000 | 500 | 20% | 학습 1회 |
| A50 = 현재 ours | 2,000 | 2,000 | 50% | 없음 (기존 `runs/full`) |

- negative 는 현재 train negative 2,000개(`negatives_v2`)에서 **범주 8개별로 같은 비율로** 뽑는다 (seed 0). → Gemini API 비용 없음
- 공개 모델의 비율은 모른다 (논문 "M = N = 1000" 이 함수당인지 전체인지 애매). 공식 210 에서 negative 는 4.8% 였다 → 공개 모델이 낮은 비율로 학습됐을 가능성 (추정). A05 는 그 상황을 흉내 내는 용도
- 필요한 코드: `build_dataset.py` 에 `--neg-fraction` (범주별 층화 추출) 옵션

**예측** (돌리기 전에 적어 둠)
- 비율이 낮을수록 기본 생성(greedy)의 거절 정확도는 떨어지고 positive 정확도는 오른다
- AUROC 와 "같은 recall 에서의 거절 정확도"는 세 모델이 비슷하다 (차이가 신뢰구간 안)
- 이렇게 나오면 → "비율은 기준선만 옮긴다" 를 주장할 수 있다
- AUROC 도 크게 달라지면 → 비율이 경계 자체에도 영향을 준다. 그대로 보고한다

### 4.1 A00: negative 0% + softmax 문턱값 (2026-11-22 추가, 논문에 없는 추가)

| 모델 | positive | negative | 거절 방법 |
|---|---|---|---|
| A00 | 2,000 (현재 train positive 전부) | 0 | `<nexa_20>` 대신 **MSP**: 1 − (함수 토큰 20개 중 최대 확률) 이 문턱값 이상이면 거절 |

- 묻는 것: 거절 클래스(`<nexa_20>`)를 negative 로 학습하는 게 경계에 도움이 되나, 아니면 negative 의 말투가 경계를 오염시키나
- MSP 는 OOD 탐지의 기본 기준선 (Hendrycks & Gimpel, ICLR 2017). 새 방법이 아니라 통제 비교용
- 학습 설정은 A50 과 같다 (3 epoch). 문장 수가 절반이라 step 도 절반 (375). A05·A20 과 같은 성격의 차이
- 코드: `src/train.py --only-kind positive`, 채점 `src/softmax_score.py`, `src/refusal_metrics.py` 의 `:msp`
- 노트북: `notebooks/09_exp_A00.ipynb`. 채점은 우선 **우리 test(dev) 와 공식 210** 만. MASSIVE·BFCL 은 7절 5단계 규칙대로 나중에
- 같은 MSP 점수를 기존 ours·공개 모델에도 계산해서 `<nexa_20>` 점수와 나란히 본다 (추가 학습 없음)

**예측** (돌리기 전에 적어 둠)
- greedy 기본 생성의 negative 정확도는 0% (`<nexa_20>` 을 배운 적이 없음). positive 정확도는 A50 과 비슷하거나 조금 높다
- dev AUROC(MSP)는 A50 의 `<nexa_20>` AUROC 보다 낮다: 같은 생성기 negative 는 말투로 거의 완벽히 가를 수 있는데 A00 은 그걸 못 배움
- negative 중 잡담·상식은 MSP 로 잘 걸러지고, 함수와 비슷한 near-miss 는 높은 확률로 호출한다 (신경망은 비슷한 입력에 확신하므로)
- 공식 210 에서는 A50 대비 차이가 dev 보다 작다 (dev 의 이점이 말투 학습에서 왔다면)

## 5. 실험 B: 내용 (Q2)

negative 개수 2,000, 비율 50%, positive 고정. **바뀌는 건 negative 의 절반(1,000개)의 내용뿐**이다.

| 모델 | negative 앞 1,000 | negative 뒤 1,000 | 배제하는 설명 |
|---|---|---|---|
| B0 = 현재 ours | 기존 negative | 기존 negative | 기준 |
| B1 다양화 | 기존 negative (B0 와 같은 1,000) | 새로 만든 near-miss, **짝 없이** 자유 생성 | "near-miss 를 많이 넣어서 좋아진 것" |
| B2 minimal pair (제안) | 기존 negative (B0 와 같은 1,000) | positive 에서 한 칸 바꾼 짝 | — |

- "새로 만든 데이터라서 좋아진 것"을 B1 이 막고, "near-miss 가 많아서 좋아진 것"도 B1 이 막는다. B2 만 오르면 **짝으로 만든 구성**이 원인이라고 말할 수 있다
- "개수가 많아서" 설명은 개수를 같게 맞춰서 막는다. "비율 때문" 은 비율을 같게 맞춰서 막는다
- B1 과 B2 의 near-miss 가 다루는 영역(어떤 기기·앱)을 비슷하게 맞춘다: B2 에서 바뀐 대상 목록을 B1 프롬프트의 세부 주제로 그대로 준다

### B2 생성 절차 (원리 한 문장: "positive 질의에서 한 칸만 바꿔 지원되지 않는 요청을 만든다")
1. positive 2,000개에서 함수별로 같은 수(50개씩)를 seed 0 으로 뽑는다
2. Gemini 에게 문장과 20개 함수 목록을 주고, **동작 / 대상 / 기기·앱 중 하나만** 바꿔 어떤 함수로도 할 수 없는 요청을 만들게 한다. 바꾼 칸 종류를 같이 출력 (칸 종류별 비율 기록)
3. 걸러내기
   - 원문과 너무 다르면 제외 (단어 편집 거리 기준, 값은 소규모 실행 후 정함)
   - 별도 호출로 Gemini 에게 "20개 함수 중 이걸 처리할 수 있는 게 있나" 를 묻고, 있다고 하면 제외
   - 기존 train / test / BFCL 문장과 정규화 후 같으면 제외
4. 사람 검토: 100개 무작위로 보고 "진짜 지원 안 되는 요청인가" 비율을 적는다. 90% 미만이면 프롬프트를 고치고 다시 (이 반복은 dev 기준으로만)
- 필요한 코드: `negatives.py` 에 `--mode minimal-pair`, `--mode near-miss-free` (B1용)
- CLAUDE.md 규칙대로 `--dry-run` → 함수 2~5개 × 10개 소규모 → 전체, `--workers` 4 이하

### 예측
- B2 가 B0·B1 보다 near-miss 거절이 오르고, 같은 recall 에서의 거절 정확도와 AUROC 가 오른다
- positive 정확도(우리 test)는 떨어지지 않는다 (떨어지면 짝 때문에 positive 경계까지 흔들린 것 → 그대로 보고)
- B1 ≈ B2 로 나오면 → 제안 방법의 이점은 "near-miss 를 넣은 것" 이지 "짝" 이 아니다. 이것도 결과로 보고한다

## 6. 지표와 판정

**주 지표** (BFCL, 그리고 dev 에서도 같이 보고)
1. 같은 recall 에서의 거절 정확도
   - positive 를 함수 쪽으로 보내는 비율(recall)을 dev 에서 **90%** 로 맞추는 문턱값 t 를 정하고, 그 t 를 BFCL 에 그대로 적용
   - 문턱값은 첫 토큰의 `<nexa_20>` 확률(`probs.jsonl` 의 p_irrelevant)에 건다
2. AUROC (p_irrelevant 로 positive / negative 를 가르는 능력, 문턱값과 무관)
3. near-miss 유형의 거절 정확도

**보조 지표**: 기본 생성의 전체 정확도, positive 함수 선택 정확도, 인자 정확도(우리 test), neg_type 별 정확도, 틀린 near-miss 의 선택 확률 중앙값

**통계**
- 같은 문항에 대한 **짝지은 bootstrap** (1,000회) 으로 두 모델의 차이와 95% 신뢰구간
- "좋아졌다" = 차이의 95% 구간이 0 을 넘는 것
- BFCL negative 117개, near-miss 는 약 23개라 구간이 넓다 → 주요 비교(B0 vs B1 vs B2)는 **학습 seed 0/1/2 세 번** 돌려 평균과 범위를 같이 보고

## 7. 실행 순서와 비용

| 단계 | 내용 | API | 학습 (A100) |
|---|---|---|---|
| 0 | 코드: `--neg-fraction`, minimal-pair / near-miss-free 생성, 짝지은 bootstrap·matched-recall 스크립트 | 없음 | — |
| 0' | MASSIVE test 만들기 (내려받기 → 후보 → 라벨 → 선정 → 사람 검토 → 확정, 3.1). 어떤 모델 결과도 보기 전에 끝낸다 | 없음 | — |
| 1 | 실험 A 데이터 만들기 → A05, A20 학습 → dev 채점 | 없음 | 2회 |
| 2 | B1, B2 소규모 생성 → 검토 → 전체 생성 → 사람 검토 100개 | 있음 (dry-run 후 추정치 기록) | — |
| 3 | B1, B2 학습 (seed 0) → dev 채점 | — | 2회 |
| 4 | B0, B1, B2 seed 1·2 추가 | — | 6회 |
| 5 | **모든 모델 확정 후** BFCL·MASSIVE 각 1회 채점 (생성 + probs), 공식 210 채점 | — | — |
| 6 | 결과 정리 (이 문서 8절, README) | — | — |

- 학습 1회 시간은 아직 안 잼 (기록할 것). 총 학습 10회
- 1단계(비율)만으로도 통신학회 원고의 분석 파트가 된다. 2단계 이후는 ICLR 워크숍 쪽

## 8. 결과 (돌린 뒤 채움)

| 모델 | dev pos / neg | dev AUROC | BFCL 거절@recall90 | BFCL AUROC | BFCL near-miss | MASSIVE 거절@recall90 | MASSIVE AUROC | MASSIVE near-miss |
|---|---|---|---|---|---|---|---|---|
| 공개 모델 | | | | | | | | |
| A05 | | | | | | | | |
| A20 | | | | | | | | |
| A50 = B0 | | | | | | | | |
| B1 | | | | | | | | |
| B2 | | | | | | | | |

## 9. 아직 정하지 않은 것

- MASSIVE test 사람 검토 (Claude 검토만 끝남). 검토로 바뀌면 모델 결과를 보기 전에 반영
- recall 기준 90% 외에 다른 값(예: 공개 모델의 기본 recall)도 같이 보고할지
- B2 에서 편집 거리 기준값, 사람 검토 합격선 90% 가 적절한지 (소규모 실행 후 정함)
- 음악 함수 과잉 거절 관련: 현재 negative 범주 7번에 "music on the phone itself" 가 있고, train positive 음악 질의 98/100 이 서비스·플레이리스트를 명시한다.
  이건 negative 문제가 아니라 **positive 쪽 표현 다양성** 문제일 수 있다 (추정). 이번 실험에서는 positive 를 고정하므로 다루지 않고, 필요하면 별도 실험 C 로 둔다

## 10. 변경 기록

| 날짜 | 바꾼 것 | 이유 |
|---|---|---|
| 2026-11-18 | 최초 작성 | |
| 2026-11-18 | 두 번째 독립 test 출처를 MASSIVE 로 정함 (3절·3.1·7절·8절, 9절 항목 정리) | BFCL 이 설계 전에 본 셋이라 완전히 깨끗하지 않고 near-miss 가 약 23개뿐. ToolBench(OpenBMB·SambaNova)도 검토했으나 질의가 RapidAPI 전용·호출 여러 번이라 positive 와 near-miss 가 거의 없어 제외 |
| 2026-11-18 | MASSIVE 후보를 "관련 의도만" → "test 파티션 전체" 로 바꿈 (3.1) | 의도 라벨에 잡음이 있어 다른 의도 속 positive 를 놓치고, 잡담·계산 negative 가 빠지기 때문. 모델 결과를 보기 전 결정 |
| 2026-11-18 | MASSIVE test 확정 (465개), 지표 스크립트 `src/refusal_metrics.py`, Colab 노트북 08 추가 | 3.1 실행. 라벨 규칙은 MASSIVE_TEST.md 2절, 모델 결과를 보기 전에 정함 |
| 2026-11-18 | **기준 모델(ours = A50·B0, 공개 모델)의 MASSIVE 결과를 봤다** (MASSIVE_TEST.md 결과 절) | 7절 5단계보다 먼저 채점함. 이후 실험 A·B 의 데이터·설정을 이 결과로 바꾸지 않는다. dev 문턱값이 MASSIVE 로 옮겨지지 않는 문제가 보였지만 6절 주 지표는 아직 바꾸지 않음 |
| 2026-11-22 | 실험 A00 추가 (4.1절): negative 0% + MSP 거절. 예측을 돌리기 전에 기록 | 연구 방향 논의 중 나온 아이디어. MASSIVE 결과를 본 뒤의 추가라서, MASSIVE·BFCL 채점은 7절 5단계까지 미룬다 |
