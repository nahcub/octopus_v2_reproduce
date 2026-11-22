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

**결과** (2026-11-22, 노트북 09 커밋 `37f823e`, seed 0 한 번. 숫자는 랩탑에서 `data/raw/eval*/` 로 다시 계산)

| 예측 | 결과 | 판정 |
|---|---|---|
| greedy negative 0%, positive 는 A50 과 비슷하거나 조금 높다 | negative 0%. positive dev 89.8% (A50 88.8%), 공식 210 75.5% (A50 67.5%) | 맞음 |
| dev AUROC(MSP) < A50 `<nexa_20>` AUROC | 0.991 vs 1.000, 차이 −0.009 [−0.016, −0.004] | 방향은 맞지만 차이가 아주 작다 |
| 잡담은 MSP 로 걸러지고 near-miss 는 호출 | dev 문턱값 뒤에도 호출로 남은 negative 13개 중 12개가 near-miss (연락처 수정·삭제 → create_contact, 손전등 밝기 → 화면 밝기, TV 볼륨 등), 1개는 "volume" 단어만 겹친 수학 질문. 분류는 문항을 읽고 정함 | 대체로 맞음 |
| 공식 210 에서는 차이가 dev 보다 작다 | AUROC 0.998 vs 0.994, 차이 +0.004 (구간에 0 포함) | 맞음. 단 공식 negative 10개라 근거 약함 |

- **공식 210 positive 를 거절한 수**: greedy 로는 A50 17/200, A00 0/200 (A00 은 그 17개 중 12개를 인자까지 정답). dev 문턱값을 옮겨 쓰면 A50 65/200 (`<nexa_20>` 17 + 문턱 48), A00 42/200. 줄어들지만 없어지지 않는다. negative 학습이 낯선 말투 positive 까지 거절하게 만든다는 가설과 같은 방향 (추정)
- **확신도**: 맞힌 dev positive 의 함수 확률 중앙값 A00 0.740, A50 0.955, 공개 모델 1.000. A00 이 덜 과신한다
- **문턱값 이식**: dev 문턱값(MSP 0.364)을 공식 210 에 쓰면 A00 positive 60% 만 통과 (A50 51%). MASSIVE 에서 본 이식 실패와 같다
- **함수 이름 지어내기**: A00 은 dev negative 24개에서 함수 토큰 대신 목록에 없는 이름을 지어 `botanical('...')` 처럼 출력했다. 그 단계 함수 토큰 최대 확률 0.03~0.10
- **결과를 본 뒤 바꾼 규칙**: 위 24개를 처음엔 MSP 점수 0(가장 확신한 호출)으로 셌다 (dev AUROC 0.931). 그 단계 상위 5개의 함수 토큰 확률로 세도록 `refusal_metrics.refusal_scores` 를 고쳤다 (0.991). 10절 변경 기록 참고
- 결과 정리 아티팩트: https://claude.ai/artifact/514GReWrmTpqXKKBCxUQW5 (비공개)

### 4.2 OE: negative 를 "어느 함수도 아님"으로 학습 (2026-11-22 추가, 논문에 없는 추가)

| 모델 | positive | negative | negative 정답 | 거절 방법 |
|---|---|---|---|---|
| A50 = ours | 2,000 | 2,000 | `<nexa_20>` (21번째 함수처럼) | `<nexa_20>` 출력 또는 그 확률 |
| A00 | 2,000 | 0 | 없음 | MSP 문턱값 |
| **OE** | 2,000 | 2,000 (A50 과 같은 문장) | 없음. 함수 토큰 20개 확률이 **고를수록** loss 가 작다 | MSP 문턱값 |

- 묻는 것 (매핑 가설): A50 은 거절을 "학습 negative 에 나온 종류의 요청 → `<nexa_20>`" 의 매핑으로 외운다 (추정).
  거절을 클래스로 주지 않고 "어느 함수에도 확신하지 않음" 으로 가르치면, 외울 거절 클래스가 없어서 함수 쪽 경계를 배울까?
- 방법은 Outlier Exposure (Hendrycks, Mazeika & Dietterich, ICLR 2019) 그대로. 함수 토큰 모델에 적용한 선행 연구는 아직 못 찾음 (추정, 확인 필요)
- loss: positive 는 그대로, negative 는 함수 토큰 자리에서 20개만 놓고 softmax 한 확률과 균등분포의 cross-entropy. 전체 = positive + 0.5 × OE (원 논문 λ)
- 바뀌는 건 negative loss 하나. 데이터·step 수(750)·나머지 설정은 A50 과 같다
- 코드: `src/train.py --neg-loss oe` (`OETrainer`, `oe_loss`), `src/preprocess/masking.build_example(oe=True)`. 노트북 `notebooks/10_exp_OE.ipynb`
- 채점: 우리 test(dev)·공식 210. MASSIVE·BFCL 은 7절 5단계 규칙대로 나중에

**예측** (돌리기 전에 적어 둠)
- greedy negative 정확도 ≈ 0% (`<nexa_20>` 을 배우지 않음). greedy positive 는 A50 과 비슷
- dev AUROC(MSP): A00(0.991) 보다 높고 A50 `<nexa_20>`(1.000) 에 가깝다. 같은 생성기 negative 를 직접 봤기 때문
- dev 문턱값 뒤 남는 실수는 A00 처럼 near-miss 에 몰리지만 수는 A00(13개) 보다 적다
- 공식 210 에 dev 문턱값을 옮겨 쓰면 거절하는 positive 수는 A00(42) 과 A50(65) 사이. 거절 클래스를 외우지 않으니 낯선 말투 positive 를 negative 쪽으로 덜 끌어당긴다 (가장 불확실한 예측)
- 문턱값 이식 문제는 남는다 (확률 하나로 거절하는 방식의 공통 한계)
- 매핑 가설 자체의 검증은 이 실험이 아니라 범주 하나를 뺀 학습 (A50 vs OE, 빠진 범주를 거절하는지) 으로 한다. 이 실험은 그 전 단계로 OE 가 같은 데이터에서 제대로 학습되는지 본다

**결과** (2026-11-22, 노트북 10, seed 0 한 번, A100 80GB, 학습 447초. 숫자는 랩탑에서 `data/raw/eval*/OE/` 로 다시 계산)

| 예측 | 결과 | 판정 |
|---|---|---|
| greedy negative ≈ 0%, positive 는 A50 과 비슷 | negative 0%, positive dev 89.0% (A50 88.8%) | 맞음 |
| dev AUROC: A00 보다 높고 A50 에 가깝다 | 0.9999 (A50 1.000, A00 0.991). OE − A50 차이 0 | 맞음 |
| 문턱값 뒤 남는 실수가 A00(13개) 보다 적다 | 0개. dev negative 400개 전부 문턱값으로 거절 (A00 이 놓친 phone_settings 7·smart_home 5 포함) | 맞음. 단 같은 생성기 negative |
| 공식 210 에서 거절하는 positive 가 A00(42) 과 A50(65) 사이 | 62/200. 숫자는 사이지만 A50 과 차이 +3 (pos_recall 차이 +0.015, 구간 [−0.065, +0.090]) | 숫자만 맞고 근거("거절 클래스를 안 외우니 덜 끌어당긴다")는 지지 안 됨 |
| 문턱값 이식 문제는 남는다 | 공식 positive 통과 69% (dev 90%) | 맞음 |

- **OE loss**: 처음부터 끝까지 2.996 ≈ log 20. negative 에서 함수 토큰 20개가 처음부터 고르고 학습 내내 고르게 유지됐다 (positive loss 는 1.17 → 0.005)
- **확신도**: dev 함수 확률 중앙값 positive 0.762 (A50 0.954, A00 0.734), negative 0.048 (최대 0.159). OE 는 A00 처럼 positive 에도 덜 확신하고, negative 와는 dev 에서 완전히 갈린다
- **공식 210 에서 거절한 positive 가 누구와 겹치나**: OE 62개 중 38개가 A00 이 거절한 42개 안에 있다 (A00 거절의 90%). A50(65개)과는 32개만 겹친다. A50 만 거절한 positive 31개, OE 만 9개.
  OE·A00 이 거절하고 A50 은 통과시킨 예: `'2023-09-05-19-00'` 같은 기계식 날짜가 든 캘린더 요청, "level 6" 밝기 요청. 우리 데이터 생성에서 금지한 표기 (README 2단계)
- **해석 (추정)**: 거절 점수가 "아는 함수와 얼마나 닮았나"(OE·A00, positive 쪽 경계)인 모델은 **낯선 말투 positive** 를 거절하고, `<nexa_20>` 모델(A50)은 그 일부와 **다른 positive** 를 거절한다.
  과잉 거절의 총량은 비슷하지만 원인이 다를 수 있다
- **A50 만 거절한 31개가 학습 negative 와 닮았나** (2026-11-22, 결과를 본 뒤의 분석, `src/neg_similarity.py`, 출력 `data/raw/eval_official/neg_similarity.txt`)
  - 유사도: 각 질의와 가장 닮은 train negative / train positive 의 코사인 유사도. 의미는 bge-small-en-v1.5 임베딩, 단어는 TF-IDF

  | 공식 positive 그룹 | n | 가장 닮은 negative (의미) | 가장 닮은 positive (의미) | negative (단어) |
  |---|---|---|---|---|
  | A50 만 거절 | 31 | 0.773 | 0.871 | 0.141 |
  | OE·A00 만 거절 | 32 | 0.736 | 0.846 | 0.112 |
  | 셋 다 거절 | 17 | 0.692 | 0.768 | 0.128 |
  | 아무도 거절 안 함 | 103 | 0.720 | 0.849 | 0.117 |

  - A50 만 거절한 positive 는 아무도 거절 안 한 positive 보다 train negative 를 더 닮았다 (의미 p = 0.014, 단어 p = 0.037, 한쪽 Mann-Whitney). 단 positive 유사도를 빼면 (negative − positive) 차이가 약해진다 (p = 0.076)
  - 함수가 몰려 있다: 문 잠금 7/10, 화면 밝기 6/10, 방해 금지 6/10. 가장 닮은 train negative 가 형제 같은 near-miss 다 (문 잠금 ↔ 차고 문, 밝기 ↔ 손전등 밝기·다크 모드, 방해 금지 ↔ 비행기 모드·다크 모드)
  - 다만 이 31개는 A50 이 greedy 로는 호출했고, `<nexa_20>` 확률이 낮은 문턱값(0.0046)을 살짝 넘은 것이다 (중앙값 0.010). 강하게 거절한 17개(중앙값 0.93)는 "셋 다 거절" 그룹에 있고 negative·positive 모두와 가장 덜 닮았다 = 말투가 낯선 것
  - 전체 positive 200개에서 A50 의 `<nexa_20>` 확률과 negative 유사도의 순위상관은 −0.19 (p = 0.006): 강한 거절은 negative 를 닮아서가 아니라 모든 학습 데이터와 멀어서 생긴다
  - **해석 (추정)**: 매핑과 맞는 약한 증거. 형제 near-miss negative 가 있는 함수에서 `<nexa_20>` 확률이 조금씩 끌려 올라간다. 하지만 과잉 거절의 큰 몫은 낯선 말투이고, 이건 거절 방식과 상관없이 세 모델이 공유한다. n = 31, seed 1개, 임베딩 하나라 근거는 약하다. 직접 검증은 실험 X (4.3절)
- 공식 negative 는 10개라 공식 AUROC 비교는 근거가 약하다. 매핑 가설 검증은 4.3절 실험 X 로

### 4.3 실험 X: negative 범주 하나 빼고 학습 — 매핑 가설 확인 (2026-11-22 추가, 논문에 없는 추가)

- **매핑 가설**: `<nexa_20>` 으로 학습한 모델은 "함수 목록에 없음" 이라는 개념이 아니라 "학습 negative 에 나온 종류의 요청 → `<nexa_20>`" 을 외운다 (추정).
  함수 토큰 모델은 추론 때 함수 목록을 입력으로 보지 않는다는 구조적 사실에서 나온 가설. 단 모델 안에는 20개 함수가 학습돼 있으므로 "20개 중 어디에도 안 맞음" 을 배울 수도 있다 → 실험으로 가른다
- negative 8개 범주 중 하나를 학습에서 뺀다. 범주 라벨: `data/neg_categories.json` (생성 기록 `negatives_v2` 에서 만듦, train 2,000·test 400 전부 매칭, test 범주마다 50개)

| 모델 | 뺀 범주 | negative 학습 |
|---|---|---|
| X_token_writing | writing (글쓰기·창작, 함수와 먼 범주) | `<nexa_20>` |
| X_oe_writing | writing | OE (4.2절) |
| X_token_smart_home | smart_home (지원 안 하는 스마트홈 기기, near-miss) | `<nexa_20>` |
| X_oe_smart_home | smart_home | OE |
| 비교: ours / OE | 없음 | 각 방식 |

- negative 1,750개 (250개 적음). positive·설정은 그대로
- 채점: 우리 test 의 범주별 negative 50개. 문턱값은 같은 test positive 90% 기준 (negative 를 안 씀). 스크립트 `src/negcat_report.py`
- 노트북 `notebooks/11_exp_X_mapping.ipynb`. seed 0 한 번
- 범주를 고른 근거 (기존 결과, 4.1절 이후 봄): 전체 학습 ours 는 우리 test 의 모든 범주를 49~50/50 greedy 거절. A00(negative 0개)도 문턱값으로 knowledge·math·smalltalk·advice·writing 49~50/50, phone_settings 43, smart_home 45.
  → 먼 범주는 매핑이 없어도 거절되므로 writing 은 "차이가 없어야 하는" 대조군, smart_home 이 시험대

**예측** (돌리기 전에 적어 둠)
- X_token_writing: 뺀 writing 을 greedy 로 여전히 대부분 거절 (≥ 40/50). 먼 범주는 다른 먼 범주(knowledge·smalltalk)의 매핑으로 덮인다
- X_token_smart_home: 뺀 smart_home 의 greedy 거절이 크게 떨어진다 (49 → 25 이하). 가장 비슷한 지원 함수(Nest 온도, 문 잠금, Nest Hub)를 부른다. **이게 나오면 매핑 가설을 뒷받침**
- X_oe_smart_home: 뺀 smart_home 의 문턱값 거절이 X_token_smart_home 의 문턱값 거절(`<nexa_20>` 확률 기준)보다 높다. 단 A00 수준(45/50) 을 넘을지는 모름
- 뺀 범주 말고 다른 범주와 positive 정확도는 거의 그대로
- 반대로 X_token_smart_home 이 smart_home 을 40/50 이상 거절하면 → 매핑보다 "20개 함수의 여집합" 쪽. 그대로 보고한다

**결과** (2026-11-22, 노트북 11, seed 0, 우리 test 범주별 50개. 표 `runs/eval/X_negcat_report.txt`)

| 범주 | ours | X_token_writing | X_token_smart_home | OE | X_oe_writing | X_oe_smart_home |
|---|---|---|---|---|---|---|
| writing (greedy / @t, AUROC) | 50 / 50, 1.000 | **\*50 / 50, 1.000** | 50 / 50, 1.000 | 0 / 50, 1.000 | **\*0 / 50, 1.000** | 0 / 50, 1.000 |
| smart_home | 49 / 50, 1.000 | 50 / 50, 1.000 | **\*45 / 50, 0.998** | 0 / 50, 1.000 | 0 / 50, 1.000 | **\*0 / 48, 0.977** |
| 나머지 6개 범주 | 전부 50 / 50 | 〃 | 〃 | 〃 | 〃 | 〃 |
| 전체 positive 정확도 | 88.75% | 90.00% | 87.75% | | 86.75% | 89.00% |

(\* = 학습에서 뺀 범주. OE 계열 greedy 는 원래 0. @t = positive 90% 통과 문턱값)

- X_token_writing: 예측대로 (50/50). 먼 범주는 빼도 그대로 거절
- **X_token_smart_home: 예측 빗나감.** 45/50 greedy 거절 (예측은 25 이하). 미리 정한 기준(40 이상이면 "20개 함수의 여집합" 쪽)에 들어간다.
  ours 49/50 과의 차이 4개는 유의하지 않다 (Fisher p = 0.20). 문턱값 기준으로는 50/50
- 놓친 5개는 모두 지원 함수와 거의 같은 요청: Samsung TV 볼륨 25 → `set_volume`, 현관 초인종 카메라 영상 3개 → `play_video_on_nest_hub`·`take_a_photo`, 차고 문 자동 닫기 → `open_or_lock_door`.
  그 외 45개(다른 가전·조명·로봇청소기 등)는 학습에서 이 범주를 본 적이 없어도 p_irr 중앙값 0.989 로 거절
- **X_oe_smart_home: 예측 빗나감.** 문턱값 거절 48/50, AUROC 0.977 로 오히려 token(50/50, 0.998)보다 약간 낮다. 놓친 2개는 token 과 같은 TV 볼륨·차고 문
- positive 정확도는 범주를 빼도 거의 그대로 (87.75~90.00%)

**해석** (추정)
- 강한 매핑 가설("학습한 negative 종류만 거절한다")은 우리 test 에서는 **지지되지 않는다.** 한 범주를 통째로 빼도 그 범주를 대부분 거절한다
- 빼서 생긴 손해는 지원 함수와 거의 겹치는 요청(기기만 다르고 동작·인자가 같음) 몇 개에 몰린다. 이런 요청은 OE 로도 못 막는다 → 경계가 "종류 목록" 이라기보다 함수 쪽 경계에 가깝고, 그 경계 바로 바깥만 negative 예시가 필요하다
- 다만 우리 test 는 학습과 같은 생성기라, 뺀 범주라도 positive 와 말투가 다르다 (A00 도 smart_home 45/50 거절). 즉 이 결과는 "범주 매핑이 아니다" 까지만 말하고,
  공식 test 에서 ours 가 과잉 거절한 원인(낯선 말투, 4.2절 유사도 분석)과 함께 보면 **모델이 배운 거절 경계는 "학습 positive 의 분포 밖" 에 가깝다** 는 쪽이 더 맞아 보인다. 이것도 추정이며 seed 1번, 범주 2개만 봄

### 4.4 실험 S: 말투를 바꿔 쓴 데이터 — 거절이 말투에 기대는가 (2026-11-22 추가, 논문에 없는 추가)

- **배경**: 실험 X(4.3절)에서 negative 범주를 통째로 빼도 거의 다 거절했다. 공식 test 에서는 낯선 말투의 positive 를 거절했고(4.2절 유사도 분석), 공개 모델은 우리 negative 를 39% 만 거절했다.
  → 두 해석이 남는다 (둘 다 추정)
  - **일반화 가설**: 모델은 "20개 함수가 아닌 것" 을 배웠고, 학습 말투가 좁아서 처음 보는 말투에서만 흔들린다. 말투를 넓혀 학습하면 **처음 보는 말투에도** 나아진다
  - **말투 암기 가설** (사용자 예측): 모델은 의미보다 말투 규칙성을 힌트로 쓴다. 말투를 넓혀도 **학습에 넣은 말투에서만** 나아지고, 처음 보는 말투에서는 그대로 실패한다
- **방법**: 같은 문장을 내용(함수·인자)은 그대로 두고 말투만 바꿔 다시 쓴다 (Gemini). positive 와 negative 를 **같은 말투들로** 바꾸므로 말투만으로는 positive/negative 를 가를 수 없다
- **말투 축 5개** (기존 생성 조건 `prompts.STYLES` 6개 — 짧은 명령·공손한 문장·구어·맥락 포함·Can you 질문·소문자 타이핑 — 에 없는 것으로 고름)

| 이름 | 말투 | 예 ("Call Mom at 6 PM") |
|---|---|---|
| howto | 방법·정보를 묻는 형태 | How do I get my phone to ring Mom at 6? |
| fragment | 동사 없는 명사구·메모 | Mom, call, 6 PM |
| need | 필요·상황만 말하고 시키지 않음 | I really need to talk to my mom around six. |
| form | 키-값·양식, 기계식 값 표기 | action=call; who=Mom; time=18:00 |
| nonnative | 비원어민 영어 (문법·어순 어색) | Please you make call to mother at 6 PM time. |

  - howto·fragment·form 은 공식 test 오답(4.2절, README 오답 표)을 보고 떠올린 말투다 → **공식 test 는 이 실험에서 참고로만** 쓰고, 최종 판정은 말투를 정할 때 보지 않은 MASSIVE·BFCL 로 한다 (모든 모델이 확정된 뒤 한 번, 8절 규칙)
- **재작성 데이터**: train 4,000개 × 5 말투, 우리 test 에서 positive 200(함수당 10)·negative 200(범주당 25)를 seed 0 으로 뽑아 × 5 말투. 원래 말투를 orig 라 부른다
- **검증**: positive 는 바꿔 쓴 문장 + 원래 인자로 "이 호출이 요청을 정확히 수행하나" 를 Gemini 로 확인 (기존 ③ 검증에서 "자연스러운가" 조건만 뺌 — fragment·form 은 일부러 부자연스럽다). negative 는 기존 negative 검증 그대로 (20개 함수 중 처리할 수 있는 게 없어야 통과). 떨어진 문장은 그 말투에서 빠진다
- **학습 조건** (모두 4,000개, 같은 원문 4,000개를 쓰고 문장마다 말투 하나를 seed 0 으로 배정. positive·negative 가 같은 비율로 말투를 나눠 가짐. 그 말투 버전이 검증에서 떨어졌으면 orig)

| 모델 | 학습 말투 |
|---|---|
| ours (기존) | orig 만 |
| S_all | orig + 5개 전부 (6개 균등) |
| S_minus_howto | orig + howto 뺀 4개 (5개 균등) |
| S_minus_fragment | orig + fragment 뺀 4개 |

  - 뺄 말투는 돌리기 전에 정함: ours 가 가장 크게 실패할 것으로 예상하는 둘 (howto 는 우리 negative 에 질문이 많아서, fragment 는 공식 test 의 동사 없는 요청 오답에서)
  - 나머지 설정은 ours 와 같음 (`<nexa_20>` 방식, seed 0)
- **채점**: 우리 test 400개 × 6 말투(orig + 5). 말투마다 positive greedy 거절률, positive 정확도, negative greedy 거절률, AUROC (irr 점수)
- **판정 규칙** (뺀 말투 k 에서, positive greedy 거절률 기준):
  메운 비율 = (ours_k − S_minus_k) / (ours_k − S_all_k). **0.5 이상이면 일반화 가설, 0.5 미만이면 말투 암기 가설.** negative 거절률도 같은 식으로 계산해 같이 보고
  (ours_k − S_all_k 가 5%p 미만이면 그 말투는 애초에 어렵지 않았던 것 → 판정 안 함)

**예측** (돌리기 전에 적어 둠)
- ours: orig 에서는 positive 거절 거의 0. howto 에서 positive greedy 거절 15% 이상, fragment 5% 이상. negative 거절은 대부분 말투에서 90% 이상이지만 fragment·form 중 하나 이상에서 90% 아래로
- S_all: 모든 말투에서 positive 거절 3% 이하, negative 거절 95% 이상. orig 정확도는 ours 와 2%p 이내
- S_minus_k (k 에서): Claude 예측은 일반화 (메운 비율 ≥ 0.5). 사용자 예측은 말투 암기 (< 0.5). **어느 쪽이든 그대로 보고**
- 한계: 바꿔 쓴 말투도 전부 Gemini 가 쓴 것이라 생성기는 하나다. 그래서 MASSIVE(사람이 쓴 문장)로 확인하는 것이 2차 결과

**데이터 생성 기록** (2026-11-23, 모델 결과 보기 전)
- train 재작성 검증 통과: positive howto 98.8%·fragment 97.8%·need 99.1%·form 96.3%·nonnative 98.7%, negative 99.8% 이상. 떨어진 문장은 그 말투 대신 원문(orig)으로 들어가 orig 가 조금 많다 (S_all positive orig 367, 나머지 316~332)
- dev: 400개 중 5개 말투 모두 통과한 **390개(positive 191, negative 199)** × 6 = 2,340개로 채점 (말투끼리 같은 원문 비교)
- dev 문장 2,340개 중 41개가 train 문장과 같다 (대부분 orig, 원래 test 의 seen_in_train 과 같은 성격). 빼지 않음
- 소규모 실행에서 고친 것: form 말투가 `action: take_photo` 처럼 함수 이름을 쓰는 지름길 → 함수 이름·snake_case 금지. howto 의 질문형 negative 는 원문이 이미 그 말투라 그대로 둠

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
| 공개 모델 | 56.8% / 39.2% | 0.944 | | | | | | |
| A00 (MSP) | 89.8% / 0% (greedy) | 0.991 | | | | | | |
| A05 | | | | | | | | |
| A20 | | | | | | | | |
| OE (MSP) | 89.0% / 0% (greedy) | 1.000 | | | | | | |
| A50 = B0 | 88.8% / 99.8% | 1.000 | | | | | | |
| B1 | | | | | | | | |
| B2 | | | | | | | | |

## 9. 아직 정하지 않은 것

- MASSIVE test 사람 검토 (Claude 검토만 끝남). 검토로 바뀌면 모델 결과를 보기 전에 반영
- recall 기준 90% 외에 다른 값(예: 공개 모델의 기본 recall)도 같이 보고할지
- B2 에서 편집 거리 기준값, 사람 검토 합격선 90% 가 적절한지 (소규모 실행 후 정함)
- 음악 함수 과잉 거절 관련: 현재 negative 범주 7번에 "music on the phone itself" 가 있고, train positive 음악 질의 98/100 이 서비스·플레이리스트를 명시한다.
  이건 negative 문제가 아니라 **positive 쪽 표현 다양성** 문제일 수 있다 (추정). 이번 실험에서는 positive 를 고정하므로 다루지 않고, 필요하면 별도 실험 C 로 둔다
- 실험 C 파일럿 (2026-11-22, 논문에 없는 추가): positive 를 시작 단어("Uhm," "Hmm," "You know," 등 8개)로 시작하게 생성한 뒤 잘라내기 (`generate.py --openers`).
  함수 3개 × 90개, 말투·후보·seed 같게 두고 비교. 문장 첫 단어 종류만 늘었고 (24 → 37), 단어 다양성·기존 train 과의 유사도는 거의 그대로 (0.336 → 0.325).
  말투 조건(STYLES)이 결과를 좌우하고 시작 단어는 앞부분만 바꾼다. 단독으로는 효과가 작음 → 실험 C 를 한다면 상황·페르소나처럼 내용을 바꾸는 쪽을 함께 검토.
  결과: `data/raw/opener_pilot_{base,fill}/`, 비교 `data/raw/opener_pilot_fill/compare.txt`

## 10. 변경 기록

| 날짜 | 바꾼 것 | 이유 |
|---|---|---|
| 2026-11-18 | 최초 작성 | |
| 2026-11-18 | 두 번째 독립 test 출처를 MASSIVE 로 정함 (3절·3.1·7절·8절, 9절 항목 정리) | BFCL 이 설계 전에 본 셋이라 완전히 깨끗하지 않고 near-miss 가 약 23개뿐. ToolBench(OpenBMB·SambaNova)도 검토했으나 질의가 RapidAPI 전용·호출 여러 번이라 positive 와 near-miss 가 거의 없어 제외 |
| 2026-11-18 | MASSIVE 후보를 "관련 의도만" → "test 파티션 전체" 로 바꿈 (3.1) | 의도 라벨에 잡음이 있어 다른 의도 속 positive 를 놓치고, 잡담·계산 negative 가 빠지기 때문. 모델 결과를 보기 전 결정 |
| 2026-11-18 | MASSIVE test 확정 (465개), 지표 스크립트 `src/refusal_metrics.py`, Colab 노트북 08 추가 | 3.1 실행. 라벨 규칙은 MASSIVE_TEST.md 2절, 모델 결과를 보기 전에 정함 |
| 2026-11-18 | **기준 모델(ours = A50·B0, 공개 모델)의 MASSIVE 결과를 봤다** (MASSIVE_TEST.md 결과 절) | 7절 5단계보다 먼저 채점함. 이후 실험 A·B 의 데이터·설정을 이 결과로 바꾸지 않는다. dev 문턱값이 MASSIVE 로 옮겨지지 않는 문제가 보였지만 6절 주 지표는 아직 바꾸지 않음 |
| 2026-11-22 | 실험 A00 추가 (4.1절): negative 0% + MSP 거절. 예측을 돌리기 전에 기록 | 연구 방향 논의 중 나온 아이디어. MASSIVE 결과를 본 뒤의 추가라서, MASSIVE·BFCL 채점은 7절 5단계까지 미룬다 |
| 2026-11-22 | **A00 결과를 본 뒤** MSP 점수 규칙 수정: 함수 토큰을 안 고른 출력은 0 대신 그 단계 상위 5개의 함수 토큰 확률로 계산 (`src/refusal_metrics.py`) | A00 이 negative 24개에서 함수 이름을 지어냈는데 0 이면 "가장 확신한 호출" 로 세져 점수의 뜻과 반대. irr 점수 규칙은 그대로. 수정 전 dev AUROC 0.931, 수정 후 0.991. 노트북 09 출력은 수정 전 숫자 |
| 2026-11-22 | 실험 OE 추가 (4.2절): negative 를 Outlier Exposure loss 로 학습 + MSP 거절. 예측을 돌리기 전에 기록 | 매핑 가설(거절을 학습 negative 종류의 매핑으로 외운다)에서 나온 처방. A00 결과를 본 뒤의 추가 |
| 2026-11-22 | 실험 X 추가 (4.3절): negative 범주 하나(writing / smart_home) 빼고 token·OE 학습. 범주 라벨 `data/neg_categories.json`, 채점 `src/negcat_report.py`. 예측을 돌리기 전에 기록 | 매핑 가설 직접 검증. 뺄 범주는 기존 ours·A00 의 범주별 결과를 보고 고름 (먼 범주 대조군 + near-miss 시험대) |
| 2026-11-22 | `OETrainer` 수정: `--grad-accum` 을 쓸 때 loss 를 누적 횟수로 나누게 (`model_accepts_loss_kwargs = False`). 노트북 10·11 은 A100 80GB 가 아니면 멈춤 | 고치기 전엔 누적 시 OE 모델의 실제 lr 이 누적 횟수배. 노트북 10 첫 실행은 40GB 런타임에서 OOM 으로 학습 전에 멈춰 결과에 영향 없음. 학습은 계속 batch 16 한 번에 (ours 와 같은 조건) |
| 2026-11-22 | OE 결과를 본 뒤 분석 추가: 공식 positive 의 거절 그룹별 train negative 유사도 (`src/neg_similarity.py`, 4.2절) | A50 만 거절한 positive 가 매핑으로 설명되는지 보려고. 사전에 계획한 분석이 아님 |
| 2026-11-22 | 실험 X 결과 기록 (4.3절). smart_home 을 뺀 token 모델이 그 범주를 45/50 거절해 예측(25 이하)이 빗나감. 놓친 5개 문항 분석은 결과를 본 뒤 추가 | 미리 정한 기준대로 "여집합 쪽" 으로 보고 |
| 2026-11-22 | 실험 S 추가 (4.4절): 같은 문장을 말투 5가지로 다시 써서 학습·채점. 예측과 판정 규칙을 돌리기 전에 기록 | 실험 X 결과를 본 뒤. 거절이 의미보다 말투에 기대는지 (사용자 예측) vs 말투를 넓히면 처음 보는 말투에도 일반화되는지 가르려고 |
