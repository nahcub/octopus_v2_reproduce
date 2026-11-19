# MASSIVE 기반 두 번째 독립 테스트셋 설명서 (논문에 없는 추가)

BFCL 셋(`BFCL_TEST.md`)과 **다른 출처**에서, 실험 설계(`EXPERIMENT_DESIGN.md`)를 정한 **뒤에**, 어떤 모델 결과도 보기 **전에** 만든 test 셋.
BFCL 은 글로 쓴 LLM 식 질의, MASSIVE 는 말로 한 음성 비서 질의라 서로 보완된다. 특히 near-miss(같은 영역인데 목록에 없는 요청)를 사람이 쓴 문장으로 충분히 모을 수 있다.

## 1. 출처
- **MASSIVE 1.1** (Amazon, `alexa/massive`), 라이선스 **CC BY 4.0** (받은 `LICENSE` 파일에서 확인). 영어 `en-US.jsonl` 만, 영어 부분은 SLURP 문장.
- **test 파티션만** 쓴다 (2,974개). 우리는 MASSIVE 로 학습하지 않지만 원래 나눔을 존중.
- **질의 문장은 고치지 않는다** (소문자, 글자로 쓴 숫자, 호출어 "alexa"·"olly" 그대로). MASSIVE 의도(intent) 라벨은 잡음이 있어서(예: `audio_volume_up` 에 "where is the car") 정답으로 쓰지 않고 후보 표시로만 썼다.

## 2. 정답 기준
BFCL_TEST.md 3절의 기준과 "라벨링하면서 추가로 정한 규칙"을 그대로 쓰고, MASSIVE 에서 새로 나온 경우는 아래처럼 정했다 (모델 결과를 보기 전).

| 경우 | 예 | 정답 |
|---|---|---|
| 일정·회의를 달력에 넣기 | "schedule a meeting with matt at eleven am on thursday" | 일정 호출. 날짜가 상대값이고 끝 시간이 없어 인자 채점 제외 |
| 날짜 없이 시각·상대 시간만 있는 "remind me" | "remind me to take my medicine at nine am", "remind me after ten minutes" | 알람 호출, 인자 채점 제외 (BFCL "상대 시간 알람" 규칙과 같음) |
| 그 밖의 리마인더 (날짜·반복·일정 앞 알림·내용 없음) | "remind me the day before my car is due" | **제외** (알람/일정 둘 다 가능) |
| 일정 조회·삭제, 초대·참석자, 반복 설정 | "cancel next meeting", "invite people to the calendar event" | 거절 (`near_miss`) |
| 알람 조회·끄기·삭제 | "what alarms are set", "turn off my alarm" | 거절 (`near_miss`) |
| 볼륨 올리기·내리기·음소거 (단계 없음) | "turn up the volume", "mute please" | 볼륨 호출, 인자 채점 제외 |
| "말하지 마", "조용히" 류 | "don't talk until i ask you to", "quiet" | **제외** (볼륨 0 / 방해 금지 / 대화 중단 모두 가능) |
| 새 메일 쓰기·보내기 | "send an email to john asking ..." | 메일 호출. 주소·제목·본문이 다 있지 않아 인자 채점 제외 |
| 답장·회신 | "reply to john's email" | 거절 (`near_miss`, 받은 메일이 필요) |
| 연락처에 이메일 추가 | "add this email to my contacts" | 거절 (`near_miss`, 연락처 함수는 전화번호) |
| 뉴스 조회 | "latest news on terrorism" | 뉴스 호출. 주제가 분명하면 인자 채점 (`query`, `language='en'`), 매체만 있거나 주제가 없으면 제외 |
| 뉴스 알림·구독 | "set notification for news about ..." | 거절 (`near_miss`) |
| 재생 제어 (다시·다음·이전 곡) | "play it again", "next song" | 거절 (`near_miss`) |
| 다음 곡으로 예약 | "play atmosphere next" | **제외** |
| 오디오북·팟캐스트·라디오 | "play the young turks podcast" | 거절 (`near_miss`, BFCL 팟캐스트 규칙과 같음) |
| 조명·플러그·청소기·커피머신 | "dim the lights" | 거절 (`near_miss`). "screen brightness" 를 명시하면 화면 밝기 호출, 화면/조명이 애매하면 제외 |
| 길 안내·경로·이동 시간 | "directions to disney land", "driving time to los angeles" | 경로 호출, 출발지가 없어 인자 채점 제외 |
| 기차 시간표·표 예매·택시 | "when is the next train to glasgow" | 거절 (`unsupported_service`) |
| 우산·옷차림처럼 날씨에 달린 질문 | "should i bring an umbrella tomorrow" | 날씨 호출. 활동 조언("should i open the sun roof")·옷 추천은 제외 |
| 음식 주문·SNS·게임·환율·주가·장소/행사 찾기 | | 거절 (`unsupported_service`) |
| 시간·날짜·정의·상식·레시피 방법·농담 | | 거절 (`chitchat_knowledge`). 레시피를 "찾아/검색해 줘"는 `unsupported_service` |
| 계산 | "what is twelve divided by four" | 거절 (`calc_science`) |

인자 정답
- 글자로 쓴 시각은 24시간 "HH:MM" 으로 ("six thirty am" → "06:30", "twelve p. m." → "12:00"). 오전/오후가 불분명하면 인자 채점 제외.
- 장소는 MASSIVE 칸 표시(`[place_name : ...]`)를 근거로 대표값을 정하고, 표기 차이를 `args_allowed` 에 넣는다.
- MASSIVE 질의는 전부 소문자라서, 문자열 인자마다 **소문자 표기를 허용 값에 자동으로 더한다** (`src/massive_labels.py`). (논문에 없는 추가)

## 3. 선정 (BFCL 과 같은 코드)
`src/bfcl_bench.py` 의 `select` 그대로: positive 는 전부, 함수당 최대 30개(넘으면 seed 0 무작위). negative 는 positive 와 같은 수, 유형별 같은 수(모자란 유형은 전부, 남는 몫은 나머지에 고르게), seed 0.

## 4. 검토
사용자 검토 대신 Claude 가 최종 셋을 전부 다시 읽었다 (사람 검토는 아직 안 함).
- 1차 초안 466개에서 애매한 7개를 라벨 원본에서 **제외**로 바꾸고 다시 뽑았다: "tell me about trump", "i want to know about the brexit", "what's been happening for the last two months" (뉴스/상식 애매), "give me some good sounds", "play youtube playlist blank" (음악/영상 애매), "make it brighter in here" (화면/조명 애매), "color of light should be dimmed bright" (의미 불명).
- 다시 뽑으며 새로 들어온 93개를 검토해 1개("i have my clothes list", 요청이 아님)를 `review.csv` 에서 제외 → 465개.
- 모델 결과는 어느 단계에서도 보지 않았다.

## 5. 숫자
| 항목 | 값 |
|---|---|
| test 파티션 | 2,974 → 같은 문장 4개 제거 → 후보 **2,970** |
| 라벨: positive / negative / 제외 | 615 / 2,036 / 319 |
| 최종 셋: positive / negative | **233 / 232 (전체 465)** |
| `score_args: true` positive | 34 |
| `seen_in_train: true` | 0 (우리 train·test 와 겹치는 문장 없음. BFCL 후보와는 2개 겹침: "what is your name", "what time is it") |
| 태그 | `wake_word` 21 |

positive 함수별 (최종 / 라벨 전체): 날씨 30/137, 음악 30/148, 뉴스 30/91, 이메일 30/67, 일정 30/60, 알람 30/55, 볼륨 30/34, 경로 18/18, 전화 2/2, YouTube 검색 2/2, 화면 밝기 1/1. 나머지 9개 함수는 0개 (MASSIVE 에 해당 의도 없음).

negative 유형별 (최종 / 라벨 전체): chitchat_knowledge 70/545, unsupported_service 70/449, near_miss 68/1,018, calc_science 23/23, coding_tech 1/1

positive 말투: 명령형 178, 질문형 55

## 6. 파일
| 파일 | 내용 | git |
|---|---|---|
| `src/massive_bench.py` | 내려받기 → 후보 추출 / build / finalize | O |
| `src/massive_labels.py` | 라벨 원본 → `labels.jsonl` (BFCL 형식 + `D <의도> <유형>` 줄) | O |
| `src/refusal_metrics.py` | AUROC, 거절@recall90, 유형별 거절, 짝지은 bootstrap 95% 구간 (BFCL·MASSIVE 공용) | O |
| `notebooks/08_eval_massive.ipynb` | Colab: 모델별 생성·확률·채점 → 지표 | O |
| `data/massive_test.jsonl` | 확정된 test 셋 465개 | O |
| `data/raw/massive/` | 원본 압축·`en-US.jsonl`·`LICENSE`·`candidates.jsonl`·`labels_src/`·`labels.jsonl`·초안·`review.csv` | X |

라벨 원본의 `D <의도> <유형>` 줄은 그 의도의 문장을 **전부 읽고** 예외를 따로 적은 뒤, 나머지를 한 유형으로 묶은 것이다 (예: `D alarm_query nm`).

## 결과 (우리 모델·공개 모델, 2026-11-18)
Colab `notebooks/08_eval_massive.ipynb` (L4 GPU, 커밋 8a00257). 원본 Drive `runs/eval_massive/`, 랩탑 복사본 `data/raw/eval_massive/`. 괄호는 bootstrap 95% 구간.

| | 우리 모델 | 공개 모델 |
|---|---|---|
| positive 함수 선택 (233, 첫 함수 토큰 기준) | 76.4% | **88.0%** |
| negative 거절 (232, 첫 함수 토큰 기준) | **85.8%** | 34.5% |
| 기본 생성 균형 정확도 | **0.811** | 0.612 |
| 주된 실패 | 놓친 호출 49 | 잘못된 호출 142 |
| AUROC (`<nexa_20>` 확률) | **0.901** (0.874~0.926) | 0.842 (0.807~0.878) |
| 차이 (공개 − 우리, 짝지은 bootstrap) | | −0.058 (−0.104~−0.015) |
| 문턱값을 MASSIVE 에서 최적으로 골랐을 때 균형 정확도 (낙관적) | 0.822 | 0.770 |
| [낙관적] MASSIVE positive recall 90% 문턱에서 거절 | **68.1%** | 48.3% (차이 95% +0.04~+0.31) |
| 같은 조건 near-miss 거절 | **69%** | 41% (차이 95% +0.07~+0.43) |

- 첫 함수 토큰 기준: 채점기는 `<nexa_20>('london')` 처럼 거절하면서 인자를 붙인 출력(우리 7개)과, 문자열 안 작은따옴표(`'king's park'`) 때문에 읽을 수 없는 출력(공개 positive 17개 중 12개는 함수가 맞음)을 틀린 것으로 센다. 거절 판단만 보려고 `probs.jsonl` 의 처음 고른 함수 토큰으로 다시 셌다.
- **dev 문턱값은 MASSIVE 로 옮겨지지 않았다.** `refuse@r`(dev 에서 recall 90% 문턱)은 우리 99.1% / 공개 91.4% 지만, 그때 MASSIVE positive recall 이 41.6% / 60.5% 로 무너진다. dev positive 의 `<nexa_20>` 확률 중앙값은 우리 0.0016, MASSIVE positive 는 0.011 (75% 지점 0.25). 말투가 달라서 positive 도 확률이 올라간다 (3.1 "주의" 의 우려가 그대로 나옴).

negative 유형별 거절 (첫 함수 토큰 기준이 아닌 채점기 기준)

| 유형 | 우리 | 공개 |
|---|---|---|
| near_miss (68) | 83.8% | **32.4%** |
| unsupported_service (70) | 77.1% | 57.1% |
| chitchat_knowledge (70) | 84.3% | 18.6% |
| calc_science (23) | 91.3% | 17.4% |

positive 함수 선택 (채점기 기준)

| 함수 | 우리 | 공개 |
|---|---|---|
| play_music_on_nest_hub (30) | **16.7%** | 96.7% |
| set_volume (30) | **63.3%** | 96.7% |
| get_trending_news (30) | 93.3% | **40.0%** |
| create_calendar_event (30) | 83.3% | 70.0% |
| get_weather_forecast (30) | 90.0% | 96.7% |
| send_email (30) | 93.3% | 86.7% |
| set_timer_alarm (30) | 86.7% | 93.3% |
| find_route_google_maps (18) | 77.8% | 83.3% |

### 해석
- **BFCL 의 "정반대 방향 실패"가 MASSIVE 에서도 재현된다.** 우리는 놓친 호출, 공개는 잘못된 호출. 공개 모델은 잡담·상식에 YouTube 검색(44번), 밝기(17번)를 부른다 (BFCL 에서도 YouTube 24번, 밝기 8번).
- **BFCL 과 반대로, 순위로 가르는 능력(AUROC)도 MASSIVE 에서는 우리가 낫다** (BFCL 우리 0.884 / 공개 0.925 → MASSIVE 0.901 / 0.842, 차이 구간이 0 을 넘음). 그래서 "공개 모델은 문턱값만 다시 잡으면 더 잘 가른다"는 BFCL 결론은 셋에 따라 달라진다.
- **우리 모델의 놓친 호출은 이번에도 특정 함수에 몰린다.** 음악 25/30 거절 (서비스 이름 없는 "play X"), 볼륨은 음소거 요청 11개를 전부 거절 ("mute please", "silence speakers"). 우리 train positive 의 말투가 좁다는 가설과 맞는다 (추정, 통제 실험 전).
- **공개 모델은 near-miss 를 확신하며 틀린다.** 틀린 near-miss 에서 고른 함수 확률 중앙값 0.86 (BFCL 0.97). 알람 조회·일정 조회·메일 조회·음악 곡 정보는 거의 다 함수를 부른다.
- 공개 모델의 뉴스 40% 는 거절 문제가 아니라 인자 형식 문제가 크다 (`language` 를 빼거나 날짜 인자를 붙임, YouTube 검색으로 보냄).

## 7. 다시 만들기
```
python -m src.massive_bench candidates
python -m src.massive_labels
python -m src.massive_bench build
python -m src.massive_bench finalize
```
