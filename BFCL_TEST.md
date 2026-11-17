# BFCL 기반 제3 테스트셋 설명서 (논문에 없는 추가)

우리 모델과 공개 모델(`NexaAIDev/Octopus-v2`)을 **둘 중 누구도 만들지 않은 질의**로 평가하기 위한 test 셋.
아직 확정되지 않은 숫자는 `__` 로 비워 두고, 만들면서 채운다.

## 1. 목적
- 기존 test(우리가 Gemini 로 생성)와 공식 평가셋(nexa 작성)은 각자 학습 데이터와 말투가 비슷해서, 각 모델이 자기 쪽 test 에서만 점수가 높다 (README "공식 평가셋으로 교차 평가").
- 제3자가 만든 질의(BFCL)로 **거절을 양방향으로** 잰다: 불러야 할 때 거절하는 실패(missed call)와, 거절해야 할 때 부르는 실패(false call).
- "20개 함수 중 무엇을 골랐나"를 고르게 재는 셋이 아니다. 함수별 문항 수가 고르지 않다 (BFCL 에 폰 비서 질의가 적음). 함수 선택 정확도는 기존 test·공식 평가셋이 맡는다.

## 2. 출처
- **BFCL v4** (Berkeley Function Calling Leaderboard, `ShishirPatil/gorilla` 저장소 `berkeley-function-call-leaderboard/bfcl_eval/data/`), 라이선스 Apache-2.0. **유일한 출처**.
- 쓰는 카테고리: `simple_python`, `multiple`, `live_simple`, `live_multiple`, `live_relevance`, `irrelevance`, `live_irrelevance`
- 안 쓰는 카테고리: parallel 계열(호출 여러 번), multi_turn 계열(여러 턴), java/javascript(코드 문법), web_search·memory·format_sensitivity(에이전트형)
- 조건: 턴 1개, 메시지 1개(사용자 메시지만). system 메시지가 붙은 문항은 우리 프롬프트에 자리가 없어서 뺀다.
- **질의 문장은 고치지 않는다.** BFCL 이 붙인 함수 목록과 정답은 쓰지 않는다 (BFCL 함수 기준이라 우리 함수와 다름). 정답은 아래 기준으로 우리가 새로 붙인다.

## 3. 정답 기준 (우리 함수 20개 + 거절)
각 질의에 다음 중 하나를 붙인다: 우리 함수 i 호출 / 거절(`<nexa_20>`) / 제외.

| 경우 | 예 | 정답 |
|---|---|---|
| 의도를 우리 함수로 실행할 수 있음 | "weather in Seoul" | 그 함수 호출 |
| 질문형 요청 | "How do I check the weather in Seoul?" | **호출** (공식 평가셋에 "How can I take a selfie with the front camera?" 가 호출 정답으로 있음) |
| 일부만 지원 (표현 못 하는 세부 요구) | "weather in Tel Aviv **in Fahrenheit**" | 호출. 표현 못 하는 부분은 무시 |
| 함수 의미가 조금 다름 | "current weather" (우리 함수는 예보) | 호출 |
| 필수 인자 정보가 질의에 없음 | "play some music" | 호출. 단 인자 채점 제외 (`score_args: false`) |
| BFCL 함수·특정 API 를 지목하거나 BFCL 환경을 전제 | "using the Geocoding API", "the drink id is 'latte'" | 우리 함수로 실행 불가면 거절 |
| 우리 함수에 없는 일 | 영화 예매, 주식 조회, 코드 작성 | 거절 |
| 판단 불가 (호출/거절 둘 다 말이 됨) | | **제외**, 사유 기록 |

판단은 함수 설명(`src/toy_api.py` 의 `DESCRIPTIONS`)에 적힌 내용만 근거로 한다.

### 라벨링하면서 추가로 정한 규칙
실제 질의를 보며 위 표로 정해지지 않는 경우가 나와서 아래처럼 통일했다 (라벨 원본의 `##` 메모에 경우마다 적어 둠).

| 경우 | 예 | 정답 |
|---|---|---|
| 두 장소 사이 경로·운전 거리·차로 걸리는 시간 | "driving distance between NYC and DC", "How long by car from Boston to NY" | 경로 함수 호출 (교통수단 언급 없으면 기본값 `driving`, 기차·대중교통이면 `transit`) |
| 교통수단 없이 "두 도시 사이 거리"만 | "distance between New York and Boston" | 제외 (경로 계산인지 지리 상식인지 애매) |
| 교통 상황·교통량 | "current traffic from Boston to NY" | 제외 (경로 결과에 포함되는지 설명에 없음) |
| 현재·미래 날씨, 기온, 습도, 강수 | "temperature in Riga now", "uk rainfall" | 날씨 함수 호출 |
| 과거 날씨, 월평균 기온, 대기질, 자외선 지수 | "average temperature for LA in December", "air quality index" | 거절 (`near_miss`) |
| 특정 곡·가수를 틀어 달라 / 듣고 싶다 | "play wrecking ball by Miley Cyrus" | 음악 함수 호출. 서비스가 거의 항상 빠져 있어 인자 채점 제외 |
| 노래를 찾아 달라·목록을 달라 (재생 요청 없음) | "find me Metal songs by Jinjer" | 거절 (`near_miss`) |
| 노래를 "검색"해 달라 / 듣고 싶다 + 찾아 달라가 섞임 | "Can you search for the song 'Shape of You'?" | 제외 (재생·YouTube 검색·거절 모두 가능) |
| 장소가 좌표로만 주어지거나 위치가 없음 | "weather at 37.86 N, 119.53 W", "weather for tomorrow" | 호출, 인자 채점 제외 |
| 볼륨을 0~10 밖 값(20, 70, 30%)으로 | "Set the volume to 70" | 호출, 인자 채점 제외 (설명의 범위 밖) |
| 상대 시간 알람·타이머 | "remind me in 5 mins" | 호출, 인자 채점 제외 |
| 일정 변경·조회, 할 일 목록, 설정된 알람 조회, 에어컨·공기청정기·세탁기 제어, 앱 실행 | "Move my session to next Friday" | 거절 (`near_miss`, 같은 영역이지만 목록에 없음) |
| 한 질의에 호출이 2번 필요 | "weather in Boston and San Francisco" | 제외 (우리 형식은 호출 1번) |
| 질의 안에 system 지시문·에이전트 프롬프트·함수 목록 JSON 이 섞임 | | 의도가 분명하면 그대로 판단, 아니면 제외 |
| 명령 프롬프트 명령을 지정한 요청 | "open camera, using the start command" | 거절 (`coding_tech`) |
| 한 단어·의미 없는 입력 | "hm", "Paris", "sdfsdf" | 거절 (`chitchat_knowledge`). 단 "on", "turn on", "3 degree" 처럼 기기 명령일 수도 있는 건 제외 |

### negative 유형
| 유형 | 뜻 |
|---|---|
| `chitchat_knowledge` | 잡담, 상식·사실 질문, 추천·조언 (애초에 명령이 아닌 말) |
| `calc_science` | 계산, 수학·과학 문제 |
| `coding_tech` | 코드 작성, 시스템·IT 작업 |
| `unsupported_service` | 무언가를 실행해 달라는 요청이지만 우리 목록에 없는 서비스 (예매, 결제, 검색 등) |
| `near_miss` | 우리 함수와 같은 영역인데 지원하지 않는 요청 (예: 날씨가 아닌 대기질, 음악이 아닌 팟캐스트 검색) |

### positive 말투
`imperative` (명령형) / `question` (질문형) 을 태그로 붙인다.

### 기타 태그
`non_english` (영어가 아닌 질의), `long` (300자 초과). 제외하지 않고 표시만 한다.

## 4. 인자 정답
- 1차 지표는 **함수 선택과 거절 여부**. 인자는 2차 지표.
- 정답이 분명한 positive 만 인자를 채점한다 (`score_args: true`). 기본값이 있는 인자는 함수 설명의 기본값으로 채운다 (기존 test 와 같음).
- `args` 는 대표값, `args_allowed` 는 인자별 허용 값 리스트 (BFCL 처럼 표기 차이를 정답으로 인정. 예: `"Tel Aviv"`, `"Tel Aviv, Israel"`). (논문에 없는 추가)
- 상대 날짜("내일")처럼 정답을 정할 수 없으면 `score_args: false`.

## 5. 선정 규칙 (모델 결과를 보기 전에 고정)
1. 후보 전부(2,706개)에 라벨을 붙인다.
2. positive: 조건에 맞는 것 **전부**. 단 함수당 최대 30개 (넘으면 seed 고정 무작위 추출).
3. negative: positive 와 **같은 수**. 유형별로 같은 수를 뽑는다 (어떤 유형이 모자라면 그 유형은 전부 쓰고 나머지를 다른 유형에 고르게 나눔). seed 고정.
4. 최종 셋을 사람이 전부 검토하고, 고친 내용을 반영해 확정한다.
- seed: `0`

## 6. 파일
| 파일 | 내용 | git |
|---|---|---|
| `src/bfcl_bench.py` | 내려받기 → 후보 추출 → (라벨 합치기) → 최종 셋 추출 → review.csv | O |
| `src/bfcl_labels.py` | 라벨 원본(압축 형식) → `labels.jsonl` | O |
| `data/raw/bfcl/candidates.jsonl` | 조건을 통과한 후보 (BFCL id, 카테고리, 질의, 키워드 표시) | X |
| `data/raw/bfcl/labels_src/*.txt` | 라벨 원본. 250개씩 11개 파일, 사람이 읽기 쉬운 압축 형식 (형식은 같은 폴더 `README.txt`) | X |
| `data/raw/bfcl/labels.jsonl` | 후보마다 라벨·유형·인자·판단 이유 | X |
| `data/raw/bfcl/bfcl_test_draft.jsonl` | 검토 전 최종 셋 | X |
| `data/raw/bfcl/review.csv` | 최종 셋 사람 검토용 | X |
| `data/bfcl_test.jsonl` | 확정된 test 셋 | O (검토 후) |

`data/bfcl_test.jsonl` 한 줄 형식 (기존 `data/test.jsonl` 에 필드 추가):
```json
{"kind": "positive", "function": "get_weather_forecast", "token_id": 2,
 "query": "...", "args": {"location": "Tel Aviv"}, "call": "<nexa_2>('Tel Aviv')<nexa_end>",
 "seen_in_train": false,
 "bfcl_id": "live_simple_12-3-0", "score_args": true,
 "args_allowed": {"location": ["Tel Aviv", "Tel Aviv, Israel"]},
 "style": "question", "neg_type": null, "tags": []}
```

## 7. 채점
- 기존 `src/evaluate.py` 로 생성·채점. `score_args: false` 문항은 함수 선택까지만 보고, `args_allowed` 가 있으면 그걸로 비교하도록 채점기를 보완한다 (기존 test 결과는 그대로 유지).
- 보고할 숫자: positive 정확도(함수 선택), negative 정확도(거절), negative 유형별·positive 말투별 정확도, 인자 정확도(`score_args` 문항만), `<nexa_20>` 확률로 계산한 AUROC, bootstrap 95% 신뢰구간.
- 대상: 우리 모델, 공개 모델. Colab 에서 실행.

## 8. 숫자 (검토 전 초안 기준. 검토 후 바뀌면 고친다)
| 항목 | 값 |
|---|---|
| 7개 카테고리 전체 문항 | 3,051 |
| 턴 1개·사용자 메시지 1개 후보 | 2,706 (메시지 여러 개 99개 제외, 카테고리 사이 같은 질의 246개 제거) |
| 라벨: positive / negative / 제외 | 250 / 2,419 / 37 |
| 최종 셋: positive / negative | **117 / 117 (전체 234)** |
| `score_args: true` positive | 62 |
| `seen_in_train: true` | 0 |
| 태그 | `non_english` 3, `long` 6 |

positive 함수별 (최종 / 라벨 전체). 9개 함수(연기 감지기, 전화, 연락처, 밝기, 방해 금지, Nest 온도, 영상 재생, 문, 사진 1개 제외)는 0~1개다.

| 함수 | 최종 / 전체 | 인자 채점 |
|---|---|---|
| get_weather_forecast | 30 / 152 (상한 적용) | 25 |
| play_music_on_nest_hub | 30 / 41 (상한 적용) | 2 |
| find_route_google_maps | 15 / 15 | 14 |
| set_timer_alarm | 15 / 15 | 9 |
| get_trending_news | 12 / 12 | 9 |
| set_volume | 4 / 4 | 0 |
| create_calendar_event | 3 / 3 | 0 |
| send_email | 3 / 3 | 1 |
| connect_to_bluetooth_device | 2 / 2 | 2 |
| send_text_message | 2 / 2 | 0 |
| take_a_photo | 1 / 1 | 0 |

negative 유형별 (최종 / 라벨 전체): chitchat_knowledge 24 / 452, calc_science 24 / 272, coding_tech 23 / 430, unsupported_service 23 / 1,151, near_miss 23 / 114

positive 말투: 질문형 62, 명령형 55

제외 37개 사유: 검색·찾기 요청이 섞여 호출/거절 애매 11, 지시문·함수 목록이 섞임 7, 두 도시 간 거리 4, 호출 2번 필요 4, 교통 상황 3, 기타(한 단어 기기 명령, 메모/알람 애매 등) 8

최종 셋 출처: live_multiple 86, live_irrelevance 51, live_simple 36, simple_python 35, irrelevance 17, multiple 8, live_relevance 1

## 9. 진행 기록
- 후보 추출: BFCL v4 7개 카테고리 내려받아 2,706개 후보 (`python -m src.bfcl_bench candidates`).
- 라벨링: Claude(이 대화의 Claude Code)가 후보 2,706개를 전부 읽고 라벨을 붙였다. 키워드 표시는 참고만 했고 정답 판단에는 쓰지 않았다.
  라벨 중 3절 표로 정해지지 않는 경우가 나와 "라벨링하면서 추가로 정한 규칙" 을 만들었다. 이 규칙은 모델 결과를 보기 전에 정했다.
- 최종 셋 추출 (`python -m src.bfcl_bench build`): 5절 규칙 그대로 → 234개. 처음 예상(약 300개)보다 적은 건 positive 가 날씨·음악 외에는 적고 함수당 30개 상한이 있어서다. 상한·비율은 바꾸지 않았다.
- 알아 둘 점
  - positive 의 절반(55/117)이 인자 채점 제외다. 대부분 음악(서비스 이름이 질의에 없음)이라 인자 정확도는 날씨·경로·알람·뉴스 위주로 계산된다.
  - BFCL `irrelevance`·`live_irrelevance` 에서 온 질의 중 우리 기준 positive 가 많다 (예: live_irrelevance 의 날씨 질의). BFCL 의 "관련 없음"은 그 문항에 붙은 함수 기준이라서다.

- 채점기 보완 (`src/evaluate.py`):
  - `score_args: false` 문항은 함수를 맞히면 결과 `function_only` 로 정답 처리. 인자 정확도에서는 뺀다.
  - `args_allowed` 가 있으면 인자마다 허용 값 중 하나와 같으면 정답 (엄격: 타입까지, 느슨: 대소문자·공백 무시).
  - 지표 추가: positive 함수 선택 정확도, negative 유형별 거절 정확도, positive 말투별 함수 선택 정확도.
  - 확인: 기존 test·공식 평가셋의 우리/공개 모델 예측 4개를 다시 채점해 `scored.jsonl`·summary 가 보완 전과 완전히 같음. selftest 에 BFCL 필드 사례 6개 추가, 통과.
  - 실행: `python -m src.evaluate score --preds <preds.jsonl> --test-file data/bfcl_test.jsonl`

- 생성 준비: 사용자 검토 전 초안(`bfcl_test_draft.jsonl`, 234개)을 Drive `MyDrive/octopus_v2/` 에 올리고 `notebooks/07_eval_bfcl.ipynb` 를 만들었다.
  Colab 에서 생성(`preds.jsonl`)·확률(`probs.jsonl`)·채점까지 한다.
  검토로 라벨만 바뀌면 생성은 다시 할 필요 없이 다시 채점만 하면 된다.

## 10. 다음 단계
1. 사용자 검토: `data/raw/bfcl/review.csv` (234줄). 맞으면 `OK(Y/N)` 비우거나 Y, 틀리면 N 과 `fix` 칸에 고칠 내용 (JSON, 예: `{"function": "irrelevant_function", "neg_type": "near_miss"}`, 빼려면 `exclude`).
2. `python -m src.bfcl_bench finalize` → `data/bfcl_test.jsonl` 확정.
3. ~~`src/evaluate.py` 보완~~ (완료, 아래 진행 기록)
4. Colab 에서 우리 모델·공개 모델 생성·채점, `<nexa_20>` 확률로 AUROC.
