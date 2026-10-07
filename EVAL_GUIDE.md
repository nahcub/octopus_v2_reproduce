# 평가 결과 보는 법

코드를 열지 않고 `eval/` 폴더만 보고 결과를 읽기 위한 안내. 결과 수치와 오답 검토 내용은 README "평가에서 정한 것" 절에 있다.

## 1. 실험 한 줄 요약
질의(영어 문장)를 주면 모델이 함수 호출 한 줄을 만든다. 함수 이름 대신 **함수마다 새 토큰 하나**(`<nexa_0>` ~ `<nexa_20>`)를 쓴다.
`google/gemma-2b` 를 train 4,000개로 전체 학습한 **우리 모델**과, 비교 기준인 **공개 모델**(`NexaAIDev/Octopus-v2`)을 같은 test 800개로 평가했다.
- test = positive 400 (함수 20개 × 20, 함수를 불러야 하는 질의) + negative 400 (함수와 무관한 질의 → `<nexa_20>()` 이 정답)
- 생성은 한 번에 1개씩, 매번 같은 답이 나오는 방식(greedy), 최대 128토큰. 채점은 생성된 글의 **첫 `<nexa_end>` 까지만** 본다

호출 형식: `<nexa_3>('a@b.com', 'Hi', 'See you')<nexa_end>` = 3번 함수(send_email)를 인자 3개로 부름. 인자는 아래 표의 순서대로 들어간다.

| 토큰 | 함수 (인자 순서) | 토큰 | 함수 (인자 순서) |
|---|---|---|---|
| 0 | take_a_photo (camera) | 11 | change_screen_brightness (level) |
| 1 | get_trending_news (query, language) | 12 | create_calendar_event (title, start_time, end_time) |
| 2 | get_weather_forecast (location) | 13 | set_volume (level, volume_type) |
| 3 | send_email (recipient, title, content) | 14 | connect_to_bluetooth_device (device_name, timeout) |
| 4 | search_youtube_videos (query) | 15 | enable_do_not_disturb (enabled) |
| 5 | find_route_google_maps (origin, destination, mode) | 16 | set_nest_temperature (target_temperature) |
| 6 | check_nest_smoke_detector_status () | 17 | play_music_on_nest_hub (music_service, music_name) |
| 7 | send_text_message (contact_name, message) | 18 | play_video_on_nest_hub (video_service, video_name) |
| 8 | make_phone_call (phone_number) | 19 | open_or_lock_door (status) |
| 9 | create_contact (name, phone_number) | **20** | **irrelevant_function ()** = "부를 함수 없음" |
| 10 | set_timer_alarm (time, label) | | |

## 2. 폴더 구조
```
eval/
├── ours/     우리 모델 결과
└── public/   공개 모델 결과        (두 폴더의 파일 구성은 같다)
    ├── summary.json    점수 요약          ← 먼저 볼 것
    ├── errors.csv      오답만 모은 표     ← 엑셀로 열기 (한글 안 깨짐)
    ├── scored.jsonl    800문항 전체 채점 결과 (정답 포함)
    ├── preds.jsonl     모델이 생성한 원문 그대로
    └── gen_args.json   생성할 때 설정 (모델, GPU, 라이브러리 버전 등)
```
`.jsonl` = 한 줄에 JSON 하나. 메모장·VS Code 로 열거나 pandas `read_json(path, lines=True)` 로 읽는다.
모든 파일의 `index` 는 `data/test.jsonl` 의 줄 번호(0부터)와 같다.

## 3. 결과 코드 (`outcome`)
문항마다 아래 중 **하나**가 붙는다. 위에서부터 차례로 검사해서 처음 걸린 것이 결과다.

| 코드 | 뜻 | 예 (정답 → 생성) |
|---|---|---|
| `correct` | 함수와 인자가 정답과 같음 | |
| `format:no_nexa_end` | `<nexa_end>` 가 없음 (128토큰에서 잘렸거나 호출이 아닌 문장) | `I can't do that.` |
| `format:unparsable` | `<nexa_i>(...)` 모양이 아니거나 괄호 안을 값으로 읽을 수 없음 | `'Dad's AirPods'` (따옴표가 안 닫힘) |
| `format:unknown_token` | 없는 함수 번호 (`<nexa_21>` 이상) | `<nexa_99>()` |
| `function:false_call` | negative 인데 함수를 부름 | `<nexa_20>()` → `<nexa_4>('...')` |
| `function:missed_call` | positive 인데 "부를 함수 없음"(`<nexa_20>`)을 냄 | `<nexa_0>('front')` → `<nexa_20>()` |
| `function:other_function` | positive 인데 다른 함수를 부름 | `<nexa_0>(...)` → `<nexa_2>('Paris')` |
| `args:count_mismatch` | 함수는 맞음. 인자 개수나 이름이 다름 | `<nexa_0>()` (인자 빠짐) |
| `args:rule_violation` | 함수는 맞음. 인자가 함수 설명의 규칙을 어김 (범위·허용값·타입) | 볼륨 `level` 11 (허용 0~10) |
| `args:valid_but_different` | 함수는 맞고 규칙도 지켰지만 정답과 값이 다름 | `'front'` → `'back'` |

같다고 보는 기준: 값과 **타입**이 모두 같아야 한다 (`70` ≠ `70.0`, `True` ≠ `1`, `'Luca'` ≠ `'luca'`).
따옴표 종류(`'a'` / `"a"`)와 이름 붙인 인자(`camera='front'`)는 차이로 보지 않는다.

## 4. `summary.json`
최상위에 `"전체"`(800개)와 `"seen_in_train=false"`(train 에 없던 질의만) 두 묶음이 있고, 안의 항목은 같다. 비율은 0~1.

| 키 | 뜻 |
|---|---|
| `n` | 문항 수 |
| `acc` | **전체 정확도 (대표 숫자)** = `correct` 비율 |
| `acc_positive` / `acc_negative` | positive만 / negative만의 정확도 |
| `format_ok` | `format:*` 이 아닌 비율 |
| `function_acc` | 함수 선택 정확도 = `correct` + `args:*` 비율 (함수는 맞힘) |
| `args_acc` | 함수를 맞힌 positive 중 인자까지 맞힌 비율 |
| `acc_lenient` | 느슨한 정확도: 문자열의 대소문자·공백·끝 문장부호(.!?) 무시, 숫자 int/float 무시 |
| `exact_string` | 호출 문자열이 정답과 글자까지 같은 비율 (따옴표 종류도 다르면 불일치) |
| `outcomes` | 결과 코드별 개수 |
| `latency_s` | (최상위) 1문항 생성 시간(초): `median` 중앙값, `p90` 90% 지점, `mean` 평균 |

## 5. `errors.csv` (오답만)
| 열 | 뜻 |
|---|---|
| `index` | test 줄 번호 |
| `outcome` | 결과 코드 (3절) |
| `lenient_correct` | `True` 면 느슨한 기준으로는 정답 (대소문자·끝 마침표 정도만 다름) |
| `seen_in_train` | `True` 면 train 에 같은 질의가 있었음 (대소문자·문장부호 무시하고 비교) |
| `function` | 정답 함수 이름 |
| `query` | 질의 |
| `gold` | 정답 호출 |
| `pred` | 생성된 호출 (첫 `<nexa_end>` 까지). `format:no_nexa_end` 이면 생성 원문 |
| `rule_errors` | `args:rule_violation` 일 때 어긴 규칙 (`;` 로 구분) |
| `review` | 빈 칸. 사람이 검토 메모를 적는 곳 |

## 6. `scored.jsonl` (전체 800줄)
한 줄 = test 원본 필드 + 생성 결과 + 채점 결과.

| 필드 | 뜻 |
|---|---|
| `kind` | `positive` / `negative` |
| `function`, `token_id` | 정답 함수 이름과 번호 (1절 표) |
| `query` | 질의 |
| `args` | 정답 인자 `{이름: 값}` |
| `call` | 정답 호출 문자열 |
| `seen_in_train` | 5절과 같음 |
| `index` | test 줄 번호 |
| `output` | 생성 원문. 앞에 공백이 하나 붙고, `<nexa_end>` 뒤에 설명 글이 이어질 수 있다 (채점 안 함) |
| `new_tokens` | 생성한 토큰 수 |
| `latency_s` | 이 문항 생성 시간(초). 처음 5번 예열 뒤, 입력 처리부터 생성 끝까지 (모델 로드 제외) |
| `pred_call` | `output` 에서 첫 `<nexa_end>` 까지 자른 것. 없으면 `null` |
| `pred_args` | 생성된 인자 `{이름: 값}`. 함수를 맞혔고 인자 개수가 맞을 때만 채워짐, 아니면 `null` |
| `rule_errors` | 어긴 규칙 목록 |
| `lenient_correct` | 느슨한 기준 정답 여부 |
| `outcome` | 결과 코드 (3절) |

`preds.jsonl` 은 이 중 `index`, `query`, `output`, `new_tokens`, `latency_s` 만 있는 원본이다.

## 7. 읽을 때 알아둘 것
- test 정답은 사람이 아니라 Gemini 가 만들었다. 오답 중에는 정답 쪽이 애매한 경우가 있다. 결과를 본 뒤 정답을 고치지 않았다
- 공개 모델은 정답 표기 규칙이 다른 데이터로 학습되어(예: 온도 `70` vs 우리 정답 `70.0`) 엄격 기준에서 불리하다. 우리 모델은 test 와 같은 방법으로 만든 train 으로 학습했다
- 대표 숫자는 `acc`(엄격). `acc_lenient` 는 참고용(논문에 없는 추가)
