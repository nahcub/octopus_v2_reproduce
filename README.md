# Octopus v2 Reproduction

[Octopus v2: On-device language model for super agent](https://arxiv.org/abs/2404.01744) 의 functional token 방식을
코드 수준에서 이해하기 위한 소규모 재현 프로젝트.

## 환경 분담
- **랩탑**: 코드 작성, 데이터 생성(LLM API), 데이터 검증, 분석 (GPU 불필요)
- **Colab**: 모델 로드 / 학습 / 평가 (CUDA GPU 필요)
- 코드는 GitHub → Colab에서 `git clone`, 체크포인트는 Google Drive
- 최종 학습 데이터(`data/train.jsonl`, `data/test.jsonl`)는 작아서 git에 포함. 생성 중간 산출물(`data/raw/`)은 제외

## 진행 단계
세부 체크리스트는 [PLAN.md](PLAN.md) 참고.
| 단계 | 내용 | 환경 | 상태 |
|---|---|---|---|
| 0 | 공개 모델 `NexaAIDev/Octopus-v2` 추론 및 구조 관찰 (`notebooks/00_explore_octopus.ipynb`) | Colab | ✅ |
| 1 | 함수 명세 21개 (공개 모델 기준, `src/toy_api.py`) | 랩탑 | ✅ |
| 2 | 데이터 생성 (positive + negative) 및 검증 (`src/datagen/`) | 랩탑 | 🔄 |
| 3 | 전처리: 특수 토큰 추가, 프롬프트 포맷, loss masking (`src/preprocess/`) | 랩탑 | 🔄 |
| 4 | 학습 (Gemma-2B) | Colab A100 | ⬜ |
| 5 | 평가 (함수 선택/인자 정확도, latency) | Colab | ⬜ |
| 6 | Ablation | Colab | ⬜ |

## 0단계에서 확인한 것 (공개 모델)
- 구조는 `GemmaForCausalLM`, vocab 256,022 = Gemma 256,000 + 특수 토큰 22개 (`<nexa_0>`~`<nexa_20>` = ID 256000~256020, `<nexa_end>` = 256021)
- 입력 임베딩과 출력층 크기 모두 `[256022, 2048]`
- `<nexa_end>` 에서 멈추지 않고 계속 생성하면 학습 타깃에 있던 `Function description: def ...` 까지 출력한다.
  이걸 이용해 토큰 번호 ↔ 함수 대응을 확인했고, HF 모델 카드의 `android_functions.txt` 순서와 일치했다
- 관련 없는 질의("Turn on bluetooth" 등)에는 `<nexa_20>` = `irrelevant_function()` 을 출력한다 → negative 샘플의 정답 형식

## 논문에서 확인한 설정
**프롬프트 / 타깃 포맷** (함수 설명까지 학습 타깃, 추론 시 `<nexa_end>`에서 중단)
```
Below is the query from the users, please choose the correct function and generate the
parameters to call the function.
Query: {query}
Response: <nexa_i>(param1, param2, ...)<nexa_end>
Function description: {function_description}
```
- 함수 N개 → `<nexa_0>` ~ `<nexa_N-1>` + `<nexa_end>` 추가, LM head를 N개 확장
- 데이터: API 20개(Android), API당 positive 1000, negative 는 "M = N = 1000" (API당인지 전체인지 논문 표현이 애매), Gemini로 생성 + 검증 후 재생성
- 학습: AdamW, lr 5e-5, warm-up 10 steps, linear scheduler, 3 epochs
- LoRA: r=16, alpha=32, `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj`
- Weighted CE loss (특수 토큰 가중치↑)는 수렴만 빨라지고 최종 성능 차이 없음 → 벤치마크는 equal weight
- 데이터 양: API당 100~1000개 권장 (Octopus-0~3: 1K full / 1K LoRA / 500 / 100)

## 이 프로젝트에서 따르는 기준 (공개 모델 `NexaAIDev/Octopus-v2`)
논문과 공개 모델이 다른 부분은 공개 모델(HF 모델 카드)을 따른다.
- **프롬프트**: 모델 카드 예제 코드의 문구 그대로 (논문의 "choose" 대신 "call")
  ```python
  f"Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {query} \n\nResponse:"
  ```
- **함수 명세**: 모델 카드의 `android_functions.txt` (20개) + `irrelevant_function` = `<nexa_0>` ~ `<nexa_20>`
  → `src/toy_api.py` (원문을 문자열로 그대로 보관)

## 데이터 생성에서 정한 것 (2단계, `src/datagen/`)
- 생성 모델: Gemini (`gemini-3.8-flash`), 논문 3.3절 흐름: 질의 생성 → 인자 생성 → 규칙 검사 + Gemini 검증 → 불합격 재생성
- 양: 함수당 positive 100 (논문 권장 최소), negative = positive 총수 (논문 "M = N")
- 정답은 공개 모델처럼 위치 인자: `<nexa_i>('a', 1)<nexa_end>`. 질의에 없는 인자는 함수 설명의 기본값으로 채움
- 질의에는 기본값 없는 인자의 정보가 모두 들어가게 생성 (번호·날짜를 모델이 지어내는 걸 배우지 않도록)
- (논문에 없는 추가) 이름·도시는 목록(`src/datagen/pools.py`)에서 균등하게 돌려 가며 후보로 주고, 전화번호는 코드로 생성.
  목록은 train/test 로 나눠서 test 의 이름·도시는 학습 때 한 번도 안 나온 값이 되게 함
- 질의에 "Hey Google" 같은 호출어는 쓰지 않음 (모델 이름 미정)
- (논문에 없는 추가) negative 는 train/test 를 따로 만들면 Gemini 가 같은 질문을 반복해 test 의 41% 가 train 과 겹쳤다.
  그래서 범주 8개 × 세부 주제를 돌려 가며, 이미 만든 질의를 보여 주면서 하나의 풀(2,794개)로 만든 뒤 범주별로 고르게 나눈다 (`negatives.py`)
- **한계**: 날짜는 질의에 연도까지 명시한다. 추론 프롬프트에 오늘 날짜가 없어서 "내일", "다음 주" 같은 상대 날짜는 정답을 정할 수 없기 때문.
  공개 모델도 이런 질의엔 임의 날짜를 출력한다.

## 데이터 생성 실행 방법
```bash
pip install google-genai python-dotenv
cp .env.example .env          # GEMINI_API_KEY 채우기
python -m src.datagen.generate --dry-run --functions create_calendar_event   # 프롬프트만 확인 (키 불필요)
python -m src.datagen.generate --run train --split train --per-function 100 --negatives 2000 --seed 0
python -m src.datagen.generate --run test  --split test  --per-function 20  --negatives 400  --seed 1
python -m src.datagen.negatives --run negatives_v2 --train 2000 --test 400   # negative 는 하나의 풀에서 생성 후 분할
python -m src.datagen.build_dataset --train-run train --test-run test --negatives-run negatives_v2
```
- 단계별 결과는 `data/raw/<run>/` (질의, 검증 결과, 탈락 사유, 비용 기록, 사람 검토용 `review.csv`). 같은 `--run` 으로 다시 실행하면 끝난 단계는 건너뜀
- 최종 파일 한 줄 형식:
  ```json
  {"kind": "positive", "function": "take_a_photo", "token_id": 0, "query": "Take a selfie", "args": {"camera": "front"}, "call": "<nexa_0>('front')<nexa_end>"}
  ```
  `test.jsonl` 에는 `seen_in_train` (train 에 같은 질의가 있는지)이 추가된다.
  흔한 문장은 실사용에서도 반복되므로 빼지 않고 표시만 하고, 5단계에서 전체 점수와 `seen_in_train=false` 점수를 둘 다 본다
- 파일 구성: `generate.py`(파이프라인) · `prompts.py`(Gemini 프롬프트) · `validate.py`(인자 규칙, 정답 문자열) · `pools.py`(이름·도시·전화번호 후보) · `llm.py`(Gemini 호출, 재시도, 비용 집계) · `negatives.py`(negative 풀 생성·분할) · `build_dataset.py`(train/test 합치기)

## 전처리에서 정한 것 (3단계, `src/preprocess/`)
**토큰 추가** (`tokens.py`, 확인: `python -m src.preprocess.tokens`)
- `<nexa_0>`~`<nexa_20>`, `<nexa_end>` 22개를 공개 모델과 같은 방식(`special=True`, `normalized=False` 인 특수 토큰)으로 원래 사전 뒤에 붙인다
- `google/gemma-2b` 토크나이저에 추가한 결과를 공개 모델 토크나이저와 비교: 크기 256,022, 토큰 번호 256000~256021, train 4,000개 토큰화 결과 모두 같음.
  사전 전체에서 다른 건 우리 토큰과 무관한 255999번 이름 하나뿐 (지금 `gemma-2b`: `<unused99>`, 공개 모델: `<start_of_image>`)
- 특수 토큰이라 `decode(..., skip_special_tokens=True)` 하면 `<nexa_*>` 가 지워진다 → 평가 때는 이 옵션을 끈다
- Gemma-2B 는 임베딩이 256,000칸이라 256,022칸으로 늘린다 (`resize_embeddings`, 모델이 필요해서 4단계에서 실행)
- **새 토큰 22칸의 시작값**: 논문에 초기화 방법이 적혀 있지 않다 (공개 모델에는 학습 후 값만 있어서 알 수 없음).
  그래서 transformers 가 칸을 늘릴 때 넣는 기본값을 쓴다. 다른 방식(함수 설명 단어 벡터 평균 등)과의 비교는 6단계 ablation 에서 한다
  - 논문 3.2절은 초기값 대신 학습으로 뜻을 배운다고 설명한다 (word2vec 처럼 주변 문맥으로. 타깃 뒤 `Function description` 이 그 문맥 역할을 한다는 건 추정)
- **loss 가중치**: 논문 벤치마크 모델처럼 모든 토큰 가중치 1 (위 "논문에서 확인한 설정"의 Weighted CE loss 항목)
