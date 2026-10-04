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
| 3 | 전처리: 특수 토큰 추가, 프롬프트 포맷, loss masking (`src/preprocess/`) | 랩탑 | ✅ |
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
  그래서 transformers 가 칸을 늘릴 때 넣는 기본값(`mean_resizing=True`: 기존 단어 벡터들의 평균·공분산을 따르는 정규분포에서 뽑기)을 쓴다.
  버전마다 기본값이 바뀔 수 있어 코드에 직접 적었다. 다른 방식(함수 설명 단어 벡터 평균 등)과의 비교는 6단계 ablation 에서 한다
  - 논문 3.2절은 초기값 대신 학습으로 뜻을 배운다고 설명한다 (word2vec 처럼 주변 문맥으로. 타깃 뒤 `Function description` 이 그 문맥 역할을 한다는 건 추정)
- **loss 가중치**: 논문 벤치마크 모델처럼 모든 토큰 가중치 1 (위 "논문에서 확인한 설정"의 Weighted CE loss 항목)

**프롬프트 포맷** (`prompt.py`, 확인: `python -m src.preprocess.prompt`)
- 문제 부분은 모델 카드 예제 코드 원문: `...call the function.

Query: {query} 

Response:`
- 정답 부분: `" " + call + "

Function description: 
" + DESCRIPTIONS[token_id] + "

" + <eos>`
  - 0단계 공개 모델 출력 모양을 따랐다 (맨 앞 공백, `Function description: ` 뒤 줄바꿈, 끝의 `

<eos>`). 학습 데이터도 이 모양이었다는 건 추정.
    논문 3.1절 형식(줄바꿈 1개)과 다르며 공개 모델을 따른다
  - 0단계 노트북의 공개 모델 출력 20개와 대조: 19개 일치 (끝까지 생성된 2개는 `<eos>` 까지 완전 일치).
    나머지 1개는 공개 모델이 `<nexa_1>` 뒤에 다른 함수 설명을 쓴 모델 쪽 실수
  - 정답에 `Function description` 을 넣는다 (논문·공개 모델 둘 다). 빼는 버전은 6단계 ablation
- 이걸 맞추다가 `toy_api.py` 의 `irrelevant_function` 설명에서 빈 줄의 공백 2칸이 빠진 걸 찾아 공개 모델 출력 원문대로 고쳤다
- 문제와 정답을 따로 토큰으로 바꿔 이어 붙이고, 정답 시작 위치(`prompt_len`)를 남긴다 (3-3 loss masking 에서 사용).
  글 전체를 한 번에 바꾼 결과와 train 4,000 / test 800 모두 같음
- 변환된 데이터를 파일로 저장하지 않는다. 몇 초면 끝나서 학습 시작할 때 이 함수로 바로 바꾼다 (양식을 고쳤을 때 파일을 다시 만드는 걸 잊는 실수 방지)
- 길이 (Gemma 토큰, `<bos>`·`<eos>` 포함): train 최소 89 / 중앙 111 / 최대 351, test 최대 335

**loss masking** (`masking.py`, 확인: `python -m src.preprocess.masking`)
- `labels` = `input_ids` 복사 후 문제 부분(앞의 `prompt_len` 개)을 -100 으로 덮는다. -100 은 PyTorch `CrossEntropyLoss` 의 `ignore_index` 기본값
- 채점 범위는 정답 전체: ` <nexa_N>(...)<nexa_end>` + `Function description` + `<eos>`. 가중치는 모두 1
- `labels` 를 직접 한 칸 밀지 않는다 (transformers CausalLM 이 안에서 민다)
- 묶음은 오른쪽을 `<pad>` 로 채우고 pad 자리는 `labels` -100, `attention_mask` 0 (`collate`)
- 확인: 채점 부분을 글자로 되돌리면 정답 글과 같음 (train 4,000 / test 800 모두), 채점 첫 토큰은 모두 `▁` 다음 `<nexa_N>`

## 학습에서 정한 것 (4단계, `src/train.py`, `notebooks/04_train.ipynb`)
- 조건: 함수당 100개 + full model training = 논문 표 2 의 **Octopus-3** (논문 표 1 정확도 98.1%)
- 논문 3.4절 그대로: AdamW, lr 5e-5, warm-up 10 steps, linear scheduler, 3 epochs. loss 가중치 모두 1
- (논문에 없는 추가) batch size 16 → 250 step/epoch, 총 750 step. weight decay 0
- GPU: Colab **A100 80GB** (고용량 RAM 런타임, 약 6.77 컴퓨팅 단위/시간). 40GB A100 에서는 일반 AdamW 가 메모리에 안 들어가서
  (모델 9.3 + 기울기 9.3 + AdamW 18.7 GiB) 80GB 를 쓴다. 메모리 옵션 없이 batch 16 그대로, 최대 67.3 GiB, 약 1.6 step/초 (750 step ≈ 8분)
- 가중치는 fp32 로 두고 계산만 bf16 (작은 lr 의 갱신이 bf16 반올림에 묻히지 않게). 저장은 bf16
- 중간 체크포인트는 저장하지 않는다 (full 학습은 옵티마이저 포함 수십 GB). 끝나면 모델·토크나이저·loss 기록을 같이 저장
- 학습 코드는 `.py` 에 두고 노트북은 `git clone` 후 실행만 한다. 랩탑에서 작은 무작위 Gemma(`--tiny`)로 끝까지 도는지 먼저 확인
