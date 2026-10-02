# Octopus v2 Reproduction

[Octopus v2: On-device language model for super agent](https://arxiv.org/abs/2404.01744) 의 functional token 방식을
코드 수준에서 이해하기 위한 소규모 재현 프로젝트.

## 환경 분담
- **랩탑**: 코드 작성, 데이터 생성(LLM API), 데이터 검증, 분석 (GPU 불필요)
- **Colab**: 모델 로드 / 학습 / 평가 (CUDA GPU 필요)
- 코드는 GitHub → Colab에서 `git clone`, 데이터·체크포인트는 Google Drive

## 진행 단계
세부 체크리스트는 [PLAN.md](PLAN.md) 참고.
| 단계 | 내용 | 환경 | 상태 |
|---|---|---|---|
| 0 | 공개 모델 `NexaAIDev/Octopus-v2` 추론 및 구조 관찰 | Colab | ⬜ |
| 1 | 함수 명세 21개 (공개 모델 기준, `src/toy_api.py`) | 랩탑 | ✅ |
| 2 | 데이터 생성 (positive + negative) 및 검증 | 랩탑 | ⬜ |
| 3 | 전처리: 특수 토큰 추가, 프롬프트 포맷, loss masking | 랩탑 | ⬜ |
| 4 | 학습 (Qwen2.5-0.5B → Gemma-2B, LoRA) | Colab | ⬜ |
| 5 | 평가 (함수 선택/인자 정확도, latency) | Colab | ⬜ |
| 6 | Ablation | Colab | ⬜ |

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
- 데이터: API 20개(Android), API당 positive 1000 / negative 1000, Gemini로 생성 + 검증 후 재생성
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
