# 진행 계획

## 0단계: 완성품부터 구경하기 · Colab · 반나절
- [x] 시작 전에 Gemma 라이선스를 HF에서 미리 승인받아 두기
- [x] HF에 공개된 `NexaAIDev/Octopus-v2`를 받아서 추론을 몇 번 돌려 보기
- [x] 토큰 번호 ↔ 함수 대응 확인 (HF 모델 카드 `android_functions.txt` 와 일치)
- [x] 토크나이저에 `<nexa_*>` 토큰이 어떻게 들어가 있는지, 출력이 어떤 형태인지 눈으로 확인
  - 목표 지점이 어디인지 먼저 보는 단계

## 1단계: 장난감 API 정의 · 랩탑 · 반나절
- [x] 공개 모델과 같은 함수 20개 + irrelevant_function 명세 (`src/toy_api.py`)

## 2단계: 데이터 생성 · 랩탑 · 1~2일
- [x] Gemini 생성 파이프라인 (`src/datagen/`): 질의 → 인자 → 규칙 검사 + Gemini 검증 → 재생성
- [x] 파일럿(2개 함수) → 2차(20개 함수 × 10)로 프롬프트 다듬기
  - 기계식 날짜 표기 금지, 호출어 금지, negative 범주 섞기, 이름·도시·전화번호 후보 목록(train/test 분리)
- [ ] 전체 생성: train 4,000 (positive 2,000 + negative 2,000), test 800 (400 + 400)
- [ ] `data/train.jsonl`, `data/test.jsonl` 저장, 사람이 `review.csv` 로 표본 검토

## 3단계: 전처리 파이프라인 · 랩탑 (CPU로 충분) · 1일
- [ ] 토큰 추가
- [ ] 프롬프트 포맷
- [ ] loss masking
- 모델 가중치 없이 토크나이저만 있으면 되므로 랩탑에서 디버깅 가능. 코드 수준의 이해는 대부분 이 단계에서 생긴다.

## 4단계: 학습 · Colab · 1~2일
- [ ] Qwen2.5-0.5B + LoRA로 시작 (무료 T4 GPU로도 동작)
- [ ] 동작 확인 후 Gemma-2B로 올리기
  - T4에서는 QLoRA(4bit) 필요, L4나 A100(Colab Pro)이 있으면 훨씬 편함
- ⚠️ 학습할 때 `modules_to_save=["embed_tokens", "lm_head"]` 꼭 넣기

## 5단계: 평가 · Colab · 1일
- [ ] 함수 선택 정확도
- [ ] 인자 정확도 (`src/datagen/validate.py` 재사용)
- [ ] 전체 점수와 `seen_in_train=false` 점수 비교 (외워서 맞히는지 확인)
- [ ] 지연 시간

## 6단계: Ablation · Colab, 분석은 랩탑 · 2~3일
- [ ] functional token 방식 vs 함수 이름을 텍스트로 생성하는 방식
- [ ] 임베딩 초기화 방식 비교
- [ ] LoRA vs full fine-tuning (가능하면)
- [ ] 학습 타깃에 `Function description` 을 붙이는 것 vs 안 붙이는 것 (논문은 붙이지만 효과를 따로 실험하지 않음)
- [ ] negative 비율 (논문 1:1 vs 더 적게)
