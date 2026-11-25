"""4단계: Gemma-2B full fine-tuning (논문 Octopus-3 조건: 함수당 100개, full model training).

학습 설정
- 논문 3.4절: AdamW, lr 5e-5, warm-up 10 steps, linear scheduler, 3 epochs. loss 가중치 모두 1 (4.6절).
- (논문에 없는 추가) batch size 16. 메모리가 모자라면 --grad-accum 으로 같은 16을 나눠 담는다.
- 데이터 변환은 3단계 함수 그대로: 토큰 추가(tokens.py) → 양식(prompt.py) → 채점표·묶음(masking.py).

실행
    # 랩탑 CPU: 아주 작은 무작위 Gemma 로 코드가 끝까지 도는지만 확인 (몇 분)
    python -m src.train --tiny --limit 8 --batch-size 2 --max-steps 3 --output-dir runs/tiny

    # Colab A100: 학습 전 점검만 (모델 로드, 칸 늘리기, 처음 loss)
    python -m src.train --check-only

    # Colab A100: 짧게 시험 학습 → 본 학습
    python -m src.train --max-steps 30 --output-dir /content/runs/try
    python -m src.train --output-dir /content/drive/MyDrive/octopus_v2/runs/full

    # 실험 A00 (논문에 없는 추가): negative 없이 positive 만. 거절은 refusal_metrics 의 msp 점수로 본다
    python -m src.train --only-kind positive --output-dir /content/drive/MyDrive/octopus_v2/runs/A00

    # 실험 OE (논문에 없는 추가): negative 를 <nexa_20> 이 아니라 "함수 토큰 20개가 고르게" 로 학습 (Outlier Exposure)
    python -m src.train --neg-loss oe --output-dir /content/drive/MyDrive/octopus_v2/runs/OE

    # 실험 G (논문에 없는 추가): 함수 토큰 없이, 프롬프트에 함수 목록 + JSON 출력 (src/general.py).
    # 문장이 약 2,350 토큰이라 batch 16 을 나눠 담는다 (실제 batch = batch-size × grad-accum = 16 그대로)
    python -m src.train --format json --batch-size 4 --grad-accum 4 --gradient-checkpointing \
        --output-dir /content/drive/MyDrive/octopus_v2/runs/G

    # 실험 ND (논문에 없는 추가, N+설명): ours 에서 함수 토큰만 함수 이름(JSON)으로 바꾼 모델.
    # 프롬프트 = ours 와 같은 문구 (목록 없음), 정답 = JSON + ours 와 같은 Function description
    python -m src.train --format json --no-list --json-desc --output-dir /content/drive/MyDrive/octopus_v2/runs/ND
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaConfig, GemmaForCausalLM, Trainer, TrainingArguments

from src.preprocess.masking import IGNORE_INDEX, build_example, collate
from src.preprocess.prompt import load_jsonl
from src.preprocess.tokens import add_functional_tokens, resize_embeddings


def load_model(name: str, tiny: bool, vocab_size: int):
    """학습할 모델. tiny 면 내려받지 않고 구조만 Gemma 인 아주 작은 무작위 모델을 만든다 (랩탑 디버깅용)."""
    if tiny:
        config = GemmaConfig(
            vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        )
        return GemmaForCausalLM(config)
    # 가중치는 fp32 로 두고 계산만 bf16 으로 한다 (TrainingArguments bf16=True). 작은 lr 의 갱신이 bf16 반올림에 묻히지 않게.
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)


def oe_loss(logits: torch.Tensor, func_pos: torch.Tensor, func_ids: list[int]) -> torch.Tensor:
    """(논문에 없는 추가) Outlier Exposure loss (Hendrycks, Mazeika & Dietterich 2019).
    negative 문장마다 함수 토큰이 나올 자리에서, 함수 토큰 20개만 놓고 다시 softmax 한 확률이
    균등분포(각 1/20)와 얼마나 다른지 (cross-entropy). 20개가 고를수록 작고, 한 함수로 쏠릴수록 크다.
    최솟값은 log 20 ≈ 3.00 (완전히 고를 때)."""
    rows = torch.arange(len(func_pos), device=logits.device)
    logit20 = logits[rows, func_pos - 1][:, func_ids].float()  # func_pos 의 토큰은 그 앞 자리(func_pos-1)가 예측한다
    return -logit20.log_softmax(-1).mean(-1).mean()


class OETrainer(Trainer):
    """positive: 지금과 같은 다음 토큰 loss. negative: labels 가 모두 -100 이고 대신 oe_loss.
    전체 loss = positive loss + oe_lambda × oe_loss."""

    def __init__(self, *a, func_ids: list[int], oe_lambda: float, **kw):
        super().__init__(*a, **kw)
        self.func_ids, self.oe_lambda = func_ids, oe_lambda
        # compute_loss 가 묶음 평균 loss 를 돌려주므로, --grad-accum 일 때 Trainer 가 누적 횟수로 나누게 한다.
        # (True 로 두면 Trainer 는 모델이 num_items_in_batch 로 이미 나눴다고 보고 안 나눠서, 사실상 lr 이 누적 횟수배가 된다)
        self.model_accepts_loss_kwargs = False
        self._parts = {"pos_loss": [], "oe_loss": []}

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        func_pos, is_neg = inputs.pop("func_pos"), inputs.pop("is_negative")
        out = model(**inputs)
        if (inputs["labels"] != IGNORE_INDEX).any():
            pos = out.loss
        else:  # 묶음이 전부 negative 면 모델 loss 가 0/0 = nan 이라 0 으로 둔다
            pos = out.logits.sum() * 0.0
        loss = pos
        if is_neg.any():
            oe = oe_loss(out.logits[is_neg], func_pos[is_neg], self.func_ids)
            loss = loss + self.oe_lambda * oe
            self._parts["oe_loss"].append(oe.item())
        self._parts["pos_loss"].append(pos.item())
        return (loss, out) if return_outputs else loss

    def log(self, logs, *a, **kw):
        # 두 loss 를 따로 기록 (log_history.json 에 남는다)
        for k, v in self._parts.items():
            if v:
                logs[k] = round(sum(v) / len(v), 4)
                v.clear()
        super().log(logs, *a, **kw)


class AnswerOnlyLossTrainer(Trainer):
    """(논문에 없는 추가, 실험 G·ND --format json. ND 는 문장이 짧아 메모리 이득은 작지만 loss 식이 같으므로 같이 쓴다) 프롬프트에 함수 목록이 들어가 문장이 약 2,350 토큰이다.
    모델 기본 loss 는 모든 자리의 사전 확률(256,000칸)을 만들어서 메모리가 크게 든다 (문장 하나에 수 GB).
    정답 자리만 골라 출력층에 통과시켜 같은 값을 계산한다 (정답 토큰 cross-entropy 를 num_items_in_batch 로 나눔 =
    transformers 기본 causal LM loss 와 같은 식). 계산 결과는 같고 메모리만 줄어든다."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        hidden = model.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        y = labels[:, 1:]
        keep = y != IGNORE_INDEX
        logits = model.lm_head(hidden[:, :-1][keep]).float()  # 자리 t 가 토큰 t+1 을 예측
        loss_sum = torch.nn.functional.cross_entropy(logits, y[keep], reduction="sum")
        n = num_items_in_batch if num_items_in_batch is not None else keep.sum()
        loss = loss_sum / n
        return (loss, None) if return_outputs else loss


@torch.no_grad()
def initial_check(model, tokenizer, examples: list[dict], device, json_format: bool = False) -> None:
    """학습 전 한 묶음의 loss. 새 토큰은 아직 뜻이 없어서 <nexa_*> 자리의 loss 가 특히 높아야 정상."""
    model.eval()
    batch = {k: v.to(device) for k, v in collate(examples, tokenizer.pad_token_id).items()}
    batch.pop("func_pos", None)
    batch.pop("is_negative", None)
    if not (batch["labels"] != IGNORE_INDEX).any():
        print("[학습 전 loss] 점검 묶음이 전부 negative(oe) 라 건너뜀")
        model.train()
        return
    out = model(**batch)
    # 모델 안에서처럼 한 칸 밀어서 자리별 loss 를 직접 계산한다
    logits = out.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
    ).view(labels.shape)
    scored = labels != IGNORE_INDEX
    if json_format:  # 새 토큰이 없다
        print(f"[학습 전 loss] 전체(모델 출력) {out.loss.item():.3f} | 채점 자리 평균 {per_token[scored].mean().item():.3f}")
        model.train()
        return
    nexa = labels >= len(tokenizer) - 22  # 새로 추가한 22개 번호
    print(f"[학습 전 loss] 전체(모델 출력) {out.loss.item():.3f} | 채점 자리 평균 {per_token[scored].mean().item():.3f}"
          f" | <nexa_*> 자리 {per_token[nexa].mean().item():.3f} | 나머지 정답 자리 {per_token[scored & ~nexa].mean().item():.3f}")
    model.train()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description="Octopus v2 full fine-tuning")
    p.add_argument("--model", default="google/gemma-2b")
    p.add_argument("--tiny", action="store_true", help="랩탑 디버깅: 작은 무작위 Gemma (토크나이저는 --model 것)")
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--limit", type=int, default=None, help="앞에서 N개만 사용 (디버깅)")
    p.add_argument("--only-kind", choices=["positive", "negative"], default=None,
                   help="(논문에 없는 추가) 이 kind 만 학습. 실험 A00: --only-kind positive (negative 0%%)")
    p.add_argument("--neg-loss", choices=["token", "oe"], default="token",
                   help="negative 학습 방식. token: <nexa_20> 을 정답으로 (논문). "
                        "oe: 함수 토큰 20개가 고르게 (논문에 없는 추가, Outlier Exposure). 거절은 refusal_metrics 의 msp 로 본다")
    p.add_argument("--oe-lambda", type=float, default=0.5, help="oe loss 가중치 (Outlier Exposure 원 논문 값 0.5)")
    p.add_argument("--exclude-neg-category", default=None,
                   help="(논문에 없는 추가) 실험 X: 이 범주의 negative 를 학습에서 뺀다 (src/neg_categories.SHORT 중 하나)")
    p.add_argument("--format", choices=["token", "json"], default="token",
                   help="(논문에 없는 추가, 실험 G) json: 함수 토큰 없이 프롬프트에 함수 목록을 넣고 JSON 으로 호출 (src/general.py)")
    p.add_argument("--no-list", action="store_true",
                   help="(논문에 없는 추가, 실험 ND) --format json 에서 프롬프트의 함수 목록을 뺀다 (문제 부분 = ours 와 같은 문구)")
    p.add_argument("--json-desc", action="store_true",
                   help="(논문에 없는 추가, 실험 ND) --format json 에서 정답 JSON 뒤에 ours 와 같은 Function description 을 붙인다")
    p.add_argument("--output-dir", default="runs/full")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16, help="한 step 에 GPU 에 올리는 문장 수")
    p.add_argument("--grad-accum", type=int, default=1, help="몇 step 을 모아 한 번 갱신할지 (실제 batch = batch-size × grad-accum)")
    p.add_argument("--optim", default="adamw_torch", help="메모리가 모자라면 adamw_bnb_8bit")
    p.add_argument("--gradient-checkpointing", action="store_true", help="메모리 절약 (대신 느려짐)")
    p.add_argument("--max-steps", type=int, default=-1, help="시험 학습용. -1 이면 epochs 만큼")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-only", action="store_true", help="학습 전 점검만 하고 끝낸다")
    args = p.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # 1) 토크나이저 + 모델, 칸 늘리기
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    json_format = args.format == "json"
    assert not (json_format and args.neg_loss == "oe"), "--format json 은 --neg-loss token 만 지원"
    assert json_format or not (args.no_list or args.json_desc), "--no-list·--json-desc 는 --format json 전용"
    nexa_ids = [] if json_format else add_functional_tokens(tokenizer)  # json: 새 토큰 없음, 사전 그대로
    model = load_model(args.model, args.tiny, vocab_size=len(tokenizer) - len(nexa_ids))
    before = model.get_input_embeddings().num_embeddings
    if not json_format:
        resize_embeddings(model, tokenizer)
    after = model.get_input_embeddings().num_embeddings
    assert after >= len(tokenizer), (after, len(tokenizer))
    assert model.get_output_embeddings().weight.shape[0] == after  # Gemma 는 입력·출력 칸을 같이 쓴다 (tie)
    print(f"[모델] 단어 뜻 칸 {before} → {after} | 사전 {len(tokenizer)} | 파라미터 {model.num_parameters() / 1e9:.2f}B")
    model.to(device)

    # 2) 데이터: 3단계 함수로 채점표까지 만든다
    rows = load_jsonl(Path(args.train_file))
    if args.only_kind:
        rows = [r for r in rows if r["kind"] == args.only_kind]
    if args.exclude_neg_category:
        from src.neg_categories import SHORT, load as load_neg_categories
        assert args.exclude_neg_category in SHORT, f"범주 이름: {SHORT}"
        cats = load_neg_categories()
        before_n = len(rows)
        rows = [r for r in rows if not (r["kind"] == "negative" and cats[r["query"]] == args.exclude_neg_category)]
        print(f"[데이터] negative 범주 '{args.exclude_neg_category}' 제외: {before_n} → {len(rows)}개")
    rows = rows[: args.limit]
    oe = args.neg_loss == "oe"
    if json_format:
        from src.general import build_example as build_json_example
        examples = [build_json_example(r, tokenizer, with_list=not args.no_list, desc=args.json_desc) for r in rows]
    else:
        examples = [build_example(r, tokenizer, oe=oe) for r in rows]
    n_neg = sum(r["kind"] == "negative" for r in rows)
    print(f"[데이터] {len(examples)}개 (negative {n_neg}), 최대 {max(len(e['input_ids']) for e in examples)} 토큰"
          f" | negative 학습 방식: {args.neg_loss}" + (f" (λ={args.oe_lambda})" if oe else ""))
    func_ids = tokenizer.convert_tokens_to_ids([f"<nexa_{i}>" for i in range(20)])  # irrelevant(<nexa_20>) 제외

    initial_check(model, tokenizer, examples[: min(2 if json_format else 8, len(examples))], device, json_format)
    if use_gpu:
        print(f"[GPU] {torch.cuda.get_device_name()} | 모델만 올렸을 때 {torch.cuda.memory_allocated() / 2**30:.1f} GiB"
              f" / 전체 {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    if args.check_only:
        return

    # 3) 학습
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        optim=args.optim,
        weight_decay=0.0,
        bf16=use_gpu,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=1 if args.max_steps > 0 else 10,
        save_strategy="no",  # 중간 저장 안 함: full 학습 체크포인트는 옵티마이저 포함 수십 GB
        report_to="none",
        remove_unused_columns=False,  # 우리 collate 가 쓰는 칸을 Trainer 가 지우지 않게
        seed=args.seed,
    )
    common = dict(model=model, args=targs, train_dataset=examples,
                  data_collator=lambda batch: collate(batch, tokenizer.pad_token_id))
    if oe:
        trainer = OETrainer(**common, func_ids=func_ids, oe_lambda=args.oe_lambda)
    elif json_format:
        trainer = AnswerOnlyLossTrainer(**common)
    else:
        trainer = Trainer(**common)
    trainer.train()
    if use_gpu:
        print(f"[GPU] 학습 중 최대 메모리 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    # 4) 저장: 모델과 토크나이저를 같이 (짝이 맞아야 한다). 용량을 줄이려고 bf16 으로 저장.
    out = Path(args.output_dir)
    model.to(torch.bfloat16).save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "train_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
