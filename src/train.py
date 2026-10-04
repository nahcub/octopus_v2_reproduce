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


@torch.no_grad()
def initial_check(model, tokenizer, examples: list[dict], device) -> None:
    """학습 전 한 묶음의 loss. 새 토큰은 아직 뜻이 없어서 <nexa_*> 자리의 loss 가 특히 높아야 정상."""
    model.eval()
    batch = {k: v.to(device) for k, v in collate(examples, tokenizer.pad_token_id).items()}
    out = model(**batch)
    # 모델 안에서처럼 한 칸 밀어서 자리별 loss 를 직접 계산한다
    logits = out.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
    ).view(labels.shape)
    scored = labels != IGNORE_INDEX
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
    nexa_ids = add_functional_tokens(tokenizer)
    model = load_model(args.model, args.tiny, vocab_size=len(tokenizer) - len(nexa_ids))
    before = model.get_input_embeddings().num_embeddings
    resize_embeddings(model, tokenizer)
    after = model.get_input_embeddings().num_embeddings
    assert after >= len(tokenizer), (after, len(tokenizer))
    assert model.get_output_embeddings().weight.shape[0] == after  # Gemma 는 입력·출력 칸을 같이 쓴다 (tie)
    print(f"[모델] 단어 뜻 칸 {before} → {after} | 사전 {len(tokenizer)} | 파라미터 {model.num_parameters() / 1e9:.2f}B")
    model.to(device)

    # 2) 데이터: 3단계 함수로 채점표까지 만든다
    rows = load_jsonl(Path(args.train_file))[: args.limit]
    examples = [build_example(r, tokenizer) for r in rows]
    print(f"[데이터] {len(examples)}개, 최대 {max(len(e['input_ids']) for e in examples)} 토큰")

    initial_check(model, tokenizer, examples[: min(8, len(examples))], device)
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
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=examples,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
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
