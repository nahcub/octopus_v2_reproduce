"""(논문에 없는 추가) 6단계 실험 2단계 학습: 모델 A 위에 LoRA 를 붙여 SFT(모델 B) 또는 DPO(모델 C) 를 1 epoch.

B 와 C 를 공정하게 비교하려고 loss 말고는 전부 같게 만든다.
- 같은 출발점: 모델 A 폴더를 그대로 불러온다 (A 를 두 번 학습하지 않는다)
- 같은 질문, 같은 순서: 같은 pairs.jsonl 을 같은 seed 로 섞는다. B 는 chosen 만, C 는 chosen + rejected
- 같은 설정: LoRA r16 α32, 논문 3.4절 모듈(q,k,v,o,up,down), AdamW, warm-up 10, linear, batch 16, 1 epoch
- 같은 lr 후보 {1e-5, 5e-5, 2e-4} 를 둘 다 시도하고 각자 val 에서 제일 좋은 것을 고른다

DPO loss (β = 0.1)
    정답 부분(labels != -100) 토큰 log-prob 을 더한 값을 logπ(y|x) 라 할 때
    loss = -log σ( β · [ (logπ(chosen) - logπ_ref(chosen)) - (logπ(rejected) - logπ_ref(rejected)) ] )
    기준 모델 π_ref = 모델 A = LoRA 를 끈 같은 모델 (disable_adapter). 그래서 모델을 두 번 올릴 필요가 없다.
    처음에는 LoRA 가 0 이라 π = π_ref → loss = log 2 ≈ 0.693 에서 시작해야 정상.

저장: LoRA 가중치(adapter)와 토크나이저만 (수십 MB). adapter_config.json 에 모델 A 경로가 적혀서
evaluate.py / probs.py 가 이 폴더를 그대로 불러온다 (A 위에 LoRA 를 합쳐서).

실행 (Colab GPU)
    python -m src.train_stage2 --mode sft --lr 5e-5 --model .../runs/A_s0 --pairs .../runs/A_s0/pairs.jsonl --output-dir .../runs/B_s0_lr5e-5
    python -m src.train_stage2 --mode dpo --lr 5e-5 --model .../runs/A_s0 --pairs .../runs/A_s0/pairs.jsonl --output-dir .../runs/C_s0_lr5e-5
    # 랩탑 CPU: 작은 무작위 모델로 끝까지 도는지만 (README "6단계" 참고)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.preprocess.masking import IGNORE_INDEX, build_example, collate
from src.preprocess.prompt import load_jsonl

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]  # 논문 3.4절


def seq_logps(model, batch: dict) -> torch.Tensor:
    """문장마다 정답 부분 토큰 log-prob 의 합. [문장 수]"""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    labels = batch["labels"][:, 1:]
    mask = labels != IGNORE_INDEX
    safe = labels.masked_fill(~mask, 0)
    out = []
    for i in range(logits.size(0)):  # 한 문장씩: 사전 25만 칸 × 길이의 fp32 복사본이 한꺼번에 생기지 않게
        lg = logits[i].float()
        tok = lg.gather(-1, safe[i].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1)
        out.append((tok * mask[i]).sum())
    return torch.stack(out)


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sft", "dpo"], required=True)
    p.add_argument("--model", required=True, help="모델 A 폴더")
    p.add_argument("--pairs", type=Path, required=True, help="dpo_pairs.py 가 만든 pairs.jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lr", type=float, required=True, help="후보 {1e-5, 5e-5, 2e-4} 중 하나")
    p.add_argument("--beta", type=float, default=0.1, help="DPO β")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16, help="한 번 갱신에 쓰는 질문 수 (DPO 는 문장이 2배)")
    p.add_argument("--micro-batch", type=int, default=4, help="GPU 에 한 번에 올리는 질문 수. batch-size 를 나눠떨어져야 함")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (디버깅)")
    p.add_argument("--max-steps", type=int, default=-1, help="시험용. -1 이면 epochs 만큼")
    p.add_argument("--gradient-checkpointing", action="store_true")
    a = p.parse_args()
    assert a.batch_size % a.micro_batch == 0, "batch-size 가 micro-batch 로 나눠떨어져야 함"

    torch.manual_seed(a.seed)
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # 1) 모델 A + LoRA. A 는 얼리고(bf16), LoRA 가중치만 학습 (peft 가 fp32 로 만든다)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if use_gpu else torch.float32)
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, target_modules=LORA_TARGETS,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(device)
    model.print_trainable_parameters()

    # 2) 데이터: chosen = 원래 정답, rejected = 같은 질문에 rejected_call
    rows = load_jsonl(a.pairs)[: a.limit]
    chosen = [build_example(r, tokenizer) for r in rows]
    rejected = [build_example({**r, "call": r["rejected_call"], "token_id": r["rejected_token_id"]}, tokenizer) for r in rows]
    for c, rj in zip(chosen, rejected):  # 문제 부분이 같아야 같은 질문의 짝이다
        n = c["labels"].count(IGNORE_INDEX)
        assert c["input_ids"][:n] == rj["input_ids"][:n]

    g = torch.Generator().manual_seed(a.seed)
    steps_per_epoch = math.ceil(len(rows) / a.batch_size)
    total = steps_per_epoch * a.epochs if a.max_steps < 0 else a.max_steps
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, a.warmup_steps, total)
    print(f"[설정] {a.mode} | 질문 {len(rows)}개 | {steps_per_epoch} step/epoch, 총 {total} step | lr {a.lr}"
          + (f" | β {a.beta}" if a.mode == "dpo" else ""))

    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_gpu else torch.autocast("cpu", enabled=False)
    log, step, t0 = [], 0, time.time()
    model.train()
    while step < total:
        order = torch.randperm(len(rows), generator=g).tolist()  # B 와 C 가 같은 seed → 같은 순서
        for s in range(0, len(order), a.batch_size):
            if step >= total:
                break
            idx = order[s : s + a.batch_size]
            stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0}
            for m in range(0, len(idx), a.micro_batch):
                mi = idx[m : m + a.micro_batch]
                w = len(mi) / len(idx)  # 마지막 묶음이 작을 때도 질문 하나의 무게가 같게
                if a.mode == "sft":
                    batch = {k: v.to(device) for k, v in collate([chosen[i] for i in mi], tokenizer.pad_token_id).items()}
                    with autocast:
                        loss = model(**batch).loss
                else:
                    batch = {k: v.to(device) for k, v in
                             collate([chosen[i] for i in mi] + [rejected[i] for i in mi], tokenizer.pad_token_id).items()}
                    with autocast:
                        pol = seq_logps(model, batch)
                        with torch.no_grad(), model.disable_adapter():
                            ref = seq_logps(model, batch)
                    n = len(mi)
                    margin = (pol[:n] - ref[:n]) - (pol[n:] - ref[n:])
                    loss = -F.logsigmoid(a.beta * margin).mean()
                    stats["acc"] += w * (margin > 0).float().mean().item()
                    stats["margin"] += w * (a.beta * margin).mean().item()
                (loss * w).backward()
                stats["loss"] += w * loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            entry = {"step": step, "lr": sched.get_last_lr()[0], **{k: round(v, 5) for k, v in stats.items()}}
            if a.mode == "sft":
                entry = {k: v for k, v in entry.items() if k not in ("acc", "margin")}
            log.append(entry)
            if step <= 3 or step % 10 == 0 or step == total:
                print(f"  step {step}/{total} | {entry} | {time.time() - t0:.0f}초")

    if use_gpu:
        print(f"[GPU] 최대 메모리 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.output_dir)
    tokenizer.save_pretrained(a.output_dir)
    (a.output_dir / "log_history.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.output_dir / "train_args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                                             ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[저장] {a.output_dir} (LoRA + 토크나이저)")


if __name__ == "__main__":
    main()
