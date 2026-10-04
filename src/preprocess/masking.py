"""3단계-3: loss masking. 정답 부분만 채점하도록 채점표(labels)를 만들고, 여러 문장을 한 묶음(batch)으로 맞춘다.

    input_ids = [<bos>, Below, ..., Response, :,  ▁, <nexa_0>, (', ..., <eos>]
    labels    = [-100,  -100,  ...,  -100,  -100, ▁, <nexa_0>, (', ..., <eos>]
                 └──── 문제: 채점 안 함 ────┘ └──────── 정답: 채점 ────────┘

- -100 은 PyTorch CrossEntropyLoss 의 ignore_index 기본값. 이 자리는 loss 계산에서 빠진다.
- labels 를 한 칸 밀지 않는다. transformers 의 CausalLM 이 안에서 밀어서 "다음 단어"와 비교한다.
- 채점 가중치는 모든 토큰 1 (논문 벤치마크 모델과 같음, README "전처리에서 정한 것").
- 묶음을 만들 때 짧은 문장 뒤를 <pad> 로 채운다 (오른쪽 채우기). pad 자리는 labels -100, attention_mask 0.
  Gemma 토크나이저 기본값은 왼쪽 채우기지만 여기서는 직접 채우므로 그 설정과 상관없다.

실행 (토크나이저만 받으므로 랩탑 CPU 로 충분):
    python -m src.preprocess.masking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.preprocess.prompt import build_target, encode, load_jsonl
from src.preprocess.tokens import add_functional_tokens

IGNORE_INDEX = -100


def build_example(row: dict, tokenizer) -> dict:
    """jsonl 한 줄 → {"input_ids", "labels"}. 문제 부분(앞의 prompt_len 개)의 labels 를 -100 으로 덮는다."""
    enc = encode(row, tokenizer)
    input_ids, n = enc["input_ids"], enc["prompt_len"]
    labels = [IGNORE_INDEX] * n + input_ids[n:]
    return {"input_ids": input_ids, "labels": labels}


def collate(examples: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """build_example 결과 여러 개 → 길이를 맞춘 텐서 묶음 (모델에 바로 넣을 수 있는 모양)."""
    width = max(len(ex["input_ids"]) for ex in examples)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for ex in examples:
        pad = width - len(ex["input_ids"])
        batch["input_ids"].append(ex["input_ids"] + [pad_token_id] * pad)
        batch["labels"].append(ex["labels"] + [IGNORE_INDEX] * pad)
        batch["attention_mask"].append([1] * len(ex["input_ids"]) + [0] * pad)
    return {k: torch.tensor(v) for k, v in batch.items()}


def check_scored_part(row: dict, tokenizer) -> bool:
    """채점하는 부분만 글자로 되돌리면 정답 글(build_target + <eos>)과 정확히 같은지."""
    ex = build_example(row, tokenizer)
    scored = [t for t in ex["labels"] if t != IGNORE_INDEX]
    return tokenizer.decode(scored) == build_target(row) + tokenizer.eos_token


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="loss masking 결과를 눈으로 확인한다.")
    p.add_argument("--model", default="google/gemma-2b")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    add_functional_tokens(tokenizer)
    data = {split: load_jsonl(Path(f"data/{split}.jsonl")) for split in ("train", "test")}

    # 1) 예시 한 줄: 자리별로 input_ids / labels 를 나란히
    row = data["train"][0]
    ex = build_example(row, tokenizer)
    n = ex["labels"].count(IGNORE_INDEX)
    print(f"query: {row['query']}")
    print(f"{'자리':>4} | {'input 토큰':<16} | labels")
    for i in list(range(3)) + ["..."] + list(range(n - 3, n + 6)) + ["..."] + list(range(len(ex["labels"]) - 2, len(ex["labels"]))):
        if i == "...":
            print("   ...")
            continue
        tok = repr(tokenizer.convert_ids_to_tokens(ex["input_ids"][i]))
        lab = ex["labels"][i]
        lab_str = "-100 (채점 안 함)" if lab == IGNORE_INDEX else f"{lab} {tokenizer.convert_ids_to_tokens(lab)!r}"
        print(f"{i:>4} | {tok:<16} | {lab_str}")
    print(f"→ 전체 {len(ex['labels'])}자리 중 채점 안 함 {n}, 채점 {len(ex['labels']) - n}")

    # 2) 묶음 예시: 짧은 문장 + 긴 문장
    rows = sorted(data["train"], key=lambda r: len(encode(r, tokenizer)["input_ids"]))
    short, long_ = build_example(rows[0], tokenizer), build_example(rows[-1], tokenizer)
    batch = collate([short, long_], tokenizer.pad_token_id)
    pad = batch["attention_mask"] == 0
    print("\n[묶음] 모양:", {k: tuple(v.shape) for k, v in batch.items()})
    print(f"   짧은 문장 뒤 pad {int(pad[0].sum())}칸 | pad 자리 labels 가 모두 -100: {bool((batch['labels'][pad] == IGNORE_INDEX).all())}"
          f" | pad 자리 input 이 모두 <pad>: {bool((batch['input_ids'][pad] == tokenizer.pad_token_id).all())}")

    # 3) 전체 검사
    print()
    for split, rows in data.items():
        ok = sum(check_scored_part(r, tokenizer) for r in rows)
        first_two = {tuple(tokenizer.convert_ids_to_tokens(
            [t for t in build_example(r, tokenizer)["labels"] if t != IGNORE_INDEX][:2])) for r in rows}
        starts_ok = all(a == "▁" and b.startswith("<nexa_") for a, b in first_two)
        print(f"[{split}] 채점 부분 == 정답 글: {ok}/{len(rows)} | 채점 첫 두 토큰이 '▁', '<nexa_N>': {starts_ok}")


if __name__ == "__main__":
    main()
