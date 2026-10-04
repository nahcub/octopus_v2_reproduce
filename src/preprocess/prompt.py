"""3단계-2: jsonl 한 줄을 모델이 읽을 글 한 덩어리(+ 토큰 번호, 문제/정답 경계)로 바꾼다.

글 모양 (· = 공백, ↵ = 줄바꿈)
    Below is the query from the users, please call the correct function and generate the parameters to call the function.↵
    ↵
    Query:·{query}·↵
    ↵
    Response:                                  ← 여기까지 문제 (모델에게 주는 부분)
    ·<nexa_0>('front')<nexa_end>↵              ← 여기부터 정답 (모델이 써야 하는 부분)
    ↵
    Function·description:·↵
    {DESCRIPTIONS[token_id] 원문}↵
    ↵
    <eos>

근거
- 문제 부분: HF 모델 카드 예제 코드의 문구 원문 (`{query}` 뒤 공백 1칸, `Response:` 뒤 없음).
- 정답 부분: 0단계에서 공개 모델이 <nexa_end> 뒤까지 생성한 출력의 모양
  (맨 앞 공백, `Function description: ` 뒤 줄바꿈, 설명 뒤 `\\n\\n<eos>`).
  학습 데이터도 이 모양이었을 거라는 건 추정 (모델이 학습 데이터 모양을 그대로 따라 쓰므로).
  논문 3.1절 형식(`<nexa_end>` 뒤 줄바꿈 1개, "choose")과는 다르며, 공개 모델을 따른다.

실행 (토크나이저만 받으므로 랩탑 CPU 로 충분):
    python -m src.preprocess.prompt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from src.preprocess.tokens import add_functional_tokens
from src.toy_api import DESCRIPTIONS

PROMPT_TEMPLATE = (
    "Below is the query from the users, please call the correct function and generate the parameters to call the function."
    "\n\nQuery: {query} \n\nResponse:"
)


def build_prompt(query: str) -> str:
    """문제 부분. 추론할 때도 이것만 모델에게 준다."""
    return PROMPT_TEMPLATE.format(query=query)


def build_target(row: dict) -> str:
    """정답 부분 (<eos> 제외). <eos> 는 encode 에서 토큰 번호로 붙인다."""
    return " " + row["call"] + "\n\nFunction description: \n" + DESCRIPTIONS[row["token_id"]] + "\n\n"


def encode(row: dict, tokenizer) -> dict:
    """jsonl 한 줄 → {"input_ids": 토큰 번호 목록, "prompt_len": 정답이 시작하는 위치}.

    문제와 정답을 따로 토큰으로 바꿔서 이어 붙인다. 그래야 경계가 정확히 남는다.
    (한꺼번에 바꿔도 결과가 같은지는 check_split_matches_joint 로 확인한다.)
    """
    prompt_ids = tokenizer(build_prompt(row["query"]))["input_ids"]  # 맨 앞에 <bos> 가 붙는다
    target_ids = tokenizer(build_target(row), add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + target_ids + [tokenizer.eos_token_id],
        "prompt_len": len(prompt_ids),
    }


def check_split_matches_joint(row: dict, tokenizer) -> bool:
    """따로 바꿔 이어 붙인 결과 == 글 전체를 한 번에 바꾼 결과 인지."""
    joint = tokenizer(build_prompt(row["query"]) + build_target(row) + tokenizer.eos_token)["input_ids"]
    return encode(row, tokenizer)["input_ids"] == joint


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _visible(text: str) -> str:
    return text.replace(" ", "·").replace("\n", "↵\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="프롬프트 포맷 결과를 눈으로 확인한다.")
    p.add_argument("--model", default="google/gemma-2b")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    add_functional_tokens(tokenizer)
    data = {split: load_jsonl(Path(f"data/{split}.jsonl")) for split in ("train", "test")}

    # 1) 예시: positive 1개, negative 1개
    pos = data["train"][0]
    neg = next(r for r in data["train"] if r["kind"] == "negative")
    for row in (pos, neg):
        enc = encode(row, tokenizer)
        ids, n = enc["input_ids"], enc["prompt_len"]
        print("=" * 70)
        print(_visible(tokenizer.decode(ids[:n])), end="")
        print("   ◀ 문제 끝 / 정답 시작 ▶")
        print(_visible(tokenizer.decode(ids[n:])))
        print(f"\n토큰 {len(ids)}개 (문제 {n} + 정답 {len(ids) - n})")
        print("경계 근처 토큰:", tokenizer.convert_ids_to_tokens(ids[n - 3 : n + 4]))

    # 2) 전체 검사
    print("=" * 70)
    for split, rows in data.items():
        bad = [r["query"] for r in rows if not check_split_matches_joint(r, tokenizer)]
        lens = sorted(len(encode(r, tokenizer)["input_ids"]) for r in rows)
        print(
            f"[{split}] {len(rows)}개 | 따로 바꾼 결과 == 한 번에 바꾼 결과: {len(rows) - len(bad)}/{len(rows)}"
            f" | 토큰 수 최소 {lens[0]}, 중앙 {lens[len(lens) // 2]}, 최대 {lens[-1]}"
        )
        for q in bad[:3]:
            print("   다름:", q)


if __name__ == "__main__":
    main()
