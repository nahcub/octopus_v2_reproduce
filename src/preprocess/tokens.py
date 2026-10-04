"""3단계-1: functional token(<nexa_0> ~ <nexa_20>, <nexa_end>)을 토크나이저에 추가한다.

공개 모델 NexaAIDev/Octopus-v2 토크나이저에서 확인한 방식을 그대로 따른다.
- 원래 사전 바로 뒤에 순서대로 붙는다 (Gemma: 256000 = <nexa_0>, ..., 256021 = <nexa_end>).
- 22개 모두 special=True, normalized=False 인 특수 토큰.
  → decode(..., skip_special_tokens=True) 하면 지워지므로 평가 때는 이 옵션을 끈다.

실행 (토크나이저만 받으므로 랩탑 CPU 로 충분):
    python -m src.preprocess.tokens
"""

from __future__ import annotations

import argparse
import sys

from transformers import AddedToken, AutoConfig, AutoTokenizer

from src.toy_api import FUNCTIONAL_TOKENS


def add_functional_tokens(tokenizer) -> list[int]:
    """토크나이저에 functional token 을 추가하고, 각 토큰의 번호(id)를 FUNCTIONAL_TOKENS 순서로 돌려준다.

    이미 들어 있는 토큰은 다시 추가되지 않으므로 여러 번 불러도 결과가 같다.
    """
    tokens = [AddedToken(t, special=True, normalized=False) for t in FUNCTIONAL_TOKENS]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens},
        replace_additional_special_tokens=False,  # 모델이 원래 갖고 있던 특수 토큰은 그대로 둔다
    )
    ids = tokenizer.convert_tokens_to_ids(FUNCTIONAL_TOKENS)
    check_functional_tokens(tokenizer, ids)
    return ids


def check_functional_tokens(tokenizer, ids: list[int]) -> None:
    """토큰이 제대로 들어갔는지 확인한다. 하나라도 틀리면 AssertionError."""
    # 1) <nexa_i> 의 번호가 연속이어야 <nexa_i> = 첫 번호 + i 로 계산할 수 있다.
    assert ids == list(range(ids[0], ids[0] + len(ids))), f"번호가 연속이 아님: {ids}"
    # 2) 각 토큰이 조각나지 않고 토큰 1개로 읽혀야 한다.
    for t, i in zip(FUNCTIONAL_TOKENS, ids):
        got = tokenizer(t, add_special_tokens=False)["input_ids"]
        assert got == [i], f"{t} 가 토큰 1개로 읽히지 않음: {got}"


def resize_embeddings(model, tokenizer) -> None:
    """모델의 단어 뜻 칸(임베딩) 수가 사전 크기보다 모자랄 때만 늘린다.

    Gemma-2B 는 칸이 256,000개로 사전 크기와 딱 맞다 → 256,022개로 늘린다 (공개 모델과 같은 크기).
    모델에 따라 칸이 미리 넉넉한 경우도 있어서 (예: Qwen2.5-0.5B 는 사전 151,665개에 칸 151,936개) 모자랄 때만 늘린다.
    새 칸의 시작값: 논문에 없어서 transformers 기본값(mean_resizing=True)을 쓴다.
    기존 단어 벡터들의 평균·공분산을 따르는 정규분포에서 뽑는다. 버전마다 기본값이 달라질 수 있어 직접 적는다.
    """
    need = len(tokenizer)
    have = model.get_input_embeddings().num_embeddings
    if need > have:
        model.resize_token_embeddings(need, mean_resizing=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="functional token 추가 결과를 눈으로 확인한다.")
    p.add_argument("--model", default="google/gemma-2b")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    example = "<nexa_0>('front')<nexa_end>"

    print(f"[추가 전] 사전 크기 {len(tokenizer)}")
    print("  ", tokenizer.tokenize(example))

    ids = add_functional_tokens(tokenizer)

    print(f"[추가 후] 사전 크기 {len(tokenizer)}")
    print("  ", tokenizer.tokenize(example))
    print(f"   {FUNCTIONAL_TOKENS[0]} = {ids[0]}, ..., {FUNCTIONAL_TOKENS[-1]} = {ids[-1]}")

    rows = AutoConfig.from_pretrained(args.model).vocab_size  # 모델 가중치 없이 칸 수만 확인
    verdict = "늘릴 필요 없음" if rows >= len(tokenizer) else f"{len(tokenizer)}칸으로 늘려야 함"
    print(f"[임베딩] 모델의 단어 뜻 칸 {rows}개 → {verdict}")


if __name__ == "__main__":
    main()
