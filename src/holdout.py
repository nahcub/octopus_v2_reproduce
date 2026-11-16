"""(논문에 없는 추가) 6단계 실험: negative 범주 hold-out 으로 train / val / test / holdout 파일을 만든다.

목적: "학습 때 한 번도 안 본 종류의 질문도 거절하나?" 를 재려면 학습에서 일부 범주를 통째로 빼야 한다.
(지금 test negative 는 train 과 같은 8개 범주라서 모델이 거의 다 거절한다 → 고칠 문제가 안 보인다.)

범주 정보는 git 에 없는 data/raw/negatives_v2/negatives.jsonl 의 variant 칸에만 있다.
train.jsonl / test.jsonl 의 negative 질의를 그 파일과 질의 문자열로 짝지어 범주를 붙인다.

만드는 파일 (--out-dir, 기본 data/holdout/)
    train.jsonl    1·2단계 학습용. 원래 train 에서 hold-out 범주 negative 를 빼고, val 로 뗀 것도 뺀 나머지
    val.jsonl      lr 고르기·임계값 고르기용. 원래 train 에서 --val-frac 만큼 (positive 는 함수별, negative 는 범주별로 고르게)
                   hold-out 범주는 들어가지 않는다 (시험 문제를 보고 공부하는 셈이 되므로)
    test.jsonl     원래 test 800개 그대로 + category 칸. 이전 결과(94.25%)와 같은 셋이라 비교 가능
                   (hold-out 범주 negative 100개도 그대로 들어 있다 → 범주별로 나눠 보면 된다)
    holdout.jsonl  hold-out 범주 negative 전부: 원래 train 에 있던 것 + test 에 있던 것 + 어디에도 안 쓴 나머지.
                   새 train 에서는 빠졌으므로 모두 처음 보는 질문이다. (test.jsonl 과 100개 겹침)
    meta.json      고른 범주, 개수, seed

모든 줄에 category (negative 만, positive 는 null) 와 seen_in_train (새 train 에 같은 질의가 있는지) 을 붙인다.
evaluate.py 의 report 가 seen_in_train 칸을 쓴다.

실행 (랩탑 CPU, 몇 초)
    python -m src.holdout                       # 기본: writing, smart_home 을 hold-out
    python -m src.holdout --holdout writing smart_home math   # 늘릴 때 (★ 점검에서 오호출이 적으면)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from src.datagen.prompts import NEGATIVE_CATEGORIES
from src.preprocess.prompt import load_jsonl

# 범주 짧은 이름 (prompts.NEGATIVE_CATEGORIES 와 같은 순서)
SHORT = ["knowledge", "math", "small_talk", "advice", "writing", "phone_settings", "other_apps", "smart_home"]
assert len(SHORT) == len(NEGATIVE_CATEGORIES)
FULL_TO_SHORT = dict(zip(NEGATIVE_CATEGORIES, SHORT))


def _key(q: str) -> str:
    return q.strip()


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdout", nargs="+", default=["writing", "smart_home"], choices=SHORT)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw", type=Path, default=Path("data/raw/negatives_v2/negatives.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/holdout"))
    a = p.parse_args()

    raw = load_jsonl(a.raw)
    cat_of = {_key(r["query"]): FULL_TO_SHORT[r["variant"]] for r in raw if r["status"] == "passed"}
    train, test = load_jsonl(Path("data/train.jsonl")), load_jsonl(Path("data/test.jsonl"))

    def tag(r: dict) -> dict:
        if r["kind"] == "negative":
            c = cat_of.get(_key(r["query"]))
            assert c is not None, f"범주를 못 찾은 negative: {r['query']!r}"
            return {**r, "category": c}
        return {**r, "category": None}

    train, test = [tag(r) for r in train], [tag(r) for r in test]
    held = set(a.holdout)

    # 1) 원래 train 에서 hold-out 범주를 뺀다
    kept = [r for r in train if r["category"] not in held]

    # 2) 남은 것에서 val 을 고르게 뗀다 (positive 는 함수별, negative 는 범주별)
    rng = random.Random(a.seed)
    groups: dict[str, list[dict]] = {}
    for r in kept:
        groups.setdefault(r["category"] or f"pos_{r['token_id']}", []).append(r)
    new_train, val = [], []
    for g in sorted(groups):
        rows = groups[g][:]
        rng.shuffle(rows)
        n_val = round(len(rows) * a.val_frac)
        val += rows[:n_val]
        new_train += rows[n_val:]
    rng.shuffle(new_train)  # 학습 순서는 Trainer 가 다시 섞지만, 파일만 봐도 범주가 섞여 있게

    # 3) hold-out 평가셋: 해당 범주 negative 전부 (train + test + 안 쓴 나머지)
    used = {_key(r["query"]) for r in train + test}
    leftover = [{"kind": "negative", "function": r["function"], "token_id": r["token_id"], "query": r["query"],
                 "args": r["args"], "call": r["call"], "category": FULL_TO_SHORT[r["variant"]]}
                for r in raw if r["status"] == "passed" and _key(r["query"]) not in used]
    holdout = [r for r in train + test + leftover if r["category"] in held]

    # 4) seen_in_train 을 새 train 기준으로 다시 붙인다 (train 에 있는 칸은 지운다)
    seen = {_key(r["query"]) for r in new_train}

    def mark(rows: list[dict]) -> list[dict]:
        return [{**r, "seen_in_train": _key(r["query"]) in seen} for r in rows]

    new_train = [{k: v for k, v in r.items() if k != "seen_in_train"} for r in new_train]
    val, test_out, holdout = mark(val), mark(test), mark(holdout)

    # 확인: hold-out 범주가 학습 쪽에 새지 않았는지
    assert not any(r["category"] in held for r in new_train + val)
    assert not any(r["seen_in_train"] for r in holdout)
    assert not {_key(r["query"]) for r in val} & seen

    a.out_dir.mkdir(parents=True, exist_ok=True)
    files = {"train": new_train, "val": val, "test": test_out, "holdout": holdout}
    for name, rows in files.items():
        _write(a.out_dir / f"{name}.jsonl", rows)

    def counts(rows: list[dict]) -> dict:
        c = Counter(r["category"] or "positive" for r in rows)
        return {"total": len(rows), **dict(sorted(c.items()))}

    meta = {"holdout": sorted(held), "val_frac": a.val_frac, "seed": a.seed, "counts": {n: counts(r) for n, r in files.items()}}
    (a.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"hold-out 범주: {sorted(held)}  (seed {a.seed}, val {a.val_frac:.0%})")
    for name, rows in files.items():
        print(f"  {name:<8} {counts(rows)}")
    print(f"[저장] {a.out_dir}/ train, val, test, holdout .jsonl + meta.json")


if __name__ == "__main__":
    main()
