"""train / test 생성 결과를 합쳐 최종 데이터셋을 만든다.

python -m src.datagen.build_dataset --train-run train --test-run test --negatives-run negatives_v2
→ data/train.jsonl, data/test.jsonl

--negatives-run 을 주면 negative 는 negatives.py 가 하나의 풀에서 나눈 train/test 로 교체한다
(train/test 를 따로 생성했을 때 negative 가 많이 겹쳤기 때문). positive 는 train/test run 의 것을 쓴다.

test 에서 train 과 같은 질의(대소문자·문장부호 무시)는 빼지 않고 seen_in_train=true 로 표시만 한다.
"방해금지 켜 줘" 같은 흔한 문장은 실제 사용에서도 반복되므로, 5단계 평가에서
전체 점수(실사용에 가까움)와 seen_in_train=false 만의 점수(처음 보는 문장에서의 실력)를 둘 다 본다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from src.datagen.generate import ROOT, _norm, _read_jsonl, _write_jsonl
from src.toy_api import FUNCTIONS


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-run", default="train")
    p.add_argument("--test-run", default="test")
    p.add_argument("--negatives-run", default=None, help="replace negatives with data/raw/<run>/{train,test}.jsonl")
    a = p.parse_args()

    train = _read_jsonl(ROOT / "data" / "raw" / a.train_run / "dataset.jsonl")
    test = _read_jsonl(ROOT / "data" / "raw" / a.test_run / "dataset.jsonl")
    if a.negatives_run:
        neg_dir = ROOT / "data" / "raw" / a.negatives_run
        train = [r for r in train if r["kind"] == "positive"] + _read_jsonl(neg_dir / "train.jsonl")
        test = [r for r in test if r["kind"] == "positive"] + _read_jsonl(neg_dir / "test.jsonl")

    train_keys = {_norm(r["query"]) for r in train}
    for r in test:
        r["seen_in_train"] = _norm(r["query"]) in train_keys

    _write_jsonl(ROOT / "data" / "train.jsonl", train)
    _write_jsonl(ROOT / "data" / "test.jsonl", test)

    tr, te = Counter(r["token_id"] for r in train), Counter(r["token_id"] for r in test)
    seen = Counter(r["token_id"] for r in test if r["seen_in_train"])
    print(f"{'token':10} {'function':34} {'train':>6} {'test':>5} {'test seen_in_train':>19}")
    for i, f in enumerate(FUNCTIONS):
        print(f"<nexa_{i}>".ljust(10), f"{f['name']:34} {tr[i]:6} {te[i]:5} {seen[i]:19}")
    n_seen = sum(seen.values())
    print(f"\ntrain {len(train)}  test {len(test)}  "
          f"test seen_in_train {n_seen} ({n_seen / max(len(test), 1):.1%})")
    print("saved: data/train.jsonl, data/test.jsonl")


if __name__ == "__main__":
    main()
