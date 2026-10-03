"""negative(irrelevant_function) 샘플을 하나의 풀로 만들어 train / test 로 나눈다.

generate.py 로 train, test 를 따로 만들었더니 Gemini 가 같은 질문을 반복해서
train negative 의 42% 가 중복으로 빠지고(2,000 중 1,735), test negative 의 41% 가 train 과 같은 문장이었다.
그래서 negative 만 다음 방식으로 다시 만든다.
1. 범주마다 세부 주제(prompts.NEGATIVE_SUBTOPICS)를 섞은 뒤 돌려 가며 배치마다 하나씩 준다.
2. 같은 범주는 순서대로 생성하면서, 이미 만든 질의를 프롬프트에 보여 주고 "주제가 다른 것"을 요청한다.
3. 전체를 하나의 풀로 만들고 중복을 제거한 뒤 범주별로 고르게 train / test 로 나눈다 → 두 세트에 같은 문장이 없다.

python -m src.datagen.negatives --run negatives_v2 --train 2000 --test 400
→ data/raw/negatives_v2/ (queries.jsonl, negatives.jsonl, train.jsonl, test.jsonl, review.csv, usage.jsonl)
이후 build_dataset.py --negatives-run negatives_v2 로 최종 파일을 만든다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from src.datagen import prompts
from src.datagen.generate import (IRRELEVANT, ROOT, _WAKE_WORD, _cached, _norm, _write_jsonl,
                                  _write_review_csv, process_negatives)
from src.toy_api import FUNCTIONS


def generate_pool(llm, target_per_category: int, batch: int, avoid_last: int, workers: int, seed: int) -> list[dict]:
    """범주마다 순서대로 배치를 생성한다 (범주끼리는 병렬). 범주당 고유 질의가 target 에 닿으면 멈춘다."""

    def run_category(ci: int) -> list[dict]:
        category = prompts.NEGATIVE_CATEGORIES[ci]
        rng = random.Random(f"{seed}-{ci}")
        subtopics = prompts.NEGATIVE_SUBTOPICS[ci][:]
        rng.shuffle(subtopics)
        seen, rows = set(), []
        max_calls = math.ceil(target_per_category / batch) * 3  # 중복이 많아도 끝없이 돌지 않게
        for call in range(max_calls):
            if len(rows) >= target_per_category:
                break
            subtopic = subtopics[call % len(subtopics)]
            avoid = [r["query"] for r in rows[-avoid_last:]] if avoid_last else None
            try:
                out = llm.json(prompts.negative_queries(batch, category, subtopic, avoid), prompts.QUERIES_SCHEMA)
            except RuntimeError as e:  # 재시도를 다 쓴 경우만 건너뛴다 (400 등은 그대로 멈춤)
                print(f"  ! [{ci}] batch failed: {e}", file=sys.stderr, flush=True)
                continue
            for q in out.get("queries", []):
                key = _norm(q)
                if key and key not in seen and not _WAKE_WORD.search(q):
                    seen.add(key)
                    rows.append({"kind": "negative", "token_id": IRRELEVANT, "function": FUNCTIONS[IRRELEVANT]["name"],
                                 "variant": category, "subtopic": subtopic, "query": q.strip()})
        print(f"   [{ci}] {len(rows):4} unique after {call + 1} calls  ({category[:45]})", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_category = list(pool.map(run_category, range(len(prompts.NEGATIVE_CATEGORIES))))

    # 범주끼리 겹친 질의 제거 후 id 부여
    out, seen = [], set()
    for rows in per_category:
        for r in rows:
            key = _norm(r["query"])
            if key not in seen:
                seen.add(key)
                out.append({"id": len(out), **r})
    print(f"   pool: {len(out)} unique negatives")
    return out


def split(rows: list[dict], n_train: int, n_test: int, seed: int) -> tuple[list[dict], list[dict]]:
    """통과한 negative 를 범주별로 고르게 test 에 먼저 배정하고, 나머지에서 train 을 뽑는다."""
    rng = random.Random(seed)
    passed = [r for r in rows if r["status"] == "passed"]
    by_cat: dict[str, list[dict]] = {}
    for r in passed:
        by_cat.setdefault(r["variant"], []).append(r)
    for v in by_cat.values():
        rng.shuffle(v)

    test, rest = [], []
    quota = n_test // len(by_cat)
    for v in by_cat.values():
        test += v[:quota]
        rest += v[quota:]
    rng.shuffle(rest)
    need = n_test - len(test)  # 범주 수로 나누어떨어지지 않는 나머지
    test += rest[:need]
    train = rest[need:][:n_train]
    if len(train) < n_train or len(test) < n_test:
        print(f"  ! short: train {len(train)}/{n_train}, test {len(test)}/{n_test}", file=sys.stderr)
    keys = ["kind", "function", "token_id", "query", "args", "call"]
    return [{k: r[k] for k in keys} for r in train], [{k: r[k] for k in keys} for r in test]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="negatives_v2")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--margin", type=float, default=1.15, help="generate this many times more than train+test")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--avoid-last", type=int, default=80, help="how many earlier queries of the category to show")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--thinking", default="low", choices=["minimal", "low", "medium", "high"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print one generation prompt without calling the API")
    a = p.parse_args()

    if a.dry_run:
        ci = len(prompts.NEGATIVE_CATEGORIES) - 3
        print(prompts.negative_queries(a.batch, prompts.NEGATIVE_CATEGORIES[ci], prompts.NEGATIVE_SUBTOPICS[ci][0],
                                       ["Turn on the Wi-Fi.", "Enable airplane mode please."]))
        return

    from src.datagen.llm import Gemini
    llm = Gemini(a.model, thinking_level=a.thinking)
    out_dir = ROOT / "data" / "raw" / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    target = math.ceil((a.train + a.test) * a.margin / len(prompts.NEGATIVE_CATEGORIES))

    print(f"① negative pool: {len(prompts.NEGATIVE_CATEGORIES)} categories × {target} each")
    queries = _cached(out_dir / "queries.jsonl", lambda: generate_pool(
        llm, target, a.batch, a.avoid_last, a.workers, a.seed))
    negatives = _cached(out_dir / "negatives.jsonl", lambda: process_negatives(llm, queries, a.batch, a.workers))

    train, test = split(negatives, a.train, a.test, a.seed)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "test.jsonl", test)
    _write_review_csv(out_dir / "review.csv", negatives, train + test)

    rejected = [r for r in negatives if r["status"] != "passed"]
    print(f"\npassed {len(negatives) - len(rejected)}/{len(negatives)}  →  train {len(train)}, test {len(test)}")
    for r in rejected[:10]:
        print(f"  rejected: {r['query']!r} -> {r['reason']}")
    cost = llm.cost_usd()
    with open(out_dir / "usage.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({**llm.usage, "model": a.model, "cost_usd": cost, "args": vars(a)}, ensure_ascii=False) + "\n")
    print(f"tokens: input {llm.usage['input']:,}  output {llm.usage['output']:,}  thought {llm.usage['thought']:,}  "
          f"({llm.usage['calls']} calls)  ≈ ${cost:.3f}")
    print(f"saved: {(out_dir / 'train.jsonl').relative_to(ROOT)}, {(out_dir / 'test.jsonl').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
