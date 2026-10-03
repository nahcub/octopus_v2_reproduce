"""2단계 데이터 생성 파이프라인 (논문 3.3절: 질의 생성 → 인자 생성 → 검증·재생성, negative 샘플).

예) 파일럿:  python -m src.datagen.generate --run pilot --functions take_a_photo,set_timer_alarm --per-function 20 --negatives 20
    프롬프트만 보기: python -m src.datagen.generate --dry-run

결과는 data/raw/<run>/ 에 단계별로 저장된다. 같은 --run 으로 다시 실행하면 이미 끝난 단계는 건너뛴다.
- queries.jsonl     생성된 질의 (중복 제거 후)
- positives.jsonl   positive 전체 + 검증 결과(status, reason)
- negatives.jsonl   negative 전체 + 검증 결과
- dataset.jsonl     통과한 것 중 요청 개수만큼 뽑은 최종 데이터
- usage.jsonl       실행별 토큰 사용량 / 비용
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.datagen import prompts
from src.datagen.pools import ValuePools
from src.datagen.validate import args_schema, check_args, render_call
from src.toy_api import FUNCTIONS, NAME_TO_INDEX

ROOT = Path(__file__).resolve().parents[2]
IRRELEVANT = NAME_TO_INDEX["irrelevant_function"]


# ---------------------------------------------------------------- 유틸

# 모델 이름이 정해지지 않았으므로 질의에서 assistant 를 부르는 호출어는 쓰지 않는다 (프롬프트로 금지 + 여기서 한 번 더 거름)
_WAKE_WORD = re.compile(r"^\W*(hey|hi|ok|okay)\s+(google|phone|assistant|siri|alexa|bixby|nexa|octopus)\b", re.I)


def _norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def _chunks(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _cached(path: Path, build):
    if path.exists():
        print(f"[skip] {path.relative_to(ROOT)} already exists")
        return _read_jsonl(path)
    rows = build()
    _write_jsonl(path, rows)
    return rows


def _run_parallel(llm, jobs: list[tuple[str, dict]], workers: int) -> list[dict | None]:
    """(prompt, schema) 목록을 병렬 호출. 재시도까지 실패한 배치는 None.
    400 등 재시도해도 소용없는 오류는 그대로 올려서 실행 전체를 멈춘다 (빈 결과가 캐시되지 않도록)."""
    done = 0
    lock = threading.Lock()

    def call(job):
        nonlocal done
        try:
            result = llm.json(*job)
        except RuntimeError as e:  # llm.json 이 재시도를 다 쓴 경우
            print(f"  ! batch failed: {e}", file=sys.stderr, flush=True)
            result = None
        with lock:
            done += 1
            print(f"   ... {done}/{len(jobs)}", flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(call, jobs))


# ---------------------------------------------------------------- ① 질의 생성

def generate_queries(llm, token_ids: list[int], per_function: int, negatives: int,
                     overgen: float, batch: int, workers: int, seed: int, split: str) -> list[dict]:
    rng = random.Random(seed)
    pools = ValuePools(split, seed)
    jobs, metas = [], []
    for tid in token_ids:
        fn = FUNCTIONS[tid]
        for b in range(math.ceil(per_function * overgen / batch)):
            style = rng.choice(prompts.STYLES)
            candidates = pools.candidates(fn["name"], batch)
            jobs.append((prompts.positive_queries(fn, batch, style, candidates), prompts.QUERIES_SCHEMA))
            metas.append({"kind": "positive", "token_id": tid, "variant": style})
    # 범주 순서를 섞어서, 배치 수가 적어도 헷갈리기 쉬운 범주(뒤쪽 셋)가 빠지지 않게 한다
    categories = rng.sample(prompts.NEGATIVE_CATEGORIES, len(prompts.NEGATIVE_CATEGORIES))
    for b in range(math.ceil(negatives * overgen / batch)):
        category = categories[b % len(categories)]
        jobs.append((prompts.negative_queries(batch, category), prompts.QUERIES_SCHEMA))
        metas.append({"kind": "negative", "token_id": IRRELEVANT, "variant": category})

    print(f"① queries: {len(jobs)} calls")
    rows, seen, dupes, wake = [], set(), 0, 0
    for meta, out in zip(metas, _run_parallel(llm, jobs, workers)):
        for q in (out or {}).get("queries", []):
            key = _norm(q)
            if not key or key in seen:
                dupes += 1
                continue
            if _WAKE_WORD.search(q):
                wake += 1
                continue
            seen.add(key)
            rows.append({"id": len(rows), **meta, "function": FUNCTIONS[meta["token_id"]]["name"], "query": q.strip()})
    print(f"   {len(rows)} unique queries ({dupes} duplicates, {wake} wake-word queries dropped)")
    return rows


# ---------------------------------------------------------------- ② 인자 생성 + ③ 검증 (positive)

def process_positives(llm, queries: list[dict], batch: int, workers: int, rounds: int) -> list[dict]:
    rows = {q["id"]: {**q, "args": None, "status": "pending", "reason": ""} for q in queries if q["kind"] == "positive"}
    feedback: dict[int, str] = {}
    pending = list(rows)

    for rnd in range(1, rounds + 1):
        if not pending:
            break
        by_fn = defaultdict(list)
        for i in pending:
            by_fn[rows[i]["token_id"]].append(rows[i])

        # ② 인자 생성 (인자 없는 함수는 호출 생략)
        jobs, groups = [], []
        for tid, items in by_fn.items():
            name = FUNCTIONS[tid]["name"]
            if not FUNCTIONS[tid]["params"]:
                for it in items:
                    it["args"] = {}
                continue
            schema = prompts.arguments_schema(args_schema(name))
            for chunk in _chunks(items, batch):
                jobs.append((prompts.arguments(FUNCTIONS[tid], chunk, feedback), schema))
                groups.append(chunk)
        print(f"② arguments (round {rnd}): {len(jobs)} calls")
        for chunk, out in zip(groups, _run_parallel(llm, jobs, workers)):
            got = {r["id"]: r["args"] for r in (out or {}).get("results", [])}
            for it in chunk:
                it["args"] = got.get(it["id"])

        # 규칙 검사
        rule_ok = []
        for i in pending:
            it = rows[i]
            errors = ["no arguments returned"] if it["args"] is None else check_args(it["function"], it["args"])
            if errors:
                it["status"], it["reason"] = "rejected_rule", "; ".join(errors)
            else:
                rule_ok.append(it)

        # ③ LLM 검증 (논문의 binary verification)
        by_fn = defaultdict(list)
        for it in rule_ok:
            by_fn[it["token_id"]].append(it)
        jobs, groups = [], []
        for tid, items in by_fn.items():
            for chunk in _chunks(items, batch):
                jobs.append((prompts.verify_positive(FUNCTIONS[tid], chunk), prompts.VERIFY_POSITIVE_SCHEMA))
                groups.append(chunk)
        print(f"③ verify (round {rnd}): {len(jobs)} calls")
        for chunk, out in zip(groups, _run_parallel(llm, jobs, workers)):
            verdicts = {r["id"]: r for r in (out or {}).get("results", [])}
            for it in chunk:
                v = verdicts.get(it["id"])
                if v is None:
                    it["status"], it["reason"] = "rejected_llm", "no verdict returned"
                elif v["valid"]:
                    it["status"], it["reason"] = "passed", ""
                else:
                    it["status"], it["reason"] = "rejected_llm", v["reason"]

        pending = [i for i in pending if rows[i]["status"] != "passed"]
        feedback = {i: rows[i]["reason"] for i in pending}
        print(f"   passed so far: {sum(r['status'] == 'passed' for r in rows.values())}/{len(rows)}")

    for it in rows.values():
        if it["status"] == "passed":
            it["call"] = render_call(it["token_id"], it["args"])
    return list(rows.values())


# ---------------------------------------------------------------- ③ 검증 (negative)

def process_negatives(llm, queries: list[dict], batch: int, workers: int) -> list[dict]:
    rows = [{**q, "args": {}, "status": "pending", "reason": ""} for q in queries if q["kind"] == "negative"]
    chunks = _chunks(rows, batch * 2)
    jobs = [(prompts.verify_negative(c), prompts.VERIFY_NEGATIVE_SCHEMA) for c in chunks]
    print(f"③ verify negatives: {len(jobs)} calls")
    for chunk, out in zip(chunks, _run_parallel(llm, jobs, workers)):
        verdicts = {r["id"]: r["function"] for r in (out or {}).get("results", [])}
        for it in chunk:
            match = verdicts.get(it["id"])
            if match is None:
                it["status"], it["reason"] = "rejected_llm", "no verdict returned"
            elif match == "none":
                it["status"], it["call"] = "passed", render_call(IRRELEVANT, {})
            else:
                it["status"], it["reason"] = "rejected_llm", f"handled by {match}"
    return rows


# ---------------------------------------------------------------- ④ 최종 선택 + 리포트

def select(positives, negatives, token_ids, per_function, n_negatives, seed) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for tid in token_ids:
        passed = [r for r in positives if r["token_id"] == tid and r["status"] == "passed"]
        rng.shuffle(passed)
        out += passed[:per_function]
    passed = [r for r in negatives if r["status"] == "passed"]
    rng.shuffle(passed)
    out += passed[:n_negatives]
    keys = ["kind", "function", "token_id", "query", "args", "call"]
    return [{k: r[k] for k in keys} for r in out]


def report(positives, negatives, dataset, token_ids, per_function, n_negatives) -> None:
    print("\n=== report ===")
    used = Counter(r["token_id"] for r in dataset)
    for tid in token_ids + [IRRELEVANT]:
        rows = [r for r in (negatives if tid == IRRELEVANT else positives) if r["token_id"] == tid]
        status = Counter(r["status"] for r in rows)
        want = n_negatives if tid == IRRELEVANT else per_function
        flag = "" if used[tid] >= want else "  << short"
        print(f"<nexa_{tid}> {FUNCTIONS[tid]['name']:34} generated {len(rows):4}  passed {status['passed']:4}  "
              f"rule-rejected {status['rejected_rule']:3}  llm-rejected {status['rejected_llm']:3}  used {used[tid]}/{want}{flag}")
    rejected = [r for r in positives + negatives if r["status"].startswith("rejected")]
    if rejected:
        print("\nsample rejections:")
        for r in rejected[:10]:
            print(f"  [{r['function']}] {r['query']!r}\n      -> {r['status']}: {r['reason']}")


def _write_review_csv(path: Path, rows: list[dict], dataset: list[dict]) -> None:
    """사람 검토용. 엑셀에서 한글이 깨지지 않게 utf-8-sig. my_check 칸은 검토자가 채운다."""
    used = {r["query"] for r in dataset}
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["used_in_dataset", "status", "function", "query", "call", "reason", "my_check"])
        for r in rows:
            w.writerow(["Y" if r["query"] in used else "", r["status"], r["function"], r["query"],
                        r.get("call", ""), r["reason"], ""])


# ---------------------------------------------------------------- main

def _dry_run(token_ids: list[int]) -> None:
    fn = FUNCTIONS[token_ids[0]]
    example = [{"id": 0, "query": "<example query>", "args": {p: "<value>" for p in fn["params"]}}]
    for title, text in [
        ("① positive queries", prompts.positive_queries(fn, 10, prompts.STYLES[0],
                                                       ValuePools("train", 0).candidates(fn["name"], 10))),
        ("① negative queries", prompts.negative_queries(10, prompts.NEGATIVE_CATEGORIES[5])),
        ("② arguments", prompts.arguments(fn, example)),
        ("③ verify positive", prompts.verify_positive(fn, example)),
        ("③ verify negative", prompts.verify_negative(example)),
    ]:
        print(f"\n{'=' * 30} {title} {'=' * 30}\n{text}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="pilot", help="output folder name under data/raw/")
    p.add_argument("--functions", default="all", help="comma-separated function names, or 'all'")
    p.add_argument("--per-function", type=int, default=20, help="positive samples to keep per function")
    p.add_argument("--negatives", type=int, default=20, help="negative samples to keep")
    p.add_argument("--overgen", type=float, default=1.5, help="generate this many times more than needed")
    p.add_argument("--batch", type=int, default=10, help="items per LLM call")
    p.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    p.add_argument("--rounds", type=int, default=2, help="argument generation attempts per query")
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--thinking", default="low", choices=["minimal", "low", "medium", "high"])
    p.add_argument("--split", default="train", choices=["train", "test"],
                   help="which half of the name/city pools to use (test values never appear in train)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print prompts without calling the API")
    a = p.parse_args()

    if a.functions == "all":
        token_ids = [i for i in range(len(FUNCTIONS)) if i != IRRELEVANT]
    else:
        names = [n.strip() for n in a.functions.split(",")]
        unknown = [n for n in names if n not in NAME_TO_INDEX or n == "irrelevant_function"]
        if unknown:
            p.error(f"unknown function(s): {unknown}")
        token_ids = sorted(NAME_TO_INDEX[n] for n in names)

    if a.dry_run:
        _dry_run(token_ids)
        return

    from src.datagen.llm import Gemini  # dry-run 은 SDK/키 없이도 동작하도록
    llm = Gemini(a.model, thinking_level=a.thinking)
    out_dir = ROOT / "data" / "raw" / a.run
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = _cached(out_dir / "queries.jsonl", lambda: generate_queries(
        llm, token_ids, a.per_function, a.negatives, a.overgen, a.batch, a.workers, a.seed, a.split))
    positives = _cached(out_dir / "positives.jsonl", lambda: process_positives(
        llm, queries, a.batch, a.workers, a.rounds))
    negatives = _cached(out_dir / "negatives.jsonl", lambda: process_negatives(
        llm, queries, a.batch, a.workers))
    dataset = select(positives, negatives, token_ids, a.per_function, a.negatives, a.seed)
    _write_jsonl(out_dir / "dataset.jsonl", dataset)
    _write_review_csv(out_dir / "review.csv", positives + negatives, dataset)

    report(positives, negatives, dataset, token_ids, a.per_function, a.negatives)
    cost = llm.cost_usd()
    usage = {**llm.usage, "model": a.model, "thinking": a.thinking, "cost_usd": cost, "args": vars(a)}
    with open(out_dir / "usage.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(usage, ensure_ascii=False) + "\n")
    print(f"\ntokens: input {llm.usage['input']:,}  output {llm.usage['output']:,}  thought {llm.usage['thought']:,}  "
          f"({llm.usage['calls']} calls)  ≈ ${cost:.3f}" if cost is not None else "")
    print(f"saved: {(out_dir / 'dataset.jsonl').relative_to(ROOT)} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
