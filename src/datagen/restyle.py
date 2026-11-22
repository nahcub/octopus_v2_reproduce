"""(논문에 없는 추가, 실험 S, EXPERIMENT_DESIGN.md 4.4절) 같은 문장을 말투만 바꿔 다시 쓴다.

내용(함수·인자)은 그대로 두고 말투만 바꾼다. positive 는 바꾼 문장 + 원래 인자로 호출이 맞는지,
negative 는 바꾼 뒤에도 20개 함수 중 처리할 수 있는 게 없는지 Gemini 로 확인한다.

    # 프롬프트만 보기
    python -m src.datagen.restyle rewrite --source dev --dry-run
    # 소규모 (함수 2개 × 5개 + negative 10개, 말투 5개)
    python -m src.datagen.restyle rewrite --source train --run restyle_pilot --functions take_a_photo,set_volume --per-function 5 --negatives 10
    # 전체
    python -m src.datagen.restyle rewrite --source train --run restyle_train
    python -m src.datagen.restyle rewrite --source dev --run restyle_dev
    # 학습·채점 파일 만들기 → data/style/
    python -m src.datagen.restyle build

결과: data/raw/<run>/<말투>.jsonl (말투마다, 이미 있으면 건너뜀), usage.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.datagen import prompts
from src.datagen.generate import ROOT, _chunks, _norm, _read_jsonl, _run_parallel, _write_jsonl
from src.toy_api import FUNCTIONS, NAME_TO_INDEX

IRRELEVANT = NAME_TO_INDEX["irrelevant_function"]

# 기존 생성 조건(prompts.STYLES: 짧은 명령, 공손한 문장, 구어, 맥락 포함, Can you 질문, 소문자 타이핑)에 없는 말투
STYLES = {
    "howto": ("Rephrase it as a question asking HOW to do it or WHETHER/WHERE something is possible or what the "
              "information is, instead of a command. E.g. 'How do I ...?', 'Is there a way to ...?', "
              "'What's the ...?', 'Where can I ...?'. Do not use 'Can you' / 'Could you'."),
    "fragment": ("Rewrite it as a terse fragment or note with no full sentence: mostly nouns and values, "
                 "little or no verb, like a sticky note or a search box entry. E.g. 'Mom, call, 6 PM'."),
    "need": ("Rewrite it as a statement of the user's need, wish or situation, without directly telling the "
             "assistant what to do. E.g. 'I really need to talk to my mom around six.', "
             "'It's way too loud in here, 15 would be better.'"),
    "form": ("Rewrite it as a structured, form-like or key-value request, as if typed into a form or generated "
             "by a program. Values may be written in machine style (e.g. 18:00, 2025-03-03) but must stay "
             "exactly equivalent. Field values for the action must be plain words, never a software or API-style "
             "identifier (no snake_case names like take_photo or send_message). E.g. 'action=call; who=Mom; time=18:00'."),
    "nonnative": ("Rewrite it in the English of a non-native speaker: slightly wrong grammar, odd word order or "
                  "word choice, missing articles, but still understandable. E.g. 'Please you make call to "
                  "mother at 6 PM time.'"),
}

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "query": {"type": "string"}},
        "required": ["id", "query"]}}},
    "required": ["results"],
}


def rewrite_prompt(style: str, items: list[dict]) -> str:
    payload = [{"id": it["id"], "query": it["query"]} for it in items]
    return f"""You are rewriting user requests to a phone assistant into a different speaking style, for a study of how wording affects a model.

Target style: {STYLES[style]}

Rules:
- Keep EXACTLY the same request: same action, same target, same app or device, and every concrete value (names, numbers, phone numbers, email addresses, dates, times, places, quoted message or email text). Do not add or drop any information.
- Only the wording style changes. Do not make the request easier or harder to understand than needed for the style.
- Do not address the assistant by name and do not add greetings.
- Never use software function or API names (e.g. take_a_photo, send_text_message, set_volume) or snake_case identifiers.
- Write each rewrite independently; vary the wording across items.

Requests (JSON):
{json.dumps(payload, ensure_ascii=False, indent=1)}"""


def verify_positive_prompt(fn: dict, items: list[dict]) -> str:
    """prompts.verify_positive 와 같되 '자연스러운가' 조건을 뺐다 (fragment·form 은 일부러 부자연스럽다)."""
    payload = [{"id": it["id"], "query": it["query"], "arguments": it["args"]} for it in items]
    return f"""You are checking data for a function-calling model. The requests are deliberately written in unusual styles (questions about how to do something, fragments, statements of need, form-like text, non-native English). Do not judge the style.
For each item, decide whether calling the API below with the given arguments correctly and completely fulfills what the user wants.
Treat a question like 'How do I ...?' or 'Is there a way to ...?' about doing something, and a statement of need, as an indirect request to do it.

API:
{fn["description"]}

Mark an item invalid if any of these is true:
- The request does not actually want what this API does (it wants something else).
- An argument is wrong, or contains information the request does not support (except documented defaults, and composed message/email text when the user asked for it).
- An argument breaks the format or allowed values in the parameter descriptions.

Items (JSON):
{json.dumps(payload, ensure_ascii=False, indent=1)}"""


# ---------------------------------------------------------------- 원문 고르기

def load_source(source: str, functions: list[str] | None, per_function: int | None, negatives: int | None,
                seed: int) -> list[dict]:
    """train: data/train.jsonl 전부. dev: data/test.jsonl 에서 positive 함수당 10, negative 범주당 25 (seed 고정).
    --functions 등을 주면 그 안에서 더 줄인다 (소규모 실행)."""
    from src.neg_categories import SHORT, load as load_cats
    rng = random.Random(seed)
    rows = _read_jsonl(ROOT / "data" / ("train.jsonl" if source == "train" else "test.jsonl"))
    for i, r in enumerate(rows):
        r["src_index"] = i
    if source == "dev":
        cats = load_cats()
        by_fn, by_cat = defaultdict(list), defaultdict(list)
        for r in rows:
            (by_fn[r["function"]] if r["kind"] == "positive" else by_cat[cats[r["query"]]]).append(r)
        rows = [r for f in sorted(by_fn) for r in rng.sample(by_fn[f], 10)]
        rows += [r for c in SHORT for r in rng.sample(by_cat[c], 25)]
    pos = [r for r in rows if r["kind"] == "positive" and (not functions or r["function"] in functions)]
    neg = [r for r in rows if r["kind"] == "negative"]
    if per_function:
        by_fn = defaultdict(list)
        for r in pos:
            by_fn[r["function"]].append(r)
        pos = [r for f in by_fn for r in rng.sample(by_fn[f], min(per_function, len(by_fn[f])))]
    if negatives is not None:
        neg = rng.sample(neg, min(negatives, len(neg)))
    out = pos + neg
    for k, r in enumerate(out):
        r["id"] = k
    return out


# ---------------------------------------------------------------- 재작성 + 검증

def _abort_if_failed(outs: list, style: str, step: str) -> None:
    """재시도까지 실패한 배치가 있으면 파일을 쓰지 않고 멈춘다 (실패를 '검증 탈락' 으로 저장하지 않도록).
    예: 2026-11-23 Gemini 월 지출 한도 429 로 nonnative 검증이 멈춤"""
    failed = sum(o is None for o in outs)
    if failed:
        raise SystemExit(f"[{style}] {step}: {failed}/{len(outs)} batches failed → 저장 안 함. 원인 해결 후 같은 명령으로 다시 실행")


def restyle(llm, items: list[dict], style: str, batch: int, workers: int) -> list[dict]:
    rng = random.Random(0)
    shuffled = items[:]
    rng.shuffle(shuffled)  # 배치 안에 여러 함수가 섞이게 (같은 함수만 모이면 문장이 서로 닮는다)
    chunks = _chunks(shuffled, batch)
    print(f"[{style}] rewrite: {len(chunks)} calls")
    new = {}
    outs = _run_parallel(llm, [(rewrite_prompt(style, c), REWRITE_SCHEMA) for c in chunks], workers)
    _abort_if_failed(outs, style, "rewrite")
    for out in outs:
        for r in (out or {}).get("results", []):
            new[r["id"]] = r["query"].strip()

    rows = []
    for it in items:
        q = new.get(it["id"])
        row = {"id": it["id"], "src_index": it["src_index"], "style": style, "kind": it["kind"],
               "function": it["function"], "token_id": it["token_id"], "orig_query": it["query"],
               "query": q or "", "args": it["args"], "call": it["call"], "status": "pending", "reason": ""}
        if not q:
            row["status"], row["reason"] = "rejected_rewrite", "no rewrite returned"
        elif _norm(q) == _norm(it["query"]):
            row["reason"] = "unchanged"  # 원문이 이미 그 말투 (예: howto 의 질문형 negative). 검증은 똑같이 받는다
        rows.append(row)

    todo = [r for r in rows if r["status"] == "pending"]
    by_fn = defaultdict(list)
    for r in todo:
        if r["kind"] == "positive":
            by_fn[r["token_id"]].append(r)
    jobs, groups = [], []
    for tid, rs in by_fn.items():
        for c in _chunks(rs, 10):
            jobs.append((verify_positive_prompt(FUNCTIONS[tid], c), prompts.VERIFY_POSITIVE_SCHEMA))
            groups.append(("positive", c))
    for c in _chunks([r for r in todo if r["kind"] == "negative"], 20):
        jobs.append((prompts.verify_negative(c), prompts.VERIFY_NEGATIVE_SCHEMA))
        groups.append(("negative", c))
    print(f"[{style}] verify: {len(jobs)} calls")
    outs = _run_parallel(llm, jobs, workers)
    _abort_if_failed(outs, style, "verify")
    for (kind, c), out in zip(groups, outs):
        got = {v["id"]: v for v in (out or {}).get("results", [])}
        for r in c:
            v = got.get(r["id"])
            if v is None:
                r["status"], r["reason"] = "rejected_llm", "no verdict returned"
            elif kind == "positive":
                r["status"], r["reason"] = ("passed", "") if v["valid"] else ("rejected_llm", v["reason"])
            else:
                r["status"], r["reason"] = ("passed", "") if v["function"] == "none" else ("rejected_llm", f"handled by {v['function']}")
    c = Counter((r["kind"], r["status"]) for r in rows)
    print(f"[{style}] " + ", ".join(f"{k}/{s} {n}" for (k, s), n in sorted(c.items())))
    return rows


def cmd_rewrite(a) -> None:
    styles = list(STYLES) if a.styles == "all" else a.styles.split(",")
    functions = a.functions.split(",") if a.functions else None
    items = load_source(a.source, functions, a.per_function, a.negatives, a.seed)
    print(f"원문 {len(items)}개 (positive {sum(r['kind'] == 'positive' for r in items)}), 말투 {styles}")
    if a.dry_run:
        print("\n" + "=" * 30 + " rewrite " + "=" * 30 + "\n" + rewrite_prompt(styles[0], items[:3]))
        pos = [r for r in items if r["kind"] == "positive"][:2]
        print("\n" + "=" * 30 + " verify positive " + "=" * 30 + "\n" + verify_positive_prompt(FUNCTIONS[pos[0]["token_id"]], pos))
        return
    from src.datagen.llm import Gemini
    llm = Gemini(a.model, thinking_level=a.thinking)
    out_dir = ROOT / "data" / "raw" / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    for st in styles:
        path = out_dir / f"{st}.jsonl"
        if path.exists():
            print(f"[skip] {path.relative_to(ROOT)} already exists")
            continue
        _write_jsonl(path, restyle(llm, items, st, a.batch, a.workers))
    cost = llm.cost_usd()
    with open(out_dir / "usage.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({**llm.usage, "model": a.model, "cost_usd": cost, "args": vars(a)}, ensure_ascii=False) + "\n")
    print(f"\n{llm.usage['calls']} calls ≈ ${cost:.3f}" if cost is not None else "")


# ---------------------------------------------------------------- 학습·채점 파일

TRAIN_SETS = {"S_all": list(STYLES), "S_minus_howto": [s for s in STYLES if s != "howto"],
              "S_minus_fragment": [s for s in STYLES if s != "fragment"]}
KEYS = ["kind", "function", "token_id", "query", "args", "call"]


def cmd_build(a) -> None:
    out = ROOT / "data" / "style"
    out.mkdir(exist_ok=True)
    train = _read_jsonl(ROOT / "data" / "train.jsonl")
    passed = defaultdict(dict)  # src_index → {style: query}
    for st in STYLES:
        for r in _read_jsonl(ROOT / "data" / "raw" / a.train_run / f"{st}.jsonl"):
            if r["status"] == "passed":
                passed[r["src_index"]][st] = r["query"]
    for name, styles in TRAIN_SETS.items():
        rng = random.Random(a.seed)
        choices = ["orig"] + styles
        # positive·negative 가 같은 비율로 말투를 나눠 갖게 kind 별로 돌아가며 배정 (섞은 순서로)
        rows, used = [], Counter()
        for kind in ("positive", "negative"):
            idx = [i for i, r in enumerate(train) if r["kind"] == kind]
            rng.shuffle(idx)
            for j, i in enumerate(idx):
                st = choices[j % len(choices)]
                q = train[i]["query"] if st == "orig" else passed[i].get(st)
                if q is None:  # 그 말투 버전이 검증에서 떨어짐 → 원문
                    st, q = "orig", train[i]["query"]
                used[(kind, st)] += 1
                rows.append({**{k: train[i][k] for k in KEYS}, "query": q, "style": st})
        rows.sort(key=lambda r: (r["kind"] != "positive", r["token_id"]))
        _write_jsonl(out / f"train_{name}.jsonl", rows)
        print(f"train_{name}.jsonl {len(rows)}개: " + ", ".join(f"{k[:3]}/{s} {n}" for (k, s), n in sorted(used.items())))

    # 채점: dev 400 × (orig + 5). 한 말투라도 검증에서 떨어진 원문은 모든 말투에서 뺀다 (말투끼리 같은 문장으로 비교)
    dev = {st: {r["src_index"]: r for r in _read_jsonl(ROOT / "data" / "raw" / a.dev_run / f"{st}.jsonl")} for st in STYLES}
    test = _read_jsonl(ROOT / "data" / "test.jsonl")
    keep = sorted(i for i in dev[next(iter(STYLES))] if all(dev[st][i]["status"] == "passed" for st in STYLES))
    rows = []
    for st in ["orig"] + list(STYLES):
        for i in keep:
            r = test[i]
            rows.append({**{k: r[k] for k in KEYS}, "query": r["query"] if st == "orig" else dev[st][i]["query"],
                         "style": st, "src_index": i, "seen_in_train": False})
    _write_jsonl(out / "dev_styled.jsonl", rows)
    n_pos = sum(test[i]["kind"] == "positive" for i in keep)
    print(f"dev_styled.jsonl {len(rows)}개 = 원문 {len(keep)} (positive {n_pos}) × {len(STYLES) + 1} 말투 "
          f"(뽑은 {len(dev['howto'])} 중 모든 말투 통과한 것)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rewrite")
    r.add_argument("--source", choices=["train", "dev"], required=True)
    r.add_argument("--run", default="restyle_pilot")
    r.add_argument("--styles", default="all")
    r.add_argument("--functions", default=None, help="쉼표로 구분 (소규모 실행)")
    r.add_argument("--per-function", type=int, default=None)
    r.add_argument("--negatives", type=int, default=None)
    r.add_argument("--batch", type=int, default=20)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--model", default="gemini-3.8-flash")
    r.add_argument("--thinking", default="low", choices=["minimal", "low", "medium", "high"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--train-run", default="restyle_train")
    b.add_argument("--dev-run", default="restyle_dev")
    b.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"rewrite": cmd_rewrite, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
