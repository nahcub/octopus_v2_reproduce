"""(논문에 없는 추가) BFCL 질의로 만드는 제3 테스트셋. 설명서는 BFCL_TEST.md.

- 출처: BFCL v4 (gorilla 저장소, Apache-2.0). 질의 문장만 쓰고, 정답은 우리 함수 20개 기준으로 새로 붙인다.
- 라벨은 data/raw/bfcl/labels.jsonl (후보 id 마다 한 줄). 이 스크립트는 라벨을 만들지 않고 합치기만 한다.

실행
    python -m src.bfcl_bench candidates     # 내려받기 → data/raw/bfcl/candidates.jsonl
    python -m src.bfcl_bench build          # labels.jsonl 합치기 → 최종 셋 추출 → review.csv, bfcl_test_draft.jsonl
    python -m src.bfcl_bench finalize       # (사람 검토 후) review.csv 반영 → data/bfcl_test.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from src.datagen.validate import check_args, render_call
from src.toy_api import FUNCTIONS, NAME_TO_INDEX

BASE = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data/"
CATEGORIES = ["simple_python", "multiple", "live_simple", "live_multiple", "live_relevance", "irrelevance", "live_irrelevance"]
OUT_DIR = Path("data/raw/bfcl")
CANDIDATES = OUT_DIR / "candidates.jsonl"
LABELS = OUT_DIR / "labels.jsonl"
DRAFT = OUT_DIR / "bfcl_test_draft.jsonl"
REVIEW = OUT_DIR / "review.csv"
FINAL = Path("data/bfcl_test.jsonl")
TRAIN = Path("data/train.jsonl")

NEG_ID = NAME_TO_INDEX["irrelevant_function"]
NEG_TYPES = ["chitchat_knowledge", "calc_science", "coding_tech", "unsupported_service", "near_miss"]
MAX_PER_FUNCTION = 30
SEED = 0

# 라벨링 때 꼼꼼히 볼 후보 표시용 (정답 판단에는 쓰지 않음)
KEYWORDS = {
    "take_a_photo": r"\b(photo|picture|selfie|camera)\b",
    "get_trending_news": r"\b(news|headlines?)\b",
    "get_weather_forecast": r"\b(weather|forecast|rain|temperature)\b",
    "send_email": r"\be-?mails?\b",
    "search_youtube_videos": r"\b(youtube|videos?)\b",
    "find_route_google_maps": r"\b(route|directions?|navigate|maps?)\b",
    "check_nest_smoke_detector_status": r"\bsmoke\b",
    "send_text_message": r"\b(text|sms|message)\b",
    "make_phone_call": r"\b(call|dial|phone)\b",
    "create_contact": r"\bcontacts?\b",
    "set_timer_alarm": r"\b(alarm|timer|remind)\b",
    "change_screen_brightness": r"\bbrightness\b",
    "create_calendar_event": r"\b(calendar|meeting|appointment|schedule)\b",
    "set_volume": r"\bvolume\b",
    "connect_to_bluetooth_device": r"\bbluetooth\b",
    "enable_do_not_disturb": r"\b(disturb|dnd|silent)\b",
    "set_nest_temperature": r"\b(thermostat|nest|heating)\b",
    "play_music_on_nest_hub": r"\b(play|song|music|track|playlist|spotify)\b",
    "play_video_on_nest_hub": r"\b(movie|show|netflix|stream)\b",
    "open_or_lock_door": r"\b(door|lock|unlock)\b",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(".!?").strip().lower()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_candidates(_args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, total, seen = [], 0, set()
    stats = Counter()
    for cat in CATEGORIES:
        path = OUT_DIR / f"BFCL_v4_{cat}.json"
        if not path.exists():
            urllib.request.urlretrieve(BASE + f"BFCL_v4_{cat}.json", path)
        for d in load_jsonl(path):
            total += 1
            q = d["question"]
            if len(q) != 1 or len(q[0]) != 1 or q[0][0]["role"] != "user":
                stats["drop:not_single_user_message"] += 1
                continue
            query = q[0][0]["content"]
            key = _norm(query)
            if key in seen:  # BFCL 안에서 같은 질의가 여러 카테고리에 나오는 경우
                stats["drop:duplicate_query"] += 1
                continue
            seen.add(key)
            low = query.lower()
            rows.append({"cid": len(rows), "bfcl_id": d["id"], "category": cat, "query": query,
                         "kw": [n for n, p in KEYWORDS.items() if re.search(p, low)]})
    write_jsonl(CANDIDATES, rows)
    print(f"total {total} -> candidates {len(rows)}  {dict(stats)}")
    print(f"with keyword hits: {sum(1 for r in rows if r['kw'])}")


def _tags(query: str) -> list[str]:
    tags = []
    if len(query) > 300:
        tags.append("long")
    letters = [c for c in query if c.isalpha()]
    if letters and sum(1 for c in letters if c.isascii()) / len(letters) < 0.8:
        tags.append("non_english")
    return tags


def merge(candidates: Path = CANDIDATES, labels_path: Path = LABELS, src_key: str = "bfcl_id") -> tuple[list[dict], Counter]:
    """candidates + labels → 라벨 붙은 행 목록. 라벨 검사도 여기서.
    src_key: 원본 데이터의 문항 id 필드 이름 (BFCL "bfcl_id", MASSIVE "massive_id"). 후보에 "tags" 가 있으면 이어 붙인다."""
    cands = {r["cid"]: r for r in load_jsonl(candidates)}
    labels = {}
    for l in load_jsonl(labels_path):
        assert l["cid"] not in labels, f"duplicate label cid {l['cid']}"
        labels[l["cid"]] = l
    missing = sorted(set(cands) - set(labels))
    assert not missing, f"{len(missing)} candidates without label, e.g. {missing[:10]}"
    train = {_norm(r["query"]) for r in load_jsonl(TRAIN)}
    out, stats = [], Counter()
    for cid, c in cands.items():
        l = labels[cid]
        label = l["label"]
        stats[f"label:{'positive' if label not in ('irrelevant_function', 'exclude') else label}"] += 1
        if label == "exclude":
            continue
        tid = NAME_TO_INDEX[label]
        row = {"kind": "negative" if tid == NEG_ID else "positive", "function": label, "token_id": tid,
               "query": c["query"], "args": {}, "call": render_call(tid, {}) if tid == NEG_ID else None,
               "seen_in_train": _norm(c["query"]) in train, src_key: c[src_key], "cid": cid,
               "score_args": False, "args_allowed": None, "style": None, "neg_type": None,
               "tags": _tags(c["query"]) + c.get("tags", []), "note": l.get("note", "")}
        if tid == NEG_ID:
            assert l.get("neg_type") in NEG_TYPES, f"cid {cid}: bad neg_type {l.get('neg_type')}"
            row["neg_type"] = l["neg_type"]
            row["score_args"] = True
        else:
            assert l.get("style") in ("imperative", "question"), f"cid {cid}: bad style"
            row["style"] = l["style"]
            if l.get("args") is not None:
                errs = check_args(label, l["args"])
                assert not errs, f"cid {cid}: {errs}"
                row["args"] = l["args"]
                row["call"] = render_call(tid, l["args"])
                row["score_args"] = bool(l.get("score_args", True))
                allowed = l.get("args_allowed") or {}
                row["args_allowed"] = {p: list(dict.fromkeys([l["args"][p], *allowed.get(p, [])]))
                                       for p in FUNCTIONS[tid]["params"]}
        out.append(row)
    return out, stats


def select(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_func = defaultdict(list)
    for r in rows:
        if r["kind"] == "positive":
            by_func[r["function"]].append(r)
    pos = []
    for f in sorted(by_func):
        group = sorted(by_func[f], key=lambda r: r["cid"])
        pos += group if len(group) <= MAX_PER_FUNCTION else rng.sample(group, MAX_PER_FUNCTION)
    # negative: positive 와 같은 수, 유형별로 같은 수 (모자란 유형은 전부, 남는 몫은 나머지에 고르게)
    by_type = {t: sorted([r for r in rows if r["neg_type"] == t], key=lambda r: r["cid"]) for t in NEG_TYPES}
    need, quota, open_types = len(pos), {t: 0 for t in NEG_TYPES}, [t for t in NEG_TYPES if by_type[t]]
    while need > 0 and open_types:
        share = max(1, need // len(open_types))
        for t in list(open_types):
            take = min(share, len(by_type[t]) - quota[t], need)
            quota[t] += take
            need -= take
            if quota[t] >= len(by_type[t]):
                open_types.remove(t)
            if need == 0:
                break
    neg = []
    for t in NEG_TYPES:
        neg += rng.sample(by_type[t], quota[t]) if quota[t] < len(by_type[t]) else by_type[t]
    return pos + neg


def build(candidates: Path = CANDIDATES, labels_path: Path = LABELS, draft: Path = DRAFT, review: Path = REVIEW,
          src_key: str = "bfcl_id") -> None:
    rows, stats = merge(candidates, labels_path, src_key)
    chosen = select(rows)
    write_jsonl(draft, chosen)
    with review.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cid", src_key, "kind", "function", "call", "score_args", "args_allowed",
                    "style", "neg_type", "tags", "query", "note", "OK(Y/N)", "fix"])
        for r in chosen:
            w.writerow([r["cid"], r[src_key], r["kind"], r["function"], r["call"] or "", r["score_args"],
                        json.dumps(r["args_allowed"], ensure_ascii=False) if r["args_allowed"] else "",
                        r["style"] or "", r["neg_type"] or "", ",".join(r["tags"]), r["query"], r["note"], "", ""])
    print(dict(stats))
    print_summary(rows, chosen)


def cmd_build(_args) -> None:
    build()


def print_summary(rows: list[dict], chosen: list[dict]) -> None:
    pos_all = Counter(r["function"] for r in rows if r["kind"] == "positive")
    neg_all = Counter(r["neg_type"] for r in rows if r["kind"] == "negative")
    pos = [r for r in chosen if r["kind"] == "positive"]
    neg = [r for r in chosen if r["kind"] == "negative"]
    print(f"labeled positive {sum(pos_all.values())}  negative {sum(neg_all.values())}")
    print(f"chosen positive {len(pos)}  negative {len(neg)}")
    print("positive by function (chosen/all):",
          {f: f"{sum(1 for r in pos if r['function'] == f)}/{n}" for f, n in sorted(pos_all.items())})
    print("negative by type (chosen/all):",
          {t: f"{sum(1 for r in neg if r['neg_type'] == t)}/{neg_all[t]}" for t in NEG_TYPES})
    print("positive style:", dict(Counter(r["style"] for r in pos)))
    print("score_args true positive:", sum(1 for r in pos if r["score_args"]))
    print("seen_in_train:", sum(1 for r in chosen if r["seen_in_train"]))
    print("tags:", dict(Counter(t for r in chosen for t in r["tags"])))


def finalize(draft: Path = DRAFT, review: Path = REVIEW, final: Path = FINAL) -> None:
    """review.csv 의 OK=N 줄은 fix 칸(JSON: 바꿀 필드)을 반영, fix 가 "exclude" 면 뺀다."""
    rows = {r["cid"]: r for r in load_jsonl(draft)}
    with review.open(encoding="utf-8-sig") as f:
        for rec in csv.DictReader(f):
            if rec["OK(Y/N)"].strip().upper() != "N":
                continue
            cid, fix = int(rec["cid"]), rec["fix"].strip()
            if fix == "exclude":
                rows.pop(cid)
                continue
            r = rows[cid]
            r.update(json.loads(fix))
            tid = NAME_TO_INDEX[r["function"]]
            r["token_id"], r["kind"] = tid, "negative" if tid == NEG_ID else "positive"
            if r["kind"] == "positive" and r["args"]:
                assert not check_args(r["function"], r["args"]), f"cid {cid}"
                r["call"] = render_call(tid, r["args"])
            elif r["kind"] == "negative":
                r["args"], r["call"] = {}, render_call(tid, {})
    write_jsonl(final, list(rows.values()))
    print(f"wrote {final} ({len(rows)} rows)")


def cmd_finalize(_args) -> None:
    finalize()


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("candidates").set_defaults(fn=cmd_candidates)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("finalize").set_defaults(fn=cmd_finalize)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
