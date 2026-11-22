"""negative 범주 라벨 (논문에 없는 추가, 실험 X: negative 범주 하나 빼고 학습).

train/test negative 는 negatives.py 가 8개 범주(prompts.NEGATIVE_CATEGORIES)로 나눠 생성했지만
data/train.jsonl·test.jsonl 에는 범주 칸이 없다. 생성 기록(data/raw/negatives_v2, git 제외)에서
질의 → 범주 표를 만들어 data/neg_categories.json 으로 커밋해 둔다 (Colab 에는 data/raw 가 없으므로).

    python -m src.neg_categories build    # 랩탑에서 한 번
    python -m src.neg_categories show     # 범주별 개수
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "neg_categories.json"

# prompts.NEGATIVE_CATEGORIES 와 같은 순서의 짧은 이름. 뒤쪽 셋이 지원 함수와 헷갈리기 쉬운 near-miss 범주
SHORT = ["knowledge", "math", "smalltalk", "advice", "writing", "phone_settings", "other_apps", "smart_home"]


def load() -> dict[str, str]:
    """{질의: 범주 짧은 이름}"""
    return json.loads(PATH.read_text(encoding="utf-8"))


def build() -> None:
    from src.datagen import prompts
    assert len(SHORT) == len(prompts.NEGATIVE_CATEGORIES)
    to_short = dict(zip(prompts.NEGATIVE_CATEGORIES, SHORT))
    raw = ROOT / "data" / "raw" / "negatives_v2" / "queries.jsonl"
    by_query = {}
    for line in raw.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_query[r["query"]] = to_short[r["variant"]]
    out = {}
    for name in ("train", "test"):
        rows = [json.loads(l) for l in (ROOT / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        neg = [r["query"] for r in rows if r["kind"] == "negative"]
        missing = [q for q in neg if q not in by_query]
        assert not missing, f"{name}: 범주를 못 찾은 negative {len(missing)}개, 예: {missing[:3]}"
        out.update({q: by_query[q] for q in neg})
    PATH.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"saved {PATH.relative_to(ROOT)} ({len(out)}개)")


def show() -> None:
    cats = load()
    for name in ("train", "test"):
        rows = [json.loads(l) for l in (ROOT / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        c = Counter(cats[r["query"]] for r in rows if r["kind"] == "negative")
        print(name, {k: c[k] for k in SHORT})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"build": build, "show": show}[sys.argv[1] if len(sys.argv) > 1 else "show"]()
