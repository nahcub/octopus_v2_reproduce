"""(논문에 없는 추가) MASSIVE 질의로 만드는 두 번째 독립 테스트셋. 설계는 EXPERIMENT_DESIGN.md 3.1절.

- 출처: MASSIVE 1.1 (Amazon, alexa/massive, CC BY 4.0) 의 영어 en-US.jsonl. 영어 부분은 SLURP 문장.
- 질의 문장(utt)만 쓰고, 정답은 우리 함수 20개 기준으로 새로 붙인다 (BFCL_TEST.md 3절 기준).
  MASSIVE 의도(intent) 라벨에는 잡음이 있어서 정답으로 쓰지 않고, 라벨링 때 참고 표시로만 둔다.
- test 파티션 전체를 후보로 쓴다. 의도 라벨이 틀린 문장 속에도 positive 가 숨어 있고,
  잡담·계산 같은 negative 유형은 qa_*, general_* 의도에서 나오기 때문.

실행
    python -m src.massive_bench candidates   # 내려받기 → data/raw/massive/candidates.jsonl
    python -m src.massive_labels             # labels_src/*.txt → labels.jsonl
    python -m src.massive_bench build        # 라벨 합치기 → 최종 셋 추출 → review.csv, massive_test_draft.jsonl
    python -m src.massive_bench finalize     # review.csv 반영 → data/massive_test.jsonl
선정 규칙(함수당 상한 30, negative 는 positive 와 같은 수·유형별 같은 수, seed 0)은 BFCL 과 같은 코드(src/bfcl_bench.py)를 쓴다.
"""

from __future__ import annotations

import argparse
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from src.bfcl_bench import KEYWORDS, _norm, build, finalize, load_jsonl, write_jsonl

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
OUT_DIR = Path("data/raw/massive")
ARCHIVE = OUT_DIR / "amazon-massive-dataset-1.1.tar.gz"
EN_US = OUT_DIR / "en-US.jsonl"
LICENSE = OUT_DIR / "LICENSE"
CANDIDATES = OUT_DIR / "candidates.jsonl"
LABELS = OUT_DIR / "labels.jsonl"
DRAFT = OUT_DIR / "massive_test_draft.jsonl"
REVIEW = OUT_DIR / "review.csv"
FINAL = Path("data/massive_test.jsonl")
PARTITION = "test"

# 압축 파일 안에서 꺼낼 것만 (나머지 51개 언어는 쓰지 않음)
MEMBERS = {"1.1/data/en-US.jsonl": EN_US, "1.1/LICENSE": LICENSE}

# 라벨링 참고용: MASSIVE 의도 → 대응할 수도 있는 우리 함수 (정답 판단에는 쓰지 않음)
INTENT_HINT = {
    "weather_query": "get_weather_forecast",
    "play_music": "play_music_on_nest_hub",
    "alarm_set": "set_timer_alarm",
    "news_query": "get_trending_news",
    "transport_query": "find_route_google_maps",
    "calendar_set": "create_calendar_event",
    "email_sendemail": "send_email",
    "email_addcontact": "create_contact",
    "audio_volume_up": "set_volume",
    "audio_volume_down": "set_volume",
    "audio_volume_mute": "set_volume",
    "audio_volume_other": "set_volume",
}

# 음성 비서 호출어. 질의에서 지우지 않고 태그만 붙인다 (질의 문장은 고치지 않는다)
WAKE_WORD = re.compile(r"\b(alexa|olly|ok google|hey google|siri)\b", re.I)


def download() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if EN_US.exists() and LICENSE.exists():
        return
    if not ARCHIVE.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for name, dest in MEMBERS.items():
            f = tar.extractfile(tar.getmember(name))
            dest.write_bytes(f.read())


def cmd_candidates(_args) -> None:
    download()
    rows, seen, stats = [], set(), Counter()
    for d in load_jsonl(EN_US):
        if d["partition"] != PARTITION:
            continue
        stats["test"] += 1
        query = d["utt"]
        key = _norm(query)
        if key in seen:  # MASSIVE test 안에서 같은 문장이 여러 번 나오는 경우
            stats["drop:duplicate_query"] += 1
            continue
        seen.add(key)
        low = query.lower()
        rows.append({"cid": len(rows), "massive_id": d["id"], "intent": d["intent"], "query": query,
                     "annot": d["annot_utt"], "hint": INTENT_HINT.get(d["intent"]),
                     "kw": [n for n, p in KEYWORDS.items() if re.search(p, low)],
                     "tags": ["wake_word"] if WAKE_WORD.search(query) else []})
    write_jsonl(CANDIDATES, rows)
    print(f"{PARTITION} {stats['test']} -> candidates {len(rows)}  {dict(stats)}")
    print(f"with intent hint: {sum(1 for r in rows if r['hint'])}  with keyword hits: {sum(1 for r in rows if r['kw'])}"
          f"  wake_word: {sum(1 for r in rows if r['tags'])}")
    print("hint:", dict(Counter(r["hint"] for r in rows if r["hint"])))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("candidates").set_defaults(fn=cmd_candidates)
    sub.add_parser("build").set_defaults(fn=lambda _a: build(CANDIDATES, LABELS, DRAFT, REVIEW, "massive_id"))
    sub.add_parser("finalize").set_defaults(fn=lambda _a: finalize(DRAFT, REVIEW, FINAL))
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
