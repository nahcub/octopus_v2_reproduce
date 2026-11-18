"""(논문에 없는 추가) data/raw/massive/labels_src/*.txt (압축 라벨) → data/raw/massive/labels.jsonl.

형식은 BFCL 라벨(src/bfcl_labels.py, data/raw/bfcl/labels_src/README.txt)과 같고, 한 가지를 더한다.
    D <intent> <type>    이 MASSIVE 의도의 후보 중 P/N/X 줄로 따로 적지 않은 것은 전부 이 라벨.
                         type: ck cs ct us nm (negative 유형) 또는 x (제외)
    D 줄은 그 의도의 문장을 전부 읽고, 예외를 따로 적은 뒤에만 쓴다 (기본값으로 몰아넣기용이 아님).

MASSIVE 질의는 전부 소문자라서, 인자 정답이 있는 positive 는 문자열 인자마다 소문자 표기를 허용 값에 자동으로 더한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.bfcl_bench import load_jsonl, write_jsonl
from src.bfcl_labels import TYPES, parse_line

SRC = Path("data/raw/massive/labels_src")
CANDIDATES = Path("data/raw/massive/candidates.jsonl")
OUT = Path("data/raw/massive/labels.jsonl")


def _add_lowercase(row: dict) -> dict:
    if not row.get("args"):
        return row
    allowed = row.get("args_allowed") or {}
    for p, v in row["args"].items():
        if isinstance(v, str):
            vals = [v, *allowed.get(p, [])]
            allowed[p] = list(dict.fromkeys(vals + [x.lower() for x in vals]))
    row["args_allowed"] = allowed
    return row


def main() -> None:
    cands = load_jsonl(CANDIDATES)
    explicit, defaults = {}, {}
    for f in sorted(SRC.glob("[0-9]*.txt")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                if line.startswith("D "):
                    _, intent, t = line.split()
                    assert intent not in defaults, f"duplicate default {intent}"
                    assert t == "x" or t in TYPES, f"bad type {t}"
                    defaults[intent] = t
                    continue
                for r in parse_line(line):
                    assert r["cid"] not in explicit, f"duplicate cid {r['cid']}"
                    explicit[r["cid"]] = _add_lowercase(r)
            except Exception as e:
                raise SystemExit(f"{f.name}:{n}: {e}\n  {line}")
    rows, missing = [], []
    known = {c["cid"] for c in cands}
    assert set(explicit) <= known, f"unknown cids {sorted(set(explicit) - known)[:10]}"
    for c in cands:
        if c["cid"] in explicit:
            rows.append(explicit[c["cid"]])
        elif c["intent"] in defaults:
            t = defaults[c["intent"]]
            rows.append({"cid": c["cid"], "label": "exclude", "note": f"{c['intent']} 기본 제외"} if t == "x" else
                        {"cid": c["cid"], "label": "irrelevant_function", "neg_type": TYPES[t], "note": f"{c['intent']} 기본"})
        else:
            missing.append((c["cid"], c["intent"]))
    assert not missing, f"{len(missing)} candidates without label, e.g. {missing[:10]}"
    write_jsonl(OUT, rows)
    print(f"{len(rows)} labels ({len(explicit)} explicit, {len(rows) - len(explicit)} by intent default) -> {OUT}")


if __name__ == "__main__":
    main()
