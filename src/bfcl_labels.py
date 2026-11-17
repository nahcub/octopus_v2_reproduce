"""(논문에 없는 추가) data/raw/bfcl/labels_src/*.txt (압축 라벨) → data/raw/bfcl/labels.jsonl. 형식은 labels_src/README.txt."""

from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path("data/raw/bfcl/labels_src")
OUT = Path("data/raw/bfcl/labels.jsonl")
TYPES = {"ck": "chitchat_knowledge", "cs": "calc_science", "ct": "coding_tech", "us": "unsupported_service", "nm": "near_miss"}


def _ids(spec: str) -> list[int]:
    out = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def parse_line(line: str) -> list[dict]:
    kind, rest = line[0], line[2:].strip()
    if kind == "N":
        rest, _, note = rest.partition("##")
        t, ids = rest.split(None, 1)
        return [{"cid": c, "label": "irrelevant_function", "neg_type": TYPES[t], "note": note.strip()} for c in _ids(ids)]
    if kind == "X":
        cid, note = (rest.split(None, 1) + [""])[:2]
        return [{"cid": int(cid), "label": "exclude", "note": note}]
    if kind == "P":
        note = ""
        if "##" in rest:
            rest, note = rest.split("##", 1)
        allowed = None
        if "||" in rest:
            rest, a = rest.split("||", 1)
            allowed = json.loads(a)
        cid, func, style, args = rest.strip().split(None, 3)
        args = args.strip()
        row = {"cid": int(cid), "label": func, "style": {"q": "question", "i": "imperative"}[style], "note": note.strip()}
        if args == "-":
            row["args"], row["score_args"] = None, False
        else:
            row["args"], row["score_args"], row["args_allowed"] = json.loads(args), True, allowed
        return [row]
    raise ValueError(line)


def main() -> None:
    rows, seen = [], set()
    for f in sorted(SRC.glob("[0-9]*.txt")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                for r in parse_line(line):
                    assert r["cid"] not in seen, f"duplicate cid {r['cid']}"
                    seen.add(r["cid"])
                    rows.append(r)
            except Exception as e:
                raise SystemExit(f"{f.name}:{n}: {e}\n  {line}")
    rows.sort(key=lambda r: r["cid"])
    with OUT.open("w", encoding="utf-8", newline="\n") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} labels -> {OUT}")


if __name__ == "__main__":
    main()
