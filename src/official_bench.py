"""(논문에 없는 추가) 공식 평가셋으로 평가: 모델 카드에 링크된 android_benchmark.xlsx → test.jsonl 형식.

- 출처: HF `NexaAIDev/Octopus-v2` 의 `android_benchmark.xlsx` (모델 카드: "manually verified").
  210개 = 함수 20개 + irrelevant_function 각 10개. Octopus 의 T 개수가 논문 표 1과 같다 (209/210 = 99.524% 등).
- 질의·정답은 엑셀 그대로 쓴다 (중복 질의도 그대로 둔다. 논문 숫자가 210개 기준이라서).
  정답은 ground_truth_params (위치 인자 리스트) 를 함수 인자 이름에 짝지어 args 로 만든다.
- 엑셀은 git 에 넣지 않는다 (CC-BY-NC). 이 스크립트가 내려받아 변환한다.

실행
    python -m src.official_bench convert                  # → data/raw/official/official_test.jsonl
    python -m src.official_bench check                    # nexa 가 기록한 Octopus 출력을 우리 채점기로 다시 채점, 엑셀 T/F 와 비교
    python -m src.evaluate generate --model ... --test-file data/raw/official/official_test.jsonl --out .../preds.jsonl
    python -m src.evaluate score --preds .../preds.jsonl --test-file data/raw/official/official_test.jsonl
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from src.datagen.validate import check_args, render_call
from src.evaluate import judge
from src.preprocess.prompt import load_jsonl
from src.toy_api import FUNCTIONS, NAME_TO_INDEX

URL = "https://huggingface.co/NexaAIDev/Octopus-v2/resolve/main/android_benchmark.xlsx"
OUT_DIR = Path("data/raw/official")
XLSX = OUT_DIR / "android_benchmark.xlsx"
TEST = OUT_DIR / "official_test.jsonl"
# 엑셀의 Octopus 출력 열 → 논문 표 1 이름 (표 2 설정)
OCTOPUS_COLUMNS = {"nexa_gemma_2b": "Octopus-0", "nexa_gemma_2b_lora": "Octopus-1",
                   "nexa_gemma_2b_50_percent": "Octopus-2", "nexa_gemma_2b_10_percent": "Octopus-3"}


def read_xlsx(path: Path) -> list[list]:
    """openpyxl 없이 첫 시트를 행 목록으로 읽는다 (빈 칸은 None)."""
    z = zipfile.ZipFile(path)
    shared = [html.unescape(re.sub(r"<[^>]+>", "", s))
              for s in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode("utf-8"), re.S)]
    rows = []
    for row in re.findall(r"<row [^>]*>(.*?)</row>", z.read("xl/worksheets/sheet1.xml").decode("utf-8"), re.S):
        cells = {}
        for ref, attrs, body in re.findall(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', row, re.S):
            v = re.search(r"<v>(.*?)</v>", body or "")
            v = html.unescape(v.group(1)) if v else None
            if v is not None and 't="s"' in attrs:
                v = shared[int(v)]
            col = 0
            for ch in ref:
                col = col * 26 + ord(ch) - 64
            cells[col - 1] = v
        rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
    return rows


def load_sheet() -> list[dict]:
    if not XLSX.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[다운로드] {URL}")
        urllib.request.urlretrieve(URL, XLSX)
    rows = read_xlsx(XLSX)
    header = rows[0]
    # 결과 열 이름이 겹치므로 (nexa_gemma_2b_result 가 두 번) 출력 열 바로 뒤 칸을 그 출력의 T/F 로 본다
    out = []
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        r = r + [None] * (len(header) - len(r))
        d = {"query": r[0], "function": r[1], "params": r[2]}
        for col, name in OCTOPUS_COLUMNS.items():
            i = header.index(col)
            d[name] = {"output": r[i], "result": r[i + 1]}
        out.append(d)
    return out


def convert(args) -> None:
    sheet = load_sheet()
    train = {r["query"].strip().lower() for r in load_jsonl(Path("data/train.jsonl"))}
    rows, problems = [], []
    for i, s in enumerate(sheet):
        tid = NAME_TO_INDEX[s["function"]]
        values = ast.literal_eval(s["params"])
        params = FUNCTIONS[tid]["params"]
        assert len(values) == len(params), f"{i}: {s['function']} 인자 개수 {values} vs {params}"
        a = dict(zip(params, values))
        if errs := check_args(s["function"], a):
            problems.append((i, s["query"][:60], errs))  # 공식 정답이 우리 규칙과 다른 경우 (정답은 그대로 둔다)
        rows.append({"kind": "negative" if s["function"] == "irrelevant_function" else "positive",
                     "function": s["function"], "token_id": tid, "query": s["query"], "args": a,
                     "call": render_call(tid, a), "seen_in_train": s["query"].strip().lower() in train})
    with TEST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[저장] {TEST}: {len(rows)}개 {dict(Counter(r['kind'] for r in rows))}, "
          f"seen_in_train {sum(r['seen_in_train'] for r in rows)}개, 질의 중복 {len(rows) - len({r['query'] for r in rows})}개")
    print(f"[우리 RULES 를 어기는 공식 정답] {len(problems)}개")
    for p in problems:
        print("  ", *p)


def check(args) -> None:
    """엑셀의 Octopus 출력을 우리 채점기로 채점 → nexa 의 T/F 와 비교. 채점 기준이 얼마나 같은지 본다."""
    sheet, test = load_sheet(), load_jsonl(TEST)
    for name in OCTOPUS_COLUMNS.values():
        strict = lenient = nexa_t = 0
        diff = []
        for s, row in zip(sheet, test):
            o = s[name]
            j = judge(row, o["output"] or "")
            ok, t = j["outcome"] == "correct", o["result"] in ("T", "t")
            strict, lenient, nexa_t = strict + ok, lenient + j["lenient_correct"], nexa_t + t
            if ok != t:
                diff.append((t, j["outcome"], row["query"][:70], row["call"][:80], (o["output"] or "")[:80]))
        n = len(test)
        print(f"\n== {name}: nexa 판정 {nexa_t}/{n} ({nexa_t / n:.3%}) | 우리 엄격 {strict}/{n} ({strict / n:.3%})"
              f" | 우리 느슨 {lenient}/{n} ({lenient / n:.3%}) | 판정 다름 {len(diff)}개")
        for t, outcome, q, gold, out in diff:
            print(f"   nexa={'T' if t else 'F'} 우리={outcome}\n     Q {q}\n     정답 {gold}\n     출력 {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="공식 평가셋 (android_benchmark.xlsx) 변환·점검")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("convert", help=f"엑셀 → {TEST}")
    sub.add_parser("check", help="엑셀의 Octopus 출력을 우리 채점기로 채점해 nexa T/F 와 비교")
    a = p.parse_args()
    {"convert": convert, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
