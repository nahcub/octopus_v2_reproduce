"""negative 범주별 거절 (논문에 없는 추가, 실험 X: negative 범주 하나 빼고 학습).

매핑 가설: <nexa_20> 으로 학습한 모델은 거절을 "학습 negative 에 나온 종류 → <nexa_20>" 으로 외운다.
그렇다면 학습에서 뺀 범주는 (다른 범주보다) 덜 거절해야 한다. OE 로 학습하면 그 차이가 작아야 한다.

우리 test(dev) 의 negative 를 범주별(각 50개)로 나눠, 모델마다
- greedy: 기본 생성이 <nexa_20> 을 냈나 (OE·A00 은 0 이 정상)
- @t: 거절 점수 ≥ 문턱값 (문턱값은 같은 test 의 positive 90% 가 통과하도록. negative 는 문턱값 정하는 데 안 씀)
- AUROC: 전체 positive 400 vs 그 범주 negative 50
을 계산한다. 학습에서 뺀 범주는 * 로 표시.

    python -m src.negcat_report --test-file data/test.jsonl \
        --model ours:irr runs/eval/ours/probs.jsonl \
        --model X_tok_writing:irr runs/eval/X_tok_writing/probs.jsonl --held-out X_tok_writing=writing ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.neg_categories import SHORT, load as load_neg_categories
from src.refusal_metrics import RECALL, auroc, load_jsonl, refusal_scores, threshold_at_recall


def report(rows: list[dict], probs: list[dict], score: str, recall: float) -> dict:
    cats = load_neg_categories()
    n = len(rows)
    s = refusal_scores(probs, n, score)
    by_idx = {p["index"]: p for p in probs}
    greedy = [int(bool(by_idx[i].get("func")) and by_idx[i]["func"]["chosen"] == "<nexa_20>") for i in range(n)]
    pos = [i for i in range(n) if rows[i]["kind"] == "positive"]
    t = threshold_at_recall([s[i] for i in pos], recall)
    out = {"threshold": t, "pos_recall": sum(s[i] < t for i in pos) / len(pos),
           "pos_greedy_refused": sum(greedy[i] for i in pos), "n_pos": len(pos), "cat": {}}
    for c in SHORT:
        idx = [i for i in range(n) if rows[i]["kind"] == "negative" and cats[rows[i]["query"]] == c]
        out["cat"][c] = {
            "n": len(idx),
            "greedy": sum(greedy[i] for i in idx),
            "at_t": sum(s[i] >= t for i in idx),
            "auroc": auroc([s[i] for i in pos], [s[i] for i in idx]),
        }
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    p.add_argument("--model", nargs=2, action="append", metavar=("NAME:SCORE", "PROBS"), required=True)
    p.add_argument("--held-out", action="append", default=[], metavar="NAME=CATEGORY",
                   help="학습에서 뺀 범주 표시 (NAME 은 :SCORE 앞부분)")
    p.add_argument("--recall", type=float, default=RECALL)
    p.add_argument("--out", type=Path, default=None, help="결과 json 저장 경로")
    a = p.parse_args()

    rows = load_jsonl(a.test_file)
    held = dict(h.split("=", 1) for h in a.held_out)
    results = {}
    for name_score, path in a.model:
        name, score = name_score.split(":")
        results[name_score] = report(rows, load_jsonl(Path(path)), score, a.recall)
        results[name_score]["held_out"] = held.get(name)

    w = 22
    print(f"\n문턱값: 같은 test 의 positive {a.recall:.0%} 가 통과하도록. 칸 = greedy 거절 / 문턱값 거절 / AUROC (각 범주 50개). * = 학습에서 뺀 범주")
    print(f"{'범주':<16}" + "".join(f"{k:>{w}}" for k in results))
    for c in SHORT:
        line = f"{c:<16}"
        for k, r in results.items():
            x = r["cat"][c]
            mark = "*" if r["held_out"] == c else " "
            line += f"{mark}{x['greedy']:>3}/{x['at_t']:>3}/{x['n']} {x['auroc']:.3f}".rjust(w)
        print(line)
    print(f"{'positive 통과':<16}" + "".join(f"{r['pos_recall']:>{w}.1%}" for r in results.values()))
    print(f"{'positive greedy거절':<16}" + "".join(f"{r['pos_greedy_refused']:>{w - 4}}/{r['n_pos']}" for r in results.values()))

    # 뺀 범주 요약: 같은 방식(token/oe) 의 전체 학습 모델과 비교할 수 있게 숫자만 모아 준다
    print("\n뺀 범주 요약 (문턱값 거절 수 / 50, AUROC)")
    for k, r in results.items():
        if r["held_out"]:
            x = r["cat"][r["held_out"]]
            others = [v for c, v in r["cat"].items() if c != r["held_out"]]
            print(f"  {k:<22} 뺀 범주 {r['held_out']:<14} {x['at_t']:>3}/50  AUROC {x['auroc']:.3f}"
                  f" | 나머지 범주 평균 {sum(v['at_t'] for v in others) / len(others):.1f}/50"
                  f"  AUROC {sum(v['auroc'] for v in others) / len(others):.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
