"""(논문에 없는 추가) 정답 유무 + softmax 확률을 같이 보는 채점. 실험 A00(negative 없이 학습) 용으로 만들었지만 어떤 모델에도 쓴다.

evaluate score 는 greedy 출력만 본다. negative 없이 학습한 모델은 <nexa_20> 을 낼 수 없어서 negative 정확도가 늘 0 이 된다.
그래서 여기서는 probs.jsonl 의 함수 토큰 확률로 거절 점수를 만들고, 문턱값을 넘으면 거절로 바꿔서 다시 채점한다.

거절 점수 (refusal_metrics.py 와 같은 정의, 높을수록 거절 쪽)
- irr: <nexa_20> 확률
- msp: 1 - (함수 토큰 <nexa_0>~<nexa_19> 중 가장 높은 확률)  (Maximum Softmax Probability, Hendrycks & Gimpel 2017)
- 함수 토큰을 한 번도 고르지 않은 출력: irr 은 0 (거절 안 함), msp 는 그 단계 상위 k개의 함수 토큰 확률로 계산
  (자세한 규칙은 refusal_metrics.refusal_scores)

문턱값 t: --dev-probs 를 주면 dev(우리 test) positive 의 90% 가 거절되지 않게 dev 에서 정한다 (refusal_metrics 와 같은 방식).
         test 와 dev 가 같은 파일이면 dev 에서 정하고 dev 에 쓰는 것이라 낙관적인 숫자다 (화면에 표시).

문턱값 적용 판정 (한 문항)
- 거절 = (점수 >= t) 또는 greedy 출력이 <nexa_20>
- negative: 거절이면 정답
- positive: 거절 안 했고 greedy 채점이 correct 면 정답 (거절했으면 threshold_refused 로 오답)

실행 (CPU, evaluate score 와 probs generate 가 끝난 뒤)
    python -m src.softmax_score --test-file data/test.jsonl --scored .../ours/scored.jsonl --probs .../ours/probs.jsonl \
        --score msp --dev-probs .../ours/probs.jsonl
출력: 화면 표 + scored.jsonl 옆에 softmax_<score>_scored.jsonl (문항별), softmax_<score>_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from src.refusal_metrics import NEG_ID, RECALL, SCORES, auroc, load_jsonl, refusal_scores, threshold_at_recall

DEV_TEST = Path("data/test.jsonl")
BINS = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]


def func_info(p: dict) -> dict:
    """probs 한 줄 → 함수 토큰 단계의 요약 (없으면 None 들)."""
    f = p.get("func")
    if not f:
        return {"top_func": None, "p_top_func": None, "p_irrelevant": None, "p_chosen": None, "chosen": None}
    d = f["dist"][:NEG_ID]
    j = max(range(len(d)), key=d.__getitem__)
    return {"top_func": j, "p_top_func": d[j], "p_irrelevant": f["p_irrelevant"], "p_chosen": f["p_chosen"], "chosen": f["chosen"]}


def judge_with_threshold(s: dict, score: float, t: float) -> str:
    greedy_refused = s["outcome"] == "correct" and s["kind"] == "negative" or s["outcome"] == "function:missed_call"
    refused = score >= t or greedy_refused
    if s["kind"] == "negative":
        return "correct" if refused else "false_call"
    if refused:
        return "threshold_refused" if not greedy_refused else "function:missed_call"
    return "correct" if s["outcome"] in ("correct", "function_only") else s["outcome"]


def med(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):.4f}" if xs else "-"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="(논문에 없는 추가) 정답 + softmax 채점")
    ap.add_argument("--test-file", type=Path, required=True)
    ap.add_argument("--scored", type=Path, required=True, help="evaluate score 가 만든 scored.jsonl")
    ap.add_argument("--probs", type=Path, required=True, help="probs generate 가 만든 probs.jsonl (같은 test)")
    ap.add_argument("--score", choices=SCORES, default="msp")
    ap.add_argument("--dev-probs", type=Path, default=None, help="dev(우리 test 800) probs. 없으면 --threshold 필요")
    ap.add_argument("--dev-test", type=Path, default=DEV_TEST)
    ap.add_argument("--threshold", type=float, default=None, help="문턱값을 직접 줄 때")
    ap.add_argument("--recall", type=float, default=RECALL)
    a = ap.parse_args()

    test = load_jsonl(a.test_file)
    scored = sorted(load_jsonl(a.scored), key=lambda s: s["index"])
    probs = load_jsonl(a.probs)
    assert len(scored) == len(test), f"scored {len(scored)} != test {len(test)}"
    scores = refusal_scores(probs, len(test), a.score)
    by_idx = {p["index"]: p for p in probs}

    if a.threshold is not None:
        t, t_from = a.threshold, "직접 지정"
    else:
        assert a.dev_probs, "--dev-probs 또는 --threshold 가 필요"
        dev_rows = load_jsonl(a.dev_test)
        ds = refusal_scores(load_jsonl(a.dev_probs), len(dev_rows), a.score)
        t = threshold_at_recall([v for v, r in zip(ds, dev_rows) if r["kind"] == "positive"], a.recall)
        t_from = f"dev positive recall {a.recall:.0%}"
        if a.dev_probs.resolve() == a.probs.resolve():
            t_from += " (⚠️ test = dev: 같은 셋에서 정하고 적용, 낙관적)"

    rows = []
    for s, v in zip(scored, scores):
        info = func_info(by_idx[s["index"]])
        rows.append({"index": s["index"], "kind": s["kind"], "function": s["function"], "query": s["query"],
                     "greedy_outcome": s["outcome"], "pred_call": s["pred_call"], **info,
                     "refusal_score": round(v, 6), "outcome_t": judge_with_threshold(s, v, t)})

    pos = [r for r in rows if r["kind"] == "positive"]
    neg = [r for r in rows if r["kind"] == "negative"]

    def acc(rs, key):
        good = ("correct", "function_only")
        return sum(r[key] in good for r in rs) / len(rs) if rs else float("nan")

    summary = {
        "score": a.score, "threshold": t, "threshold_from": t_from, "n": len(rows),
        "auroc": auroc([r["refusal_score"] for r in pos], [r["refusal_score"] for r in neg]),
        "greedy": {"acc": acc(rows, "greedy_outcome"), "acc_positive": acc(pos, "greedy_outcome"), "acc_negative": acc(neg, "greedy_outcome")},
        "threshold_applied": {"acc": acc(rows, "outcome_t"), "acc_positive": acc(pos, "outcome_t"), "acc_negative": acc(neg, "outcome_t")},
        "outcomes_t": dict(sorted(Counter(r["outcome_t"] for r in rows).items())),
        "no_function_token": sum(r["top_func"] is None for r in rows),
    }

    print(f"test {a.test_file} (positive {len(pos)}, negative {len(neg)}) | 점수 {a.score} | 문턱값 t = {t:.4g} ({t_from})")
    rule = "점수 0 으로 둠" if a.score == "irr" else "그 단계 상위 k개의 함수 토큰 확률로 점수 계산"
    print(f"함수 토큰을 고르지 않은 출력: {summary['no_function_token']}개 ({rule})\n")
    print(f"{'':<26}{'greedy':>10}{'문턱값 적용':>14}")
    for k, label in (("acc", "전체 정확도"), ("acc_positive", "  positive"), ("acc_negative", "  negative")):
        print(f"{label:<26}{summary['greedy'][k]:>10.2%}{summary['threshold_applied'][k]:>14.2%}")
    print(f"{'AUROC (문턱값 무관)':<26}{summary['auroc']:>10.3f}")
    print(f"\n[문턱값 적용 결과 종류] {summary['outcomes_t']}")

    print(f"\n[softmax 확률 중앙값]  p_top = 함수 20개 중 최대 확률, p_irr = <nexa_20> 확률, score = 거절 점수({a.score})")
    groups = {
        "positive greedy 정답": [r for r in pos if r["greedy_outcome"] in ("correct", "function_only")],
        "positive greedy 오답": [r for r in pos if r["greedy_outcome"] not in ("correct", "function_only")],
        "negative greedy 정답(거절)": [r for r in neg if r["greedy_outcome"] == "correct"],
        "negative greedy 오답(호출)": [r for r in neg if r["greedy_outcome"] != "correct"],
    }
    print(f"{'':<28}{'개수':>6}{'p_top':>10}{'p_irr':>10}{'score':>10}")
    for name, rs in groups.items():
        print(f"{name:<28}{len(rs):>6}{med([r['p_top_func'] for r in rs]):>10}{med([r['p_irrelevant'] for r in rs]):>10}"
              f"{med([r['refusal_score'] for r in rs]):>10}")

    print(f"\n[거절 점수 분포] 구간별 개수")
    print(f"{'구간':<16}{'positive':>10}{'negative':>10}")
    hist = []
    for lo, hi in zip(BINS, BINS[1:]):
        np_, nn = (sum(lo <= r["refusal_score"] < hi for r in rs) for rs in (pos, neg))
        hist.append({"lo": lo, "hi": min(hi, 1.0), "positive": np_, "negative": nn})
        print(f"[{lo:.2f}, {min(hi, 1.0):.2f}{')' if hi < 1 else ']'}{'':<4}{np_:>10}{nn:>10}")
    summary["histogram"] = hist

    print("\n[negative 중 가장 확신하며 호출한 10개] (거절 점수 낮은 순)")
    for r in sorted(neg, key=lambda r: r["refusal_score"])[:10]:
        top = f"<nexa_{r['top_func']}> p={r['p_top_func']:.3f}" if r["top_func"] is not None else "함수 토큰 없음"
        print(f"  #{r['index']} score={r['refusal_score']:.4f} {top} | {r['query'][:80]}")
    print("\n[positive 중 가장 거절 쪽인 10개] (거절 점수 높은 순)")
    for r in sorted(pos, key=lambda r: -r["refusal_score"])[:10]:
        print(f"  #{r['index']} score={r['refusal_score']:.4f} 정답 {r['function']} | greedy {r['greedy_outcome']} | {r['query'][:70]}")

    out = a.scored.parent
    with (out / f"softmax_{a.score}_scored.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / f"softmax_{a.score}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[저장] {out}/softmax_{a.score}_scored.jsonl, softmax_{a.score}_summary.json")


if __name__ == "__main__":
    main()
