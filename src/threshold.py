"""(논문에 없는 추가) 6단계 실험: 거절 지표 + "임계값만 바꾸는" 기준선. 학습 없음, CPU 로 충분.

거절 점수 s
    함수 토큰을 고르는 자리에서 21개 함수 토큰 확률(probs.jsonl 의 func.dist)만 놓고
    s = P(<nexa_20>) / (21개 확률 합)   ← "함수 토큰 중에서 거절을 얼마나 원하나" (0~1)
    함수 토큰을 아예 안 고른 줄(형식 오류)은 s 가 없다 → 임계값을 적용하지 않고, AUROC 에서도 뺀다.

1) AUROC: negative(거절해야 함)와 positive(불러야 함)를 s 로 얼마나 잘 나누나. 임계값과 상관없는 점수.
   1.0 = 완벽히 나눔, 0.5 = 동전 던지기. 임계값을 옮기는 것만으로는 이 값이 안 바뀐다.
2) 임계값 기준선 (A + τ): 원래 출력(greedy)에 규칙 하나를 더한다.
       s ≥ τ 이면 원래 함수를 불렀더라도 거절(<nexa_20>()) 로 바꾼다.
   τ 를 낮출수록 오호출은 줄고 과잉거절은 는다 (줄다리기).
   τ 고르기: val 에서 전체 정확도가 가장 높은 τ. 동점이면 가장 큰 τ (원래 출력을 가장 덜 바꾸는 쪽).
   --match-overrefusal R 을 주면 대신 "test positive 과잉거절률이 R 이하인 가장 작은 τ" 를 쓴다
   → C 와 과잉거절을 똑같이 맞춰 놓고 오호출을 비교할 때.

입력: 모델·평가셋마다 폴더 하나. 그 안에 같은 질의 순서로 만든
    probs.jsonl   (python -m src.probs generate ... --test-file <평가셋> --out <폴더>/probs.jsonl)
    scored.jsonl  (python -m src.evaluate score ... 이 만든 것)

실행
    python -m src.threshold --val runs/eval/A/val --eval test=runs/eval/A/test holdout=runs/eval/A/holdout
    python -m src.threshold --val ... --eval ... --match-overrefusal 0.05 --out runs/eval/A/threshold.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from src.evaluate import NEG_ID
from src.preprocess.prompt import load_jsonl


def load_dir(d: Path) -> list[dict]:
    """probs.jsonl + scored.jsonl → 줄마다 {kind, category, outcome, s}."""
    probs = {r["index"]: r for r in load_jsonl(d / "probs.jsonl")}
    rows = []
    for sc in load_jsonl(d / "scored.jsonl"):
        pr = probs.get(sc["index"])
        assert pr is not None, f"{d}: index {sc['index']} 가 probs.jsonl 에 없음"
        assert pr["query"] == sc["query"], f"{d}: index {sc['index']} 질의가 다름 (다른 파일로 만든 듯)"
        f = pr["func"]
        s = f["dist"][NEG_ID] / sum(f["dist"]) if f and sum(f["dist"]) > 0 else None
        rows.append({"kind": sc["kind"], "category": sc.get("category"), "outcome": sc["outcome"], "s": s})
    return rows


def refused(r: dict, tau: float | None) -> bool:
    """최종적으로 거절했나. 원래 거절(<nexa_20> 선택)했거나, 임계값 규칙에 걸렸거나."""
    if r["kind"] == "negative":
        orig = r["outcome"] == "correct"  # negative 정답 = <nexa_20>() 를 정확히 출력
    else:
        orig = r["outcome"] == "function:missed_call"
    return orig or (tau is not None and r["s"] is not None and r["s"] >= tau)


def correct(r: dict, tau: float | None) -> bool:
    if r["kind"] == "negative":
        return refused(r, tau)
    return r["outcome"] == "correct" and not refused(r, tau)


def rates(rows: list[dict], tau: float | None) -> dict:
    neg = [r for r in rows if r["kind"] == "negative"]
    pos = [r for r in rows if r["kind"] == "positive"]
    out = {"n": len(rows), "acc": _mean([correct(r, tau) for r in rows])}
    if neg:
        out["misfire"] = _mean([not refused(r, tau) for r in neg])  # 오호출률: 거절했어야 하는데 안 함 (형식 오류 포함)
    if pos:
        out["overrefusal"] = _mean([refused(r, tau) for r in pos])  # 과잉거절률: 불렀어야 하는데 거절
    by_cat = defaultdict(list)
    for r in neg:
        by_cat[r["category"]].append(not refused(r, tau))
    if len(by_cat) > 1 or None not in by_cat:
        out["misfire_by_category"] = {c: _mean(v) for c, v in sorted(by_cat.items(), key=lambda x: str(x[0]))}
    return out


def auroc(neg_scores: list[float], pos_scores: list[float]) -> float:
    """P(무작위 negative 의 s > 무작위 positive 의 s), 동점은 절반. (Mann-Whitney U / n1·n2)"""
    if not neg_scores or not pos_scores:
        return float("nan")
    allv = sorted([(v, 1) for v in neg_scores] + [(v, 0) for v in pos_scores])
    rank_sum, i = 0.0, 0
    while i < len(allv):  # 같은 값은 평균 순위
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    n1, n2 = len(neg_scores), len(pos_scores)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n2)


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def pick_tau(val: list[dict]) -> float:
    cands = sorted({r["s"] for r in val if r["s"] is not None} | {math.inf})
    best = max(cands, key=lambda t: (rates(val, t)["acc"], t))  # 동점이면 큰 τ
    return best


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val", type=Path, required=True, help="val 평가 폴더 (τ 고르기용)")
    p.add_argument("--eval", nargs="+", required=True, help="이름=폴더 (예: test=runs/eval/A/test holdout=...)")
    p.add_argument("--match-overrefusal", type=float, default=None,
                   help="τ 를 val 대신 'test positive 과잉거절률 ≤ R' 인 가장 작은 τ 로 고른다")
    p.add_argument("--positives-from", default="test", help="AUROC·과잉거절 맞추기에 쓸 positive 가 있는 평가셋 이름")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    val = load_dir(a.val)
    sets = {}
    for item in a.eval:
        name, d = item.split("=", 1)
        sets[name] = load_dir(Path(d))
    pos_rows = [r for r in sets[a.positives_from] if r["kind"] == "positive"]

    if a.match_overrefusal is None:
        tau, how = pick_tau(val), "val 전체 정확도 최대"
    else:
        # positive 점수 "바로 위" 값들이 후보: τ 가 그 positive 를 막 안 건드리게 되는 지점
        cands = sorted({math.nextafter(r["s"], math.inf) for r in pos_rows if r["s"] is not None} | {0.0, math.inf})
        ok = [t for t in cands if rates(pos_rows, t)["overrefusal"] <= a.match_overrefusal]
        tau, how = min(ok), f"{a.positives_from} 과잉거절 ≤ {a.match_overrefusal:.1%} 인 가장 작은 τ"

    result = {"tau": tau if tau != math.inf else None, "tau_rule": how, "sets": {}}
    print(f"τ = {tau:.4f} ({how})" if tau != math.inf else f"τ = 없음 (규칙을 안 쓰는 게 가장 좋음, {how})")
    print(f"\n{'평가셋':<10}{'':<8}{'정확도':>9}{'오호출':>9}{'과잉거절':>10}{'AUROC':>9}{'s 없음':>8}")
    for name, rows in [("val", val)] + list(sets.items()):
        neg_s = [r["s"] for r in rows if r["kind"] == "negative" and r["s"] is not None]
        own_pos = [r["s"] for r in rows if r["kind"] == "positive" and r["s"] is not None]
        pos_s = own_pos or [r["s"] for r in pos_rows if r["s"] is not None]  # holdout 처럼 negative 만 있으면 test positive 와 비교
        au = auroc(neg_s, pos_s)
        missing = sum(r["s"] is None for r in rows)
        res = {"auroc": au, "auroc_positives": "own" if own_pos else a.positives_from, "no_score": missing,
               "greedy": rates(rows, None), "with_tau": rates(rows, tau if tau != math.inf else None)}
        result["sets"][name] = res
        for label, m in (("원래", res["greedy"]), ("+τ", res["with_tau"])):
            cell = lambda k: f"{m[k]:>9.2%}" if k in m else f"{'-':>9}"
            print(f"{name if label == '원래' else '':<10}{label:<8}{cell('acc')}{cell('misfire')}{cell('overrefusal'):>10}"
                  + (f"{au:>9.4f}{missing:>8}" if label == "원래" else ""))
        cats = res["with_tau"].get("misfire_by_category")
        if cats:
            print(" " * 18 + "오호출(+τ) 범주별: " + ", ".join(f"{c} {v:.1%}" for c, v in cats.items()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[저장] {a.out}")


if __name__ == "__main__":
    main()
