"""(논문에 없는 추가) 거절 경계 지표. 설계는 EXPERIMENT_DESIGN.md 6절.

`<nexa_20>`(irrelevant_function) 확률 p_irrelevant (probs.jsonl 의 func.p_irrelevant) 하나로 positive / negative 를 가른다.
- AUROC: negative 의 p_irrelevant 가 positive 보다 높을 확률 (같으면 0.5). 문턱값과 무관
- 거절@recall90: dev(우리 test) positive 의 90% 가 함수 쪽(p_irrelevant < t)으로 가게 하는 문턱값 t 를 dev 에서 정하고,
  그 t 를 test 에 그대로 적용했을 때 test negative 중 p_irrelevant >= t 인 비율. test positive 쪽 recall 도 같이 보고
- negative 유형별: 같은 t 로 유형마다 거절 비율
- 신뢰구간: test 문항을 복원 추출하는 bootstrap 1,000회 (seed 0), 95% 구간. 모델이 둘 이상이면 첫 모델과의
  차이를 **같은 표본**으로 뽑아 계산한다 (짝지은 bootstrap). 문턱값 t 는 dev 에서 한 번 정한 값을 고정
- 함수 토큰을 한 번도 고르지 않은 출력(p_irrelevant 가 null)은 거절하지 않은 것으로 보고 0 으로 둔다

실행 (CPU 로 충분)
    python -m src.refusal_metrics --test data/massive_test.jsonl \
        --model ours  .../eval_massive/ours/probs.jsonl  .../eval/ours/probs.jsonl \
        --model public .../eval_massive/public/probs.jsonl .../eval/public/probs.jsonl
    --model 다음은 이름, test probs, dev probs (dev 는 data/test.jsonl 로 만든 probs. 없으면 - 로 두고 AUROC 만 본다)
    이름 끝에 :msp 를 붙이면 그 모델은 p_irrelevant 대신 MSP 점수를 쓴다 (예: --model A00:msp ...). 기본은 :irr
    (논문에 없는 추가: msp 점수, 실험 A00 용)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DEV_TEST = Path("data/test.jsonl")
NEG_ID = 20  # <nexa_20> = irrelevant_function (src/evaluate.py 와 같은 값)
RECALL = 0.90
N_BOOT = 1000
SEED = 0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


SCORES = ("irr", "msp")


def refusal_scores(probs: list[dict], n: int, score: str = "irr") -> list[float]:
    """문항마다 거절 점수 (높을수록 거절 쪽).
    irr: <nexa_20> 확률 (p_irrelevant)
    msp: 1 - (함수 토큰 <nexa_0>~<nexa_19> 중 가장 높은 확률). Maximum Softmax Probability (Hendrycks & Gimpel 2017).
         <nexa_20> 을 쓰지 않으므로 negative 없이 학습한 모델(실험 A00)에도 쓸 수 있다. 확률은 전체 사전 기준 (다시 정규화하지 않음)
    함수 토큰을 한 번도 고르지 않은 출력(func 가 null)
    - irr: 0 (거절 안 함). 기존 규칙 그대로
    - msp: 함수 토큰을 골랐어야 할 단계(공백이 아닌 첫 토큰)의 상위 k개에서 함수 토큰 최대 확률을 쓴다.
           상위 k개에 함수 토큰이 없으면 k번째 확률을 상한으로 쓴다 (실제 점수보다 낮거나 같음 = 보수적).
           (2026-11-22 추가: A00 은 negative 24개에서 목록에 없는 함수 이름을 지어내 `botanical('...')` 처럼 출력했다.
            함수 토큰 확률이 낮았다는 뜻인데 0 으로 두면 "가장 확신한 호출" 로 세게 된다)
    """
    assert len(probs) == n, f"probs {len(probs)} != test {n} (생성이 끝나지 않았거나 다른 test 파일)"
    by_idx = {p["index"]: p for p in probs}
    out = []
    for i in range(n):
        f = by_idx[i].get("func")
        if not f and score == "msp":
            out.append(1.0 - no_func_max(by_idx[i]))
        elif not f:
            out.append(0.0)
        elif score == "irr":
            out.append(f.get("p_irrelevant") or 0.0)
        else:
            out.append(1.0 - max(f["dist"][:NEG_ID]))
    return out


FUNC_TOKENS = {f"<nexa_{i}>" for i in range(NEG_ID)}


def no_func_max(p: dict) -> float:
    """함수 토큰을 안 고른 출력: 공백이 아닌 첫 단계의 상위 k개 중 함수 토큰 최대 확률 (없으면 k번째 확률)."""
    steps = p.get("steps") or []
    step = next((s for s in steps if s["token"].strip("▁ \n\r")), steps[0] if steps else None)
    if not step:
        return 0.0
    in_top = [v for t, v in step["top"] if t in FUNC_TOKENS]
    return max(in_top) if in_top else step["top"][-1][1]


def p_irr(probs: list[dict], n: int) -> list[float]:
    return refusal_scores(probs, n, "irr")


def auroc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney: P(neg 점수 > pos 점수) + 0.5 P(같음)."""
    allv = sorted([(v, 0) for v in pos] + [(v, 1) for v in neg])
    i = 0
    rank_sum_neg = 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        rank_sum_neg += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    n_neg, n_pos = len(neg), len(pos)
    return (rank_sum_neg - n_neg * (n_neg + 1) / 2) / (n_neg * n_pos)


def threshold_at_recall(dev_pos: list[float], recall: float) -> float:
    """positive 의 recall 비율이 p < t 가 되는 가장 작은 t (같은 값이 많으면 recall 이 조금 넘을 수 있다)."""
    s = sorted(dev_pos)
    k = min(len(s) - 1, max(0, int(round(recall * len(s))) - 1))
    return s[k] + 1e-12


def metrics(rows: list[dict], scores: list[float], idx: list[int], t: float | None) -> dict:
    pos = [scores[i] for i in idx if rows[i]["kind"] == "positive"]
    neg = [scores[i] for i in idx if rows[i]["kind"] == "negative"]
    m = {"auroc": auroc(pos, neg)}
    if t is not None:
        m["refuse@r"] = sum(v >= t for v in neg) / len(neg)
        m["pos_recall"] = sum(v < t for v in pos) / len(pos)
        # 우리 test·공식 210 에는 neg_type 칸이 없다 → 유형별은 건너뜀
        types = sorted({rows[i].get("neg_type") for i in idx if rows[i]["kind"] == "negative"} - {None})
        for ty in types:
            vs = [scores[i] for i in idx if rows[i].get("neg_type") == ty]
            m[f"refuse:{ty}"] = sum(v >= t for v in vs) / len(vs)
    return m


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--dev-test", type=Path, default=DEV_TEST)
    p.add_argument("--model", nargs=3, action="append", metavar=("NAME", "TEST_PROBS", "DEV_PROBS"), required=True)
    p.add_argument("--recall", type=float, default=RECALL)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    a = p.parse_args()

    rows = load_jsonl(a.test)
    dev_rows = load_jsonl(a.dev_test) if a.dev_test.exists() else None
    models = []
    for name, tp, dp in a.model:
        score = name.rsplit(":", 1)[1] if ":" in name else "irr"
        assert score in SCORES, f"{name}: 점수 종류는 {SCORES} 중 하나 (이름 끝에 :msp 처럼)"
        s = refusal_scores(load_jsonl(Path(tp)), len(rows), score)
        t = None
        if dp != "-":
            ds = refusal_scores(load_jsonl(Path(dp)), len(dev_rows), score)
            t = threshold_at_recall([v for v, r in zip(ds, dev_rows) if r["kind"] == "positive"], a.recall)
        models.append((name, s, t))

    full = list(range(len(rows)))
    point = {name: metrics(rows, s, full, t) for name, s, t in models}
    rng = random.Random(SEED)
    pos_idx = [i for i in full if rows[i]["kind"] == "positive"]
    neg_idx = [i for i in full if rows[i]["kind"] == "negative"]
    boots = {name: [] for name, _, _ in models}
    for _ in range(a.n_boot):
        # positive / negative 를 따로 복원 추출 (비율 유지, AUROC 가 늘 정의되게)
        idx = [rng.choice(pos_idx) for _ in pos_idx] + [rng.choice(neg_idx) for _ in neg_idx]
        for name, s, t in models:
            boots[name].append(metrics(rows, s, idx, t))

    def ci(vals: list[float]) -> str:
        v = sorted(vals)
        return f"[{v[int(0.025 * len(v))]:.3f}, {v[int(0.975 * len(v)) - 1]:.3f}]"

    print(f"test: {a.test} (positive {len(pos_idx)}, negative {len(neg_idx)}), recall 기준 {a.recall:.0%}, bootstrap {a.n_boot}회")
    for name, _, t in models:
        print(f"\n== {name}" + (f"  (dev 문턱값 t = {t:.4g})" if t is not None else "  (dev probs 없음: AUROC 만)"))
        for k, v in point[name].items():
            print(f"  {k:<32}{v:>8.3f}  95% {ci([b[k] for b in boots[name] if k in b])}")
    base = models[0][0]
    for name, _, _ in models[1:]:
        print(f"\n== 차이 {name} - {base} (짝지은 bootstrap)")
        for k in point[name]:
            if k not in point[base]:
                continue
            # 문항이 아주 적은 유형(예: coding_tech 1개)은 표본에 안 뽑히는 회차가 있어서 둘 다 있는 회차만 쓴다
            d = [bn[k] - bb[k] for bn, bb in zip(boots[name], boots[base]) if k in bn and k in bb]
            lo, hi = sorted(d)[int(0.025 * len(d))], sorted(d)[int(0.975 * len(d)) - 1]
            mark = "  *" if lo > 0 or hi < 0 else ""
            print(f"  {k:<32}{point[name][k] - point[base][k]:>+8.3f}  95% [{lo:+.3f}, {hi:+.3f}]{mark}")
    print("\n* = 95% 구간이 0 을 포함하지 않음")


if __name__ == "__main__":
    main()
