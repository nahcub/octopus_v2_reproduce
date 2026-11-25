"""말투별 거절 (논문에 없는 추가, 실험 S, EXPERIMENT_DESIGN.md 4.4절).

data/style/dev_styled.jsonl (우리 test 400개 × orig + 말투 5개) 에 대한 probs.jsonl 을 받아, 모델·말투마다
- pos 거절: positive 중 greedy 로 <nexa_20> 을 낸 비율 (과잉 거절)
- pos 정확: positive 중 출력이 정답 호출과 글자까지 같은 비율 (엄격)
- neg 거절: negative 중 greedy 로 <nexa_20> 을 낸 비율
- AUROC: 그 말투의 positive vs negative, irr 점수
를 계산하고, 뺀 말투에서 "메운 비율" = (ours − S_minus) / (ours − S_all) 을 낸다 (판정 규칙: 0.5 이상이면 일반화 가설).

    python -m src.style_report --model ours runs/eval/style/ours/probs.jsonl --model S_all ... --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.refusal_metrics import auroc, load_jsonl, refusal_scores

STYLES = ["orig", "howto", "fragment", "need", "form", "nonnative"]
HELD = {"S_minus_howto": "howto", "S_minus_fragment": "fragment", "S_minus_form": "form"}


def report(rows: list[dict], probs: list[dict]) -> dict:
    n = len(rows)
    s = refusal_scores(probs, n, "irr")
    by = {p["index"]: p for p in probs}
    refused = [bool(by[i].get("func")) and by[i]["func"]["chosen"] == "<nexa_20>" for i in range(n)]
    exact = [by[i]["output"].strip() == rows[i]["call"].strip() for i in range(n)]
    out = {}
    for st in STYLES:
        pos = [i for i in range(n) if rows[i]["style"] == st and rows[i]["kind"] == "positive"]
        neg = [i for i in range(n) if rows[i]["style"] == st and rows[i]["kind"] == "negative"]
        out[st] = {"n_pos": len(pos), "n_neg": len(neg),
                   "pos_refused": sum(refused[i] for i in pos) / len(pos),
                   "pos_exact": sum(exact[i] for i in pos) / len(pos),
                   "neg_refused": sum(refused[i] for i in neg) / len(neg),
                   "auroc": auroc([s[i] for i in pos], [s[i] for i in neg])}
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-file", type=Path, default=Path("data/style/dev_styled.jsonl"))
    p.add_argument("--model", nargs=2, action="append", metavar=("NAME", "PROBS"), required=True)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    rows = load_jsonl(a.test_file)
    res = {name: report(rows, load_jsonl(Path(path))) for name, path in a.model}

    first = next(iter(res.values()))["orig"]
    print(f"\n우리 test 원문 {first['n_pos']}+{first['n_neg']}개를 말투만 바꿔 채점. * = 학습에서 뺀 말투")
    for key, title in [("pos_refused", "positive 를 거절한 비율 (낮을수록 좋음)"), ("neg_refused", "negative 를 거절한 비율 (높을수록 좋음)"),
                       ("pos_exact", "positive 엄격 정확도"), ("auroc", "AUROC (irr 점수, positive vs negative)")]:
        print(f"\n[{title}]")
        print(f"{'모델':<18}" + "".join(f"{st:>11}" for st in STYLES))
        for name, r in res.items():
            line = f"{name:<18}"
            for st in STYLES:
                mark = "*" if HELD.get(name) == st else " "
                v = r[st][key]
                line += f"{mark}{v:.3f}".rjust(11) if key == "auroc" else f"{mark}{v:.1%}".rjust(11)
            print(line)

    print("\n[판정] 뺀 말투에서 메운 비율 = (ours − S_minus) / (ours − S_all). 0.5 이상 → 일반화 가설, 미만 → 말투 암기 가설")
    verdict = {}
    if "ours" in res and "S_all" in res:
        for name, k in HELD.items():
            if name not in res:
                continue
            for key, sign in (("pos_refused", 1), ("neg_refused", -1), ("pos_exact", -1)):
                o, m, al = (res[x][k][key] for x in ("ours", name, "S_all"))
                gap = sign * (o - al)
                if gap < 0.05:
                    msg = f"ours−S_all 차이 {gap:+.1%}p < 5%p → 판정 안 함"
                    ratio = None
                else:
                    ratio = sign * (o - m) / gap
                    msg = f"메운 비율 {ratio:.2f} → {'일반화' if ratio >= 0.5 else '말투 암기'}"
                verdict[f"{name}:{key}"] = ratio
                print(f"  {name:<18} 말투 {k:<9} {key:<12} ours {o:.1%}  {name} {m:.1%}  S_all {al:.1%} | {msg}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"results": res, "verdict": verdict}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
