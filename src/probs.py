"""(논문에 없는 추가) 생성 단계마다 softmax 확률을 저장하고 살펴본다. 오답(특히 false_call) 분석용.

evaluate.py 의 generate 와 같은 조건(같은 모델 로드, 같은 프롬프트, greedy, batch 1, <nexa_end>/<eos> 에서 멈춤)으로
다시 생성하면서 매 단계의 logits 를 받아 softmax 한다. 지연 시간은 재지 않는다.
맞은 것·틀린 것 모두 test 전체를 저장한다.

함수 토큰이 첫 단계에 나오지 않을 수 있다: 출력이 " <nexa_0>(...)" 처럼 공백으로 시작하면 첫 단계는 공백 토큰(▁)이다.
그래서 "첫 단계" 가 아니라 "처음으로 함수 토큰(<nexa_0>~<nexa_20>)을 고른 단계" 를 따로 찾아 func 에 적는다.

probs.jsonl 한 줄:
    {"index": 0, "query": "...", "output": " <nexa_0>('front')<nexa_end>", "same_as_preds": true,
     "func": {"step": 1, "chosen": "<nexa_0>", "p_chosen": 0.99, "p_irrelevant": 0.001, "dist": [21개]},
     "steps": [{"token": "▁", "p": 0.98, "top": [["▁", 0.98], ["<nexa_0>", 0.01], ...]}, ...]}
  steps: 단계마다 고른 토큰, 그 확률, 상위 k개(기본 5) [토큰, 확률]
  func:  함수 토큰을 처음 고른 단계. dist 는 그 단계에서 함수 토큰 21개 각각의 확률 (합이 1 이 아님, 나머지는 다른 토큰).
         p_irrelevant = dist[20] (<nexa_20> = irrelevant_function). 함수 토큰을 안 골랐으면 null
  same_as_preds: --preds 를 주면 다시 생성한 글이 preds.jsonl 의 output 과 같은지

실행 (Colab GPU)
    python -m src.probs generate --model NexaAIDev/Octopus-v2 --preds .../eval/public/preds.jsonl --out .../eval/public/probs.jsonl
    python -m src.probs report --probs .../eval/public/probs.jsonl --scored .../eval/public/scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from src.evaluate import END, NEG_ID, load_for_generation
from src.preprocess.prompt import load_jsonl
from src.toy_api import FUNCTIONS


def generate(args) -> None:
    import torch

    from src.preprocess.prompt import build_prompt

    test = load_jsonl(args.test_file)[: args.limit]
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {p["index"] for p in load_jsonl(out)} if out.exists() else set()
    preds = {p["index"]: p["output"] for p in load_jsonl(args.preds)} if args.preds else {}
    todo = [(i, r) for i, r in enumerate(test) if i not in done]
    print(f"[확률] {len(test)}개 중 이미 {len(done)}개 있음 → {len(todo)}개 생성 → {out}")
    if not todo:
        return

    tokenizer, model, device = load_for_generation(args.model, tiny=False)
    stop = [tokenizer.convert_tokens_to_ids(END), tokenizer.eos_token_id]
    func_ids = tokenizer.convert_tokens_to_ids([f"<nexa_{i}>" for i in range(len(FUNCTIONS))])
    func_set = set(func_ids)

    def tok(t) -> str:  # decode 대신 토큰 그대로: 공백이 ▁ 로 보여서 공백 토큰을 알아볼 수 있다
        return tokenizer.convert_ids_to_tokens(int(t))

    with out.open("a", encoding="utf-8") as f:
        for k, (i, r) in enumerate(todo, 1):
            ids = tokenizer(build_prompt(r["query"]), return_tensors="pt").to(device)
            with torch.no_grad():
                res = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     eos_token_id=stop, pad_token_id=tokenizer.pad_token_id,
                                     output_logits=True, return_dict_in_generate=True)
            new = res.sequences[0, ids["input_ids"].shape[1]:]
            steps, func = [], None
            for s, (t, logits) in enumerate(zip(new, res.logits)):
                p = torch.softmax(logits[0].float(), dim=-1)  # bf16 그대로 하면 작은 확률이 뭉개진다
                top = torch.topk(p, args.top_k)
                steps.append({"token": tok(t), "p": round(p[t].item(), 6),
                              "top": [[tok(j), round(v.item(), 6)] for v, j in zip(top.values, top.indices)]})
                if func is None and int(t) in func_set:
                    dist = p[func_ids].tolist()
                    func = {"step": s, "chosen": tok(t), "p_chosen": round(p[t].item(), 6),
                            "p_irrelevant": round(dist[NEG_ID], 6), "dist": [round(v, 6) for v in dist]}
            text = tokenizer.decode(new, skip_special_tokens=False)
            row = {"index": i, "query": r["query"], "output": text}
            if i in preds:
                row["same_as_preds"] = text == preds[i]
            f.write(json.dumps({**row, "func": func, "steps": steps}, ensure_ascii=False) + "\n")
            f.flush()
            if k <= 3 or k % 100 == 0 or k == len(todo):
                fp = (f"{func['chosen']}(단계 {func['step']}) p={func['p_chosen']:.3f}, <nexa_{NEG_ID}> p={func['p_irrelevant']:.4f}"
                      if func else "함수 토큰 없음")
                print(f"  {k}/{len(todo)} | {fp} | 생성:{text!r:.80}")
    print(f"[저장] {out}")


def report(args) -> None:
    """probs.jsonl + evaluate score 의 scored.jsonl → 결과 종류별 확률 비교, false_call 하나씩."""
    pr = {r["index"]: r for r in load_jsonl(args.probs)}
    sc = {r["index"]: r for r in load_jsonl(args.scored)}
    diff = [i for i, r in pr.items() if r.get("same_as_preds") is False]
    if diff:
        print(f"⚠️ preds.jsonl 과 출력이 다른 줄 {len(diff)}개: {diff[:20]} (아래 채점 결과와 안 맞을 수 있음)\n")
    where = Counter(r["func"]["step"] if r["func"] else None for r in pr.values())
    print(f"함수 토큰을 고른 단계 (0: 첫 토큰, 1: 공백 하나 뒤, None: 함수 토큰 없음): {dict(where)}\n")

    groups = {
        "negative 정답": lambda s: s["kind"] == "negative" and s["outcome"] == "correct",
        "negative false_call": lambda s: s["outcome"] == "function:false_call",
        "positive 정답": lambda s: s["kind"] == "positive" and s["outcome"] == "correct",
        "positive 함수 틀림": lambda s: s["outcome"] in ("function:missed_call", "function:other_function"),
    }
    print(f"{'':<22}{'개수':>6}{'고른 토큰 p 중앙값':>18}{'최소':>8}{'<nexa_20> p 중앙값':>20}{'최대':>8}")
    for name, cond in groups.items():
        fs = [pr[i]["func"] for i in pr if i in sc and cond(sc[i]) and pr[i]["func"]]
        if not fs:
            print(f"{name:<22}{0:>6}")
            continue
        pc, pi = [f["p_chosen"] for f in fs], [f["p_irrelevant"] for f in fs]
        print(f"{name:<22}{len(fs):>6}{statistics.median(pc):>18.3f}{min(pc):>8.3f}{statistics.median(pi):>20.4f}{max(pi):>8.4f}")

    print("\n[false_call 하나씩] 함수 토큰 단계의 상위 함수 3개")
    for i in sorted(pr):
        if i not in sc or sc[i]["outcome"] != "function:false_call" or not pr[i]["func"]:
            continue
        f = pr[i]["func"]
        top3 = sorted(enumerate(f["dist"]), key=lambda x: -x[1])[:3]
        print(f"#{i} {f['chosen']} p={f['p_chosen']:.3f} | <nexa_{NEG_ID}> p={f['p_irrelevant']:.4f} | "
              + ", ".join(f"<nexa_{j}> {p:.3f}" for j, p in top3) + f"\n    Q: {sc[i]['query'][:110]}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="(논문에 없는 추가) 생성 단계별 softmax 확률")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="다시 생성하며 단계별 상위 k개 확률 저장 → probs.jsonl (GPU)")
    g.add_argument("--model", default="/content/drive/MyDrive/octopus_v2/runs/full", help="학습한 모델 폴더 또는 HF 이름")
    g.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    g.add_argument("--out", type=Path, required=True, help="probs.jsonl 경로. 이미 있으면 남은 것만 이어서 생성")
    g.add_argument("--preds", type=Path, default=None, help="같은 모델의 preds.jsonl. 주면 출력이 같은지 표시")
    g.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (디버깅)")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--top-k", type=int, default=5, help="단계마다 저장할 상위 토큰 수")
    r = sub.add_parser("report", help="probs.jsonl + scored.jsonl 로 결과 종류별 확률 보기")
    r.add_argument("--probs", type=Path, required=True)
    r.add_argument("--scored", type=Path, required=True, help="evaluate score 가 만든 scored.jsonl")
    args = p.parse_args()
    generate(args) if args.cmd == "generate" else report(args)


if __name__ == "__main__":
    main()
