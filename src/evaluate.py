"""5단계: 평가. test.jsonl 질의마다 모델로 생성하고(generate), 정답과 비교해 채점한다(score).

생성 (generate, GPU)
- 한 번에 1개씩 (batch 1). 묶어서 생성하면 pad 를 왼쪽에 채워야 하고 결과가 미세하게 달라질 수 있어서.
  같은 실행에서 지연 시간도 잰다.
- greedy (do_sample=False), <nexa_end> 또는 <eos> 에서 멈춤, max_new_tokens 128 (test 정답 최대 70토큰).
- 지연 시간 (논문 4.1절 조건: A100, 양자화 없음): 처음 몇 개는 예열로 버리고, 토큰화 ~ 생성 끝까지를
  cuda.synchronize 로 GPU 작업이 실제로 끝난 시점까지 잰다. 모델 로드 시간은 뺀다.
  attention 은 transformers 기본값(sdpa). 논문은 flash attention 을 썼다고만 적혀 있다.
- 한 줄씩 바로 파일에 쓴다. 중간에 끊기면 같은 명령으로 다시 실행해 남은 것만 이어서 한다.

채점 순서 (한 줄마다 하나의 결과로 분류)
    format:no_nexa_end      <nexa_end> 가 없음 (잘렸거나 호출 형식이 아님)
    format:unparsable       <nexa_i>(...) 모양이 아니거나 괄호 안을 값으로 읽을 수 없음
    format:unknown_token    <nexa_21> 이상처럼 없는 함수 번호
    function:false_call     negative 인데 함수를 부름
    function:missed_call    positive 인데 irrelevant_function 을 부름
    function:other_function positive 인데 다른 함수를 부름
    args:count_mismatch     함수는 맞았지만 인자 개수·이름이 다름
    args:rule_violation     함수는 맞았지만 인자가 함수 설명의 규칙(validate.RULES)을 어김
    args:valid_but_different 함수는 맞았고 규칙도 지켰지만 정답과 다른 값
    correct                 함수와 인자가 정답과 같음 (값과 타입이 모두 같아야 함)

- 생성된 글은 첫 <nexa_end> 까지만 본다 (그 뒤 Function description 은 채점하지 않음).
- (논문에 없는 추가) 느슨한 정확도: 문자열 인자의 대소문자·연속 공백·앞뒤 공백·끝 문장부호(.!?)를 무시하고,
  숫자는 int/float 를 구분하지 않는다. 대표 숫자는 엄격한 정확도.

preds.jsonl 한 줄 (generate 가 만든다):
    {"index": 0, "query": "...", "output": " <nexa_0>('front')<nexa_end>", "new_tokens": 5, "latency_s": 0.21}

실행
    # 랩탑 CPU: 채점 코드 자체 점검 / 작은 무작위 Gemma 로 생성 코드가 끝까지 도는지만 확인
    python -m src.evaluate selftest
    python -m src.evaluate generate --tiny --limit 3 --max-new-tokens 8 --out runs/eval/tiny/preds.jsonl

    # Colab GPU: 생성 (우리 모델 / 공개 모델)
    python -m src.evaluate generate --model /content/drive/MyDrive/octopus_v2/runs/full --out .../eval/ours/preds.jsonl
    python -m src.evaluate generate --model NexaAIDev/Octopus-v2 --out .../eval/public/preds.jsonl

    # 채점 (CPU 로 충분)
    python -m src.evaluate score --preds runs/eval/ours/preds.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from src.datagen.validate import check_args, render_call
from src.preprocess.prompt import load_jsonl
from src.toy_api import FUNCTIONS, NAME_TO_INDEX, parse_call

END = "<nexa_end>"
NEG_ID = NAME_TO_INDEX["irrelevant_function"]


def extract_call(output: str) -> str | None:
    """생성된 글에서 첫 <nexa_end> 까지만 잘라 돌려준다. <nexa_end> 가 없으면 None."""
    end = output.find(END)
    return None if end < 0 else output[: end + len(END)].strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(".!?").strip().lower()


def same_value(a, b, lenient: bool = False) -> bool:
    """엄격: 타입까지 같아야 함 (True ≠ 1, 72 ≠ 72.0). 느슨: 문자열은 _norm 후 비교, 숫자는 int/float 무시."""
    if lenient:
        if isinstance(a, str) and isinstance(b, str):
            return _norm(a) == _norm(b)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)):
            return a == b
    return type(a) is type(b) and a == b


def same_args(a: dict, b: dict, lenient: bool = False) -> bool:
    return a.keys() == b.keys() and all(same_value(a[k], b[k], lenient) for k in a)


def judge(row: dict, output: str) -> dict:
    """test 한 줄 + 생성된 글 → {"outcome", "pred_call", "pred_args", "rule_errors", "lenient_correct"}."""
    res = {"pred_call": extract_call(output), "pred_args": None, "rule_errors": [], "lenient_correct": False}
    if res["pred_call"] is None:
        return {**res, "outcome": "format:no_nexa_end"}
    try:
        tid, pos, kw = parse_call(res["pred_call"])
    except ValueError:
        return {**res, "outcome": "format:unparsable"}
    if tid >= len(FUNCTIONS):
        return {**res, "outcome": "format:unknown_token"}

    gold = row["token_id"]
    if tid != gold:
        kind = "false_call" if gold == NEG_ID else "missed_call" if tid == NEG_ID else "other_function"
        return {**res, "outcome": f"function:{kind}"}

    params = FUNCTIONS[tid]["params"]
    args = dict(zip(params, pos)) | kw
    if len(pos) > len(params) or len(pos) + len(kw) != len(params) or set(args) != set(params):
        return {**res, "outcome": "args:count_mismatch"}
    res["pred_args"] = args
    res["lenient_correct"] = same_args(args, row["args"], lenient=True)
    if same_args(args, row["args"]):
        return {**res, "outcome": "correct"}
    res["rule_errors"] = check_args(FUNCTIONS[tid]["name"], args)
    return {**res, "outcome": "args:rule_violation" if res["rule_errors"] else "args:valid_but_different"}


def summarize(scored: list[dict]) -> dict:
    """채점된 줄 목록 → 숫자들. 비율은 0~1."""
    n = len(scored)
    pos = [s for s in scored if s["kind"] == "positive"]
    neg = [s for s in scored if s["kind"] == "negative"]
    func_ok = [s for s in scored if s["outcome"] == "correct" or s["outcome"].startswith("args:")]
    pos_func_ok = [s for s in func_ok if s["kind"] == "positive"]

    def rate(part, whole):
        return len(part) / len(whole) if whole else float("nan")

    def acc(rows):
        return rate([s for s in rows if s["outcome"] == "correct"], rows)

    return {
        "n": n,
        "format_ok": rate([s for s in scored if not s["outcome"].startswith("format:")], scored),
        "function_acc": rate(func_ok, scored),
        "args_acc": acc(pos_func_ok),  # 함수를 맞힌 positive 중 인자까지 맞힌 비율
        "acc": acc(scored),
        "acc_positive": acc(pos),
        "acc_negative": acc(neg),
        "acc_lenient": rate([s for s in scored if s["lenient_correct"]], scored),
        "exact_string": rate([s for s in scored if s["pred_call"] == s["call"]], scored),
        "outcomes": dict(sorted(Counter(s["outcome"] for s in scored).items())),
    }


METRIC_NAMES = {
    "n": "개수",
    "format_ok": "형식 OK",
    "function_acc": "함수 선택 정확도",
    "args_acc": "인자 정확도 (함수 맞힌 positive 중)",
    "acc": "전체 정확도 ★",
    "acc_positive": "  positive",
    "acc_negative": "  negative",
    "acc_lenient": "느슨한 정확도 (논문에 없는 추가)",
    "exact_string": "호출 문자열 완전 일치",
}


def score(test: list[dict], preds: list[dict]) -> list[dict]:
    """preds 를 index 로 test 와 짝지어 한 줄씩 채점한다."""
    assert len({p["index"] for p in preds}) == len(preds), "preds 에 같은 index 가 두 번 있음"
    if len(preds) != len(test):
        print(f"⚠️ 일부만 채점: preds {len(preds)}개 / test {len(test)}개\n")
    scored = []
    for p in sorted(preds, key=lambda p: p["index"]):
        row = test[p["index"]]
        assert p["query"] == row["query"], f"index {p['index']} 의 질의가 다름"
        scored.append({**row, **{k: v for k, v in p.items() if k != "query"}, **judge(row, p["output"])})
    return scored


def report(scored: list[dict]) -> dict:
    """화면에 점수표를 찍고 summary dict 를 돌려준다."""
    subsets = {"전체": scored, "seen_in_train=false": [s for s in scored if not s["seen_in_train"]]}
    summary = {name: summarize(rows) for name, rows in subsets.items()}

    print(f"{'':<34}" + "".join(f"{name:>22}" for name in subsets))
    for key, label in METRIC_NAMES.items():
        cells = [summary[name][key] for name in subsets]
        print(f"{label:<34}" + "".join(f"{c:>22}" if key == "n" else f"{c:>21.2%} " for c in cells))
    print("\n[결과 종류별 개수]")
    for outcome in sorted({o for s in summary.values() for o in s["outcomes"]}):
        print(f"  {outcome:<34}" + "".join(f"{summary[name]['outcomes'].get(outcome, 0):>22}" for name in subsets))

    print(f"\n[함수별] {'함수':<34}{'개수':>6}{'함수 선택':>10}{'전체 정확도':>12}")
    for i, f in enumerate(FUNCTIONS):
        rows = [s for s in scored if s["token_id"] == i]
        if rows:
            m = summarize(rows)
            print(f"  <nexa_{i}> {f['name']:<34}{m['n']:>6}{m['function_acc']:>10.1%}{m['acc']:>12.1%}")

    lat = [s["latency_s"] for s in scored if s.get("latency_s") is not None]
    if lat:
        lat.sort()
        summary["latency_s"] = {"median": statistics.median(lat), "p90": lat[int(0.9 * (len(lat) - 1))], "mean": statistics.fmean(lat)}
        toks = [s["new_tokens"] for s in scored if s.get("new_tokens") is not None]
        print(f"\n[지연 시간] 중앙값 {summary['latency_s']['median']:.3f}초 | 90% {summary['latency_s']['p90']:.3f}초"
              f" | 평균 {summary['latency_s']['mean']:.3f}초 | 생성 토큰 평균 {statistics.fmean(toks):.1f}")
    return summary


def save(scored: list[dict], summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scored.jsonl").open("w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    # 사람이 엑셀로 볼 오답 목록 (utf-8-sig: 엑셀에서 한글이 안 깨짐)
    with (out / "errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "outcome", "lenient_correct", "seen_in_train", "function", "query", "gold", "pred", "rule_errors", "review"])
        for s in scored:
            if s["outcome"] != "correct":
                w.writerow([s["index"], s["outcome"], s["lenient_correct"], s["seen_in_train"], s["function"], s["query"],
                            s["call"], s["pred_call"] or s["output"], "; ".join(s["rule_errors"]), ""])
    print(f"\n[저장] {out}/scored.jsonl, summary.json, errors.csv")


def load_for_generation(name: str, tiny: bool):
    """(토크나이저, 모델, 장치). 토크나이저는 모델과 같이 저장된 것을 쓴다 (<nexa_*> 가 이미 들어 있어야 함)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.preprocess.tokens import add_functional_tokens, check_functional_tokens, resize_embeddings
    from src.toy_api import FUNCTIONAL_TOKENS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tiny:  # 랩탑 디버깅: 학습 안 된 작은 무작위 Gemma (출력은 의미 없음)
        from src.train import load_model

        add_functional_tokens(tokenizer)
        model = load_model(name, tiny=True, vocab_size=len(tokenizer) - len(FUNCTIONAL_TOKENS))
        resize_embeddings(model, tokenizer)
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    # 토큰 번호가 학습 때와 같은지 (<nexa_0> = 256000, ...). 다르면 채점이 전부 틀어진다
    check_functional_tokens(tokenizer, tokenizer.convert_tokens_to_ids(FUNCTIONAL_TOKENS))
    return tokenizer, model.to(device).eval(), device


def generate(args) -> None:
    import platform
    import subprocess
    import time

    import torch
    import transformers

    from src.preprocess.prompt import build_prompt

    test = load_jsonl(args.test_file)[: args.limit]
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {p["index"] for p in load_jsonl(out)} if out.exists() else set()
    todo = [(i, r) for i, r in enumerate(test) if i not in done]
    print(f"[생성] {len(test)}개 중 이미 {len(done)}개 있음 → {len(todo)}개 생성 → {out}")
    if not todo:
        return

    tokenizer, model, device = load_for_generation(args.model, args.tiny)
    stop = [tokenizer.convert_tokens_to_ids(END), tokenizer.eos_token_id]
    cuda = device.type == "cuda"

    def run(query: str) -> tuple[str, int, float]:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ids = tokenizer(build_prompt(query), return_tensors="pt").to(device)
        with torch.no_grad():
            seq = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=stop, pad_token_id=tokenizer.pad_token_id)
        if cuda:
            torch.cuda.synchronize()
        new = seq[0, ids["input_ids"].shape[1]:]
        # skip_special_tokens=False: True 면 <nexa_*> 가 지워진다
        return tokenizer.decode(new, skip_special_tokens=False), len(new), time.perf_counter() - t0

    for _, r in todo[: args.warmup]:  # 예열: 첫 몇 번은 GPU 준비 때문에 느려서 버린다
        run(r["query"])

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    meta = {
        "model": args.model, "tiny": args.tiny, "test_file": str(args.test_file), "limit": args.limit,
        "max_new_tokens": args.max_new_tokens, "warmup": args.warmup, "dtype": str(model.dtype),
        "attn_implementation": model.config._attn_implementation,
        "device": torch.cuda.get_device_name() if cuda else platform.processor(),
        "transformers": transformers.__version__, "torch": torch.__version__, "git_commit": commit,
    }
    (out.parent / "gen_args.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[설정] {meta}")

    with out.open("a", encoding="utf-8") as f:
        for k, (i, r) in enumerate(todo, 1):
            text, n_new, sec = run(r["query"])
            f.write(json.dumps({"index": i, "query": r["query"], "output": text, "new_tokens": n_new,
                                "latency_s": round(sec, 4)}, ensure_ascii=False) + "\n")
            f.flush()
            if k <= 3 or k % 100 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} | {sec:.3f}초 | Q: {r['query'][:60]}\n     정답: {r['call']}\n     생성:{text!r:.120}")
    print(f"[저장] {out}")


def selftest(test: list[dict]) -> None:
    """정답·망가뜨린 예측을 직접 만들어 채점 결과가 기대와 같은지 확인한다."""
    # 1) 정답 그대로 → 모두 correct. 일부는 <nexa_end> 뒤에 Function description 이 이어져도 같아야 함
    preds = [{"index": i, "query": r["query"], "output": " " + r["call"] + ("\n\nFunction description: ..." if i % 2 else "")}
             for i, r in enumerate(test)]
    m = summarize(score(test, preds))
    assert m["acc"] == m["acc_lenient"] == m["exact_string"] == m["function_acc"] == 1.0, m
    print(f"[selftest] 정답 그대로 {m['n']}개 → 전체 정확도 {m['acc']:.0%}")

    def first(name):
        return next(r for r in test if r["function"] == name)

    photo, neg = first("take_a_photo"), first("irrelevant_function")
    text, vol, dnd, temp = first("send_text_message"), first("set_volume"), first("enable_do_not_disturb"), first("set_nest_temperature")
    other_cam = "back" if photo["args"]["camera"] == "front" else "front"
    t = temp["args"]["target_temperature"]
    temp_alt = float(t) if isinstance(t, int) else int(t)  # 72 ↔ 72.0 (float 정답이 정수값이라고 가정, 아래 assert 로 확인)
    assert temp_alt == t, t

    cases = [  # (정답 줄, 생성된 글, 기대 결과, 기대 느슨한 정답 여부)
        (photo, render_call(0, {"camera": other_cam}), "args:valid_but_different", False),
        (photo, f"<nexa_0>(\"{photo['args']['camera']}\")<nexa_end>", "correct", True),  # 큰따옴표여도 값이 같으면 정답
        (photo, f"<nexa_0>(camera='{photo['args']['camera']}')<nexa_end>", "correct", True),  # 이름 붙인 인자
        (photo, "<nexa_0>()<nexa_end>", "args:count_mismatch", False),
        (photo, f"<nexa_0>('{photo['args']['camera']}', 1)<nexa_end>", "args:count_mismatch", False),
        (photo, "<nexa_20>()<nexa_end>", "function:missed_call", False),
        (photo, render_call(2, {"location": "Paris"}), "function:other_function", False),
        (neg, render_call(0, {"camera": "front"}), "function:false_call", False),
        (photo, photo["call"].replace(END, ""), "format:no_nexa_end", False),  # 잘림
        (photo, "I can't do that.", "format:no_nexa_end", False),
        (photo, "<nexa_0>(front)<nexa_end>", "format:unparsable", False),
        (photo, "Sure: <nexa_0>('front')<nexa_end>", "format:unparsable", False),
        (photo, "<nexa_99>()<nexa_end>", "format:unknown_token", False),
        (vol, render_call(vol["token_id"], {**vol["args"], "level": 11}), "args:rule_violation", False),
        (dnd, render_call(dnd["token_id"], {"enabled": int(dnd["args"]["enabled"])}), "args:rule_violation", False),  # True ≠ 1
        (text, render_call(text["token_id"], {**text["args"], "message": "  " + text["args"]["message"].upper() + "!! "}),
         "args:valid_but_different", True),
        (temp, render_call(temp["token_id"], {"target_temperature": temp_alt}), "args:valid_but_different", True),
    ]
    for row, output, want, want_lenient in cases:
        got = judge(row, output)
        assert (got["outcome"], got["lenient_correct"]) == (want, want_lenient), (output, got, want, want_lenient)
        print(f"[selftest] {want:<26} 느슨={str(want_lenient):<5} ← {output!r:.70}")
    print(f"[selftest] 통과 ({len(cases)}개 사례)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Octopus v2 평가 (생성, 채점)")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="test 질의마다 생성 → preds.jsonl (GPU)")
    g.add_argument("--model", default="/content/drive/MyDrive/octopus_v2/runs/full", help="학습한 모델 폴더 또는 HF 이름")
    g.add_argument("--tiny", action="store_true", help="랩탑 디버깅: 작은 무작위 Gemma (--model 은 google/gemma-2b 로 바뀜)")
    g.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    g.add_argument("--out", type=Path, required=True, help="preds.jsonl 경로. 이미 있으면 남은 것만 이어서 생성")
    g.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (디버깅)")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--warmup", type=int, default=5, help="지연 시간 측정 전 버리는 생성 횟수")
    s = sub.add_parser("score", help="preds.jsonl 채점")
    s.add_argument("--preds", type=Path, required=True)
    s.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    s.add_argument("--out", type=Path, default=None, help="기본: preds 와 같은 폴더")
    t = sub.add_parser("selftest", help="채점 코드 자체 점검")
    t.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    args = p.parse_args()

    if args.cmd == "generate":
        if args.tiny:
            args.model = "google/gemma-2b"
        generate(args)
        return
    test = load_jsonl(args.test_file)
    if args.cmd == "selftest":
        selftest(test)
        return
    scored = score(test, load_jsonl(args.preds))
    summary = report(scored)
    save(scored, summary, args.out or args.preds.parent)


if __name__ == "__main__":
    main()
