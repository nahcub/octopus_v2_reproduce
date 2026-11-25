"""(논문에 없는 추가) 실험 G: 함수 토큰 대신 "함수 설명을 프롬프트에 넣고 JSON 으로 호출하는" 일반 모델.

묻는 것: 지금까지 본 거절 현상(말투 의존, near-miss 오호출, 문턱값 이식 실패)이 함수 토큰 방식이라서 생기나,
함수 호출 모델이면 공통으로 생기나. 출력 방식만 바꾸고 나머지(베이스, train 4,000, 학습 설정, test)는 ours 와 같다.

글 모양 (· = 공백, ↵ = 줄바꿈)
    Below is the query from the users, please call the correct function and generate the parameters to call the function.↵
    ↵
    Functions:↵
    {"name": "take_a_photo", "description": "...", "parameters": {...}}↵      ← 21개 (토큰 번호 순서, 한 줄에 하나)
    ...↵
    {"name": "irrelevant_function", ...}↵
    ↵
    Query:·{query}·↵
    ↵
    Response:                                                                  ← 여기까지 문제
    ·{"name": "change_screen_brightness", "arguments": {"level": 5}}<eos>     ← 정답 (채점 부분)

정한 것 (모두 결과를 보기 전)
- 지시문·Query·Response 부분은 ours(prompt.py) 와 같은 문구. 그 사이에 함수 목록만 끼운다
- 함수 목록: BFCL·OpenAI 식 JSON 스키마 (name, description, parameters). toy_api.DESCRIPTIONS 원문에서 옮긴다
  - description = 설명 첫 단락, 인자 description = 원문 인자 줄 그대로 (기본값·형식 설명 포함)
  - 타입: str→string, int→integer, float→number, bool→boolean, enum→string + 원문에 따옴표로 적힌 선택지
  - 모든 인자를 required 로 둔다 (ours 정답도 기본값까지 다 채운다)
- 거절 = irrelevant_function 을 21번째 함수로 목록에 넣고 {"name": "irrelevant_function", "arguments": {}} 로 학습.
  ours 의 <nexa_20> 과 맞춘다 (거절이 자유 문장이면 채점·점수가 흔들린다)
- 정답 뒤에 Function description 을 붙이지 않는다 (ours 와의 차이. 설명은 이미 프롬프트에 있다)
- 새 토큰을 추가하지 않는다. gemma 원래 사전 그대로

실험 ND = N+설명 (2026-11-25 추가): ours 에서 함수 토큰만 함수 이름(JSON)으로 바꾼 모델. 학습: train.py --format json --no-list --json-desc
- 문제 부분: ours(prompt.py) 와 글자까지 같다 (함수 목록 없음, build_prompt(with_list=False))
- 정답: ` {JSON}` + ours 와 글자까지 같은 `\n\nFunction description: \n{설명 원문}\n\n` + <eos> (build_target(desc=True))
- 생성: ours 가 <nexa_end> 에서 멈추듯 JSON 뒤 빈 줄("\n\n")에서 멈춘다. G 에도 같은 규칙 (G 정답에는 빈 줄이 없어 채점에 영향 없음)
- generate/probs 는 모델 폴더의 train_args.json 을 읽어 목록을 넣을지 정한다 (--no-list 로 직접 지정 가능)
3모델 비교 (EXPERIMENT_DESIGN.md 4.5절): ours → ND (토큰 → 이름) → G (설명을 외우기 → 입력에서 보고 고르기)

거절 점수 (probs)
- 함수 이름 21개를 하나씩 정답 자리에 넣어 보고, 그 이름(+ 닫는 따옴표)을 쓸 log 확률을 더한다
  → 21개로 softmax = 함수 토큰 모델의 dist 와 같은 모양. probs.jsonl 형식도 probs.py 와 같게 써서
  refusal_metrics.py·threshold 계산을 그대로 쓴다
- 차이 (논문에 적을 것): ours 의 dist 는 전체 사전 기준 확률, 여기는 21개 안에서 다시 나눈 값 (합 1).
  AUROC 는 모델 안에서의 순위만 보므로 영향이 없지만, 문턱값 숫자는 두 모델끼리 직접 비교하지 않는다
- 이름 길이 보정은 하지 않는다 (진짜 문장 확률). 이름별 log 확률 합(logp)도 같이 저장해 둔다

실행
    python -m src.general stats                          # 프롬프트 길이 (토크나이저만, CPU)
    python -m src.general show                           # 예시 한 줄을 눈으로 확인
    python -m src.general generate --model <폴더> --test-file data/test.jsonl --out .../preds.jsonl   (GPU)
    python -m src.evaluate score --preds .../preds.jsonl --out ...                              (기존 채점 그대로)
    python -m src.general probs --model <폴더> --test-file data/test.jsonl --preds .../preds.jsonl --out .../probs.jsonl  (GPU)
      probs.jsonl 의 func.chosen·output 은 preds(greedy 생성)에서 옮겨 적는다 (probs.py 와 같은 뜻).
      21개 점수의 1등은 func.argmax 에 따로 둔다
    학습은 python -m src.train --format json ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.preprocess.prompt import PROMPT_TEMPLATE, load_jsonl
from src.toy_api import DESCRIPTIONS, FUNCTIONS, NAME_TO_INDEX

NEG_ID = NAME_TO_INDEX["irrelevant_function"]
END = "<nexa_end>"  # 기존 채점기(evaluate.judge)에 넘길 때만 쓴다. 모델은 이 토큰을 모른다
_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
_PARAM = re.compile(r"^\s*-\s*(\w+)\s*\(([^)]*)\):\s*(.*)$")


def tool_schema(i: int) -> dict:
    """toy_api 의 함수 설명 원문 → JSON 스키마 하나."""
    f = FUNCTIONS[i]
    body = f["description"].split('"""')[1]
    summary_lines = []
    for line in body.strip().splitlines():  # 첫 빈 줄(공백만 있는 줄 포함)까지가 요약
        if not line.strip():
            break
        summary_lines.append(line.strip())
    summary = " ".join(summary_lines)
    props = {}
    for line in body.splitlines():
        m = _PARAM.match(line)
        if not m:
            continue
        name, typ, text = m.groups()
        if typ == "enum":
            options = list(dict.fromkeys(re.findall(r"['\"]([^'\"]+)['\"]", text)))  # 기본값 문장에 같은 값이 또 나와서 중복 제거
            props[name] = {"type": "string", "enum": options, "description": text}
        else:
            props[name] = {"type": _TYPES[typ], "description": text}
    assert list(props) == f["params"], (f["name"], list(props), f["params"])
    return {"name": f["name"], "description": summary,
            "parameters": {"type": "object", "properties": props, "required": f["params"]}}


TOOLS = [tool_schema(i) for i in range(len(FUNCTIONS))]
TOOLS_TEXT = "\n".join(json.dumps(t, ensure_ascii=False) for t in TOOLS)
_HEAD, _TAIL = PROMPT_TEMPLATE.split("\n\nQuery:")
assert _TAIL.startswith(" {query}"), PROMPT_TEMPLATE
GENERAL_TEMPLATE = _HEAD + "\n\nFunctions:\n" + TOOLS_TEXT.replace("{", "{{").replace("}", "}}") + "\n\nQuery:" + _TAIL
TARGET_PREFIX = ' {"name": "'  # 정답의 앞부분. 함수 이름이 바로 뒤에 온다
STOP = "\n\n"  # 생성 멈춤 (JSON 은 한 줄이라 안에 빈 줄이 없다)


def build_prompt(query: str, with_list: bool = True) -> str:
    """with_list=False (실험 ND): 함수 목록 없이 ours(prompt.py) 와 글자까지 같은 문제 부분.
    모델은 함수 이름·인자를 학습 데이터에서만 배운다 (ours 가 <nexa_i> 의 뜻을 학습에서만 배우듯)."""
    return (GENERAL_TEMPLATE if with_list else PROMPT_TEMPLATE).format(query=query)


def read_with_list(model_dir, override: bool | None) -> bool:
    """generate/probs 에서 함수 목록을 넣을지. --no-list 를 주면 False, 아니면 학습 폴더의 train_args.json
    (train.py 가 저장) 의 no_list 를 따른다. 파일이 없으면 G 로 보고 True. 학습과 다른 프롬프트로 평가하는 실수를 막는다."""
    if override is not None:
        return override
    f = Path(model_dir) / "train_args.json"
    if f.exists():
        return not json.loads(f.read_text(encoding="utf-8")).get("no_list", False)
    return True


def target_json(row: dict) -> str:
    """정답 JSON. 인자 순서는 함수 정의 순서."""
    f = FUNCTIONS[row["token_id"]]
    args = {p: row["args"][p] for p in f["params"]}
    return json.dumps({"name": f["name"], "arguments": args}, ensure_ascii=False)


def build_target(row: dict, desc: bool = False) -> str:
    """desc=True (실험 ND): JSON 뒤에 ours(prompt.build_target) 와 글자까지 같은 Function description 부분을 붙인다."""
    text = " " + target_json(row)
    if desc:
        text += "\n\nFunction description: \n" + DESCRIPTIONS[row["token_id"]] + "\n\n"
    return text


def build_example(row: dict, tokenizer, with_list: bool = True, desc: bool = False) -> dict:
    """masking.build_example 과 같은 모양 ({"input_ids", "labels"}). 문제 부분은 -100."""
    from src.preprocess.masking import IGNORE_INDEX

    prompt_ids = tokenizer(build_prompt(row["query"], with_list))["input_ids"]
    target_ids = tokenizer(build_target(row, desc), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {"input_ids": prompt_ids + target_ids, "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids}


def to_call(output: str) -> str:
    """생성된 JSON → 기존 채점기가 읽는 `<nexa_i>(k=v, ...)<nexa_end>` 문자열.
    JSON 이 아니거나 모양이 틀리면 원문을 그대로 돌려준다 (채점기가 format 오류로 센다)."""
    text = output.strip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return output
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not isinstance(obj.get("arguments", {}), dict):
        return output
    name = obj["name"]
    tid = NAME_TO_INDEX.get(name, len(FUNCTIONS))  # 없는 이름 → 없는 토큰 번호 = format:unknown_token
    args = obj.get("arguments", {})
    if tid < len(FUNCTIONS) and set(args) == set(FUNCTIONS[tid]["params"]):
        # 인자 이름이 정확히 맞으면 정답(data 의 call)과 같은 위치 인자 표기로 → 글자 비교(style_report 등)도 같게 된다
        from src.datagen.validate import render_call
        return render_call(tid, args)
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())  # 인자가 틀리면 이름을 남겨 채점기가 count_mismatch 로 센다
    return f"<nexa_{tid}>({kw}){END}"


# ---------------------------------------------------------------- GPU 부분

def load(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return tokenizer, model, device


def generate(a) -> None:
    import time

    import torch

    test = load_jsonl(a.test_file)[: a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = {p["index"] for p in load_jsonl(a.out)} if a.out.exists() else set()
    todo = [(i, r) for i, r in enumerate(test) if i not in done]
    print(f"[생성] {len(test)}개 중 이미 {len(done)}개 → {len(todo)}개 → {a.out}")
    if not todo:
        return
    with_list = read_with_list(a.model, a.with_list)
    print(f"[프롬프트] 함수 목록 {'있음 (G)' if with_list else '없음 (ND)'}")
    tokenizer, model, device = load(a.model)
    with a.out.open("a", encoding="utf-8") as f:
        for k, (i, r) in enumerate(todo, 1):
            t0 = time.perf_counter()
            ids = tokenizer(build_prompt(r["query"], with_list), return_tensors="pt").to(device)
            with torch.no_grad():
                # JSON 뒤 빈 줄에서 멈춘다 (ND 는 그 뒤에 설명을 쓰도록 학습됨. ours 가 <nexa_end> 에서 멈추는 것과 같은 뜻)
                seq = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                     stop_strings=[STOP], tokenizer=tokenizer)
            new = seq[0, ids["input_ids"].shape[1]:]
            raw = tokenizer.decode(new, skip_special_tokens=True)
            # output = 기존 채점기용으로 바꾼 글, raw_output = 모델이 실제로 쓴 JSON
            f.write(json.dumps({"index": i, "query": r["query"], "output": to_call(raw), "raw_output": raw,
                                "new_tokens": len(new), "latency_s": round(time.perf_counter() - t0, 4)},
                               ensure_ascii=False) + "\n")
            f.flush()
            if k <= 3 or k % 100 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} | Q: {r['query'][:60]}\n     정답: {target_json(r)}\n     생성: {raw[:120]!r}")
    print(f"[저장] {a.out}")


def candidate_ids(tokenizer, query: str, with_list: bool = True) -> tuple[list[int], list[list[int]]]:
    """(공통 앞부분, 함수 21개 각각의 뒷부분) 토큰. 이름+닫는 따옴표까지 붙여 한 번에 토큰화한 뒤
    21개가 공통으로 갖는 앞부분에서 자른다 → 학습 때와 같은 토큰 경계를 쓴다."""
    texts = [build_prompt(query, with_list) + TARGET_PREFIX + f["name"] + '"' for f in FUNCTIONS]
    full = [tokenizer(t)["input_ids"] for t in texts]
    n = 0
    while all(len(x) > n for x in full) and len({x[n] for x in full}) == 1:
        n += 1
    return full[0][:n], [x[n:] for x in full]


def score_candidates(model, prefix: list[int], conts: list[list[int]], device) -> list[float]:
    """이름마다 log P(뒷부분 | 앞부분). 앞부분은 한 번만 계산하고 (KV cache), 21개를 한 묶음으로 이어서 계산한다."""
    import torch

    with torch.no_grad():
        # logits_to_keep=1: 앞부분 끝 자리의 확률만 만든다 (2,350 자리 × 256,000칸을 다 만들면 수 GB)
        out = model(input_ids=torch.tensor([prefix], device=device), use_cache=True, logits_to_keep=1)
        first = torch.log_softmax(out.logits[0, -1].float(), -1)  # 뒷부분 첫 토큰의 확률은 앞부분 끝에서 나온다
        cache = out.past_key_values
        width = max(len(c) for c in conts)
        pad = 0
        ids = torch.tensor([c + [pad] * (width - len(c)) for c in conts], device=device)
        valid = torch.tensor([[1] * len(c) + [0] * (width - len(c)) for c in conts], device=device)
        cache.batch_repeat_interleave(len(conts))
        mask = torch.cat([torch.ones(len(conts), len(prefix), dtype=valid.dtype, device=device), valid], dim=1)
        logits = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True).logits.float()
        lp = torch.log_softmax(logits, -1)
        scores = []
        for j, c in enumerate(conts):
            s = first[c[0]].item()
            for t in range(1, len(c)):
                s += lp[j, t - 1, c[t]].item()
            scores.append(s)
    return scores


def probs(a) -> None:
    import math

    test = load_jsonl(a.test_file)[: a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = {p["index"] for p in load_jsonl(a.out)} if a.out.exists() else set()
    todo = [(i, r) for i, r in enumerate(test) if i not in done]
    print(f"[확률] {len(test)}개 중 이미 {len(done)}개 → {len(todo)}개 → {a.out}")
    if not todo:
        return
    preds = {p["index"]: p for p in load_jsonl(a.preds)}
    missing = [i for i, _ in todo if i not in preds]
    assert not missing, f"preds 에 없는 index {missing[:5]}... (generate 를 먼저 끝까지)"
    with_list = read_with_list(a.model, a.with_list)
    print(f"[프롬프트] 함수 목록 {'있음 (G)' if with_list else '없음 (ND)'}")
    tokenizer, model, device = load(a.model)
    with a.out.open("a", encoding="utf-8") as f:
        for k, (i, r) in enumerate(todo, 1):
            assert preds[i]["query"] == r["query"], f"index {i} 질의가 preds 와 다름"
            prefix, conts = candidate_ids(tokenizer, r["query"], with_list)
            logp = score_candidates(model, prefix, conts, device)
            m = max(logp)
            z = sum(math.exp(x - m) for x in logp)
            dist = [math.exp(x - m) / z for x in logp]
            best = max(range(len(dist)), key=dist.__getitem__)
            # chosen = greedy 생성이 실제로 고른 함수 (probs.py 와 같은 뜻. 보고서들이 "greedy 거절" 을 이걸로 센다).
            # 21개 점수의 1등(argmax)은 따로 둔다. 이름을 못 읽는 출력이면 chosen 은 None
            out_call = preds[i]["output"]
            m_tok = re.match(r"^<nexa_(\d+)>", out_call.strip())
            chosen = f"<nexa_{m_tok.group(1)}>" if m_tok else None
            func = {"step": None, "chosen": chosen, "argmax": f"<nexa_{best}>",
                    "p_chosen": round(dist[int(m_tok.group(1))], 6) if m_tok and int(m_tok.group(1)) < len(dist) else None,
                    "p_irrelevant": round(dist[NEG_ID], 6), "dist": [round(x, 6) for x in dist]}
            f.write(json.dumps({"index": i, "query": r["query"], "output": out_call, "score_kind": "candidate_softmax21",
                                "logp": [round(x, 4) for x in logp], "func": func}, ensure_ascii=False) + "\n")
            f.flush()
            if k <= 3 or k % 100 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} | 생성 {chosen} | 점수 1등 {FUNCTIONS[best]['name']} {dist[best]:.3f}"
                      f" | 거절 {dist[NEG_ID]:.4f} | 정답 {r['function']} | Q: {r['query'][:50]}")
    print(f"[저장] {a.out}")


# ---------------------------------------------------------------- 확인용 (CPU)

def stats(a) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    for name, with_list, desc in (("G", True, False), ("ND", False, True)):
        print(f"--- {name}: 함수 목록 {'있음' if with_list else '없음'}, 정답 뒤 설명 {'있음' if desc else '없음'}")
        for split in ("train", "test"):
            rows = load_jsonl(Path(f"data/{split}.jsonl"))
            lens = sorted(len(build_example(r, tokenizer, with_list, desc)["input_ids"]) for r in rows)
            tl = sorted(len(tokenizer(build_target(r, desc), add_special_tokens=False)["input_ids"]) + 1 for r in rows)
            print(f"[{split}] {len(rows)}개 | 전체 토큰 최소 {lens[0]} / 중앙 {lens[len(lens) // 2]} / 최대 {lens[-1]}"
                  f" | 정답 부분 중앙 {tl[len(tl) // 2]} / 최대 {tl[-1]}")
        head = len(tokenizer(build_prompt("", with_list))["input_ids"])
        print(f"프롬프트 (질의 빼고): {head} 토큰")


def show(a) -> None:
    rows = load_jsonl(Path("data/train.jsonl"))
    for r in (rows[0], next(x for x in rows if x["kind"] == "negative")):
        print("=" * 70)
        p = build_prompt(r["query"])
        print(p[:400] + "\n   ... (함수 목록 생략) ...\n" + p[-200:] + "   ◀ 문제 끝 / 정답 시작 ▶")
        print(build_target(r))
        print("기존 채점기용:", to_call(target_json(r)), "| 원래 정답:", r["call"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats", help="프롬프트 길이 (토크나이저만)")
    s.add_argument("--model", default="google/gemma-2b")
    sub.add_parser("show", help="예시 한 줄")
    for name in ("generate", "probs"):
        g = sub.add_parser(name)
        g.add_argument("--model", required=True)
        g.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
        g.add_argument("--out", type=Path, required=True)
        g.add_argument("--limit", type=int, default=None)
        g.add_argument("--max-new-tokens", type=int, default=128)
        g.add_argument("--no-list", dest="with_list", action="store_const", const=False, default=None,
                       help="함수 목록 없는 프롬프트 (실험 ND). 안 주면 모델 폴더의 train_args.json 을 따른다")
        if name == "probs":
            g.add_argument("--preds", type=Path, required=True, help="같은 test 로 generate 한 preds.jsonl (greedy 선택·출력을 옮겨 적음)")
    a = p.parse_args()
    {"stats": stats, "show": show, "generate": generate, "probs": probs}[a.cmd](a)


if __name__ == "__main__":
    main()
