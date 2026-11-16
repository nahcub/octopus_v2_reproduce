"""(논문에 없는 추가) 6단계 실험: 2단계 학습용 (chosen, rejected) 짝을 만든다. B(SFT)와 C(DPO)가 같은 파일을 쓴다.

    질문            chosen (정답)        rejected (이렇게 답하면 안 됨)
    negative        <nexa_20>()          모델 A 가 함수 토큰 자리에서 <nexa_0>~<nexa_19> 중 확률 1등인 함수
                                         → 그 토큰을 넣고 인자는 A 가 greedy 로 이어서 쓴 것
    positive        정답 호출            <nexa_20>()   (과잉거절을 막는 쪽)

- B 는 chosen 만 쓰고(SFT), C 는 둘 다 쓴다(DPO). 질문 집합이 완전히 같아서 다른 건 loss 하나뿐이다.
- 모델 A 는 train 을 거의 외웠으므로 실제로 틀리는 일은 드물다. 그래서 "틀린다면 이렇게 틀렸을" 1등 후보를 쓴다.
  새 질문을 만들지 않는다 (rejected 는 A 가 스스로 만든 것).
- 함수 토큰 자리: 정답 글은 " <nexa_i>..." 라서 첫 토큰이 공백(▁), 그다음이 함수 토큰이다 (masking.py 에서 확인).
  그래서 프롬프트 + ▁ 까지 넣고 다음 토큰 확률을 본다.
- 1등 함수로 이어 쓴 글이 <nexa_end> 로 끝나는 올바른 호출이 아니면 2등, 3등 함수로 다시 시도한다. 셋 다 안 되면 뺀다.

출력 pairs.jsonl 한 줄 = 원래 train 줄 + 아래 칸
    rejected_call       "<nexa_7>('...')<nexa_end>"
    rejected_token_id   7
    p_irrelevant        A 가 그 질문에서 매긴 함수 토큰 중 <nexa_20> 비율 (분석용, positive 는 null)
    rejected_rank       몇 등 함수로 만들었나 (1 = 1등, positive 는 null)

실행 (Colab GPU, negative 약 1,350개 생성 → 수 분)
    python -m src.dpo_pairs --model /content/drive/MyDrive/octopus_v2/runs/A_s0 \\
        --train-file data/holdout/train.jsonl --out /content/drive/MyDrive/octopus_v2/runs/A_s0/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.evaluate import END, NEG_ID, load_for_generation
from src.preprocess.prompt import build_prompt, load_jsonl
from src.toy_api import FUNCTIONS, parse_call

REFUSE_CALL = f"<nexa_{NEG_ID}>(){END}"


def main() -> None:
    import torch

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="모델 A 폴더 (1단계 SFT 결과)")
    p.add_argument("--train-file", type=Path, default=Path("data/holdout/train.jsonl"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--tries", type=int, default=3, help="올바른 호출이 나올 때까지 몇 등 함수까지 시도할지")
    a = p.parse_args()

    rows = load_jsonl(a.train_file)[: a.limit]
    tokenizer, model, device = load_for_generation(a.model, tiny=False)
    stop = [tokenizer.convert_tokens_to_ids(END), tokenizer.eos_token_id]
    func_ids = tokenizer.convert_tokens_to_ids([f"<nexa_{i}>" for i in range(len(FUNCTIONS))])
    space_id = tokenizer.convert_tokens_to_ids("▁")
    assert space_id != tokenizer.unk_token_id, "공백 토큰(▁)을 찾지 못함"

    out, skipped = [], []
    n_neg = sum(r["kind"] == "negative" for r in rows)
    k = 0
    for r in rows:
        if r["kind"] == "positive":
            out.append({**r, "rejected_call": REFUSE_CALL, "rejected_token_id": NEG_ID, "p_irrelevant": None, "rejected_rank": None})
            continue
        k += 1
        prompt = tokenizer(build_prompt(r["query"]))["input_ids"] + [space_id]
        ctx = torch.tensor([prompt], device=device)
        with torch.no_grad():
            logits = model(input_ids=ctx).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)[func_ids]
        p_irr = (probs[NEG_ID] / probs.sum()).item()
        order = [i for i in torch.argsort(probs, descending=True).tolist() if i != NEG_ID][: a.tries]

        made = None
        for rank, fid in enumerate(order, 1):
            ids = torch.tensor([prompt + [func_ids[fid]]], device=device)
            with torch.no_grad():
                seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.max_new_tokens,
                                     do_sample=False, eos_token_id=stop, pad_token_id=tokenizer.pad_token_id)
            text = f"<nexa_{fid}>" + tokenizer.decode(seq[0, ids.shape[1]:], skip_special_tokens=False)
            end = text.find(END)
            if end < 0:
                continue
            call = text[: end + len(END)]
            try:
                tid, _, _ = parse_call(call)
            except ValueError:
                continue
            if tid == fid:
                made = (call, fid, rank)
                break
        if made is None:
            skipped.append(r["query"])
        else:
            out.append({**r, "rejected_call": made[0], "rejected_token_id": made[1], "p_irrelevant": round(p_irr, 6),
                        "rejected_rank": made[2]})
        if k <= 3 or k % 200 == 0 or k == n_neg:
            got = made[0] if made else "(실패)"
            print(f"  negative {k}/{n_neg} | P(<nexa_20>) {p_irr:.4f} | Q: {r['query'][:60]}\n     rejected: {got}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ranks = [r["rejected_rank"] for r in out if r["rejected_rank"]]
    print(f"[저장] {a.out} | 짝 {len(out)}개 (positive {len(out) - len(ranks)}, negative {len(ranks)})"
          f" | negative 실패로 뺌 {len(skipped)}개 | 1등 함수로 만든 비율 {ranks.count(1) / max(len(ranks), 1):.1%}")
    for q in skipped[:5]:
        print("   뺌:", q)


if __name__ == "__main__":
    main()
