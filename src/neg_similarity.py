"""(논문에 없는 추가) 공식 210 positive 를 "누가 거절했나" 로 나눠, train negative / train positive 와의 유사도를 비교한다.
매핑 가설: ours(<nexa_20>) 만 거절한 positive 는 train negative 와 더 닮았어야 한다.

랩탑, 저장소 루트에서. 필요: data/raw/eval_official/{ours,A00,OE}/ 의 probs.jsonl, softmax_*_summary.json (dev 문턱값),
임베딩 모델 BAAI/bge-small-en-v1.5 (HF 캐시)
    python -m src.neg_similarity | tee data/raw/eval_official/neg_similarity.txt
"""
import json, os, sys, statistics as st
os.environ["USE_TF"] = "0"
import numpy as np, torch
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
from src.refusal_metrics import refusal_scores

L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
test = L("data/raw/official/official_test.jsonl"); n = len(test)
train = L("data/train.jsonl")
tneg = [r["query"] for r in train if r["kind"] == "negative"]
tpos = [r["query"] for r in train if r["kind"] == "positive"]
pos_idx = [i for i in range(n) if test[i]["kind"] == "positive"]

T = {m: (sc, json.load(open(f"data/raw/eval_official/{m}/softmax_{sc}_summary.json", encoding="utf-8"))["threshold"])
     for m, sc in (("ours", "irr"), ("A00", "msp"), ("OE", "msp"))}  # 각 모델의 dev 문턱값 (positive recall 90%)
ref, S = {}, {}
for m, (sc, t) in T.items():
    pr = L(f"data/raw/eval_official/{m}/probs.jsonl"); S[m] = refusal_scores(pr, n, sc)
    byi = {p["index"]: p for p in pr}
    g = lambda i: bool(byi[i].get("func")) and byi[i]["func"]["chosen"] == "<nexa_20>"
    ref[m] = {i for i in pos_idx if S[m][i] >= t or g(i)}
o, a, e = ref["ours"], ref["A00"], ref["OE"]
groups = {
    "ours 만 거절": sorted(o - a - e),
    "OE·A00 만 거절 (ours 통과)": sorted((a | e) - o),
    "셋 다 거절": sorted(o & a & e),
    "아무도 거절 안 함": sorted(set(pos_idx) - o - a - e),
}

# 임베딩: bge-small (의미), TF-IDF (단어)
tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5"); mdl = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()
@torch.no_grad()
def emb(xs):
    out = []
    for k in range(0, len(xs), 64):
        b = tok(xs[k:k + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
        h = mdl(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1))
    return torch.cat(out).numpy()
Q = [test[i]["query"] for i in range(n)]
EQ, EN, EP = emb(Q), emb(tneg), emb(tpos)
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(tneg + tpos + Q)
TQ, TN, TP = vec.transform(Q), vec.transform(tneg), vec.transform(tpos)

def nn(Aq, B, sparse=False):
    sim = (Aq @ B.T).toarray() if sparse else Aq @ B.T
    return sim.max(1), sim.argmax(1)
sn_e, an_e = nn(EQ, EN); sp_e, _ = nn(EQ, EP)
sn_t, an_t = nn(TQ, TN, True); sp_t, _ = nn(TQ, TP, True)
margin_e = sn_e - sp_e  # +면 positive 보다 negative 를 더 닮음

print(f"공식 positive 200개. 거절 수 ours {len(o)}, A00 {len(a)}, OE {len(e)}\n")
print(f"{'그룹':<26}{'n':>4}  {'neg 유사도(의미)':>14}{'pos 유사도(의미)':>14}{'neg−pos':>9}{'neg 유사도(단어)':>14}")
for gname, idx in groups.items():
    if not idx: continue
    f = lambda arr: st.median(float(arr[i]) for i in idx)
    print(f"{gname:<26}{len(idx):>4}  {f(sn_e):>14.3f}{f(sp_e):>14.3f}{f(margin_e):>+9.3f}{f(sn_t):>14.3f}")

g1, g2 = groups["ours 만 거절"], groups["아무도 거절 안 함"]
for name, arr in (("neg 유사도(의미)", sn_e), ("neg−pos", margin_e), ("neg 유사도(단어)", sn_t)):
    u = mannwhitneyu([arr[i] for i in g1], [arr[i] for i in g2], alternative="greater")
    print(f"ours 만 거절 > 아무도 거절 안 함 ({name}): p = {u.pvalue:.3g}")
r = spearmanr([S["ours"][i] for i in pos_idx], [sn_e[i] for i in pos_idx])
r2 = spearmanr([S["ours"][i] for i in pos_idx], [margin_e[i] for i in pos_idx])
r3 = spearmanr([S["OE"][i] for i in pos_idx], [margin_e[i] for i in pos_idx])
print(f"\nspearman: ours p_irr vs neg 유사도 {r.statistic:+.2f} (p={r.pvalue:.2g}), vs neg−pos {r2.statistic:+.2f} (p={r2.pvalue:.2g}) | OE msp vs neg−pos {r3.statistic:+.2f} (p={r3.pvalue:.2g})")

print("\n=== ours 만 거절한 positive: 가장 닮은 train negative (의미 기준) ===")
for i in sorted(g1, key=lambda i: -sn_e[i]):
    print(f"[{sn_e[i]:.2f}/{sp_e[i]:.2f} p_irr {S['ours'][i]:.3f}] {test[i]['function']:<26} {Q[i][:85]}\n      ↳ neg: {tneg[an_e[i]][:85]}")
