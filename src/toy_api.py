"""1단계: 함수 명세 (공개 모델 NexaAIDev/Octopus-v2 의 21개 functional token 에 맞춤).

FUNCTIONS[i] 가 <nexa_i> 에 대응한다. 순서(=토큰 번호)는 공개 모델과 같아야 하므로 바꾸지 말 것.

출처
- <nexa_0> ~ <nexa_19>: data/android_functions.txt (HF 모델 카드 NexaAIDev/Octopus-v2 의 원본, 수정 금지).
  파일 안의 순서가 곧 토큰 번호다. 0단계 Colab 질의 출력에서 확인한 대응
  (0, 2, 3, 4, 7, 8, 10, 11, 12, 15, 16, 17, 19, 20번)과 모두 일치함.
- <nexa_20> irrelevant_function: 파일에 없음. 공개 모델 출력 원문을 옮김.
  (논문에는 4.1절 비교 모델용 프롬프트에만 이름이 나온다.)

각 항목의 description 은 학습 타깃의 `Function description:` 뒤에 붙는 텍스트 원문 그대로다.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

FUNCTIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "android_functions.txt"

IRRELEVANT_FUNCTION = '''def irrelevant_function():
  """
  If user query is not related to any of the predefined functions, this function will be called.

  Args:

  Returns:
  """'''


def _load_functions() -> list[dict]:
    text = FUNCTIONS_FILE.read_text(encoding="utf-8")
    blocks = [b.strip("\n") for b in re.split(r"\n(?=def )", text) if b.strip()]
    descriptions = blocks + [IRRELEVANT_FUNCTION]
    functions = []
    for desc in descriptions:
        m = re.match(r"def\s+(\w+)\((.*?)\):", desc)
        functions.append({
            "name": m.group(1),
            "params": [p.strip() for p in m.group(2).split(",") if p.strip()],
            "description": desc,
        })
    return functions


FUNCTIONS: list[dict] = _load_functions()
FUNCTIONAL_TOKENS = [f"<nexa_{i}>" for i in range(len(FUNCTIONS))] + ["<nexa_end>"]
NAME_TO_INDEX = {f["name"]: i for i, f in enumerate(FUNCTIONS)}


_CALL = re.compile(r"^<nexa_(\d+)>\((.*)\)<nexa_end>$", re.DOTALL)


def parse_call(text: str) -> tuple[int, tuple, dict]:
    """`<nexa_i>(args)<nexa_end>` → (i, positional args, keyword args). 형식이 틀리면 ValueError."""
    m = _CALL.match(text.strip())
    if not m:
        raise ValueError(f"not a functional-token call: {text!r}")
    try:
        call = ast.parse(f"f({m.group(2)})", mode="eval").body
        args = tuple(ast.literal_eval(a) for a in call.args)
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"unparsable arguments in {text!r}: {e}") from e
    return int(m.group(1)), args, kwargs


if __name__ == "__main__":
    for i, f in enumerate(FUNCTIONS):
        print(f"<nexa_{i}>".ljust(11), f"{f['name']}({', '.join(f['params'])})")
