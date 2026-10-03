"""인자 규칙: 함수 설명(src/toy_api.py)에 적힌 타입·선택지·형식·범위만 옮긴 것.

2단계에서는 생성된 인자를 검사하고, 5단계에서는 모델 출력 채점에 재사용한다.
"""

from __future__ import annotations

import re

from src.toy_api import FUNCTIONS

_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DATETIME = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])-([01]\d|2[0-3])-[0-5]\d$")  # YYYY-MM-DD-HH-MM
_E164 = re.compile(r"^\+[1-9]\d{6,14}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ISO639_1 = re.compile(r"^[a-z]{2}$")


def _nonempty(v):
    return v.strip() != ""


# 함수 이름 → 인자 이름 → (JSON 타입, 선택지 또는 None, 추가 검사 또는 None)
# JSON 타입과 선택지는 Gemini 구조화 출력 스키마에도 그대로 쓰인다.
RULES: dict[str, dict[str, tuple]] = {
    "take_a_photo": {"camera": ("string", ["front", "back"], None)},
    "get_trending_news": {"query": ("string", None, _nonempty),
                          "language": ("string", None, _ISO639_1.match)},
    "get_weather_forecast": {"location": ("string", None, _nonempty)},
    "send_email": {"recipient": ("string", None, _EMAIL.match),
                   "title": ("string", None, _nonempty),
                   "content": ("string", None, _nonempty)},
    "search_youtube_videos": {"query": ("string", None, _nonempty)},
    "find_route_google_maps": {"origin": ("string", None, _nonempty),
                               "destination": ("string", None, _nonempty),
                               "mode": ("string", ["driving", "walking", "bicycling", "transit"], None)},
    "check_nest_smoke_detector_status": {},
    "send_text_message": {"contact_name": ("string", None, _nonempty),
                          "message": ("string", None, _nonempty)},
    "make_phone_call": {"phone_number": ("string", None, _E164.match)},
    "create_contact": {"name": ("string", None, _nonempty),
                       "phone_number": ("string", None, _E164.match)},
    "set_timer_alarm": {"time": ("string", None, _HHMM.match),
                        "label": ("string", None, _nonempty)},
    "change_screen_brightness": {"level": ("integer", None, lambda v: 0 <= v <= 10)},
    "create_calendar_event": {"title": ("string", None, _nonempty),
                              "start_time": ("string", None, _DATETIME.match),
                              "end_time": ("string", None, _DATETIME.match)},
    "set_volume": {"level": ("integer", None, lambda v: 0 <= v <= 10),
                   "volume_type": ("string", ["ring", "media", "alarm"], None)},
    "connect_to_bluetooth_device": {"device_name": ("string", None, _nonempty),
                                    "timeout": ("integer", None, lambda v: v > 0)},
    "enable_do_not_disturb": {"enabled": ("boolean", None, None)},
    "set_nest_temperature": {"target_temperature": ("number", None, lambda v: 50 <= v <= 90)},
    "play_music_on_nest_hub": {"music_service": ("string", None, _nonempty),
                               "music_name": ("string", None, _nonempty)},
    "play_video_on_nest_hub": {"video_service": ("string", None, _nonempty),
                               "video_name": ("string", None, _nonempty)},
    "open_or_lock_door": {"status": ("string", ["open", "lock"], None)},
    "irrelevant_function": {},
}
assert [f["name"] for f in FUNCTIONS] == list(RULES), "RULES must follow FUNCTIONS order"
for _f in FUNCTIONS:
    assert _f["params"] == list(RULES[_f["name"]]), f"param mismatch: {_f['name']}"

_PY_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def args_schema(name: str) -> dict:
    """구조화 출력용 JSON 스키마 (인자 객체 하나)."""
    props = {}
    for param, (typ, choices, _) in RULES[name].items():
        props[param] = {"type": typ, **({"enum": choices} if choices else {})}
    return {"type": "object", "properties": props, "required": list(props)}


def check_args(name: str, args: dict) -> list[str]:
    """인자 dict 를 검사해 문제 목록을 돌려준다. 빈 리스트면 통과."""
    rules = RULES[name]
    errors = []
    if set(args) != set(rules):
        return [f"params {sorted(args)} != expected {list(rules)}"]
    for param, (typ, choices, check) in rules.items():
        v = args[param]
        if isinstance(v, bool) and typ != "boolean" or not isinstance(v, _PY_TYPES[typ]):
            errors.append(f"{param}: expected {typ}, got {v!r}")
        elif choices and v not in choices:
            errors.append(f"{param}: {v!r} not in {choices}")
        elif check and not check(v):
            errors.append(f"{param}: bad value {v!r}")
    if name == "create_calendar_event" and not errors and args["end_time"] <= args["start_time"]:
        errors.append("end_time must be after start_time")
    return errors


def render_call(token_id: int, args: dict) -> str:
    """학습 타깃의 정답 부분. 공개 모델처럼 위치 인자로 쓴다: <nexa_i>('a', 1)<nexa_end>"""
    params = FUNCTIONS[token_id]["params"]
    return f"<nexa_{token_id}>({', '.join(repr(args[p]) for p in params)})<nexa_end>"
