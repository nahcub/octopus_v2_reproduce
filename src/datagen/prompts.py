"""Gemini 에 보내는 프롬프트. 논문 3.3절의 세 단계(질의 생성 → 인자 생성 → 검증)와 negative 생성."""

from __future__ import annotations

import json

from src.toy_api import FUNCTIONS

# 배치마다 바꿔 끼우는 말투 조건 (다양성 확보)
STYLES = [
    "short imperative commands (3-8 words)",
    "polite full sentences",
    "casual spoken style, as if talking to a voice assistant",
    "longer requests that include some context about why the user wants it",
    "questions, e.g. 'Can you ...?' or 'Could you ...?'",
    "terse typing, may be lowercase with little punctuation",
]

# 형식이 까다로운 함수에만 붙이는 질의 조건. 추론 시 모델은 오늘 날짜를 모르므로 날짜는 질의에 명시한다.
QUERY_HINTS = {
    "make_phone_call": "Each request must contain the phone number to call, including the country code.",
    "create_contact": "Each request must contain the person's full name (first and last) and phone number with country code.",
    "send_email": "Each request must contain the recipient's email address.",
    "create_calendar_event": "Each request must state an explicit calendar date including the year "
                             "(e.g. 'on March 3, 2025'), a start time, and an end time or duration. "
                             "Do not use relative dates like 'tomorrow'.",
    "set_timer_alarm": "Each request must state an explicit clock time, e.g. '6:45', 'seven thirty', 'noon'.",
    "change_screen_brightness": "Each request must state the target brightness as a number from 0 to 10.",
    "set_volume": "Each request must state the target level as a number from 0 to 10 and which volume "
                  "(ringtone, media, or alarm).",
    "set_nest_temperature": "Each request must state a target temperature in Fahrenheit between 50 and 90.",
}

# negative 질의 배치마다 바꿔 끼우는 범주. 뒤쪽 셋은 지원 API 와 헷갈리기 쉬운 것들.
NEGATIVE_CATEGORIES = [
    "general knowledge questions",
    "math, unit conversion, or calculation requests",
    "small talk and chit-chat",
    "requests for recommendations or advice (food, travel, shopping, health)",
    "writing or creative tasks (poems, summaries, translations)",
    "phone settings or features that are NOT in the list, e.g. turning Wi-Fi, Bluetooth, or the flashlight "
    "on or off, airplane mode, deleting or listing alarms, reading messages",
    "tasks for apps or services that are NOT in the list, e.g. ordering food, booking a ride, banking, "
    "posting on social media, music on the phone itself",
    "smart home devices or actions that are NOT in the list, e.g. lights, robot vacuum, TV, garage door, "
    "security camera",
]


def _summary(fn: dict) -> str:
    """함수 이름 + docstring 첫 문장."""
    first = fn["description"].split('"""')[1].strip().splitlines()[0]
    return f"- {fn['name']}: {first}"


SUPPORTED_FUNCTIONS = [f for f in FUNCTIONS if f["name"] != "irrelevant_function"]


def positive_queries(fn: dict, n: int, style: str, candidates: list[str] | None = None) -> str:
    hint = QUERY_HINTS.get(fn["name"], "")
    extra = "".join(f"\n- {c}" for c in [hint, *(candidates or [])] if c)
    if candidates:
        extra += ("\n- Use a different value from the lists above in each request; "
                  "do not invent other names, numbers, or cities.")
    return f"""You are creating training data for an on-device assistant that maps a user's request to an Android API call.

API:
{fn["description"]}

Write {n} different requests that a smartphone user might say or type, each of which can be fully handled by a single call to this API.

Requirements:
- Each request must include the information needed for every parameter, except parameters whose description states a default value; those may be left unstated.{extra}
- Write dates, times, and numbers the way people naturally say or type them (e.g. "3 pm", "half past two", "March 3rd, 2025"). Never use the API's technical formats such as "2025-03-03" or "14:00-15:30".
- Style for this batch: {style}.
- Vary the situation, the wording, and the concrete values (names, places, times, topics, services). Not every request needs to mention optional parameters. Do not repeat a request.
- English only. Each request is a single message from the user, not a dialogue.
- Do not address the assistant with a wake word or name (no "Hey Google", "Hey phone", "Hey assistant", "Siri", etc.)."""


def negative_queries(n: int, category: str) -> str:
    supported = "\n".join(_summary(f) for f in SUPPORTED_FUNCTIONS)
    return f"""You are creating training data for an on-device assistant. The assistant can ONLY do the following:
{supported}

Write {n} different requests that a smartphone user might say or type which CANNOT be handled by any of the functions above.

Requirements:
- Category for this batch: {category}.
- Make them realistic things people ask a phone assistant. Vary wording and length.
- English only. Each request is a single message from the user, not a dialogue.
- Do not address the assistant with a wake word or name (no "Hey Google", "Hey phone", "Hey assistant", "Siri", etc.)."""


def arguments(fn: dict, items: list[dict], feedback: dict[int, str] | None = None) -> str:
    feedback = feedback or {}
    payload = [{"id": it["id"], "query": it["query"],
                **({"previous_attempt_rejected_because": feedback[it["id"]]} if it["id"] in feedback else {})}
               for it in items]
    return f"""Fill in the arguments for calling the API below, for each user request.

API:
{fn["description"]}

Rules:
- Give a value for every parameter. If the request does not mention a parameter that has a documented default, use that default.
- Follow the formats and allowed values in the parameter descriptions exactly (time and date formats, phone number format, option names).
- Take values from the request. Do not invent facts that the request does not state. If the user asks you to compose a message or email body without giving the exact text, write a short, appropriate one.
- If an item has "previous_attempt_rejected_because", fix that problem.

Requests (JSON):
{json.dumps(payload, ensure_ascii=False, indent=1)}"""


def verify_positive(fn: dict, items: list[dict]) -> str:
    payload = [{"id": it["id"], "query": it["query"], "arguments": it["args"]} for it in items]
    return f"""You are checking training data for a function-calling model.
For each item, decide whether calling the API below with the given arguments correctly and completely fulfills the user's request.

API:
{fn["description"]}

Mark an item invalid if any of these is true:
- The request is not actually asking for what this API does.
- An argument is wrong, or contains information the request does not support (except documented defaults, and composed message/email text when the user asked for it).
- An argument breaks the format or allowed values in the parameter descriptions.
- The request is unnatural, ambiguous, or not something a real user would say.

Items (JSON):
{json.dumps(payload, ensure_ascii=False, indent=1)}"""


def verify_negative(items: list[dict]) -> str:
    descriptions = "\n\n".join(f["description"] for f in SUPPORTED_FUNCTIONS)
    payload = [{"id": it["id"], "query": it["query"]} for it in items]
    return f"""You are checking training data for a function-calling model. The model supports ONLY these functions:

{descriptions}

For each user request below, answer which ONE of these functions could fulfill it, or "none" if no function can.
Answer a function name only if that function really does what the user asks (e.g. turning Bluetooth on is not the same as connecting to a device, and opening or closing a garage door is not a smart-lock action).

Requests (JSON):
{json.dumps(payload, ensure_ascii=False, indent=1)}"""


# ---- 구조화 출력 스키마

QUERIES_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}


def arguments_schema(args_schema: dict) -> dict:
    item = {"type": "object", "properties": {"id": {"type": "integer"}, "args": args_schema},
            "required": ["id", "args"]}
    return {"type": "object", "properties": {"results": {"type": "array", "items": item}}, "required": ["results"]}


VERIFY_POSITIVE_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "valid": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["id", "valid", "reason"]}}},
    "required": ["results"],
}

VERIFY_NEGATIVE_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"},
                       "function": {"type": "string", "enum": [f["name"] for f in SUPPORTED_FUNCTIONS] + ["none"]}},
        "required": ["id", "function"]}}},
    "required": ["results"],
}
