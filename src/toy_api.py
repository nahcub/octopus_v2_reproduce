"""1단계: 장난감 API 정의.

Octopus v2 논문의 Android API를 본뜬 8개 함수. 실제 동작은 하지 않는 stub이며,
모델이 학습할 대상은 "어떤 함수를 고르고 어떤 인자를 넣느냐"뿐이다.

출처 구분
- [논문 A.1] 부록 A.1 "Android function examples"의 시그니처·docstring을 그대로 옮김.
  (PDF 줄바꿈만 풀고, 둥근 따옴표 ’ 를 ' 로 바꿈.) 타입 힌트는 검증용으로 추가했고,
  docstring에 적힌 내용(선택지, 타입)만 반영했다.
- [직접 작성] 논문 3.2절 "Android system API"에 기능 이름만 나오는 것(texting, setting alarms,
  creating calendar entries)을 논문 docstring 문체에 맞춰 직접 작성.

- `API_REGISTRY[i]` 가 functional token `<nexa_i>` 에 대응한다. 순서를 바꾸면 토큰 의미가 바뀌므로
  데이터 생성 이후에는 순서를 고정할 것 (새 함수는 맨 뒤에 추가).
- docstring은 그대로 학습 타깃의 `Function description:` 부분이 된다 (`function_description`).
- `parse_call` / `validate_call` 은 2단계 데이터 검증과 5단계 인자 정확도 평가에서 재사용한다.
"""

from __future__ import annotations

import ast
import inspect
import re
from typing import Callable, Literal, Optional, get_args, get_origin, get_type_hints


# ---------------------------------------------------------------- [논문 A.1]

def take_a_photo(camera: Literal["front", "back"] = "back",
                 resolution: Literal["720p", "1080p", "4K"] = "1080p") -> str:
    """
    Captures a photo using the specified camera and resolution settings.
    Parameters:
    - camera (str, optional): Specifies the camera to use. Can be 'front' or 'back'. The default is 'back'. Optional to provide.
    - resolution (str, optional): Sets the photo resolution. Options include '720p', '1080p', and '4K'. The default is '1080p'. Optional to provide.
    Returns:
    - str: The string contains the file path of the captured photo if successful, or an error message if not. Example: '/storage/emulated/0/Pictures/MyApp/IMG_20240310_123456.jpg'
    """
    return "/storage/emulated/0/Pictures/MyApp/IMG_20240310_123456.jpg"


def get_trending_news(category: Optional[str] = None, region: str = 'US', language: str = 'en',
                      max_results: int = 5) -> list:
    """
    Fetches trending news articles based on category, region, and language.
    Parameters:
    - category (str, optional): News category to filter by, by default use None for all categories. Optional to provide.
    - region (str, optional): ISO 3166-1 alpha-2 country code for region-specific news, by default, uses 'US'. Optional to provide.
    - language (str, optional): ISO 639-1 language code for article language, by default uses 'en'. Optional to provide.
    - max_results (int, optional): Maximum number of articles to return, by default, uses 5. Optional to provide.
    Returns:
    - list[str]: A list of strings, each representing an article. Each string contains the article's heading and URL.
    """
    return [f"headline {i} - https://news.example.com/{i}" for i in range(max_results)]


def get_weather_forecast(location: str, days: int = 1) -> list:
    """
    Provides a weather forecast for a specified location over a given number of days. Each day's forecast includes a brief description of the expected weather conditions.
    Parameters:
    - location (str): The location for which the weather forecast is desired. Can be a city name, ZIP code, or other location identifiers.
    - days (int, optional): The number of days to include in the forecast, starting from today. The default is 1 day. Optional to provide.
    Returns:
    - list[str]: A list of strings, each representing the weather forecast for one day. Each string includes the date and a brief description of the weather conditions. Formatted in 'YYYY-MM-DD: Description' format.
    """
    return ["2024-03-10: Sunny"] * days


def send_email(recipient: str, subject: str, body: str, attachments: Optional[list[str]] = None,
               cc: Optional[list[str]] = None, bcc: Optional[list[str]] = None) -> bool:
    """
    Sends an email with optional attachments, CC, and BCC.
    Parameters:
    - recipient (str): Primary recipient's email address.
    - subject (str): Email subject line.
    - body (str): Main email body content.
    - attachments (list of str, optional): A list of file paths representing files to attach to the email. Defaults to None, indicating no attachments. Optional to provide.
    - cc (list of str, optional): A list of email addresses to include in the Carbon Copy (CC) field. Defaults to None. Optional to provide.
    - bcc (list of str, optional): A list of email addresses to include in the Blind Carbon Copy (BCC) field. Defaults to None. Optional to provide.
    Returns:
    - bool: True if the email was sent successfully, False otherwise.
    """
    # 논문 원문은 Returns: 아래가 비어 있음 → 반환값 설명 한 줄만 직접 채움
    return True


def search_youtube_videos(query: str, max_results: int = 10,
                          search_filter: Literal["Relevance", "Upload date", "View Count", "Rating"] = "Relevance") -> list:
    """
    Searches YouTube for videos matching a query.
    Parameters:
    - query (str): Search query.
    - max_results (int, optional): Maximum number of search results, by default, use 10. Optional to provide.
    - search_filter (enum, optional): Filter for search results, chosen from 'Relevance', 'Upload date', 'View Count', 'Rating'. By default, use 'Relevance'. Optional to provide.
    Returns:
    - list[str]: A list of strings, each string includes video names and URLs.
    """
    return [f"video {i} - https://www.youtube.com/watch?v={i}" for i in range(max_results)]


# ---------------------------------------------------------------- [직접 작성] 3.2절 기능 목록 기반

def send_text_message(phone_number: str, message: str) -> bool:
    """
    Sends a text message (SMS) to the specified phone number.
    Parameters:
    - phone_number (str): The recipient's phone number, digits with an optional leading '+' and dashes. Example: '+1-415-555-0123'.
    - message (str): The content of the text message.
    Returns:
    - bool: True if the message was sent successfully, False otherwise.
    """
    return True


def set_alarm(time: str, label: Optional[str] = None) -> bool:
    """
    Sets an alarm for a specific time of day.
    Parameters:
    - time (str): The alarm time in 24-hour 'HH:MM' format. Example: '07:30'.
    - label (str, optional): A label displayed when the alarm goes off. Defaults to None. Optional to provide.
    Returns:
    - bool: True if the alarm was set successfully, False otherwise.
    """
    return True


def create_calendar_event(title: str, start_time: str, end_time: str, location: Optional[str] = None) -> bool:
    """
    Creates a new event in the user's calendar.
    Parameters:
    - title (str): The title of the event.
    - start_time (str): The start time of the event in 'YYYY-MM-DD HH:MM' format.
    - end_time (str): The end time of the event in 'YYYY-MM-DD HH:MM' format. Must be after start_time.
    - location (str, optional): The location of the event. Defaults to None. Optional to provide.
    Returns:
    - bool: True if the event was created successfully, False otherwise.
    """
    return True


# <nexa_i> ↔ API_REGISTRY[i]. 순서 고정!
API_REGISTRY: list[Callable] = [
    take_a_photo,            # <nexa_0>  [논문 A.1]
    get_trending_news,       # <nexa_1>  [논문 A.1]
    get_weather_forecast,    # <nexa_2>  [논문 A.1]
    send_email,              # <nexa_3>  [논문 A.1]
    search_youtube_videos,   # <nexa_4>  [논문 A.1]
    send_text_message,       # <nexa_5>  [직접 작성]
    set_alarm,               # <nexa_6>  [직접 작성]
    create_calendar_event,   # <nexa_7>  [직접 작성]
]
NAME_TO_INDEX = {fn.__name__: i for i, fn in enumerate(API_REGISTRY)}
FUNCTIONAL_TOKENS = [f"<nexa_{i}>" for i in range(len(API_REGISTRY))] + ["<nexa_end>"]


# 타입 힌트만으로 표현 못 하는 값 제약. 각 docstring에 적힌 형식만 검사한다.
_NONEMPTY = lambda v: v.strip() != ""
_ISO_COUNTRY = re.compile(r"^[A-Z]{2}$")       # ISO 3166-1 alpha-2
_ISO_LANG = re.compile(r"^[a-z]{2}$")          # ISO 639-1
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?\d[\d-]{6,18}\d$")
_TIME_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DATETIME = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]) ([01]\d|2[0-3]):[0-5]\d$")
_emails = lambda v: all(_EMAIL.match(x) for x in v)

VALUE_CONSTRAINTS: dict[str, dict[str, Callable[[object], bool]]] = {
    "get_trending_news": {"category": _NONEMPTY, "region": lambda v: bool(_ISO_COUNTRY.match(v)),
                          "language": lambda v: bool(_ISO_LANG.match(v)), "max_results": lambda v: v >= 1},
    "get_weather_forecast": {"location": _NONEMPTY, "days": lambda v: v >= 1},
    "send_email": {"recipient": lambda v: bool(_EMAIL.match(v)), "subject": _NONEMPTY, "body": _NONEMPTY,
                   "cc": _emails, "bcc": _emails},
    "search_youtube_videos": {"query": _NONEMPTY, "max_results": lambda v: v >= 1},
    "send_text_message": {"phone_number": lambda v: bool(_PHONE.match(v)), "message": _NONEMPTY},
    "set_alarm": {"time": lambda v: bool(_TIME_HHMM.match(v))},
    "create_calendar_event": {"title": _NONEMPTY,
                              "start_time": lambda v: bool(_DATETIME.match(v)),
                              "end_time": lambda v: bool(_DATETIME.match(v))},
}


def function_description(fn: Callable) -> str:
    """학습 타깃의 `Function description:` 에 들어갈 문자열. 논문 부록처럼 타입 힌트 없는 시그니처 + docstring."""
    sig = inspect.signature(fn)
    params = ", ".join(name + (f"={p.default!r}" if p.default is not p.empty else "")
                       for name, p in sig.parameters.items())
    doc = "\n".join(("    " + line).rstrip() for line in inspect.getdoc(fn).splitlines())
    return f"def {fn.__name__}({params}):\n    \"\"\"\n{doc}\n    \"\"\""


_CALL = re.compile(r"^<nexa_(\d+)>\((.*)\)<nexa_end>$", re.DOTALL)


def parse_call(text: str) -> tuple[int, tuple, dict]:
    """`<nexa_i>(args)<nexa_end>` → (i, positional args, keyword args). 형식이 틀리면 ValueError."""
    m = _CALL.match(text.strip())
    if not m:
        raise ValueError(f"not a functional-token call: {text!r}")
    idx = int(m.group(1))
    try:
        call = ast.parse(f"f({m.group(2)})", mode="eval").body
        args = tuple(ast.literal_eval(a) for a in call.args)
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"unparsable arguments in {text!r}: {e}") from e
    return idx, args, kwargs


def _type_ok(value: object, hint: object) -> bool:
    origin = get_origin(hint)
    if origin is Literal:
        return value in get_args(hint)
    if origin is list:
        (item,) = get_args(hint)
        return isinstance(value, list) and all(_type_ok(x, item) for x in value)
    if origin is not None:  # Optional[...] / Union[...]
        return any(_type_ok(value, h) for h in get_args(hint))
    if hint is type(None):
        return value is None
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, hint)


def validate_call(text: str) -> list[str]:
    """호출 문자열을 검사해 문제 목록을 돌려준다. 빈 리스트면 유효."""
    try:
        idx, args, kwargs = parse_call(text)
    except ValueError as e:
        return [str(e)]
    if not 0 <= idx < len(API_REGISTRY):
        return [f"unknown function token <nexa_{idx}>"]

    fn = API_REGISTRY[idx]
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
    except TypeError as e:
        return [f"{fn.__name__}: {e}"]

    errors = []
    hints = get_type_hints(fn)
    constraints = VALUE_CONSTRAINTS.get(fn.__name__, {})
    for name, value in bound.arguments.items():
        if not _type_ok(value, hints[name]):
            errors.append(f"{fn.__name__}.{name}: bad type/value {value!r}")
        elif value is not None and name in constraints and not constraints[name](value):
            errors.append(f"{fn.__name__}.{name}: constraint violated {value!r}")
    if fn is create_calendar_event and not errors:
        a = bound.arguments
        if a["end_time"] <= a["start_time"]:  # 같은 포맷이라 문자열 비교로 충분
            errors.append("create_calendar_event: end_time must be after start_time")
    return errors


if __name__ == "__main__":
    for i, fn in enumerate(API_REGISTRY):
        print(f"<nexa_{i}> {fn.__name__}{inspect.signature(fn)}")
    print("\n" + function_description(take_a_photo))
