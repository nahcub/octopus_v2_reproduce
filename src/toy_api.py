"""1단계: 함수 명세 (공개 모델 NexaAIDev/Octopus-v2 의 21개 functional token 에 맞춤).

DESCRIPTIONS[i] 가 <nexa_i> 에 대응한다. 순서(=토큰 번호)는 공개 모델과 같아야 하므로 바꾸지 말 것.
각 문자열은 학습 타깃의 `Function description:` 뒤에 붙는 텍스트 원문 그대로다. 공백·줄바꿈까지 수정 금지.

출처
- <nexa_0> ~ <nexa_19>: HF 모델 카드 NexaAIDev/Octopus-v2 의 android_functions.txt 원문.
  파일 안의 순서가 곧 토큰 번호다. 0단계 Colab 질의 출력에서 확인한 대응
  (0, 2, 3, 4, 7, 8, 10, 11, 12, 15, 16, 17, 19, 20번)과 모두 일치함.
- <nexa_20> irrelevant_function: android_functions.txt 에 없음. 공개 모델 출력 원문을 옮김.
  (논문에는 4.1절 비교 모델용 프롬프트에만 이름이 나온다.)
"""

from __future__ import annotations

import ast
import re


DESCRIPTIONS: list[str] = [
    # <nexa_0>
    '''def take_a_photo(camera):
    """
    Captures a photo using the specified camera and resolution settings.

    Parameters:
    - camera (str): Specifies the camera to use. Can be 'front' or 'back'. The default is 'back'.

    Returns:
    - str: The string contains the file path of the captured photo if successful, or an error message if not. Example: '/storage/emulated/0/Pictures/MyApp/IMG_20240310_123456.jpg'
    """''',

    # <nexa_1>
    '''def get_trending_news(query, language):
    """
    Retrieves a collection of trending news articles relevant to a specified query and language.

    Parameters:
    - query (str): Topic for news articles.
    - language (str): ISO 639-1 language code. The default language is English ('en'), but it can be set to any valid ISO 639-1 code to accommodate different language preferences (e.g., 'es' for Spanish, 'fr' for French).

    Returns:
    - list[str]: A list of strings, where each string represents a single news article. Each article representation includes the article's title and its URL, allowing users to easily access the full article for detailed information.
    """''',

    # <nexa_2>
    '''def get_weather_forecast(location):
    """
    Provides a weather forecast for a specified location over a given number of days. Each day's forecast includes a brief description of the expected weather conditions.

    Parameters:
    - location (str): The location for which the weather forecast is desired. Can be a city name, ZIP code, or other location identifiers.

    Returns:
    - list[str]: A list of strings, each representing the weather forecast for one day. Each string includes the date and a brief description of the weather conditions. Formatted in 'YYYY-MM-DD: Description' format.
    """''',

    # <nexa_3>
    '''def send_email(recipient, title, content):
    """
    Sends an email to a specified recipient with a given title and content.

    Parameters:
    - recipient (str): The email address of the recipient.
    - title (str): The subject line of the email. This is a brief summary or title of the email's purpose or content.
    - content (str): The main body text of the email. It contains the primary message, information, or content that is intended to be communicated to the recipient.

    Returns:
    """''',

    # <nexa_4>
    '''def search_youtube_videos(query):
    """
    Searches YouTube for videos matching a query.

    Parameters:
    - query (str): Search query.

    Returns:
    - list[str]: A list of strings, each string includes video names and URLs.
    """''',

    # <nexa_5>
    '''def find_route_google_maps(origin, destination, mode):
    """
    Computes a route using Google Maps from an origin to a destination.

    Parameters:
    - origin (str): Starting location.
    - destination (str): Target location.
    - mode (enum): Mode of transportation, options include 'driving', 'walking', 'bicycling', and 'transit'. The default mode is 'driving'.

    Returns:
    - List[str]:  The string provides the route details.
    """''',

    # <nexa_6>
    '''def check_nest_smoke_detector_status():
    """
    Check the current status of a Nest Smoke Detector.

    Parameters:

    Returns:
    - str: A message with the detailed smoke detector status. This message includes information about battery level, sensor functionality, and connectivity status.
    """''',

    # <nexa_7>
    '''def send_text_message(contact_name, message):
    """
    Sends a text message to the specified contact.

    Parameters:
    - contact_name (str): The name of the recipient contact.
    - message (str): The content of the message to be sent. This is what the recipient will receive.

    Returns:
    """''',

    # <nexa_8>
    '''def make_phone_call(phone_number):
    """
    Initiates a phone call to the given phone number. It can handle both international and domestic numbers.

    Parameters:
    - phone_number (str): phone number of the contact. The phone number should be provided in a standard format, preferably in E.164 format (e.g., +12345678900 for an international format).

    Returns:
    """''',

    # <nexa_9>
    '''def create_contact(name, phone_number):
    """
    Creates a new contact entry in the device's address book.

    Parameters:
    - name (str): Full name of the contact. This should include first and last name.
    - phone_number (str): phone number of the contact. The phone number should be provided in a standard format, preferably in E.164 format (e.g., +12345678900 for an international format).

    Returns:
    """''',

    # <nexa_10>
    '''def set_timer_alarm(time, label):
    """
    Sets a timer or alarm for a specified time.

    Parameters:
    - time (str): Alarm time in "HH:MM" 24-hour format. For example, "07:12" for 7:12 AM.
    - label (str): Custom label for the alarm, default is "alarm".

    Returns:
    """''',

    # <nexa_11>
    '''def change_screen_brightness(level):
    """
    Adjust the device's screen brightness level.

    Parameters:
    - level (int): Desired brightness level, on a scale from 0 (minimum) to 10 (maximum).

    Returns:
    """''',

    # <nexa_12>
    '''def create_calendar_event(title, start_time, end_time):
    """
    Schedules a new event in the calendar.

    Parameters:
    - title (str): Event title.
    - start_time (str): Event start time as a string in ISO 8601 format "YYYY-MM-DD-HH-MM". For example, "2022-12-31-23-59" for 11:59 PM on December 31, 2022.
    - end_time (str): Event end time as a string in ISO 8601 format "YYYY-MM-DD-HH-MM". Must be after start_time. For example, "2023-01-01-00-00" for 12:00 AM on January 1, 2023.

    Returns:
    """''',

    # <nexa_13>
    '''def set_volume(level, volume_type):
    """
    Sets the volume level for a specified type : "ring" , "media" , "alarm".

    Parameters:
    - level (int): Target volume level, from 0 (mute) to 10 (maximum).
    - volume_type (enum): The category of volume to adjust, select from "ring" , "media" , "alarm".

    Returns:
    """''',

    # <nexa_14>
    '''def connect_to_bluetooth_device(device_name, timeout):
    """
    Attempts to connect to a specified Bluetooth device within a given timeout.

    Parameters:
    - device_name (str): The name of the Bluetooth device to connect to.
    - timeout (int): The maximum time, in seconds, to attempt a connection. The default is 10 seconds.

    Returns:
    - bool: Returns True if the device was successfully connected within the timeout period, False otherwise.
    """''',

    # <nexa_15>
    '''def enable_do_not_disturb(enabled):
    """
    Toggles the Do Not Disturb mode on or off.

    Parameters:
    - enabled (bool): True to enable, False to disable Do Not Disturb mode.

    Returns:
    """''',

    # <nexa_16>
    '''def set_nest_temperature(target_temperature):
    """
    Sets a Nest Thermostat to a desired temperature.

    Parameters:
    - target_temperature (float): Target temperature in degrees Fahrenheit, ranging from 50 to 90.

    Returns:
    """''',

    # <nexa_17>
    '''def play_music_on_nest_hub(music_service, music_name):
    """
    Streams music on a Google Nest Audio device from a specified service.

    Parameters:
    - music_service (str): Music streaming service name.
    - music_name (str): Playlist name to play.

    Returns:
    """''',

    # <nexa_18>
    '''def play_video_on_nest_hub(video_service, video_name):
    """
    Streams video on a Google Nest Hub device from a specified service.

    Parameters:
    - video_service (str): Video streaming service name.
    - video_name (str): Video playlist name to play.

    Returns:
    """''',

    # <nexa_19>
    '''def open_or_lock_door(status):
    """
    Opens or locks a door using a Google-compatible smart lock.

    Parameters:
    - status (enum): Desired action, chosen from 'open' or 'lock'.

    Returns:
    """''',

    # <nexa_20>
    '''def irrelevant_function():
  """
  If user query is not related to any of the predefined functions, this function will be called.

  Args:

  Returns:
  """''',
]


def _parse_signature(description: str) -> tuple[str, list[str]]:
    m = re.match(r"def\s+(\w+)\((.*?)\):", description)
    return m.group(1), [p.strip() for p in m.group(2).split(",") if p.strip()]


FUNCTIONS: list[dict] = [
    {"name": name, "params": params, "description": desc}
    for desc in DESCRIPTIONS
    for name, params in [_parse_signature(desc)]
]
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
