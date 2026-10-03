"""질의에 들어갈 값 후보 (사람 이름, 전화번호, 도시).

Gemini 가 스스로 값을 고르면 같은 이름·번호·장소를 반복한다 (2차 실행에서 함수당 20개 중 이름 6개 중복).
그래서 배치마다 후보를 무작위로 뽑아 프롬프트에 주고 "필요하면 이 중에서 골라 써라"고 한다.

목록은 train / test 로 나눈다. test 질의의 이름·도시는 학습 때 한 번도 안 본 값이 되므로,
5단계 평가에서 "처음 보는 값도 질의에서 정확히 꺼내 오는가"를 측정할 수 있다.
이름은 이름(first)과 성(last)을 각각 나눠서 test 의 전체 이름이 train 과 완전히 겹치지 않게 한다.
전화번호는 코드로 무작위 생성하므로 목록이 없고, split 마다 다른 난수열을 쓴다.
"""

from __future__ import annotations

import random

FIRST_NAMES = [
    # 영어권
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason", "Isabella", "Logan",
    "Mia", "Lucas", "Harper", "Owen", "Ella", "Caleb", "Grace", "Wyatt", "Chloe", "Nathan",
    "Zoe", "Ryan", "Lily", "Connor", "Hannah", "Dylan", "Nora", "Gavin", "Audrey", "Tyler",
    # 스페인어·포르투갈어권
    "Mateo", "Valentina", "Santiago", "Camila", "Diego", "Lucia", "Javier", "Sofia", "Rafael", "Ines",
    # 유럽
    "Luca", "Giulia", "Pierre", "Camille", "Lars", "Freya", "Anton", "Katarina", "Pavel", "Elena",
    # 아시아
    "Hiroshi", "Yuki", "Minjun", "Seoyeon", "Wei", "Mei", "Arjun", "Priya", "Thanh", "Linh",
    # 중동·아프리카
    "Omar", "Layla", "Yusuf", "Amira", "Kwame", "Amara", "Tariq", "Zainab", "Chidi", "Nia",
    # 2배 확장분 (같은 문화권 비율 유지)
    "Jack", "Amelia", "Henry", "Charlotte", "Samuel", "Evelyn", "Benjamin", "Abigail", "Daniel", "Scarlett",
    "Jacob", "Madison", "Isaac", "Natalie", "Julian", "Claire", "Aaron", "Violet", "Evan", "Ruby",
    "Alejandro", "Gabriela", "Carlos", "Mariana", "Andres", "Daniela", "Joao", "Beatriz", "Pedro", "Renata",
    "Matteo", "Chiara", "Hugo", "Margaux", "Felix", "Clara", "Sven", "Astrid", "Mikhail", "Anastasia",
    "Tomas", "Zofia", "Niklas", "Ingrid", "Dimitri",
    "Kenji", "Sakura", "Jiho", "Haeun", "Jun", "Xiu", "Rohan", "Ananya", "Minh", "Kavya", "Ravi", "Aiko",
    "Karim", "Yasmin", "Hamza", "Fatima", "Kofi", "Adaeze", "Samir", "Leila", "Tunde", "Imani", "Mustafa",
    "Noura", "Jabari",
]

LAST_NAMES = [
    "Smith", "Johnson", "Brown", "Miller", "Davis", "Wilson", "Anderson", "Taylor", "Thomas", "Moore",
    "Martin", "Thompson", "White", "Harris", "Clark", "Lewis", "Walker", "Hall", "Young", "King",
    "Garcia", "Rodriguez", "Martinez", "Lopez", "Gonzalez", "Perez", "Sanchez", "Ramirez", "Torres", "Silva",
    "Rossi", "Bianchi", "Dubois", "Lefebvre", "Muller", "Schmidt", "Novak", "Kowalski", "Jensen", "Larsen",
    "Tanaka", "Sato", "Kim", "Park", "Chen", "Wang", "Nguyen", "Tran", "Sharma", "Patel",
    "Khan", "Haddad", "Rahman", "Okafor", "Mensah", "Abebe", "Hassan", "Ali", "Cohen", "Levi",
    "O'Brien", "Murphy", "Kelly", "Fischer", "Weber", "Costa", "Ferreira", "Ivanov", "Petrov", "Yilmaz",
    # 2배 확장분
    "Robinson", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Carter", "Mitchell", "Roberts",
    "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart", "Morris", "Reed",
    "Hernandez", "Diaz", "Morales", "Ortiz", "Castillo", "Vargas", "Romero", "Alvarez", "Mendoza", "Oliveira",
    "Santos", "Pereira",
    "Ricci", "Romano", "Moreau", "Laurent", "Becker", "Hoffmann", "Wagner", "Nielsen", "Lindqvist", "Horvat",
    "Dvorak", "Popescu", "Papadopoulos", "Andersen",
    "Suzuki", "Watanabe", "Yamamoto", "Lee", "Choi", "Jung", "Liu", "Zhang", "Huang", "Pham", "Le", "Gupta",
    "Reddy", "Iyer",
    "Farouk", "Nasser", "Aziz", "Adeyemi", "Boateng", "Kamau", "Mwangi", "Diallo", "Toure", "Demir",
]

CITIES = [
    # 북미
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego",
    "Dallas", "Austin", "Seattle", "Denver", "Boston", "Nashville", "Portland", "Las Vegas", "Atlanta",
    "Miami", "Minneapolis", "New Orleans", "Salt Lake City", "Pittsburgh", "Honolulu", "Anchorage",
    "Toronto", "Vancouver", "Montreal", "Calgary", "Mexico City", "Guadalajara",
    # 남미
    "Sao Paulo", "Rio de Janeiro", "Buenos Aires", "Santiago", "Lima", "Bogota", "Medellin", "Quito",
    # 유럽
    "London", "Manchester", "Edinburgh", "Dublin", "Paris", "Lyon", "Marseille", "Berlin", "Munich",
    "Hamburg", "Amsterdam", "Rotterdam", "Brussels", "Zurich", "Geneva", "Vienna", "Prague", "Warsaw",
    "Krakow", "Budapest", "Rome", "Milan", "Florence", "Naples", "Madrid", "Barcelona", "Seville",
    "Valencia", "Lisbon", "Porto", "Copenhagen", "Stockholm", "Oslo", "Helsinki", "Reykjavik", "Athens",
    "Istanbul", "Bucharest",
    # 아시아·오세아니아
    "Tokyo", "Osaka", "Kyoto", "Seoul", "Busan", "Beijing", "Shanghai", "Hong Kong", "Taipei", "Singapore",
    "Bangkok", "Chiang Mai", "Hanoi", "Ho Chi Minh City", "Kuala Lumpur", "Jakarta", "Bali", "Manila",
    "Mumbai", "Delhi", "Bangalore", "Kathmandu", "Dubai", "Abu Dhabi", "Doha", "Tel Aviv",
    "Sydney", "Melbourne", "Brisbane", "Perth", "Auckland", "Wellington",
    # 아프리카
    "Cairo", "Marrakech", "Casablanca", "Lagos", "Nairobi", "Accra", "Addis Ababa", "Cape Town",
    "Johannesburg", "Zanzibar",
]

# (국가번호, 번호 길이 규칙) — 각 나라 휴대폰 번호 형식을 흉내 낸 가짜 번호. E.164 (+ 와 숫자 7~15자리)
_PHONE_FORMATS = [
    ("1", lambda r: f"{r.randint(2, 9)}{r.randint(0, 9)}{r.randint(0, 9)}{r.randint(2, 9)}{_digits(r, 6)}"),  # 미국·캐나다
    ("44", lambda r: f"7{_digits(r, 9)}"),            # 영국
    ("49", lambda r: f"1{r.choice('567')}{_digits(r, 9)}"),  # 독일
    ("33", lambda r: f"{r.choice('67')}{_digits(r, 8)}"),    # 프랑스
    ("34", lambda r: f"6{_digits(r, 8)}"),            # 스페인
    ("39", lambda r: f"3{_digits(r, 9)}"),            # 이탈리아
    ("81", lambda r: f"{r.choice(['70', '80', '90'])}{_digits(r, 8)}"),  # 일본
    ("82", lambda r: f"10{_digits(r, 8)}"),           # 한국
    ("91", lambda r: f"{r.randint(6, 9)}{_digits(r, 9)}"),   # 인도
    ("61", lambda r: f"4{_digits(r, 8)}"),            # 호주
    ("55", lambda r: f"119{_digits(r, 8)}"),          # 브라질
    ("52", lambda r: f"55{_digits(r, 8)}"),           # 멕시코
]


def _digits(r: random.Random, n: int) -> str:
    return "".join(str(r.randint(0, 9)) for _ in range(n))


def _split(values: list[str], split: str) -> list[str]:
    """고정된 규칙(5개마다 1개를 test)으로 나눈다. 시드와 무관하게 항상 같은 결과."""
    test = values[::5]
    return test if split == "test" else [v for v in values if v not in test]


class _Cycle:
    """목록을 섞은 뒤 순서대로 꺼내고, 다 쓰면 다시 섞는다 → 모든 값이 거의 같은 횟수로 쓰인다.
    (배치마다 무작위 추출하면 어떤 값은 여러 번, 어떤 값은 0번 나오는 쏠림이 생긴다.)
    처음부터 섞는 이유: 목록이 문화권별로 묶여 있어 그냥 순서대로 자르면 배치 하나가 한 문화권 이름만 받는다."""

    def __init__(self, values: list[str], rng: random.Random):
        self.values, self.rng, self.queue = list(values), rng, []

    def take(self, n: int) -> list[str]:
        out = []
        while len(out) < n:
            if not self.queue:
                self.queue = self.values[:]
                self.rng.shuffle(self.queue)
            v = self.queue.pop()
            if v in out:  # 다시 섞는 경계에서 같은 배치에 같은 값이 두 번 들어가는 것만 막는다
                self.queue.insert(0, v)
                continue
            out.append(v)
        return out


class ValuePools:
    def __init__(self, split: str, seed: int):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        # split 마다 다른 난수열 → train/test 전화번호가 사실상 겹치지 않음
        self.rng = random.Random(f"{seed}-{split}")
        self.first = _Cycle(_split(FIRST_NAMES, split), self.rng)
        self.last = _Cycle(_split(LAST_NAMES, split), self.rng)
        self.cities = _Cycle(_split(CITIES, split), self.rng)

    def full_names(self, n: int) -> list[str]:
        return [f"{f} {l}" for f, l in zip(self.first.take(n), self.last.take(n))]

    def first_names(self, n: int) -> list[str]:
        return self.first.take(n)

    def phone_numbers(self, n: int) -> list[str]:
        out = []
        for _ in range(n):
            code, rest = self.rng.choice(_PHONE_FORMATS)
            out.append(f"+{code}{rest(self.rng)}")
        return out

    def cities_sample(self, n: int) -> list[str]:
        return self.cities.take(n)

    def candidates(self, function: str, n: int) -> list[str]:
        """함수별로 질의 생성 프롬프트에 넣을 후보 안내 문장들. 해당 없는 함수는 빈 리스트."""
        lines = []
        if function == "send_text_message":
            lines.append("Contact names to choose from (a relation like 'Mom' is also fine sometimes): "
                         + ", ".join(self.first_names(n)))
        if function == "create_contact":
            lines.append("Full names to choose from: " + ", ".join(self.full_names(n)))
        if function in ("create_contact", "make_phone_call"):
            lines.append("Phone numbers to choose from (the user may say or type them in any natural way, "
                         "with spaces, dashes, or spelled-out digits): " + ", ".join(self.phone_numbers(n)))
        if function == "get_weather_forecast":
            lines.append("Cities to choose from (a ZIP code or a more specific place in the city is also fine): "
                         + ", ".join(self.cities_sample(n)))
        if function == "find_route_google_maps":
            lines.append("Set the routes in these cities, using real streets, landmarks, or venues there: "
                         + ", ".join(self.cities_sample(n)))
        return lines
