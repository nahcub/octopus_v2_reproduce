"""Gemini 호출 래퍼: JSON 구조화 출력 + 재시도 + 토큰 사용량 집계."""

from __future__ import annotations

import json
import os
import threading
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

# USD / 1M tokens, 일반(비 Batch) 요청. thought 토큰은 출력 요금으로 청구된다.
# 출처: ai.google.dev/gemini-api/docs/pricing (2026-10 확인). 3.8 Flash 는 2027-01-01 부터 두 배.
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}

_NO_RETRY_STATUS = {400, 401, 403, 404}


class Gemini:
    def __init__(self, model: str, thinking_level: str = "low", max_retries: int = 5, timeout: float = 120):
        load_dotenv()
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY not found (.env or environment variable).")
        # SDK 자체 재시도(429/5xx 를 조용히 최대 4번 재시도)를 끄고, 재시도는 아래 json() 에서만 로그와 함께 한다.
        # 두 겹으로 재시도하면 요청이 불어나 429 가 오래 지속된다 (첫 파일럿이 18분 멈춘 원인).
        self.client = genai.Client(
            api_key=key, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
        self.model = model
        self.thinking_level = thinking_level
        self.max_retries = max_retries
        self.timeout = timeout  # 초. 응답 없이 멈춘 요청은 이 시간 후 재시도
        self.usage = {"calls": 0, "input": 0, "output": 0, "thought": 0}
        self._lock = threading.Lock()

    def json(self, prompt: str, schema: dict) -> dict:
        """prompt 를 보내고 schema 에 맞는 JSON 을 파싱해 돌려준다."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                interaction = self.client.interactions.create(
                    model=self.model,
                    input=prompt,
                    generation_config={"thinking_level": self.thinking_level},
                    response_format={"type": "text", "mime_type": "application/json", "schema": schema},
                    store=False,  # 서버에 대화 기록을 남기지 않음
                    timeout=self.timeout,
                )
                self._add_usage(interaction.usage)
                return json.loads(interaction.output_text)
            except json.JSONDecodeError as e:
                last_error = e
            except Exception as e:  # noqa: BLE001 - SDK 예외 종류가 여러 개라 상태 코드로 판별
                status = getattr(e, "status_code", None) or getattr(e, "code", None)
                if status in _NO_RETRY_STATUS:
                    raise
                last_error = e
            if attempt == self.max_retries - 1:
                break
            # 429(한도 초과)·503(서버 바쁨)은 바로 다시 보내면 악화되므로 10, 20, 40, 60초로 늘려 가며 기다린다
            wait = min(10 * 2 ** attempt, 60)
            print(f"  ! retry {attempt + 1}/{self.max_retries} in {wait}s: {type(last_error).__name__}: "
                  f"{str(last_error)[:300]}", flush=True)
            time.sleep(wait)
        raise RuntimeError(f"Gemini call failed after {self.max_retries} attempts: {last_error}")

    def _add_usage(self, usage) -> None:
        if usage is None:
            return
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input"] += usage.total_input_tokens or 0
            self.usage["output"] += usage.total_output_tokens or 0
            self.usage["thought"] += usage.total_thought_tokens or 0

    def cost_usd(self) -> float | None:
        if self.model not in PRICES:
            return None
        price_in, price_out = PRICES[self.model]
        u = self.usage
        return (u["input"] * price_in + (u["output"] + u["thought"]) * price_out) / 1e6
