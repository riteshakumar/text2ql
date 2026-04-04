from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """Minimal OpenAI-compatible chat completions adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 30,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("TEXT2QL_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or TEXT2QL_API_KEY.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        raw = self._request_with_retries(request)

        decoded = json.loads(raw)
        choices = decoded.get("choices", [])
        if not choices:
            raise RuntimeError("LLM provider returned no choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("LLM provider returned invalid content payload")
        return content.strip()

    def _request_with_retries(self, request: urllib.request.Request) -> str:
        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 or attempt == attempts - 1:
                    break
                retry_after = exc.headers.get("Retry-After")
                wait_seconds = self._retry_delay(attempt, retry_after)
                time.sleep(wait_seconds)
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                time.sleep(self._retry_delay(attempt, None))

        raise RuntimeError(f"LLM provider request failed: {last_error}") from last_error

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return self.retry_backoff_seconds * (attempt + 1)
