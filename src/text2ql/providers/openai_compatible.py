from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Minimal OpenAI-compatible chat completions adapter.

    Parameters
    ----------
    api_key:
        API key. Defaults to ``OPENAI_API_KEY`` or ``TEXT2QL_API_KEY`` env vars.
    model:
        Model identifier sent in every request.
    base_url:
        API base URL.  Defaults to the OpenAI endpoint.
    timeout_seconds:
        Per-request HTTP timeout.
    max_retries:
        Number of additional retry attempts on 429 / transient errors.
    retry_backoff_seconds:
        Base backoff multiplier (attempt × backoff).
    use_structured_output:
        When ``True``, :meth:`complete_structured` sends the JSON schema via
        ``response_format: {"type": "json_schema", ...}`` (requires a model
        that supports the Structured Outputs feature, e.g. ``gpt-4o-2024-08-06``
        or later).  Falls back to plain ``complete()`` for unsupported models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 30,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        use_structured_output: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("TEXT2QL_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or TEXT2QL_API_KEY.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.use_structured_output = use_structured_output
        logger.debug(
            "OpenAICompatibleProvider initialised: model=%s base_url=%s "
            "use_structured_output=%s",
            model,
            base_url,
            use_structured_output,
        )

    def _build_request(self, system_prompt: str, user_prompt: str) -> urllib.request.Request:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        return urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    def _build_structured_request(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
    ) -> urllib.request.Request:
        """Build a request that uses ``response_format: json_schema``."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "query_intent",
                    "schema": json_schema,
                    "strict": True,
                },
            },
        }
        return urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    @staticmethod
    def _parse_response(raw: str) -> str:
        decoded = json.loads(raw)
        choices = decoded.get("choices", [])
        if not choices:
            raise RuntimeError("LLM provider returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("LLM provider returned invalid content payload")
        return content.strip()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        logger.debug("complete(): model=%s", self.model)
        raw = self._request_with_retries(self._build_request(system_prompt, user_prompt))
        return self._parse_response(raw)

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
    ) -> str:
        """Complete with JSON Schema-constrained output (Structured Outputs).

        When :attr:`use_structured_output` is ``True``, sends the schema via
        ``response_format: json_schema`` so the model is forced to emit valid
        JSON matching the schema.

        When ``False`` (or when the model doesn't support the feature), falls
        back to a plain :meth:`complete` call and lets the downstream
        constrained parser validate the output.
        """
        if not self.use_structured_output:
            logger.debug(
                "complete_structured(): use_structured_output=False; "
                "using plain complete() for model=%s",
                self.model,
            )
            return self.complete(system_prompt, user_prompt)

        logger.debug(
            "complete_structured(): using json_schema response_format for model=%s",
            self.model,
        )
        try:
            request = self._build_structured_request(system_prompt, user_prompt, json_schema)
            raw = self._request_with_retries(request)
            return self._parse_response(raw)
        except (RuntimeError, urllib.error.HTTPError) as exc:
            # Structured output may not be supported by the model/endpoint.
            # Fall back to plain completion and let the parser handle it.
            logger.warning(
                "complete_structured(): structured output request failed (%s); "
                "falling back to plain complete()",
                exc,
            )
            return self.complete(system_prompt, user_prompt)

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        """Async completion with non-blocking retries — no thread held during backoff."""
        logger.debug("acomplete(): model=%s", self.model)
        request = self._build_request(system_prompt, user_prompt)
        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                raw = await asyncio.to_thread(
                    urllib.request.urlopen, request, self.timeout_seconds
                )
                return self._parse_response(raw.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 or attempt == attempts - 1:
                    logger.warning(
                        "acomplete(): HTTP %d on attempt %d/%d",
                        exc.code,
                        attempt + 1,
                        attempts,
                    )
                    break
                retry_after = exc.headers.get("Retry-After")
                delay = self._retry_delay(attempt, retry_after)
                logger.warning(
                    "acomplete(): rate-limited (429) on attempt %d/%d; "
                    "retrying in %.1fs",
                    attempt + 1,
                    attempts,
                    delay,
                )
                await asyncio.sleep(delay)
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    logger.error(
                        "acomplete(): network error on final attempt %d: %s",
                        attempt + 1,
                        exc,
                    )
                    break
                delay = self._retry_delay(attempt, None)
                logger.warning(
                    "acomplete(): network error on attempt %d/%d; retrying in %.1fs: %s",
                    attempt + 1,
                    attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(f"LLM provider request failed: {last_error}") from last_error

    async def acomplete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
    ) -> str:
        """Async variant of :meth:`complete_structured`."""
        if not self.use_structured_output:
            return await self.acomplete(system_prompt, user_prompt)

        logger.debug(
            "acomplete_structured(): using json_schema response_format for model=%s",
            self.model,
        )
        try:
            request = self._build_structured_request(system_prompt, user_prompt, json_schema)
            attempts = self.max_retries + 1
            last_error: Exception | None = None
            for attempt in range(attempts):
                try:
                    raw = await asyncio.to_thread(
                        urllib.request.urlopen, request, self.timeout_seconds
                    )
                    return self._parse_response(raw.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    if exc.code != 429 or attempt == attempts - 1:
                        break
                    retry_after = exc.headers.get("Retry-After")
                    await asyncio.sleep(self._retry_delay(attempt, retry_after))
                except urllib.error.URLError as exc:
                    last_error = exc
                    if attempt == attempts - 1:
                        break
                    await asyncio.sleep(self._retry_delay(attempt, None))
            raise RuntimeError(f"LLM provider structured request failed: {last_error}") from last_error
        except (RuntimeError, urllib.error.HTTPError) as exc:
            logger.warning(
                "acomplete_structured(): falling back to plain acomplete(): %s", exc
            )
            return await self.acomplete(system_prompt, user_prompt)

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
                    logger.warning(
                        "_request_with_retries(): HTTP %d on attempt %d/%d",
                        exc.code,
                        attempt + 1,
                        attempts,
                    )
                    break
                retry_after = exc.headers.get("Retry-After")
                delay = self._retry_delay(attempt, retry_after)
                logger.warning(
                    "_request_with_retries(): rate-limited (429) on attempt %d/%d; "
                    "retrying in %.1fs",
                    attempt + 1,
                    attempts,
                    delay,
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    logger.error(
                        "_request_with_retries(): network error on final attempt %d: %s",
                        attempt + 1,
                        exc,
                    )
                    break
                delay = self._retry_delay(attempt, None)
                logger.warning(
                    "_request_with_retries(): network error on attempt %d/%d; "
                    "retrying in %.1fs: %s",
                    attempt + 1,
                    attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)

        raise RuntimeError(f"LLM provider request failed: {last_error}") from last_error

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return self.retry_backoff_seconds * (attempt + 1)
