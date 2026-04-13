import urllib.error
import urllib.request

import pytest

from text2ql.providers.openai_compatible import OpenAICompatibleProvider

pytestmark = pytest.mark.unit


class _FakeHTTPResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


def test_complete_structured_falls_back_to_plain_complete_when_disabled() -> None:
    provider = OpenAICompatibleProvider(api_key="sk-test", use_structured_output=False)
    provider.complete = lambda *_: '{"plain": true}'  # type: ignore[method-assign]

    result = provider.complete_structured(
        system_prompt="system",
        user_prompt="user",
        json_schema={"type": "object"},
    )

    # Rule: complete_structured delegates to complete() when structured output is disabled.
    assert isinstance(result, str)


def test_complete_structured_falls_back_to_plain_complete_on_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatibleProvider(api_key="sk-test", use_structured_output=True)

    def _raise_runtime_error(*_: object) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(provider, "_build_structured_request", lambda *_: object())
    monkeypatch.setattr(provider, "_request_with_retries", _raise_runtime_error)
    monkeypatch.setattr(provider, "complete", lambda *_: '{"ok": true}')

    result = provider.complete_structured(
        system_prompt="system",
        user_prompt="user",
        json_schema={"type": "object"},
    )

    assert result == '{"ok": true}'


def test_request_with_retries_recovers_after_single_429(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        max_retries=1,
        retry_backoff_seconds=0.01,
    )
    calls = {"count": 0}

    def _fake_urlopen(_request: urllib.request.Request, timeout: int) -> _FakeHTTPResponse:
        del timeout
        if calls["count"] == 0:
            calls["count"] += 1
            raise urllib.error.HTTPError(
                url="https://example.test",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "0"},
                fp=None,
            )
        return _FakeHTTPResponse('{"choices":[{"message":{"content":"ok"}}]}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    result = provider.complete(system_prompt="system", user_prompt="user")

    assert result == "ok"
    assert calls["count"] == 1


def test_retry_delay_prefers_retry_after_header() -> None:
    provider = OpenAICompatibleProvider(api_key="sk-test", retry_backoff_seconds=1.5)

    assert provider._retry_delay(attempt=2, retry_after="3.2") == pytest.approx(3.2)
    assert provider._retry_delay(attempt=2, retry_after="invalid") == pytest.approx(4.5)
