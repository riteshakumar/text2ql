import json

import pytest

from text2ql import Text2QL
from text2ql.providers.base import LLMProvider

pytestmark = pytest.mark.unit


class StubProvider(LLMProvider):
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(self.payload)


class InvalidProvider(LLMProvider):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "not-json"


def test_llm_mode_uses_provider_with_constrained_output() -> None:
    service = Text2QL(
        provider=StubProvider(
            {
                "entity": "customers",
                "fields": ["email", "status"],
                "filters": {"status": "active", "limit": 3},
                "explanation": "Parsed by adapter.",
                "confidence": 0.93,
            }
        )
    )

    result = service.generate(
        "show top 3 clients with mail state active",
        schema={"entities": ["customers"], "fields": ["id", "email", "status"]},
        context={"mode": "llm"},
    )

    assert "customers(limit: 3, status: \"active\")" in result.query
    assert result.metadata["mode"] == "llm"
    assert result.confidence == pytest.approx(0.93)


def test_llm_mode_falls_back_to_deterministic_on_invalid_output() -> None:
    service = Text2QL(provider=InvalidProvider())

    result = service.generate("list users", context={"mode": "llm"})

    assert result.metadata["mode"] == "deterministic"
    assert "user" in result.query
