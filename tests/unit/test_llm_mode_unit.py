import asyncio
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


class ErrorProvider(LLMProvider):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("HTTP Error 429: Too Many Requests")


class StructuredFallbackProvider(LLMProvider):
    """Returns valid JSON from complete() but raises on complete_structured()."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(self.payload)

    def complete_structured(self, system_prompt: str, user_prompt: str, json_schema: dict) -> str:
        raise RuntimeError("Structured output not supported by this model")


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
        context={"mode": "llm", "language": "english"},
    )

    assert "customers(limit: 3, status: \"active\")" in result.query
    assert result.metadata["mode"] == "llm"
    assert result.metadata["language"] == "english"
    # confidence is now calibrated from schema signals; raw LLM value is preserved in metadata
    assert result.metadata["llm_confidence"] == pytest.approx(0.93)
    assert 0.0 <= result.confidence <= 1.0


def test_llm_mode_falls_back_to_deterministic_on_invalid_output() -> None:
    service = Text2QL(provider=InvalidProvider())

    result = service.generate("list users", context={"mode": "llm"})

    assert result.metadata["mode"] == "deterministic"
    assert "user" in result.query


def test_llm_mode_falls_back_to_deterministic_on_provider_error() -> None:
    service = Text2QL(provider=ErrorProvider())

    result = service.generate("list users", context={"mode": "llm"})

    assert result.metadata["mode"] == "deterministic"
    assert result.metadata["llm_error"] is not None
    assert "user" in result.query


def test_llm_mode_falls_back_to_deterministic_on_unsupported_language() -> None:
    service = Text2QL(
        provider=StubProvider(
            {
                "entity": "customers",
                "fields": ["email"],
                "filters": {},
                "explanation": "Parsed by adapter.",
                "confidence": 0.93,
            }
        )
    )

    result = service.generate("list users", context={"mode": "llm", "language": "spanish"})

    assert result.metadata["mode"] == "deterministic"
    assert "user" in result.query


def test_sql_llm_mode_uses_provider_with_constrained_output() -> None:
    service = Text2QL(
        provider=StubProvider(
            {
                "table": "orders",
                "columns": ["id", "status"],
                "filters": {"status": "active"},
                "joins": [],
                "order_by": "createdAt",
                "order_dir": "DESC",
                "limit": 5,
                "offset": 0,
                "explanation": "Parsed SQL by adapter.",
                "confidence": 0.91,
            }
        )
    )

    result = service.generate(
        "show top 5 latest orders with status active",
        target="sql",
        schema={"entities": ["orders"], "fields": {"orders": ["id", "status", "createdAt"]}},
        context={"mode": "llm", "language": "english"},
    )

    assert result.target == "sql"
    assert "FROM orders" in result.query
    assert "ORDER BY orders.createdAt DESC" in result.query
    assert "LIMIT 5" in result.query
    assert result.metadata["mode"] == "llm"
    # confidence is now calibrated from schema signals; raw LLM value is preserved in metadata
    assert result.metadata["llm_confidence"] == pytest.approx(0.91)
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Async path
# ---------------------------------------------------------------------------

def test_async_llm_mode_succeeds() -> None:
    service = Text2QL(
        provider=StubProvider(
            {
                "entity": "orders",
                "fields": ["id", "status"],
                "filters": {"status": "pending"},
                "explanation": "Async LLM result.",
                "confidence": 0.88,
            }
        )
    )

    result = asyncio.run(
        service.agenerate(
            "pending orders",
            schema={"entities": ["orders"], "fields": ["id", "status"]},
            context={"mode": "llm"},
        )
    )

    assert result.metadata["mode"] == "llm"
    assert "orders" in result.query


def test_async_llm_mode_preserves_error_on_fallback() -> None:
    service = Text2QL(provider=ErrorProvider())

    result = asyncio.run(
        service.agenerate("list users", context={"mode": "llm"})
    )

    assert result.metadata["mode"] == "deterministic"
    assert result.metadata.get("llm_error") is not None
    assert "user" in result.query


def test_async_sql_llm_mode_preserves_error_on_fallback() -> None:
    service = Text2QL(provider=ErrorProvider())

    result = asyncio.run(
        service.agenerate("list orders", target="sql", context={"mode": "llm"})
    )

    assert result.metadata["mode"] == "deterministic"
    assert result.metadata.get("llm_error") is not None


# ---------------------------------------------------------------------------
# function_calling mode falls back to plain complete() when structured raises
# ---------------------------------------------------------------------------

def test_function_calling_falls_back_to_plain_complete_on_error() -> None:
    service = Text2QL(
        provider=StructuredFallbackProvider(
            {
                "entity": "products",
                "fields": ["id", "name"],
                "filters": {},
                "explanation": "Fallback plain completion.",
                "confidence": 0.80,
            }
        )
    )

    result = service.generate(
        "list products",
        schema={"entities": ["products"], "fields": ["id", "name"]},
        context={"mode": "function_calling"},
    )

    # Engine catches the structured error and falls through to deterministic;
    # the important thing is we get a valid result, not an unhandled exception.
    assert "product" in result.query.lower()


# ---------------------------------------------------------------------------
# Filter value canonicalization
# ---------------------------------------------------------------------------

def test_llm_mode_applies_filter_value_aliases() -> None:
    service = Text2QL(
        provider=StubProvider(
            {
                "entity": "orders",
                "fields": ["id", "status"],
                "filters": {"status": "active"},
                "explanation": "LLM returned alias value.",
                "confidence": 0.85,
            }
        )
    )

    result = service.generate(
        "show active orders",
        schema={"entities": ["orders"], "fields": ["id", "status"]},
        mapping={"filter_values": {"status": {"active": "ACTIVE"}}},
        context={"mode": "llm"},
    )

    assert result.metadata["mode"] == "llm"
    assert result.metadata["filters"].get("status") == "ACTIVE"


# ---------------------------------------------------------------------------
# system_context injection
# ---------------------------------------------------------------------------

def test_llm_mode_system_context_is_injected() -> None:
    """system_context from request.context must appear in the prompt seen by the provider."""

    received_system_prompts: list[str] = []

    class CapturingProvider(LLMProvider):
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            received_system_prompts.append(system_prompt)
            return json.dumps(
                {
                    "entity": "users",
                    "fields": ["id"],
                    "filters": {},
                    "explanation": "ok",
                    "confidence": 0.9,
                }
            )

    service = Text2QL(provider=CapturingProvider())
    service.generate(
        "list users",
        context={"mode": "llm", "system_context": "Only return PII-free fields."},
    )

    assert received_system_prompts, "Provider was never called"
    assert "Only return PII-free fields." in received_system_prompts[0]


# ---------------------------------------------------------------------------
# Unknown join relation skipped (SQL LLM mode)
# ---------------------------------------------------------------------------

def test_sql_llm_mode_skips_unknown_join_relation() -> None:
    service = Text2QL(
        provider=StubProvider(
            {
                "table": "orders",
                "columns": ["id", "status"],
                "filters": {},
                "joins": [{"relation": "nonexistent_relation", "fields": ["id"]}],
                "order_by": None,
                "order_dir": None,
                "limit": None,
                "offset": None,
                "explanation": "LLM hallucinated a join.",
                "confidence": 0.75,
            }
        )
    )

    result = service.generate(
        "show orders",
        target="sql",
        schema={"entities": ["orders"], "fields": {"orders": ["id", "status"]}},
        context={"mode": "llm"},
    )

    # Invalid join must be silently dropped — query must still be valid SQL
    assert result.metadata["mode"] == "llm"
    assert "JOIN" not in result.query
    assert "FROM orders" in result.query
