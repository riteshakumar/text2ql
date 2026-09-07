from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from text2ql.schema_config import NormalizedSchemaConfig
from text2ql.types import QueryRequest, QueryResult, ValidationError

logger = logging.getLogger(__name__)


class QueryEngine(ABC):
    """Converts a normalized request into a target query language."""

    @abstractmethod
    def generate(self, request: QueryRequest) -> QueryResult:
        """Generate a query."""

    async def agenerate(self, request: QueryRequest) -> QueryResult:
        """Async generate — default runs sync generate in a thread pool."""
        return await asyncio.to_thread(self.generate, request)

    def parse_to_ir(self, request: QueryRequest):
        """Return the exact compiled intent; direct LLM SQL has no portable IR."""
        result = self.generate(request)
        if result.ir is None or not result.executable:
            raise ValidationError("No compiled intent available", [result.explanation])
        return result.ir

    def _finish_result(self, result: QueryResult, request: QueryRequest, config: NormalizedSchemaConfig) -> QueryResult:
        from text2ql.query_validation import validate_graphql, validate_sql

        notes = result.metadata.get("validation_notes", [])
        if notes:
            if self.strict_validation:
                raise ValidationError("Query intent could not be preserved", list(notes))
            result.status = "needs_review"
        if result.target == "sql":
            if result.ir is not None:
                from text2ql.renderers import SQLIRRenderer
                result.query = SQLIRRenderer(request.context.get("dialect", "sqlite")).render(result.ir)
            validate_sql(result.query, config if self.strict_validation or result.metadata.get("mode") == "llm_direct" else None,
                         dialect=request.context.get("dialect", "sqlite"))
        else:
            validate_graphql(result.query, config, operation_name=request.context.get("operation_name"))
        if result.metadata.get("mode") == "llm_direct":
            result.confidence = 0.0
            result.metadata["confidence_kind"] = "unavailable"
        else:
            result.metadata["confidence_kind"] = "heuristic"
            result.metadata["heuristic_score"] = result.confidence
        if result.ir is not None:
            result.ir.source_text = request.text
        result.metadata["validation_status"] = result.status
        return result

    @staticmethod
    def _clarification(prompt: str, config: NormalizedSchemaConfig, target: str) -> QueryResult | None:
        from text2ql.engines.text_utils import contains_entity_token
        if not prompt.strip():
            reason = "Please describe the records or calculation you want."
        elif config.entities:
            vocabulary = set(config.entities) | set(config.fields) | set(config.entity_aliases) | set(config.field_aliases) | set(config.filter_key_aliases)
            vocabulary.update(name for name in ("own", "hold", "holding", "position") if any(term in str(field).lower() for field in config.fields for term in ("ticker", "symbol", "quantity", "share")))
            vocabulary.update(alias for values in config.filter_value_aliases.values() for alias in values)
            for intent in config.keyword_intents:
                words = intent.get("keywords", [])
                vocabulary.update([words] if isinstance(words, str) else words)
            # Match plural words and human spellings of compound schema names.
            for term in list(vocabulary):
                words = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(term))
                vocabulary.update(word for word in re.split(r"[^a-zA-Z]+", words) if len(word) > 2)
            if any(contains_entity_token(prompt.lower(), str(term)) for term in vocabulary):
                return None
            compact = re.sub(r"[^a-z]", "", prompt.lower())
            if any(len(term) > 3 and re.sub(r"[^a-z]", "", term.lower()) in compact for term in config.entities):
                return None
            if config.default_entity and re.fullmatch(
                r"(?:please )?(?:show|list|fetch|get|count)(?: me)?(?: all)?(?: records| rows| items)?[.!?]?",
                prompt.strip().lower(),
            ):
                return None
            reason = "Which table or field should this request use?"
        else:
            return None
        return QueryResult(query="", target=target, confidence=0.0, explanation=reason,
                           status="needs_clarification", metadata={"candidates": list(config.entities), "confidence_kind": "unavailable"})

    def _failed_llm(self, error: str, target: str) -> QueryResult:
        if self.strict_validation:
            raise ValidationError("LLM generation failed", [error])
        return QueryResult(query="", target=target, confidence=0.0, explanation=error,
                           status="needs_clarification", metadata={"llm_error": error, "confidence_kind": "unavailable"})

    @staticmethod
    def _apply_system_context(system_prompt: str, context: dict[str, Any]) -> str:
        """Append optional caller-supplied system context to the base system prompt."""
        extra = context.get("system_context")
        if not isinstance(extra, str):
            return system_prompt
        cleaned = extra.strip()
        if not cleaned:
            return system_prompt
        return f"{system_prompt}\n\nAdditional system context:\n{cleaned}"

    @staticmethod
    def _extract_entity_from_text(lowered: str) -> str:
        """Heuristically extract the most likely entity name from raw query text.

        Used only when no schema entities are declared.  Avoids hardcoded domain
        lists by tokenising the query and skipping common stop-words; basic
        singularisation (strip trailing *s*) converts plural nouns to their root
        form so that "list users" → "user".
        """
        _STOP_WORDS = frozenset({
            "list", "show", "get", "fetch", "find", "display", "give", "tell",
            "me", "all", "the", "a", "an", "of", "from", "with", "where",
            "and", "or", "by", "in", "for", "is", "are", "was", "were",
            "have", "has", "had", "my", "your", "their", "its", "our",
            "what", "which", "who", "how", "top", "latest", "first", "last",
            "recent", "new", "old",
        })
        tokens = re.findall(r"[a-z][a-z0-9_]*", lowered)
        for token in tokens:
            if token in _STOP_WORDS or len(token) < 3:
                continue
            if token.endswith("s") and len(token) > 3:
                return token[:-1]
            return token
        return "items"


def compute_deterministic_confidence(
    entity: str,
    fields: list[str],
    filters: dict[str, Any],
    validation_notes: list[str],
    config: NormalizedSchemaConfig,
    *,
    extra_signals: dict[str, Any] | None = None,
) -> float:
    """Compute a runtime confidence score for deterministic mode.

    Signals (in order of contribution):
    - Schema provided: base certainty the engine had real vocabulary to match against.
    - Entity resolution: exact schema name > alias > semantic fallback > pure guess.
    - Field coverage: fraction of selected fields that appear in the entity's schema.
    - Filters: reward for finding meaningful constraints (more = better signal).
    - Extra engine signals: aggregations/nested (GraphQL) or joins/order_by (SQL).
    - Validation penalty: deduct for each issue caught during post-generation validation.
    """
    extra = extra_signals or {}
    has_schema = bool(config.entities)

    score = (
        0.30
        + _schema_score(has_schema)
        + _entity_score(entity, config, has_schema)
        + _field_score(entity, fields, config)
        + _filter_score(filters)
        + _extra_signal_score(extra)
        - _validation_penalty(validation_notes)
    )
    return round(min(0.97, max(0.15, score)), 4)


def _schema_score(has_schema: bool) -> float:
    return 0.10 if has_schema else 0.0


def _entity_score(entity: str, config: NormalizedSchemaConfig, has_schema: bool) -> float:
    if entity in config.entities:
        return 0.20
    if entity in set(config.entity_aliases.values()):
        return 0.16
    return 0.05 if has_schema else 0.12


def _field_score(entity: str, fields: list[str], config: NormalizedSchemaConfig) -> float:
    if not fields:
        return 0.01
    schema_fields = set(config.fields_by_entity.get(entity, config.fields))
    if not schema_fields:
        return 0.08
    matched = sum(1 for f in fields if f in schema_fields)
    return 0.15 * (matched / len(fields))


def _filter_score(filters: dict[str, Any]) -> float:
    if not filters:
        return 0.0
    return 0.10 + 0.03 * min(len(filters) - 1, 3)


def _extra_signal_score(extra: dict[str, Any]) -> float:
    return (
        (0.03 if extra.get("aggregations") else 0.0)
        + (0.03 if extra.get("nested") else 0.0)
        + (0.03 if extra.get("joins") else 0.0)
        + (0.02 if extra.get("order_by") else 0.0)
    )


def _validation_penalty(validation_notes: list[str]) -> float:
    return min(0.20, 0.05 * len(validation_notes))
