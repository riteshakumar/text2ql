from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any

from text2ql.schema_config import NormalizedSchemaConfig
from text2ql.types import QueryRequest, QueryResult


class QueryEngine(ABC):
    """Converts a normalized request into a target query language."""

    @abstractmethod
    def generate(self, request: QueryRequest) -> QueryResult:
        """Generate a query."""

    async def agenerate(self, request: QueryRequest) -> QueryResult:
        """Async generate — default runs sync generate in a thread pool."""
        return await asyncio.to_thread(self.generate, request)

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
