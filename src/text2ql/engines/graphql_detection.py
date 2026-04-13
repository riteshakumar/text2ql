from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text2ql.engines.graphql import GraphQLEngine
    from text2ql.schema_config import NormalizedSchemaConfig


def detect_entity(engine: "GraphQLEngine", text: str, config: "NormalizedSchemaConfig") -> str:
    lowered = text.lower()
    owned_asset = engine._detect_owned_asset(lowered)
    holdings_entity = engine._resolve_holdings_entity(config)
    if owned_asset and holdings_entity:
        return holdings_entity
    special_entity = engine._resolve_special_entity(lowered, config)
    if special_entity is not None:
        return special_entity

    alias_or_name_entity = engine._resolve_entity_by_alias_or_name(lowered, config)
    if alias_or_name_entity is not None:
        return alias_or_name_entity

    semantic_entity = engine._resolve_entity_by_semantic_field_match(lowered, config)
    if semantic_entity is not None:
        return semantic_entity

    if config.default_entity:
        return config.default_entity

    # Prefer the first schema-declared entity over generic text extraction.
    if config.entities:
        return config.entities[0]

    # Last resort when no schema is provided: extract a noun-like token.
    return engine._extract_entity_from_text(lowered)


def detect_fields(
    engine: "GraphQLEngine",
    text: str,
    config: "NormalizedSchemaConfig",
    entity: str,
) -> list[str]:
    lowered = text.lower()
    common = ["id", "name", "title", "email", "createdAt", "status", "price"]

    schema_fields = engine._fields_for_entity(config, entity)
    owned_asset = engine._detect_owned_asset(lowered)
    if owned_asset and engine._entity_looks_like_holdings(entity, schema_fields):
        owned_fields = engine._resolve_holdings_fields(schema_fields)
        if owned_fields:
            return owned_fields

    if not schema_fields:
        return engine._detect_common_fields(lowered, common)

    selected = engine._select_fields_from_schema(lowered, schema_fields, config)
    if selected:
        return selected
    if engine._entity_looks_like_holdings(entity, schema_fields):
        contextual = engine._resolve_holdings_context_fields(lowered, schema_fields)
        if contextual:
            return contextual
    semantic_fields = engine._resolve_fields_by_semantic_match(lowered, schema_fields)
    if semantic_fields:
        return semantic_fields
    return config.default_fields or schema_fields[:3]


def detect_aggregations(
    engine: "GraphQLEngine",
    text: str,
    config: "NormalizedSchemaConfig",
    entity: str,
) -> list[dict[str, str]]:
    lowered = text.lower()
    aggregations: list[dict[str, str]] = []
    candidate_fields = engine._fields_for_entity(config, entity)

    if re.search(r"\bcount\b", lowered) or (
        re.search(r"\bhow many\b", lowered) and engine._detect_owned_asset(lowered) is None
    ):
        aggregations.append({"function": "count", "field": ""})

    for fn_name in ["sum", "avg", "min", "max"]:
        if not re.search(rf"\b{fn_name}\b", lowered):
            continue
        metric_field = engine._detect_metric_field(lowered, candidate_fields)
        aggregations.append({"function": fn_name, "field": metric_field})

    return aggregations
