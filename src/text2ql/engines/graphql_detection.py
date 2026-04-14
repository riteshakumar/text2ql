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
    metric_entity = _infer_entity_from_metric_intent(engine, lowered, config)
    if metric_entity is not None and metric_entity != alias_or_name_entity:
        return metric_entity
    if alias_or_name_entity is not None:
        return alias_or_name_entity

    inferred_from_values = _infer_entity_from_filter_value_aliases(engine, lowered, config)
    if inferred_from_values is not None:
        return inferred_from_values

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


def _infer_entity_from_metric_intent(
    engine: "GraphQLEngine",
    lowered: str,
    config: "NormalizedSchemaConfig",
) -> str | None:
    phrase_to_fields: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("net worth", ("netWorth", "regulatoryNetWorth", "totalMarketVal", "marketVal")),
        ("market value", ("totalMarketVal", "marketVal", "fidelityTotalMktVal", "nonFidelityTotalMktVal")),
        ("gain loss", ("totalGainLoss", "todaysGainLoss", "netWorthChg")),
    )
    for phrase, field_candidates in phrase_to_fields:
        if phrase not in lowered:
            continue
        return _best_entity_for_field_candidates(engine, config, field_candidates)
    return None


def _best_entity_for_field_candidates(
    engine: "GraphQLEngine",
    config: "NormalizedSchemaConfig",
    field_candidates: tuple[str, ...],
) -> str | None:
    candidates: list[tuple[str, int, int]] = []
    wanted = {str(field).lower() for field in field_candidates}
    for entity in getattr(config, "entities", []):
        fields = engine._fields_for_entity(config, entity)
        if not fields:
            continue
        lowered_fields = {str(field).lower() for field in fields}
        score = sum(1 for field in wanted if field in lowered_fields)
        if score <= 0:
            continue
        candidates.append((str(entity), score, len(fields)))
    if not candidates:
        return None
    max_score = max(score for _, score, _ in candidates)
    top = [(entity, width) for entity, score, width in candidates if score == max_score]
    narrowest = min(width for _, width in top)
    narrowed = [entity for entity, width in top if width == narrowest]
    if len(narrowed) == 1:
        return narrowed[0]
    return None


def _infer_entity_from_filter_value_aliases(
    engine: "GraphQLEngine",
    lowered: str,
    config: "NormalizedSchemaConfig",
) -> str | None:
    value_aliases = getattr(config, "filter_value_aliases", {})
    if not isinstance(value_aliases, dict) or not value_aliases:
        return None

    scores: dict[str, float] = {}
    widths: dict[str, int] = {}
    for canonical, alias_map in value_aliases.items():
        if not isinstance(alias_map, dict):
            continue
        if not any(_matches_value_alias(lowered, str(alias)) for alias in alias_map.keys()):
            continue
        for entity in getattr(config, "entities", []):
            support = _entity_filter_key_support_score(engine, config, entity, str(canonical))
            if support <= 0:
                continue
            scores[entity] = scores.get(entity, 0.0) + support
            widths[entity] = min(
                widths.get(entity, 10_000),
                _entity_specificity_width(engine, config, entity),
            )

    if not scores:
        return None

    max_score = max(scores.values())
    top = [entity for entity, score in scores.items() if score == max_score]
    if len(top) == 1:
        return top[0]

    narrowest_width = min(widths.get(entity, 10_000) for entity in top)
    narrowed = [entity for entity in top if widths.get(entity, 10_000) == narrowest_width]
    if len(narrowed) == 1:
        return narrowed[0]
    return None


def _entity_filter_key_support_score(
    engine: "GraphQLEngine",
    config: "NormalizedSchemaConfig",
    entity: str,
    candidate_key: str,
) -> float:
    canonical = str(candidate_key).lower()
    args = [str(arg).lower() for arg in getattr(config, "args_by_entity", {}).get(entity, [])]
    fields = [str(field).lower() for field in engine._fields_for_entity(config, entity)]
    in_args = canonical in args
    in_fields = canonical in fields
    if not in_args and not in_fields:
        return 0.0
    # Prefer entities that can both filter and project by the canonical key.
    return 1.25 if in_fields else 1.0


def _entity_specificity_width(
    engine: "GraphQLEngine",
    config: "NormalizedSchemaConfig",
    entity: str,
) -> int:
    args = {str(arg) for arg in getattr(config, "args_by_entity", {}).get(entity, [])}
    fields = {str(field) for field in engine._fields_for_entity(config, entity)}
    width = len(args.union(fields))
    return width or 10_000


def _matches_value_alias(lowered: str, alias: str) -> bool:
    alias_text = str(alias).strip().lower()
    if not alias_text:
        return False
    if re.search(rf"\b{re.escape(alias_text)}\b", lowered):
        return True
    alias_tokens = [token for token in re.findall(r"[a-z0-9]+", alias_text) if len(token) >= 4]
    if not alias_tokens:
        return False
    lowered_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    if len(alias_tokens) <= 3:
        return all(token in lowered_tokens for token in alias_tokens)
    overlap = sum(1 for token in alias_tokens if token in lowered_tokens)
    return overlap >= 2


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
    entity_defaults = config.default_fields_by_entity.get(entity, [])
    if entity_defaults:
        defaults = [field for field in entity_defaults if field in schema_fields]
        if defaults:
            return defaults
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
