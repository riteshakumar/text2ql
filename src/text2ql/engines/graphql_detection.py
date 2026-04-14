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

    mentioned_entities = _entities_mentioned_by_alias_or_name(engine, lowered, config)
    alias_or_name_entity = mentioned_entities[0] if mentioned_entities else None
    metric_entity = _infer_entity_from_metric_intent(engine, lowered, config)
    if metric_entity is not None and metric_entity != alias_or_name_entity:
        return metric_entity

    if mentioned_entities:
        disambiguated = _disambiguate_mentioned_entity(engine, lowered, config, mentioned_entities)
        if disambiguated is not None:
            return disambiguated
        return mentioned_entities[0]

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


def _entities_mentioned_by_alias_or_name(
    engine: "GraphQLEngine",
    lowered: str,
    config: "NormalizedSchemaConfig",
) -> list[str]:
    mentions: list[tuple[int, int, str]] = []
    seen: set[str] = set()

    for alias, canonical in engine._sorted_alias_pairs(config.entity_aliases):
        position = _match_position(engine, lowered, alias)
        if position is None:
            continue
        lowered_canonical = str(canonical).lower()
        if lowered_canonical in seen:
            continue
        mentions.append((position, -len(str(alias)), str(canonical)))
        seen.add(lowered_canonical)

    for entity in config.entities:
        position = _match_position(engine, lowered, str(entity).lower())
        if position is None:
            continue
        lowered_entity = str(entity).lower()
        if lowered_entity in seen:
            continue
        mentions.append((position, -len(str(entity)), str(entity)))
        seen.add(lowered_entity)

    mentions.sort(key=lambda item: (item[0], item[1], item[2]))
    return [entity for _, _, entity in mentions]


def _match_position(engine: "GraphQLEngine", lowered: str, token: str) -> int | None:
    token_text = str(token).strip().lower()
    if not token_text:
        return None

    variants: list[str] = [token_text]
    if token_text.endswith("s") and len(token_text) > 3:
        variants.append(token_text[:-1])
    elif len(token_text) > 2:
        variants.append(token_text + "s")

    best_pos: int | None = None
    for candidate in variants:
        match = re.search(rf"\b{re.escape(candidate)}\b", lowered)
        if match is None:
            continue
        if best_pos is None or match.start() < best_pos:
            best_pos = match.start()

    if best_pos is not None:
        return best_pos
    return 10_000 if engine._contains_entity_token(lowered, token_text) else None


def _disambiguate_mentioned_entity(
    engine: "GraphQLEngine",
    lowered: str,
    config: "NormalizedSchemaConfig",
    candidates: list[str],
) -> str | None:
    if len(candidates) <= 1:
        return None

    scored: list[tuple[float, str]] = []
    for entity in candidates:
        score = _entity_composite_score(engine, lowered, config, entity)
        scored.append((score, entity))
    scored.sort(key=lambda item: item[0], reverse=True)

    if not scored:
        return None
    if len(scored) == 1:
        return scored[0][1]

    best_score, best_entity = scored[0]
    second_score, _ = scored[1]
    if best_score >= second_score + 0.15:
        return best_entity
    return None


def _entity_composite_score(
    engine: "GraphQLEngine",
    lowered: str,
    config: "NormalizedSchemaConfig",
    entity: str,
) -> float:
    fields = engine._fields_for_entity(config, entity)
    field_score = engine._score_fields_for_prompt(lowered, fields)
    entity_score = engine._score_entity_name_for_prompt(lowered, entity)
    return (1.6 * field_score) + (1.0 * entity_score)


def _infer_entity_from_metric_intent(
    engine: "GraphQLEngine",
    lowered: str,
    config: "NormalizedSchemaConfig",
) -> str | None:
    phrase_to_fields: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("available cash", ("cashOnly", "cashWithMargin", "cashWithoutEquity", "cash", "withoutMarginImpact")),
        ("cash available", ("cashOnly", "cashWithMargin", "cashWithoutEquity", "cash", "withoutMarginImpact")),
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
        return _enrich_selected_fields(engine, lowered, schema_fields, selected)
    if engine._entity_looks_like_holdings(entity, schema_fields):
        contextual = engine._resolve_holdings_context_fields(lowered, schema_fields)
        if contextual:
            return contextual
    semantic_fields = engine._resolve_fields_by_semantic_match(lowered, schema_fields)
    if semantic_fields:
        return _enrich_selected_fields(engine, lowered, schema_fields, semantic_fields)
    entity_defaults = config.default_fields_by_entity.get(entity, [])
    if entity_defaults:
        defaults = [field for field in entity_defaults if field in schema_fields]
        if defaults:
            return defaults
    return config.default_fields or schema_fields[:3]


def _enrich_selected_fields(
    engine: "GraphQLEngine",
    lowered: str,
    schema_fields: list[str],
    selected: list[str],
) -> list[str]:
    out = list(dict.fromkeys(selected))
    if not schema_fields:
        return out

    desired_min = _desired_projection_width(lowered)
    semantic_ranked = _rank_semantic_fields(engine, lowered, schema_fields)

    if _prompt_requests_name(lowered) and not any(_is_name_like_field(field) for field in out):
        name_candidate = next((field for _, field in semantic_ranked if _is_name_like_field(field)), None)
        if name_candidate is not None:
            out.insert(0, name_candidate)

    if len(out) < desired_min:
        for score, field in semantic_ranked:
            if score <= 0:
                continue
            if field in out:
                continue
            out.append(field)
            if len(out) >= desired_min:
                break

    return list(dict.fromkeys(out))


def _rank_semantic_fields(
    engine: "GraphQLEngine",
    lowered: str,
    schema_fields: list[str],
) -> list[tuple[float, str]]:
    ranked: list[tuple[float, str]] = []
    for field in schema_fields:
        score = engine._score_field_for_prompt(lowered, field)
        if score > 0:
            ranked.append((score, field))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _desired_projection_width(lowered: str) -> int:
    if " and " in lowered:
        return 2
    if re.search(r"\bnames?\b", lowered) is not None:
        return 2
    return 1


def _prompt_requests_name(lowered: str) -> bool:
    return re.search(r"\bnames?\b", lowered) is not None


def _is_name_like_field(field: str) -> bool:
    lowered = str(field).lower()
    return (
        lowered == "name"
        or lowered.endswith("name")
        or lowered.endswith("_name")
        or "name_" in lowered
        or "fullname" in lowered
    )


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
    if (
        re.search(r"\b(?:have|has|with)\s+more than\s+\d+\b", lowered)
        and not any(agg.get("function") == "count" for agg in aggregations)
    ):
        aggregations.append({"function": "count", "field": ""})

    for fn_name in ["sum", "avg", "min", "max"]:
        if not re.search(rf"\b{fn_name}\b", lowered):
            continue
        metric_field = engine._detect_metric_field(lowered, candidate_fields)
        aggregations.append({"function": fn_name, "field": metric_field})

    return aggregations
