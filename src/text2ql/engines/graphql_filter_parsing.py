from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text2ql.engines.graphql import GraphQLEngine
    from text2ql.schema_config import NormalizedSchemaConfig


def detect_filters(
    engine: "GraphQLEngine",
    text: str,
    config: "NormalizedSchemaConfig",
    entity: str,
) -> dict[str, Any]:
    lowered = text.lower()
    where_clause = engine._extract_where_clause(lowered)
    filters = engine._extract_limit_filters(lowered)

    filter_key_aliases = {"status": "status"}
    filter_key_aliases.update(config.filter_key_aliases)

    engine._apply_alias_key_filters(
        filters=filters,
        lowered=lowered,
        where_clause=where_clause,
        config=config,
        entity=entity,
        filter_key_aliases=filter_key_aliases,
    )
    engine._apply_alias_value_filters(
        filters=filters,
        lowered=lowered,
        config=config,
        entity=entity,
    )
    engine._apply_owned_asset_filter(
        filters=filters,
        lowered=lowered,
        config=config,
        entity=entity,
        filter_key_aliases=filter_key_aliases,
    )
    engine._apply_advanced_filters(
        filters=filters,
        lowered=lowered,
        candidate_fields=engine._fields_for_entity(config, entity),
    )
    return filters
