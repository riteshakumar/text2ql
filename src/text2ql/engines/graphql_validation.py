from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from text2ql.types import ValidationError

if TYPE_CHECKING:
    from text2ql.engines.graphql import GraphQLEngine
    from text2ql.schema_config import NormalizedSchemaConfig

logger = logging.getLogger(__name__)


def validate_components(
    engine: "GraphQLEngine",
    entity: str,
    fields: list[str],
    filters: dict[str, Any],
    aggregations: list[dict[str, str]],
    nested: list[dict[str, Any]],
    config: "NormalizedSchemaConfig",
) -> tuple[str, list[str], dict[str, Any], list[dict[str, str]], list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    validated_entity = engine._resolve_entity_for_schema(entity, config, notes)
    allowed_fields = engine._resolve_allowed_fields(validated_entity, config)
    validated_fields = engine._validate_fields(
        fields=fields,
        allowed_fields=allowed_fields,
        default_fields=config.default_fields_by_entity.get(validated_entity, config.default_fields),
        entity=validated_entity,
        notes=notes,
        aggregation_only=not fields and bool(aggregations),
    )
    allowed_args = engine._resolve_allowed_args(validated_entity, config)
    validated_filters = engine._validate_filters(
        filters=filters,
        allowed_args=allowed_args,
        entity=validated_entity,
        config=config,
        notes=notes,
    )

    # Contradiction detection — same field with conflicting plain-equality values
    from text2ql.engines.sql import _detect_contradictory_filters

    contradiction_notes = _detect_contradictory_filters(validated_filters)
    if contradiction_notes:
        for note in contradiction_notes:
            logger.warning("GraphQLEngine [%s]: %s", validated_entity, note)
        notes.extend(contradiction_notes)
        if engine.strict_validation:
            raise ValidationError(
                f"Contradictory filters detected for entity '{validated_entity}'",
                contradiction_notes,
            )

    validated_aggregations = engine._validate_aggregations(
        aggregations=aggregations,
        allowed_fields=allowed_fields,
        entity=validated_entity,
        notes=notes,
    )
    validated_nested = engine._validate_nested_nodes(
        nested=nested,
        entity=validated_entity,
        config=config,
        notes=notes,
    )

    return (
        validated_entity,
        validated_fields,
        validated_filters,
        validated_aggregations,
        validated_nested,
        notes,
    )
