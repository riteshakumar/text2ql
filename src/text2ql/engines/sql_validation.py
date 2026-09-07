from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from text2ql.types import ValidationError

if TYPE_CHECKING:
    from text2ql.engines.sql import SQLEngine, _RelationJoin

logger = logging.getLogger(__name__)


def validate_components(
    engine: "SQLEngine",
    table: str,
    columns: list[str],
    filters: dict[str, Any],
    joins: list["_RelationJoin"],
    order_by: str | None,
    order_dir: str | None,
    config: Any,
    coerce_values: bool = True,
) -> tuple[str, list[str], dict[str, Any], list["_RelationJoin"], str | None, str | None, list[str]]:
    notes: list[str] = []
    table = engine._resolve_table(table, config, notes)
    allowed_columns_ordered = engine._columns_for_table(config, table)
    allowed_columns = set(allowed_columns_ordered)
    if allowed_columns:
        notes.extend(f"dropped invalid column '{column}'" for column in columns if column not in allowed_columns)
        columns = [column for column in columns if column in allowed_columns] or allowed_columns_ordered[:2]
    columns = columns or ["id"]
    allowed_filter_keys = engine._allowed_filter_keys(config, table, allowed_columns)
    filters = engine._validate_filters(filters, allowed_filter_keys, notes)
    if coerce_values:
        engine._coerce_filter_values(filters, config, table, notes, known_filter_keys=allowed_filter_keys)

    # Contradiction detection — same field assigned conflicting plain-equality values
    from text2ql.engines.sql import _detect_contradictory_filters

    contradiction_notes = _detect_contradictory_filters(filters)
    if contradiction_notes:
        for note in contradiction_notes:
            logger.warning("SQLEngine [%s]: %s", table, note)
        notes.extend(contradiction_notes)
        if engine.strict_validation:
            raise ValidationError(
                f"Contradictory filters detected for table '{table}'",
                contradiction_notes,
            )

    if allowed_columns and order_by and order_by not in allowed_columns:
        notes.append(f"dropped invalid orderBy '{order_by}' for '{table}'")
        order_by = None
        order_dir = None

    joins = engine._validate_joins(joins, config, table, notes)
    if engine.strict_validation and notes:
        raise ValidationError("SQL components failed validation", notes)
    return table, columns, filters, joins, order_by, order_dir, notes
