from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text2ql.engines.sql import SQLEngine


def detect_filters(engine: "SQLEngine", lowered: str, config: Any, table: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    where_clause = engine._extract_where_clause(lowered) or lowered
    engine._apply_alias_filters(filters, where_clause, lowered, config, table)
    engine._apply_advanced_filters(filters, lowered)
    engine._apply_schema_inferred_filters(filters, lowered, config, table)
    grouped = engine._parse_grouped_filters(lowered)
    if grouped:
        filters.update(grouped)
    return filters
