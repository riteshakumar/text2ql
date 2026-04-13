from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text2ql.engines.sql import SQLEngine


def detect_table(engine: "SQLEngine", lowered: str, config: Any) -> str:
    for alias, canonical in engine._sorted_alias_pairs(config.entity_aliases):
        if engine._contains_token(lowered, alias):
            return canonical
    for entity in config.entities:
        if engine._contains_entity_token(lowered, entity.lower()):
            return entity
    if config.default_entity:
        return config.default_entity
    return config.entities[0] if config.entities else engine._extract_entity_from_text(lowered)


def detect_columns(engine: "SQLEngine", lowered: str, config: Any, table: str) -> list[str]:
    allowed = engine._columns_for_table(config, table)
    selected: list[str] = []
    for column in allowed:
        if engine._contains_column_reference(lowered, column):
            selected.append(column)
    for alias, canonical in engine._sorted_alias_pairs(config.field_aliases):
        if canonical in allowed and engine._contains_column_reference(lowered, alias):
            selected.append(canonical)
    selected = engine._unique_in_order(selected)
    if selected:
        return selected
    if config.default_fields:
        defaults = [field for field in config.default_fields if field in allowed]
        if defaults:
            return defaults
    return allowed[:3] if allowed else ["id"]


def detect_order(engine: "SQLEngine", lowered: str, selected_columns: list[str]) -> tuple[str | None, str | None]:
    if "latest order" in lowered:
        return None, None
    if any(token in lowered for token in ("latest", "newest", "most recent")):
        return engine._detect_order_field(lowered, selected_columns), "DESC"
    highest = re.search(r"\bhighest\s+([A-Za-z_]\w*)\b", lowered)
    if highest:
        return highest.group(1), "DESC"
    lowest = re.search(r"\blowest\s+([A-Za-z_]\w*)\b", lowered)
    if lowest:
        return lowest.group(1), "ASC"
    return None, None
