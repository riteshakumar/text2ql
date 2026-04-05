from __future__ import annotations

from typing import Any

from text2ql.types import QueryResult


def execute_query_result_on_json(
    result: QueryResult,
    payload: dict[str, Any],
    root_key: str | None = None,
) -> tuple[list[Any], str]:
    """Apply QueryResult metadata (entity/fields/filters) to a JSON payload."""
    root: Any = payload
    if root_key and isinstance(payload, dict):
        root = payload.get(root_key, payload)

    entity = str(result.metadata.get("entity") or result.metadata.get("table") or "")
    fields = [str(field) for field in (result.metadata.get("fields") or result.metadata.get("columns") or [])]
    filters = result.metadata.get("filters", {})
    aggregations = result.metadata.get("aggregations", [])
    if not isinstance(filters, dict):
        filters = {}
    if not isinstance(aggregations, list):
        aggregations = []

    nodes = _find_entity_nodes(root, entity)
    if not nodes:
        return [], f"Entity '{entity}' not found in payload."

    limit = _coerce_limit(filters, result.metadata)
    offset = _coerce_offset(filters, result.metadata)
    rows: list[Any] = []
    for node in nodes:
        if isinstance(node, list):
            rows.extend(_project_fields(item, fields) for item in node if _matches_filters(item, filters))
            continue
        if isinstance(node, dict):
            if _matches_filters(node, filters):
                rows.append(_project_fields(node, fields))
            continue
        rows.append(node)

    if offset is not None:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return [], f"Entity '{entity}' found but filtered out by {filters}."
    if aggregations:
        return [_evaluate_aggregations(rows, aggregations)], ""
    return rows, ""


def _evaluate_aggregations(rows: list[Any], aggregations: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for agg in aggregations:
        if not isinstance(agg, dict):
            continue
        function = str(agg.get("function", "")).lower()
        field = str(agg.get("field", ""))
        if function == "count":
            output["count"] = len(rows)
            continue
        if not field:
            continue
        values = _numeric_values(rows, field)
        if not values:
            output[f"{function}_{field}"] = None
            continue
        if function == "sum":
            output[f"sum_{field}"] = sum(values)
        elif function == "avg":
            output[f"avg_{field}"] = sum(values) / len(values)
        elif function == "min":
            output[f"min_{field}"] = min(values)
        elif function == "max":
            output[f"max_{field}"] = max(values)
    return output


def _numeric_values(rows: list[Any], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get(field)
        try:
            if raw is None:
                continue
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


def _find_entity_nodes(node: Any, entity: str) -> list[Any]:
    matches: list[Any] = []
    variants = _entity_key_variants(entity)

    def walk(current: Any) -> None:
        if isinstance(current, dict):
            for key, value in current.items():
                if key in variants:
                    matches.append(value)
                walk(value)
        elif isinstance(current, list):
            for item in current:
                walk(item)

    walk(node)
    return matches


def _entity_key_variants(entity: str) -> set[str]:
    return {entity, _singular(entity), _plural(entity)}


def _singular(entity: str) -> str:
    if entity.endswith("ies") and len(entity) > 3:
        return f"{entity[:-3]}y"
    if entity.endswith("s") and len(entity) > 1:
        return entity[:-1]
    return entity


def _plural(entity: str) -> str:
    return entity if entity.endswith("s") else f"{entity}s"


def _coerce_limit(filters: dict[str, Any], metadata: dict[str, Any] | None = None) -> int | None:
    value = filters.get("limit")
    if value is None:
        value = filters.get("first")
    if value is None:
        value = (metadata or {}).get("limit")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _coerce_offset(filters: dict[str, Any], metadata: dict[str, Any] | None = None) -> int | None:
    value = filters.get("offset")
    if value is None:
        value = filters.get("after")
    if value is None:
        value = (metadata or {}).get("offset")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _project_fields(row: Any, fields: list[str]) -> Any:
    if not isinstance(row, dict) or not fields:
        return row
    return {field: _lookup_field_value(row, field) for field in fields}


def _matches_filters(row: Any, filters: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return True
    for key, expected in filters.items():
        if key in {"limit", "offset", "first", "after", "orderBy", "orderDirection", "orderDir"}:
            continue
        if key in {"and", "or", "not"} and isinstance(expected, list):
            clauses = [item for item in expected if isinstance(item, dict)]
            if not clauses:
                continue
            if key == "and" and not all(_matches_filters(row, clause) for clause in clauses):
                return False
            if key == "or" and not any(_matches_filters(row, clause) for clause in clauses):
                return False
            if key == "not" and any(_matches_filters(row, clause) for clause in clauses):
                return False
            continue
        if key.endswith("_gte"):
            field = key[:-4]
            try:
                if float(_lookup_field_value(row, field)) < float(expected):
                    return False
            except (TypeError, ValueError):
                return False
            continue
        if key.endswith("_lte"):
            field = key[:-4]
            try:
                if float(_lookup_field_value(row, field)) > float(expected):
                    return False
            except (TypeError, ValueError):
                return False
            continue
        if key.endswith("_gt"):
            field = key[:-3]
            try:
                if float(_lookup_field_value(row, field)) <= float(expected):
                    return False
            except (TypeError, ValueError):
                return False
            continue
        if key.endswith("_lt"):
            field = key[:-3]
            try:
                if float(_lookup_field_value(row, field)) >= float(expected):
                    return False
            except (TypeError, ValueError):
                return False
            continue
        if key.endswith("_ne"):
            field = key[:-3]
            if _stringify(_lookup_field_value(row, field)) == _stringify(expected):
                return False
            continue
        if key.endswith("_in") and isinstance(expected, list):
            field = key[:-3]
            if _stringify(_lookup_field_value(row, field)) not in {_stringify(item) for item in expected}:
                return False
            continue
        if key.endswith("_nin") and isinstance(expected, list):
            field = key[:-4]
            if _stringify(_lookup_field_value(row, field)) in {_stringify(item) for item in expected}:
                return False
            continue
        if _stringify(_lookup_field_value(row, key)) != _stringify(expected):
            return False
    return True


def _lookup_field_value(row: Any, key: str) -> Any:
    if not isinstance(row, dict):
        return None
    if key in row:
        return row.get(key)
    for value in row.values():
        if isinstance(value, dict):
            nested = _lookup_field_value(value, key)
            if nested is not None:
                return nested
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)
