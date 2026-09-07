"""Closed JSON schemas for provider-native structured intent output."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

SCALAR = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"}]}


def strict_intent_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = deepcopy(schema)

    def close(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties", {})
            for name in ("filters", "subquery_filters"):
                if name in properties:
                    properties[name] = {"type": "array", "items": {"$ref": "#/$defs/filter"}}
            if properties.get("value") == {}:
                properties["value"] = deepcopy(SCALAR)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(properties)
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(schema)
    schema["$defs"] = {"filter": {
        "type": "object",
        "properties": {
            "field": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "operator": {"type": "string", "enum": ["=", "!=", ">", ">=", "<", "<=", "in", "nin", "is_null", "and", "or", "not"]},
            "value": {"anyOf": SCALAR["anyOf"] + [{"type": "array", "items": deepcopy(SCALAR)}]},
            "children": {"type": "array", "items": {"$ref": "#/$defs/filter"}},
        },
        "required": ["field", "operator", "value", "children"],
        "additionalProperties": False,
    }}
    return schema
