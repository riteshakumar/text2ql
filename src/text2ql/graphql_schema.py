"""Import standard GraphQL SDL and introspection responses."""
from __future__ import annotations

from typing import Any

from graphql import build_client_schema, build_schema, get_named_type, is_enum_type, is_object_type, is_scalar_type


def import_graphql_schema(payload: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    document = payload.get("introspection", payload)
    if isinstance(document, dict) and isinstance(document.get("data"), dict):
        document = document["data"]
    if isinstance(payload.get("sdl"), str):
        actual = build_schema(payload["sdl"])
    elif isinstance(document, dict) and "__schema" in document:
        actual = build_client_schema(document)
    else:
        return payload, None
    query_type = actual.query_type
    if query_type is None:
        raise ValueError("GraphQL schema must declare a query type")
    fields: dict[str, list[str]] = {}
    relations: dict[str, Any] = {}
    args: dict[str, list[str]] = {}
    query: dict[str, Any] = {}
    types: dict[str, Any] = {}
    for name, typename in actual.type_map.items():
        if name.startswith("__"):
            continue
        if is_enum_type(typename):
            types[name] = {"enumValues": list(typename.values)}
        elif is_object_type(typename):
            types[name] = {"fields": {key: str(value.type) for key, value in typename.fields.items()}}
    for root, field in query_type.fields.items():
        named = get_named_type(field.type)
        query[root] = {"type": str(field.type), "args": {key: str(arg.type) for key, arg in field.args.items()}}
        args[root] = list(field.args)
        if is_object_type(named):
            fields[root] = [name for name, item in named.fields.items() if is_scalar_type(get_named_type(item.type)) or is_enum_type(get_named_type(item.type))]
            relations[root] = {
                name: {"target": get_named_type(item.type).name,
                       "fields": list(get_named_type(item.type).fields), "args": list(item.args)}
                for name, item in named.fields.items() if is_object_type(get_named_type(item.type))
            }
    normalized = {**payload, "entities": list(query_type.fields), "fields": fields,
                  "args": args, "relations": relations, "introspection": {"query": query, "types": types}}
    return normalized, actual
