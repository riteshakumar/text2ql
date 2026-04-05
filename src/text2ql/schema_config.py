from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NormalizedRelation:
    name: str
    target: str
    fields: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedSchemaConfig:
    entities: list[str] = field(default_factory=list)
    entity_aliases: dict[str, str] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)
    fields_by_entity: dict[str, list[str]] = field(default_factory=dict)
    field_aliases: dict[str, str] = field(default_factory=dict)
    filter_key_aliases: dict[str, str] = field(default_factory=dict)
    filter_value_aliases: dict[str, dict[str, str]] = field(default_factory=dict)
    args_by_entity: dict[str, list[str]] = field(default_factory=dict)
    relations_by_entity: dict[str, dict[str, NormalizedRelation]] = field(default_factory=dict)
    introspection_query_args: dict[str, dict[str, str]] = field(default_factory=dict)
    introspection_entity_fields: dict[str, set[str]] = field(default_factory=dict)
    introspection_relation_targets: dict[str, dict[str, str]] = field(default_factory=dict)
    default_entity: str | None = None
    default_fields: list[str] = field(default_factory=list)


def normalize_schema_config(
    schema: dict[str, Any] | None, mapping: dict[str, Any] | None = None
) -> NormalizedSchemaConfig:
    config = NormalizedSchemaConfig()
    schema = schema or {}
    mapping = mapping or {}

    entities, entity_aliases = _parse_entities(schema.get("entities"))
    config.entities = entities
    config.entity_aliases.update(entity_aliases)

    fields, fields_by_entity, field_aliases = _parse_fields(schema.get("fields"))
    config.fields = fields
    config.fields_by_entity = fields_by_entity
    config.field_aliases.update(field_aliases)
    config.args_by_entity = _parse_args(schema.get("args"))
    config.relations_by_entity = _parse_relations(schema.get("relations"))
    intro_query_args, intro_fields, intro_relation_targets = _parse_introspection(
        schema.get("introspection")
    )
    config.introspection_query_args = intro_query_args
    config.introspection_entity_fields = intro_fields
    config.introspection_relation_targets = intro_relation_targets

    default_entity = schema.get("default_entity")
    if isinstance(default_entity, str) and default_entity.strip():
        config.default_entity = default_entity.strip()

    default_fields = schema.get("default_fields")
    if isinstance(default_fields, list):
        config.default_fields = [str(v) for v in default_fields if str(v).strip()]

    _infer_from_arbitrary_payload(config, schema)

    _apply_mapping_payload(config, schema.get("mapping"))
    _apply_mapping_payload(config, schema.get("mappings"))
    _apply_mapping_payload(config, mapping)
    return config


def infer_schema_from_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Infer a text2ql-compatible schema object from arbitrary nested JSON payload."""
    normalized = normalize_schema_config(payload)
    return {
        "entities": normalized.entities,
        "fields": normalized.fields_by_entity or normalized.fields,
        "args": normalized.args_by_entity,
        "default_entity": normalized.default_entity,
        "default_fields": normalized.default_fields,
    }


def _apply_mapping_payload(config: NormalizedSchemaConfig, payload: Any) -> None:
    if not isinstance(payload, dict):
        return

    config.entity_aliases.update(_build_alias_map(payload.get("entities")))
    config.entity_aliases.update(_build_alias_map(payload.get("entity_aliases")))

    config.field_aliases.update(_build_alias_map(payload.get("fields")))
    config.field_aliases.update(_build_alias_map(payload.get("field_aliases")))

    config.filter_key_aliases.update(_build_alias_map(payload.get("filters")))
    config.filter_key_aliases.update(_build_alias_map(payload.get("filter_aliases")))

    value_aliases = payload.get("filter_values")
    if isinstance(value_aliases, dict):
        for filter_key, aliases in value_aliases.items():
            if not isinstance(filter_key, str):
                continue
            alias_map = _build_alias_map(aliases)
            if not alias_map:
                continue
            config.filter_value_aliases.setdefault(filter_key.lower(), {}).update(alias_map)


def _parse_entities(entities_payload: Any) -> tuple[list[str], dict[str, str]]:
    entities: list[str] = []
    aliases: dict[str, str] = {}

    if isinstance(entities_payload, list):
        for item in entities_payload:
            _accumulate_entity_item(item, entities, aliases)
        return entities, aliases

    if isinstance(entities_payload, dict):
        _accumulate_entity_alias_map(entities_payload, entities, aliases)

    return entities, aliases


def _parse_fields(fields_payload: Any) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    fields: list[str] = []
    fields_by_entity: dict[str, list[str]] = {}
    aliases: dict[str, str] = {}

    if isinstance(fields_payload, list):
        for item in fields_payload:
            _accumulate_field_item(item, fields, aliases)
        return fields, fields_by_entity, aliases

    if isinstance(fields_payload, dict):
        if _is_fields_by_entity_payload(fields_payload):
            _accumulate_fields_by_entity(fields_payload, fields, fields_by_entity)
            return fields, fields_by_entity, aliases
        _accumulate_field_alias_map(fields_payload, fields, aliases)

    return fields, fields_by_entity, aliases


def _parse_args(args_payload: Any) -> dict[str, list[str]]:
    args_by_entity: dict[str, list[str]] = {}
    if not isinstance(args_payload, dict):
        return args_by_entity
    for entity, args in args_payload.items():
        if not isinstance(entity, str) or not isinstance(args, list):
            continue
        normalized = [str(v).strip() for v in args if str(v).strip()]
        if normalized:
            args_by_entity[entity] = normalized
    return args_by_entity


def _parse_relations(
    relations_payload: Any,
) -> dict[str, dict[str, NormalizedRelation]]:
    relations_by_entity: dict[str, dict[str, NormalizedRelation]] = {}
    if not isinstance(relations_payload, dict):
        return relations_by_entity

    for source_entity, relation_map in relations_payload.items():
        if not isinstance(source_entity, str):
            continue
        entity_relations = _parse_entity_relations(relation_map)

        if entity_relations:
            relations_by_entity[source_entity] = entity_relations

    return relations_by_entity


def _parse_introspection(
    payload: Any,
) -> tuple[dict[str, dict[str, str]], dict[str, set[str]], dict[str, dict[str, str]]]:
    query_args: dict[str, dict[str, str]] = {}
    entity_fields: dict[str, set[str]] = {}
    relation_targets: dict[str, dict[str, str]] = {}
    if not isinstance(payload, dict):
        return query_args, entity_fields, relation_targets

    query_payload = payload.get("query")
    types_payload = payload.get("types")
    if not isinstance(query_payload, dict) or not isinstance(types_payload, dict):
        return query_args, entity_fields, relation_targets

    for entity, spec in query_payload.items():
        _parse_introspection_query_entry(
            entity=entity,
            spec=spec,
            types_payload=types_payload,
            query_args=query_args,
            entity_fields=entity_fields,
            relation_targets=relation_targets,
        )

    return query_args, entity_fields, relation_targets


def _clean_graphql_type_name(raw_type: Any) -> str | None:
    if not isinstance(raw_type, str):
        return None
    cleaned = raw_type.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("[", "").replace("]", "").replace("!", "").strip()
    return cleaned or None


def _build_alias_map(payload: Any) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    if not isinstance(payload, dict):
        return alias_map

    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            _add_string_alias(alias_map, key, value)
            continue
        if isinstance(value, list):
            _add_list_aliases(alias_map, key, value)
    return alias_map


def _accumulate_entity_item(item: Any, entities: list[str], aliases: dict[str, str]) -> None:
    if isinstance(item, str):
        entities.append(item)
        return
    if not isinstance(item, dict):
        return
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        return
    canonical = name.strip()
    entities.append(canonical)
    for alias in _coerce_aliases(item.get("aliases")):
        aliases[alias.lower()] = canonical


def _accumulate_entity_alias_map(
    payload: dict[str, Any], entities: list[str], aliases: dict[str, str]
) -> None:
    alias_map = _build_alias_map(payload)
    aliases.update(alias_map)
    for canonical in alias_map.values():
        if canonical not in entities:
            entities.append(canonical)


def _accumulate_field_item(item: Any, fields: list[str], aliases: dict[str, str]) -> None:
    if isinstance(item, str):
        fields.append(item)
        return
    if not isinstance(item, dict):
        return
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        return
    canonical = name.strip()
    fields.append(canonical)
    for alias in _coerce_aliases(item.get("aliases")):
        aliases[alias.lower()] = canonical


def _is_fields_by_entity_payload(fields_payload: dict[str, Any]) -> bool:
    return all(isinstance(value, list) for value in fields_payload.values())


def _accumulate_fields_by_entity(
    payload: dict[str, Any],
    fields: list[str],
    fields_by_entity: dict[str, list[str]],
) -> None:
    for entity, entity_fields in payload.items():
        if not isinstance(entity, str):
            continue
        normalized = [str(v) for v in entity_fields if str(v).strip()]
        fields_by_entity[entity] = normalized
        for field_name in normalized:
            if field_name not in fields:
                fields.append(field_name)


def _accumulate_field_alias_map(
    payload: dict[str, Any],
    fields: list[str],
    aliases: dict[str, str],
) -> None:
    alias_map = _build_alias_map(payload)
    aliases.update(alias_map)
    for canonical in alias_map.values():
        if canonical not in fields:
            fields.append(canonical)


def _parse_entity_relations(relation_map: Any) -> dict[str, NormalizedRelation]:
    if not isinstance(relation_map, dict):
        return {}
    entity_relations: dict[str, NormalizedRelation] = {}
    for relation_name, relation_spec in relation_map.items():
        relation = _parse_single_relation(relation_name, relation_spec)
        if relation is not None:
            entity_relations[relation_name] = relation
    return entity_relations


def _parse_single_relation(relation_name: Any, relation_spec: Any) -> NormalizedRelation | None:
    if not isinstance(relation_name, str):
        return None
    if isinstance(relation_spec, dict):
        target = relation_spec.get("target", relation_name)
        fields = relation_spec.get("fields", [])
        args = relation_spec.get("args", [])
        aliases = relation_spec.get("aliases", [])
    else:
        target = relation_name
        fields = []
        args = []
        aliases = []

    target_name = target.strip() if isinstance(target, str) and target.strip() else relation_name
    return NormalizedRelation(
        name=relation_name,
        target=target_name,
        fields=[str(v).strip() for v in fields if str(v).strip()],
        args=[str(v).strip() for v in args if str(v).strip()],
        aliases=[str(v).strip() for v in aliases if str(v).strip()],
    )


def _parse_introspection_query_entry(
    entity: Any,
    spec: Any,
    types_payload: dict[str, Any],
    query_args: dict[str, dict[str, str]],
    entity_fields: dict[str, set[str]],
    relation_targets: dict[str, dict[str, str]],
) -> None:
    if not isinstance(entity, str) or not isinstance(spec, dict):
        return
    args_payload = spec.get("args")
    if isinstance(args_payload, dict):
        query_args[entity] = {
            str(arg): str(arg_type)
            for arg, arg_type in args_payload.items()
            if str(arg).strip()
        }

    graphql_type = _clean_graphql_type_name(spec.get("type"))
    if not graphql_type:
        return
    type_spec = types_payload.get(graphql_type)
    if not isinstance(type_spec, dict):
        return
    fields_payload = type_spec.get("fields")
    if not isinstance(fields_payload, dict):
        return

    entity_fields[entity] = {str(field) for field in fields_payload if str(field).strip()}
    nested_targets = _extract_nested_targets(fields_payload, types_payload)
    if nested_targets:
        relation_targets[entity] = nested_targets


def _extract_nested_targets(
    fields_payload: dict[str, Any], types_payload: dict[str, Any]
) -> dict[str, str]:
    nested_targets: dict[str, str] = {}
    for field_name, field_type in fields_payload.items():
        cleaned = _clean_graphql_type_name(field_type)
        if cleaned and cleaned in types_payload:
            nested_targets[str(field_name)] = cleaned
    return nested_targets


def _add_string_alias(alias_map: dict[str, str], key: str, value: str) -> None:
    alias = key.strip().lower()
    canonical = value.strip()
    if alias and canonical:
        alias_map[alias] = canonical


def _add_list_aliases(alias_map: dict[str, str], key: str, value: list[Any]) -> None:
    canonical = key.strip()
    if not canonical:
        return
    for alias in _coerce_aliases(value):
        alias_map[alias.lower()] = canonical


def _coerce_aliases(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [str(v).strip() for v in payload if str(v).strip()]
    return []


def _infer_from_arbitrary_payload(config: NormalizedSchemaConfig, payload: dict[str, Any]) -> None:
    if config.entities or config.fields or config.fields_by_entity:
        return
    if not isinstance(payload, dict) or not payload:
        return

    entities_to_fields = _collect_entities_and_fields(payload)
    if not entities_to_fields:
        return

    config.entities = sorted(entities_to_fields.keys())
    config.fields_by_entity = {
        entity: sorted(fields) for entity, fields in entities_to_fields.items() if fields
    }
    if config.fields_by_entity:
        all_fields: list[str] = []
        seen: set[str] = set()
        for fields in config.fields_by_entity.values():
            for field in fields:
                if field in seen:
                    continue
                seen.add(field)
                all_fields.append(field)
        config.fields = all_fields

    if not config.default_entity and config.entities:
        config.default_entity = "accounts" if "accounts" in config.entities else config.entities[0]
    if not config.default_fields and config.default_entity:
        config.default_fields = config.fields_by_entity.get(config.default_entity, [])[:5]
    if not config.args_by_entity:
        config.args_by_entity = _build_generic_args(config.fields_by_entity)


def _collect_entities_and_fields(payload: dict[str, Any]) -> dict[str, set[str]]:
    ignored_root_keys = {
        "entities",
        "fields",
        "args",
        "relations",
        "introspection",
        "mapping",
        "mappings",
        "default_entity",
        "default_fields",
    }
    out: dict[str, set[str]] = {}

    def walk(node: Any, entity_hint: str | None) -> None:
        if isinstance(node, dict):
            if entity_hint:
                out.setdefault(entity_hint, set()).update(str(k) for k in node.keys())
            for key, value in node.items():
                if entity_hint is None and key in ignored_root_keys:
                    continue
                if isinstance(value, dict):
                    if entity_hint:
                        out.setdefault(entity_hint, set()).update(str(k) for k in value.keys())
                    walk(value, str(key))
                    continue
                if isinstance(value, list):
                    first = _first_dict(value)
                    if first is not None:
                        if entity_hint:
                            out.setdefault(entity_hint, set()).update(str(k) for k in first.keys())
                        walk(first, str(key))
                    elif entity_hint:
                        out.setdefault(entity_hint, set()).add(str(key))
                    continue
                if entity_hint:
                    out.setdefault(entity_hint, set()).add(str(key))
            return

        if isinstance(node, list):
            first = _first_dict(node)
            if first is not None:
                walk(first, entity_hint)

    walk(payload, None)
    return out


def _first_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, dict):
            return item
    return None


def _build_generic_args(fields_by_entity: dict[str, list[str]]) -> dict[str, list[str]]:
    args_by_entity: dict[str, list[str]] = {}
    for entity, fields in fields_by_entity.items():
        args = {"limit", "status"}
        for field in fields:
            args.update({field, f"{field}_gte", f"{field}_lte", f"{field}_in"})
        args_by_entity[entity] = sorted(args)
    return args_by_entity
