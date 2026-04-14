from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NormalizedRelation:
    name: str
    target: str
    on: str = ""
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
    introspection_enum_values: dict[str, set[str]] = field(default_factory=dict)
    default_entity: str | None = None
    default_fields: list[str] = field(default_factory=list)
    default_fields_by_entity: dict[str, list[str]] = field(default_factory=dict)
    #: Domain-specific keyword → entity routing rules.  Each entry is a dict
    #: with a ``"keywords"`` key (str or list[str]) that must all appear in the
    #: lowered query, plus either ``"find_entity_by_name"`` (str) or
    #: ``"find_entity_with_fields"`` (list[str]) and an optional
    #: ``"preferred_entity_names"`` (list[str]).  Replaces the old hardcoded
    #: financial-domain rules in ``GraphQLEngine._resolve_special_entity``.
    keyword_intents: list[dict] = field(default_factory=list)


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
    intro_query_args, intro_fields, intro_relation_targets, intro_enum_values = _parse_introspection(
        schema.get("introspection")
    )
    config.introspection_query_args = intro_query_args
    config.introspection_entity_fields = intro_fields
    config.introspection_relation_targets = intro_relation_targets
    config.introspection_enum_values = intro_enum_values

    default_entity = schema.get("default_entity")
    if isinstance(default_entity, str) and default_entity.strip():
        config.default_entity = default_entity.strip()

    default_fields = schema.get("default_fields")
    if isinstance(default_fields, list):
        config.default_fields = [str(v) for v in default_fields if str(v).strip()]
    default_fields_by_entity = schema.get("default_fields_by_entity")
    if isinstance(default_fields_by_entity, dict):
        for entity, fields in default_fields_by_entity.items():
            if not isinstance(entity, str) or not isinstance(fields, list):
                continue
            normalized = [str(v) for v in fields if str(v).strip()]
            if normalized:
                config.default_fields_by_entity[entity] = normalized

    keyword_intents = schema.get("keyword_intents")
    if isinstance(keyword_intents, list):
        config.keyword_intents = [ki for ki in keyword_intents if isinstance(ki, dict)]

    _infer_from_arbitrary_payload(config, schema)

    _apply_mapping_payload(config, schema.get("mapping"))
    _apply_mapping_payload(config, schema.get("mappings"))
    _apply_mapping_payload(config, mapping)
    _auto_discover_args(config, schema, mapping)
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
        "default_fields_by_entity": normalized.default_fields_by_entity,
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
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, set[str]],
    dict[str, dict[str, str]],
    dict[str, set[str]],
]:
    query_args: dict[str, dict[str, str]] = {}
    entity_fields: dict[str, set[str]] = {}
    relation_targets: dict[str, dict[str, str]] = {}
    enum_values: dict[str, set[str]] = {}
    if not isinstance(payload, dict):
        return query_args, entity_fields, relation_targets, enum_values

    query_payload = payload.get("query")
    types_payload = payload.get("types")
    if not isinstance(query_payload, dict) or not isinstance(types_payload, dict):
        return query_args, entity_fields, relation_targets, enum_values

    enum_values = _extract_enum_values(types_payload)

    for entity, spec in query_payload.items():
        _parse_introspection_query_entry(
            entity=entity,
            spec=spec,
            types_payload=types_payload,
            query_args=query_args,
            entity_fields=entity_fields,
            relation_targets=relation_targets,
        )

    return query_args, entity_fields, relation_targets, enum_values


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
    if isinstance(relation_map, list):
        entity_relations: dict[str, NormalizedRelation] = {}
        for item in relation_map:
            if not isinstance(item, dict):
                continue
            relation_name = item.get("name") or item.get("target")
            relation = _parse_single_relation(relation_name, item)
            if relation is not None:
                entity_relations[str(relation.name)] = relation
        return entity_relations
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
        on = relation_spec.get("on", "")
        fields = relation_spec.get("fields", [])
        args = relation_spec.get("args", [])
        aliases = relation_spec.get("aliases", [])
    else:
        target = relation_name
        on = ""
        fields = []
        args = []
        aliases = []

    target_name = target.strip() if isinstance(target, str) and target.strip() else relation_name
    on_clause = on.strip() if isinstance(on, str) else ""
    return NormalizedRelation(
        name=relation_name,
        target=target_name,
        on=on_clause,
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
    if _has_explicit_schema(config):
        return
    if not isinstance(payload, dict) or not payload:
        return

    entities_to_fields = _collect_entities_and_fields(payload)
    if not entities_to_fields:
        return

    _apply_inferred_entities_and_fields(config, entities_to_fields)
    _apply_inferred_defaults(config)
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
            _walk_payload_dict(
                node=node,
                entity_hint=entity_hint,
                ignored_root_keys=ignored_root_keys,
                out=out,
                walk_fn=walk,
            )
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
        args = {"limit", "offset", "first", "after", "status", "orderBy", "orderDirection", "orderDir"}
        for field in fields:
            args.update(
                {
                    field,
                    f"{field}_gte",
                    f"{field}_lte",
                    f"{field}_gt",
                    f"{field}_lt",
                    f"{field}_ne",
                    f"{field}_in",
                    f"{field}_nin",
                }
            )
        args_by_entity[entity] = sorted(args)
    return args_by_entity


def _extract_enum_values(types_payload: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for type_name, type_spec in types_payload.items():
        values = _extract_enum_values_for_type(type_name, type_spec)
        if values:
            out[type_name] = values
    return out


def _extract_enum_values_for_type(type_name: Any, type_spec: Any) -> set[str]:
    if not isinstance(type_name, str) or not isinstance(type_spec, dict):
        return set()
    values_payload = type_spec.get("enumValues", type_spec.get("values"))
    if not isinstance(values_payload, list):
        return set()
    return {name for item in values_payload if (name := _enum_item_name(item))}


def _enum_item_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        name = item.get("name")
        if isinstance(name, str):
            return name
    return ""


def _auto_discover_args(
    config: NormalizedSchemaConfig,
    schema: dict[str, Any],
    mapping: dict[str, Any],
) -> None:
    entities = _candidate_entities_for_args(config)
    if not entities:
        return

    discovered: dict[str, set[str]] = {entity: set(config.args_by_entity.get(entity, [])) for entity in entities}
    _merge_introspection_args(discovered, config)
    _merge_hint_payload_args(discovered, entities, config, schema, mapping)
    _merge_filter_alias_args(discovered, entities, config)
    _ensure_default_pagination_args(discovered, entities, config)
    _write_discovered_args(config, discovered)


def _merge_discovered_from_hint_payload(
    discovered: dict[str, set[str]],
    entities: list[str],
    config: NormalizedSchemaConfig,
    payload: Any,
) -> None:
    if isinstance(payload, list):
        _merge_list_hint_payload(discovered, entities, config, payload)
        return
    if not isinstance(payload, dict):
        return
    if all(isinstance(value, list) for value in payload.values()):
        _merge_entity_list_hint_payload(discovered, payload)
        return
    # Alias-like maps: keys are aliases and values are canonicals.
    _merge_alias_hint_payload(discovered, entities, config, payload)


def _has_explicit_schema(config: NormalizedSchemaConfig) -> bool:
    return bool(config.entities or config.fields or config.fields_by_entity)


def _apply_inferred_entities_and_fields(
    config: NormalizedSchemaConfig,
    entities_to_fields: dict[str, set[str]],
) -> None:
    config.entities = sorted(entities_to_fields.keys())
    config.fields_by_entity = {entity: sorted(fields) for entity, fields in entities_to_fields.items() if fields}
    if config.fields_by_entity:
        config.fields = _flatten_unique_fields(config.fields_by_entity)


def _flatten_unique_fields(fields_by_entity: dict[str, list[str]]) -> list[str]:
    all_fields: list[str] = []
    seen: set[str] = set()
    for fields in fields_by_entity.values():
        for field in fields:
            if field in seen:
                continue
            seen.add(field)
            all_fields.append(field)
    return all_fields


def _apply_inferred_defaults(config: NormalizedSchemaConfig) -> None:
    if not config.default_entity and config.entities:
        config.default_entity = "accounts" if "accounts" in config.entities else config.entities[0]
    if not config.default_fields and config.default_entity:
        config.default_fields = config.fields_by_entity.get(config.default_entity, [])[:5]
    if config.default_entity and config.default_fields:
        config.default_fields_by_entity.setdefault(config.default_entity, list(config.default_fields))


def _walk_payload_dict(
    node: dict[str, Any],
    entity_hint: str | None,
    ignored_root_keys: set[str],
    out: dict[str, set[str]],
    walk_fn: Any,
) -> None:
    if entity_hint:
        out.setdefault(entity_hint, set()).update(str(k) for k in node.keys())
    for key, value in node.items():
        if entity_hint is None and key in ignored_root_keys:
            continue
        _walk_payload_entry(key, value, entity_hint, out, walk_fn)


def _walk_payload_entry(
    key: str,
    value: Any,
    entity_hint: str | None,
    out: dict[str, set[str]],
    walk_fn: Any,
) -> None:
    if isinstance(value, dict):
        if entity_hint:
            out.setdefault(entity_hint, set()).update(str(k) for k in value.keys())
        walk_fn(value, str(key))
        return
    if isinstance(value, list):
        first = _first_dict(value)
        if first is not None:
            if entity_hint:
                out.setdefault(entity_hint, set()).update(str(k) for k in first.keys())
            walk_fn(first, str(key))
        elif entity_hint:
            out.setdefault(entity_hint, set()).add(str(key))
        return
    if entity_hint:
        out.setdefault(entity_hint, set()).add(str(key))


def _candidate_entities_for_args(config: NormalizedSchemaConfig) -> list[str]:
    return list(dict.fromkeys(config.entities + list(config.fields_by_entity.keys())))


def _merge_introspection_args(discovered: dict[str, set[str]], config: NormalizedSchemaConfig) -> None:
    for entity, arg_map in config.introspection_query_args.items():
        discovered.setdefault(entity, set()).update(arg_map.keys())


def _merge_hint_payload_args(
    discovered: dict[str, set[str]],
    entities: list[str],
    config: NormalizedSchemaConfig,
    schema: dict[str, Any],
    mapping: dict[str, Any],
) -> None:
    for payload in (schema, mapping):
        if not isinstance(payload, dict):
            continue
        for hint_key in ("filters", "filterable_fields", "query_args", "params", "parameters", "where"):
            _merge_discovered_from_hint_payload(discovered, entities, config, payload.get(hint_key))


def _merge_filter_alias_args(
    discovered: dict[str, set[str]],
    entities: list[str],
    config: NormalizedSchemaConfig,
) -> None:
    for canonical in config.filter_key_aliases.values():
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        canonical_name = canonical.strip()
        for entity in entities:
            entity_fields = _candidate_fields_for_entity(config, entity)
            if canonical_name in entity_fields:
                discovered.setdefault(entity, set()).add(canonical_name)


def _ensure_default_pagination_args(
    discovered: dict[str, set[str]],
    entities: list[str],
    config: NormalizedSchemaConfig,
) -> None:
    pagination_args = {"limit", "offset", "first", "after", "orderBy", "orderDirection", "orderDir"}
    for entity in entities:
        if not discovered.get(entity):
            discovered[entity] = set(_candidate_fields_for_entity(config, entity))
        discovered[entity].update(pagination_args)


def _write_discovered_args(
    config: NormalizedSchemaConfig,
    discovered: dict[str, set[str]],
) -> None:
    for entity, args in discovered.items():
        normalized = sorted(arg for arg in args if isinstance(arg, str) and arg.strip())
        if normalized:
            config.args_by_entity[entity] = normalized


def _allowed_fields_for_entity(config: NormalizedSchemaConfig, entity: str) -> set[str]:
    return set(_candidate_fields_for_entity(config, entity)) | set(
        config.introspection_query_args.get(entity, {}).keys()
    )


def _merge_list_hint_payload(
    discovered: dict[str, set[str]],
    entities: list[str],
    config: NormalizedSchemaConfig,
    payload: list[Any],
) -> None:
    normalized = {str(item).strip() for item in payload if str(item).strip()}
    if not normalized:
        return
    for entity in entities:
        allowed_for_entity = _allowed_fields_for_entity(config, entity)
        scoped = {item for item in normalized if item in allowed_for_entity}
        if scoped:
            discovered.setdefault(entity, set()).update(scoped)


def _merge_entity_list_hint_payload(
    discovered: dict[str, set[str]],
    payload: dict[str, Any],
) -> None:
    for entity, values in payload.items():
        if not isinstance(entity, str):
            continue
        normalized = {str(item).strip() for item in values if str(item).strip()}
        if normalized:
            discovered.setdefault(entity, set()).update(normalized)


def _merge_alias_hint_payload(
    discovered: dict[str, set[str]],
    entities: list[str],
    config: NormalizedSchemaConfig,
    payload: dict[str, Any],
) -> None:
    for canonical in payload.values():
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        canonical_name = canonical.strip()
        for entity in entities:
            if canonical_name in _allowed_fields_for_entity(config, entity):
                discovered.setdefault(entity, set()).add(canonical_name)


def _candidate_fields_for_entity(config: NormalizedSchemaConfig, entity: str) -> list[str]:
    entity_fields = config.fields_by_entity.get(entity, [])
    if entity_fields:
        return [field for field in entity_fields if isinstance(field, str) and field.strip()]
    if config.fields_by_entity:
        # When we have per-entity fields, avoid falling back to global fields.
        # That fallback makes "empty" entities look filterable/selectable in SQL.
        return []
    return [field for field in config.fields if isinstance(field, str) and field.strip()]
