from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NormalizedSchemaConfig:
    entities: list[str] = field(default_factory=list)
    entity_aliases: dict[str, str] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)
    fields_by_entity: dict[str, list[str]] = field(default_factory=dict)
    field_aliases: dict[str, str] = field(default_factory=dict)
    filter_key_aliases: dict[str, str] = field(default_factory=dict)
    filter_value_aliases: dict[str, dict[str, str]] = field(default_factory=dict)
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

    default_entity = schema.get("default_entity")
    if isinstance(default_entity, str) and default_entity.strip():
        config.default_entity = default_entity.strip()

    default_fields = schema.get("default_fields")
    if isinstance(default_fields, list):
        config.default_fields = [str(v) for v in default_fields if str(v).strip()]

    _apply_mapping_payload(config, schema.get("mapping"))
    _apply_mapping_payload(config, schema.get("mappings"))
    _apply_mapping_payload(config, mapping)
    return config


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
            if isinstance(item, str):
                entities.append(item)
                continue
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    canonical = name.strip()
                    entities.append(canonical)
                    for alias in _coerce_aliases(item.get("aliases")):
                        aliases[alias.lower()] = canonical
    elif isinstance(entities_payload, dict):
        alias_map = _build_alias_map(entities_payload)
        aliases.update(alias_map)
        for canonical in alias_map.values():
            if canonical not in entities:
                entities.append(canonical)

    return entities, aliases


def _parse_fields(fields_payload: Any) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    fields: list[str] = []
    fields_by_entity: dict[str, list[str]] = {}
    aliases: dict[str, str] = {}

    if isinstance(fields_payload, list):
        for item in fields_payload:
            if isinstance(item, str):
                fields.append(item)
                continue
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    canonical = name.strip()
                    fields.append(canonical)
                    for alias in _coerce_aliases(item.get("aliases")):
                        aliases[alias.lower()] = canonical
    elif isinstance(fields_payload, dict):
        if all(isinstance(value, list) for value in fields_payload.values()):
            for entity, entity_fields in fields_payload.items():
                if not isinstance(entity, str):
                    continue
                normalized = [str(v) for v in entity_fields if str(v).strip()]
                fields_by_entity[entity] = normalized
                for field_name in normalized:
                    if field_name not in fields:
                        fields.append(field_name)
        else:
            alias_map = _build_alias_map(fields_payload)
            aliases.update(alias_map)
            for canonical in alias_map.values():
                if canonical not in fields:
                    fields.append(canonical)

    return fields, fields_by_entity, aliases


def _build_alias_map(payload: Any) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    if not isinstance(payload, dict):
        return alias_map

    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            alias = key.strip().lower()
            canonical = value.strip()
            if alias and canonical:
                alias_map[alias] = canonical
            continue
        if isinstance(value, list):
            canonical = key.strip()
            if not canonical:
                continue
            for alias in _coerce_aliases(value):
                alias_map[alias.lower()] = canonical
    return alias_map


def _coerce_aliases(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [str(v).strip() for v in payload if str(v).strip()]
    return []
