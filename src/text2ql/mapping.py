from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from text2ql.schema_config import infer_schema_from_json_payload, normalize_schema_config


def generate_hybrid_mapping(
    schema_payload: dict[str, Any] | None,
    data_payload: dict[str, Any] | None,
    overrides: dict[str, Any] | None = None,
    max_value_aliases_per_field: int = 200,
) -> dict[str, Any]:
    """Generate hybrid mapping: auto baseline + optional overrides + provenance."""
    schema_payload = schema_payload or {}
    data_payload = data_payload or {}
    overrides = overrides or {}

    schema_config = normalize_schema_config(schema_payload)
    inferred_data_schema = infer_schema_from_json_payload(data_payload)

    baseline = _build_baseline_mapping(
        entities=schema_config.entities or list((inferred_data_schema.get("entities") or [])),
        fields_by_entity=_as_fields_by_entity(
            schema_config.fields_by_entity or inferred_data_schema.get("fields")
        ),
        data_payload=data_payload,
        max_value_aliases_per_field=max_value_aliases_per_field,
    )

    merged = _merge_mapping(baseline, overrides)
    return {
        **merged,
        "mapping_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "provenance": _build_provenance(baseline, overrides),
            "overrides_applied": bool(overrides),
        },
    }


def _build_baseline_mapping(
    entities: list[str],
    fields_by_entity: dict[str, list[str]],
    data_payload: dict[str, Any],
    max_value_aliases_per_field: int,
) -> dict[str, Any]:
    entity_aliases: dict[str, str] = {}
    field_aliases: dict[str, str] = {}
    filter_aliases: dict[str, str] = {}
    filter_values: dict[str, dict[str, str]] = {}

    for entity in entities:
        if not isinstance(entity, str):
            continue
        singular = _singular(entity)
        plural = _plural(entity)
        if singular != entity:
            entity_aliases[singular.lower()] = entity
        if plural != entity:
            entity_aliases[plural.lower()] = entity
        entity_aliases[entity.lower()] = entity

    all_fields = {field for fields in fields_by_entity.values() for field in fields}
    for field in sorted(all_fields):
        variants = _field_alias_variants(field)
        for variant in variants:
            if variant != field.lower():
                field_aliases[variant] = field
        filter_aliases[field.lower()] = field

    # Common generic aliases.
    if "status" in {field.lower() for field in all_fields}:
        filter_aliases.setdefault("state", "status")
    if "symbol" in {field.lower() for field in all_fields}:
        filter_aliases.setdefault("ticker", "symbol")
        filter_aliases.setdefault("stock", "symbol")

    observed_values = _collect_string_values_by_key(data_payload, limit=max_value_aliases_per_field)
    canonical_field_lookup = {field.lower(): field for field in all_fields}
    for raw_key, values in observed_values.items():
        canonical_field = canonical_field_lookup.get(raw_key.lower())
        if not canonical_field:
            continue
        alias_map: dict[str, str] = {}
        for value in values:
            alias = value.lower()
            alias_map[alias] = value
        if alias_map:
            filter_values[canonical_field] = alias_map

    return {
        "entities": entity_aliases,
        "fields": field_aliases,
        "filters": filter_aliases,
        "filter_values": filter_values,
    }


def _merge_mapping(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "entities": dict(base.get("entities", {})),
        "fields": dict(base.get("fields", {})),
        "filters": dict(base.get("filters", {})),
        "filter_values": {
            key: dict(value) for key, value in (base.get("filter_values", {}) or {}).items()
        },
    }
    for section in ["entities", "fields", "filters"]:
        values = overrides.get(section)
        if isinstance(values, dict):
            merged[section].update(values)

    override_values = overrides.get("filter_values")
    if isinstance(override_values, dict):
        for field, alias_map in override_values.items():
            if not isinstance(alias_map, dict):
                continue
            merged["filter_values"].setdefault(field, {}).update(alias_map)
    return merged


def _build_provenance(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    provenance = {
        "entities": {key: "auto" for key in (base.get("entities") or {})},
        "fields": {key: "auto" for key in (base.get("fields") or {})},
        "filters": {key: "auto" for key in (base.get("filters") or {})},
        "filter_values": {
            field: {key: "auto" for key in alias_map}
            for field, alias_map in (base.get("filter_values") or {}).items()
        },
    }
    for section in ["entities", "fields", "filters"]:
        values = overrides.get(section)
        if not isinstance(values, dict):
            continue
        for key in values:
            provenance[section][key] = "override"

    override_values = overrides.get("filter_values")
    if isinstance(override_values, dict):
        for field, alias_map in override_values.items():
            if not isinstance(alias_map, dict):
                continue
            provenance["filter_values"].setdefault(field, {})
            for key in alias_map:
                provenance["filter_values"][field][key] = "override"
    return provenance


def _as_fields_by_entity(payload: Any) -> dict[str, list[str]]:
    if isinstance(payload, dict):
        out: dict[str, list[str]] = {}
        for entity, fields in payload.items():
            if isinstance(entity, str) and isinstance(fields, list):
                out[entity] = [str(field) for field in fields if str(field).strip()]
        return out
    if isinstance(payload, list):
        return {"items": [str(field) for field in payload if str(field).strip()]}
    return {}


def _collect_string_values_by_key(payload: dict[str, Any], limit: int) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    cleaned = value.strip()
                    if cleaned:
                        out.setdefault(str(key), set())
                        if len(out[str(key)]) < limit:
                            out[str(key)].add(cleaned)
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return out


def _field_alias_variants(field: str) -> set[str]:
    variants: set[str] = {field.lower()}
    split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", field).strip()
    split_camel = split_camel.replace("_", " ")
    tokens = [token.lower() for token in split_camel.split() if token.strip()]
    if tokens:
        variants.add(" ".join(tokens))
        variants.add("_".join(tokens))
        variants.add("".join(tokens))
        # Small generic acronym expansions.
        expanded = [
            {"acct": "account", "txn": "transaction", "qty": "quantity", "mkt": "market", "val": "value"}.get(
                token,
                token,
            )
            for token in tokens
        ]
        variants.add(" ".join(expanded))
        variants.add("_".join(expanded))
    return variants


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return f"{word[:-3]}y"
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def _plural(word: str) -> str:
    return word if word.endswith("s") else f"{word}s"
