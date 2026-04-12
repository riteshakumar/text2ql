from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from text2ql.schema_config import NormalizedSchemaConfig

ENGLISH_GRAPHQL_SYSTEM_PROMPT = (
    "You are a GraphQL intent extractor. Return only valid JSON with keys: "
    "entity (string), fields (array of strings), filters (object), "
    "aggregations (array of objects with function and field), "
    "nested (array of objects with relation and fields), "
    "explanation (string), confidence (number in [0,1]). "
    "For aggregations use: {\"function\": \"COUNT\", \"field\": \"*\"} or "
    "{\"function\": \"SUM\", \"field\": \"amount\"}. "
    "For filters with comparisons use suffix keys: age_gt (>), age_gte (>=), price_lt (<), price_lte (<=), field_ne (!=). "
    "For nested relations use the exact relation name from the schema. "
    "Example filter: {\"age_gt\": 20, \"status\": \"active\"}."
)

ENGLISH_SQL_SYSTEM_PROMPT = (
    "You are a SQL intent extractor. Return only valid JSON with keys: "
    "table (string), columns (array of strings), filters (object), joins (array), "
    "aggregations (array of objects with function and field), "
    "order_by (string|null), order_dir (ASC|DESC|null), limit (number|null), "
    "offset (number|null), explanation (string), confidence (number in [0,1]). "
    "For aggregations use: {\"function\": \"COUNT\", \"field\": \"*\"} or "
    "{\"function\": \"SUM\", \"field\": \"amount\"}. "
    "For filters with comparisons use suffix keys: age_gt, salary_gte, price_lt, credits_lte. "
    "For joins, use the relation name exactly as it appears in the schema relations. "
    "Example filter: {\"age_gt\": 20, \"status\": \"active\"}."
)

ENGLISH_GRAPHQL_USER_TEMPLATE = """Convert this request into GraphQL intent JSON.

Request:
{text}

Available entities:
{entities}

Available fields:
{fields}

Available relations (for nested, use these exact relation names):
{relations}

Field mapping aliases:
{field_aliases}

Filter mapping aliases:
{filter_aliases}

Rules:
- For filter comparisons use suffix keys: age_gt (>), age_gte (>=), price_lt (<), price_lte (<=), field_ne (!=)
- For aggregations like COUNT, SUM, AVG, MIN, MAX — add them to the "aggregations" array
- For nested relation fetches use the exact relation name from "Available relations"
- fields should only list non-aggregated scalar fields
"""

ENGLISH_SQL_USER_TEMPLATE = """Convert this request into SQL intent JSON.

Request:
{text}

Available tables:
{entities}

Available columns:
{fields}

Available relations (for joins, use these exact relation names):
{relations}

Field mapping aliases:
{field_aliases}

Filter mapping aliases:
{filter_aliases}

Rules:
- For WHERE comparisons use suffix keys in filters: age_gt (>), age_gte (>=), age_lt (<), age_lte (<=), field_ne (!=)
- For aggregations like COUNT, SUM, AVG, MIN, MAX — add them to the "aggregations" array
- For JOINs use the exact relation name from "Available relations"
- columns should only list non-aggregated SELECT columns
"""

# Maximum number of entities/fields/aliases to include per prompt.  Large
# schemas would otherwise exceed the model's context window and cause silent
# truncation or an API error that falls through to deterministic mode.
_MAX_PROMPT_ENTITIES = 50
_MAX_PROMPT_FIELDS = 100
_MAX_PROMPT_ALIASES = 100

SUPPORTED_PROMPT_LANGUAGES = {"english"}
_LANGUAGE_ALIASES = {
    "english": "english",
    "en": "english",
}

# ---------------------------------------------------------------------------
# JSON Schemas for function-calling / structured-output mode
#
# These describe the exact shape that the LLM must emit when the provider
# supports ``response_format: json_schema`` (OpenAI Structured Outputs) or an
# equivalent function-calling mechanism.  They mirror the fields parsed by
# ``parse_graphql_intent()`` / ``parse_sql_intent()`` in constrained.py.
# ---------------------------------------------------------------------------

GRAPHQL_INTENT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "entity": {
            "type": "string",
            "description": "The primary GraphQL entity/type to query.",
        },
        "fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Non-aggregated fields to select on the entity.",
        },
        "filters": {
            "type": "object",
            "additionalProperties": True,
            "description": "Key-value filter arguments. Use suffix keys for comparisons: age_gt, price_lte, field_ne.",
        },
        "aggregations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "function": {"type": "string", "enum": ["COUNT", "SUM", "AVG", "MIN", "MAX"]},
                    "field": {"type": "string"},
                    "alias": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["function", "field"],
                "additionalProperties": False,
            },
            "description": "Aggregation expressions like COUNT(*), SUM(amount).",
        },
        "nested": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relation": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "string"}},
                    "filters": {"type": "object", "additionalProperties": True},
                },
                "required": ["relation"],
                "additionalProperties": True,
            },
            "description": "Nested relation fetches. Use exact relation names from the schema.",
        },
        "explanation": {
            "type": "string",
            "description": "Human-readable explanation of the generated intent.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score in [0, 1].",
        },
    },
    "required": ["entity", "fields", "filters", "aggregations", "nested", "explanation", "confidence"],
    "additionalProperties": False,
}

SQL_INTENT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "table": {
            "type": "string",
            "description": "The primary SQL table to query.",
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to SELECT.",
        },
        "filters": {
            "type": "object",
            "additionalProperties": True,
            "description": "Key-value WHERE clause filters.",
        },
        "joins": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relation": {"type": "string"},
                    "alias": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "string"}},
                    "filters": {"type": "object", "additionalProperties": True},
                },
                "required": ["relation"],
                "additionalProperties": True,
            },
            "description": "JOIN descriptors.",
        },
        "order_by": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Column to ORDER BY, or null.",
        },
        "order_dir": {
            "anyOf": [{"type": "string", "enum": ["ASC", "DESC"]}, {"type": "null"}],
            "description": "Sort direction.",
        },
        "limit": {
            "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
            "description": "LIMIT value, or null.",
        },
        "offset": {
            "anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}],
            "description": "OFFSET value, or null.",
        },
        "explanation": {
            "type": "string",
            "description": "Human-readable explanation.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score in [0, 1].",
        },
        "aggregations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "function": {"type": "string", "enum": ["COUNT", "SUM", "AVG", "MIN", "MAX"]},
                    "field": {"type": "string"},
                    "alias": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["function", "field"],
                "additionalProperties": False,
            },
            "description": "Aggregation expressions like COUNT(*), SUM(amount).",
        },
    },
    "required": [
        "table",
        "columns",
        "filters",
        "joins",
        "aggregations",
        "order_by",
        "order_dir",
        "limit",
        "offset",
        "explanation",
        "confidence",
    ],
    "additionalProperties": False,
}


def build_graphql_prompts(
    text: str,
    config: NormalizedSchemaConfig,
    template: str | None = None,
    language: str = "english",
) -> tuple[str, str]:
    resolved_language = resolve_language(language)
    entities = config.entities or ["user", "customer", "order", "product", "items"]
    fields = config.fields or ["id", "name", "title", "email", "status", "price"]
    field_aliases = config.field_aliases
    filter_aliases = config.filter_key_aliases
    if len(entities) > _MAX_PROMPT_ENTITIES:
        logger.warning(
            "build_graphql_prompts: truncating entities from %d to %d",
            len(entities), _MAX_PROMPT_ENTITIES,
        )
        entities = entities[:_MAX_PROMPT_ENTITIES]
    if len(fields) > _MAX_PROMPT_FIELDS:
        logger.warning(
            "build_graphql_prompts: truncating fields from %d to %d",
            len(fields), _MAX_PROMPT_FIELDS,
        )
        fields = fields[:_MAX_PROMPT_FIELDS]
    if len(field_aliases) > _MAX_PROMPT_ALIASES:
        field_aliases = dict(list(field_aliases.items())[:_MAX_PROMPT_ALIASES])
    if len(filter_aliases) > _MAX_PROMPT_ALIASES:
        filter_aliases = dict(list(filter_aliases.items())[:_MAX_PROMPT_ALIASES])

    # Build relations dict: {entity: [relation_name, ...]}
    relations_by_entity = getattr(config, "relations_by_entity", {})
    relations: dict[str, list[str]] = {
        ent: list(rel_map.keys())
        for ent, rel_map in relations_by_entity.items()
        if rel_map
    }

    user_template = template or ENGLISH_GRAPHQL_USER_TEMPLATE
    user_prompt = user_template.format(
        text=text.strip(),
        entities=json.dumps(entities),
        fields=json.dumps(fields),
        relations=json.dumps(relations),
        field_aliases=json.dumps(field_aliases),
        filter_aliases=json.dumps(filter_aliases),
    )
    if resolved_language != "english":
        # Future extension point once additional languages are introduced.
        raise ValueError(f"Unsupported prompt language '{language}'")
    return ENGLISH_GRAPHQL_SYSTEM_PROMPT, user_prompt


def build_sql_prompts(
    text: str,
    config: NormalizedSchemaConfig,
    template: str | None = None,
    language: str = "english",
) -> tuple[str, str]:
    resolved_language = resolve_language(language)
    entities = config.entities or ["users", "customers", "orders", "products", "items"]
    fields = config.fields or ["id", "name", "createdAt", "status", "price", "amount"]
    field_aliases = config.field_aliases
    filter_aliases = config.filter_key_aliases
    if len(entities) > _MAX_PROMPT_ENTITIES:
        logger.warning(
            "build_sql_prompts: truncating entities from %d to %d",
            len(entities), _MAX_PROMPT_ENTITIES,
        )
        entities = entities[:_MAX_PROMPT_ENTITIES]
    if len(fields) > _MAX_PROMPT_FIELDS:
        logger.warning(
            "build_sql_prompts: truncating fields from %d to %d",
            len(fields), _MAX_PROMPT_FIELDS,
        )
        fields = fields[:_MAX_PROMPT_FIELDS]
    if len(field_aliases) > _MAX_PROMPT_ALIASES:
        field_aliases = dict(list(field_aliases.items())[:_MAX_PROMPT_ALIASES])
    if len(filter_aliases) > _MAX_PROMPT_ALIASES:
        filter_aliases = dict(list(filter_aliases.items())[:_MAX_PROMPT_ALIASES])

    # Build relations dict: {table: [relation_name, ...]}
    relations_by_entity = getattr(config, "relations_by_entity", {})
    relations: dict[str, list[str]] = {
        tbl: list(rel_map.keys())
        for tbl, rel_map in relations_by_entity.items()
        if rel_map
    }

    user_template = template or ENGLISH_SQL_USER_TEMPLATE
    user_prompt = user_template.format(
        text=text.strip(),
        entities=json.dumps(entities),
        fields=json.dumps(fields),
        relations=json.dumps(relations),
        field_aliases=json.dumps(field_aliases),
        filter_aliases=json.dumps(filter_aliases),
    )
    if resolved_language != "english":
        raise ValueError(f"Unsupported prompt language '{language}'")
    return ENGLISH_SQL_SYSTEM_PROMPT, user_prompt


def resolve_prompt_template(context: dict[str, Any]) -> str | None:
    template = context.get("prompt_template")
    if isinstance(template, str) and template.strip():
        return template
    return None


def resolve_language(language: str | None) -> str:
    if language is None:
        return "english"
    normalized = str(language).strip().lower()
    resolved = _LANGUAGE_ALIASES.get(normalized)
    if not resolved:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {', '.join(sorted(SUPPORTED_PROMPT_LANGUAGES))}"
        )
    return resolved
