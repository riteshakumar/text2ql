from __future__ import annotations

import json
from typing import Any

from text2ql.schema_config import NormalizedSchemaConfig

ENGLISH_GRAPHQL_SYSTEM_PROMPT = (
    "You are a GraphQL intent extractor. Return only valid JSON with keys: "
    "entity (string), fields (array of strings), filters (object), "
    "explanation (string), confidence (number in [0,1])."
)

ENGLISH_SQL_SYSTEM_PROMPT = (
    "You are a SQL intent extractor. Return only valid JSON with keys: "
    "table (string), columns (array of strings), filters (object), joins (array), "
    "order_by (string|null), order_dir (ASC|DESC|null), limit (number|null), "
    "offset (number|null), explanation (string), confidence (number in [0,1])."
)

ENGLISH_GRAPHQL_USER_TEMPLATE = """Convert this request into GraphQL intent JSON.

Request:
{text}

Available entities:
{entities}

Available fields:
{fields}

Field mapping aliases:
{field_aliases}

Filter mapping aliases:
{filter_aliases}
"""

ENGLISH_SQL_USER_TEMPLATE = """Convert this request into SQL intent JSON.

Request:
{text}

Available tables:
{entities}

Available columns:
{fields}

Field mapping aliases:
{field_aliases}

Filter mapping aliases:
{filter_aliases}
"""

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
            "description": "Fields to select on the entity.",
        },
        "filters": {
            "type": "object",
            "additionalProperties": True,
            "description": "Key-value filter arguments for the query.",
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
    "required": ["entity", "fields", "filters", "explanation", "confidence"],
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
    },
    "required": [
        "table",
        "columns",
        "filters",
        "joins",
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
    user_template = template or ENGLISH_GRAPHQL_USER_TEMPLATE
    user_prompt = user_template.format(
        text=text.strip(),
        entities=json.dumps(entities),
        fields=json.dumps(fields),
        field_aliases=json.dumps(config.field_aliases),
        filter_aliases=json.dumps(config.filter_key_aliases),
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
    user_template = template or ENGLISH_SQL_USER_TEMPLATE
    user_prompt = user_template.format(
        text=text.strip(),
        entities=json.dumps(entities),
        fields=json.dumps(fields),
        field_aliases=json.dumps(config.field_aliases),
        filter_aliases=json.dumps(config.filter_key_aliases),
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
