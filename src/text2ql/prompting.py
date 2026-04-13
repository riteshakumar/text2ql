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
    "distinct (bool, true when question asks for unique values), "
    "having (array of post-aggregation conditions: [{\"function\":\"COUNT\",\"field\":\"*\",\"operator\":\">\",\"value\":5}]), "
    "subqueries (array of NOT IN/IN conditions: [{\"type\":\"not_in\",\"column\":\"id\",\"subquery_table\":\"tbl\",\"subquery_column\":\"col\"}]), "
    "order_by (string|null), order_dir (ASC|DESC|null), limit (number|null), "
    "offset (number|null), explanation (string), confidence (number in [0,1]). "
    "For aggregations use: {\"function\": \"COUNT\", \"field\": \"*\"} or "
    "{\"function\": \"SUM\", \"field\": \"amount\"}. "
    "For filters with comparisons use suffix keys: age_gt, salary_gte, price_lt, credits_lte. "
    "For joins, use the relation name exactly as it appears in the schema relations. "
    "Use HAVING for post-aggregation filters (e.g. count > 5). "
    "Use subqueries NOT IN when the question excludes rows based on another table. "
    "Set distinct=true when the question asks for unique/distinct values. "
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
        "distinct": {
            "type": "boolean",
            "description": "True when the query should use SELECT DISTINCT.",
        },
        "having": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "function": {"type": "string", "enum": ["COUNT", "SUM", "AVG", "MIN", "MAX"]},
                    "field": {"type": "string"},
                    "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "=", "!="]},
                    "value": {},
                },
                "required": ["function", "field", "operator", "value"],
                "additionalProperties": False,
            },
            "description": "Post-aggregation HAVING conditions.",
        },
        "subqueries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["not_in", "in"]},
                    "column": {"type": "string"},
                    "subquery_table": {"type": "string"},
                    "subquery_column": {"type": "string"},
                    "subquery_filters": {"type": "object", "additionalProperties": True},
                },
                "required": ["type", "column", "subquery_table", "subquery_column"],
                "additionalProperties": False,
            },
            "description": "NOT IN / IN subquery conditions for exclusion logic.",
        },
    },
    "required": [
        "table",
        "columns",
        "filters",
        "joins",
        "aggregations",
        "distinct",
        "having",
        "subqueries",
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


# ---------------------------------------------------------------------------
# Direct query generation prompts (mode="llm")
#
# Unlike the intent-extraction prompts above, these ask the LLM to write the
# full query directly.  The engine returns the raw query string without passing
# it through the deterministic compiler, so subqueries, HAVING, DISTINCT, and
# any other SQL/GraphQL construct are supported natively.
# ---------------------------------------------------------------------------

ENGLISH_SQL_DIRECT_SYSTEM_PROMPT = (
    "You are an expert SQL query writer. "
    "Given a natural language question and a database schema, write a single valid SQL SELECT query. "
    "Rules:\n"
    "- Output ONLY the SQL query, no explanation, no markdown fences.\n"
    "- Use standard SQL syntax compatible with SQLite.\n"
    "- Always quote table and column names with double-quotes to avoid reserved-word conflicts.\n"
    "- Use JOIN ... ON ... syntax for multi-table queries.\n"
    "- Use subqueries (NOT IN, EXISTS) when the question requires exclusion or correlated logic.\n"
    "- Use HAVING for post-aggregation filters.\n"
    "- Use DISTINCT when the question asks for unique values.\n"
    "- End the query with a semicolon."
)

ENGLISH_SQL_DIRECT_USER_TEMPLATE = """Write a SQL query for the following request.

Request: {text}

Database schema:
Tables: {tables}

Columns per table:
{columns}

Foreign key relations:
{relations}

SQL query:"""

ENGLISH_GRAPHQL_DIRECT_SYSTEM_PROMPT = (
    "You are an expert GraphQL query writer. "
    "Given a natural language question and a GraphQL schema, write a single valid GraphQL query. "
    "Rules:\n"
    "- Output ONLY the GraphQL query, no explanation, no markdown fences.\n"
    "- Use standard GraphQL syntax.\n"
    "- Use nested selections for related entities.\n"
    "- Use filter arguments where needed.\n"
    "- Use aliases for aggregated fields (e.g. totalCount: count)."
)

ENGLISH_GRAPHQL_DIRECT_USER_TEMPLATE = """Write a GraphQL query for the following request.

Request: {text}

Available types: {entities}

Fields per type:
{fields}

Relations:
{relations}

GraphQL query:"""


def build_sql_direct_prompts(
    text: str,
    config: NormalizedSchemaConfig,
    language: str = "english",
) -> tuple[str, str]:
    """Build prompts for direct SQL generation (mode='llm').

    The LLM is asked to write the full SQL query rather than a structured
    intent JSON.  This enables subqueries, HAVING, DISTINCT, and any other
    SQL construct that the compiler does not support.
    """
    resolve_language(language)  # validate

    tables = config.entities or []
    columns_by_table: dict[str, list[str]] = {}
    for entity in tables:
        cols = config.fields_by_entity.get(entity, []) if hasattr(config, "fields_by_entity") else []
        if not cols and hasattr(config, "args_by_entity"):
            cols = config.args_by_entity.get(entity, [])
        columns_by_table[entity] = cols

    relations_by_entity = getattr(config, "relations_by_entity", {})
    relations_text_parts: list[str] = []
    for tbl, rel_map in relations_by_entity.items():
        for rel_name, rel in rel_map.items():
            on = getattr(rel, "on", None) or f"{tbl}.? = {rel.target}.?"
            relations_text_parts.append(f"  {tbl} → {rel.target} (via {on})")
    relations_text = "\n".join(relations_text_parts) if relations_text_parts else "  (none)"

    columns_text = "\n".join(
        f"  {tbl}: {', '.join(cols)}" for tbl, cols in columns_by_table.items()
    ) or "  (none)"

    user_prompt = ENGLISH_SQL_DIRECT_USER_TEMPLATE.format(
        text=text.strip(),
        tables=", ".join(tables),
        columns=columns_text,
        relations=relations_text,
    )
    return ENGLISH_SQL_DIRECT_SYSTEM_PROMPT, user_prompt


def build_graphql_direct_prompts(
    text: str,
    config: NormalizedSchemaConfig,
    language: str = "english",
) -> tuple[str, str]:
    """Build prompts for direct GraphQL generation (mode='llm').

    The LLM writes the full GraphQL query rather than a structured intent JSON,
    enabling nested selections and complex filter expressions.
    """
    resolve_language(language)  # validate

    entities = config.entities or []
    fields_text_parts: list[str] = []
    for entity in entities:
        cols: list[str] = []
        if hasattr(config, "fields_by_entity"):
            cols = config.fields_by_entity.get(entity, [])
        if not cols and hasattr(config, "args_by_entity"):
            cols = config.args_by_entity.get(entity, [])
        fields_text_parts.append(f"  {entity}: {', '.join(cols) if cols else '(none)'}")

    relations_by_entity = getattr(config, "relations_by_entity", {})
    relations_text_parts = []
    for ent, rel_map in relations_by_entity.items():
        for rel_name, rel in rel_map.items():
            relations_text_parts.append(f"  {ent}.{rel_name} → {rel.target}")
    relations_text = "\n".join(relations_text_parts) if relations_text_parts else "  (none)"

    user_prompt = ENGLISH_GRAPHQL_DIRECT_USER_TEMPLATE.format(
        text=text.strip(),
        entities=", ".join(entities),
        fields="\n".join(fields_text_parts) or "  (none)",
        relations=relations_text,
    )
    return ENGLISH_GRAPHQL_DIRECT_SYSTEM_PROMPT, user_prompt


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
