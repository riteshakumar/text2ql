from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from text2ql.schema_config import NormalizedSchemaConfig
from text2ql.schema_selection import select_prompt_schema

ENGLISH_GRAPHQL_SYSTEM_PROMPT = (
    "You are a GraphQL intent extractor. Return only valid JSON with keys: "
    "entity (string), fields (array of strings), filters (array of predicate objects), "
    "aggregations (array of objects with function and field), "
    "nested (array of objects with relation and fields), "
    "distinct (bool, true when query asks for unique values), "
    "having (array of post-aggregation conditions: [{\"function\":\"COUNT\",\"field\":\"*\",\"operator\":\">\",\"value\":5}]), "
    "explanation (string), confidence (number in [0,1]). "
    "For aggregations use: {\"function\": \"COUNT\", \"field\": \"*\"} or "
    "{\"function\": \"SUM\", \"field\": \"amount\"}. "
    "Preserve filter value types and boolean grouping. "
    "For nested relations use the exact relation name from the schema. "
    "Set distinct=true when the question asks for unique/distinct values. "
    "Use having for post-aggregation conditions like 'more than 5 orders'. "
    'Example filter: [{"field":"age","operator":">","value":20,"children":[]}].'
)

ENGLISH_SQL_SYSTEM_PROMPT = (
    "You are a SQL intent extractor. Return only valid JSON with keys: "
    "table (string), columns (array of strings), filters (array of predicate objects), joins (array), "
    "aggregations (array of objects with function and field), "
    "distinct (bool, true when question asks for unique values), "
    "having (array of post-aggregation conditions: [{\"function\":\"COUNT\",\"field\":\"*\",\"operator\":\">\",\"value\":5}]), "
    "subqueries (array of NOT IN/IN conditions: [{\"type\":\"not_in\",\"column\":\"id\",\"subquery_table\":\"tbl\",\"subquery_column\":\"col\"}]), "
    "order_by (string|null), order_dir (ASC|DESC|null), limit (number|null), "
    "offset (number|null), explanation (string), confidence (number in [0,1]). "
    "For aggregations use: {\"function\": \"COUNT\", \"field\": \"*\"} or "
    "{\"function\": \"SUM\", \"field\": \"amount\"}. "
    "Preserve filter value types and boolean grouping. "
    "For joins, use the relation name exactly as it appears in the schema relations. "
    "Use HAVING for post-aggregation filters (e.g. count > 5). "
    "Use subqueries NOT IN when the question excludes rows based on another table. "
    "Set distinct=true when the question asks for unique/distinct values. "
    'Example filter: [{"field":"age","operator":">","value":20,"children":[]}].'
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
- Use filter objects with field, operator, value, children; use groups for and/or/not
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
- Use filter objects with field, operator, value, children; use groups for and/or/not
- For aggregations like COUNT, SUM, AVG, MIN, MAX — add them to the "aggregations" array
- For JOINs use the exact relation name from "Available relations"
- columns should only list non-aggregated SELECT columns
"""

# Select complete relevant entities and relationship paths within this budget.
_MAX_PROMPT_ENTITIES = 50

ENGLISH_GRAPHQL_SYSTEM_PROMPT += ' Filters use an array of predicate objects with field, operator, value, children. For scalar predicates children=[]; for and/or/not groups field=null, value=null and children contains predicates. Use operator is_null with value=null for null checks.'
ENGLISH_SQL_SYSTEM_PROMPT += ' Filters use an array of predicate objects with field, operator, value, children. For scalar predicates children=[]; for and/or/not groups field=null, value=null and children contains predicates. Use operator is_null with value=null for null checks.'

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
        "distinct": {
            "type": "boolean",
            "description": "True when the query should return unique/distinct values.",
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
            "description": "Post-aggregation filter conditions.",
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
    "required": ["entity", "fields", "filters", "aggregations", "nested", "distinct", "having", "explanation", "confidence"],
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


from text2ql.structured_schema import strict_intent_schema

GRAPHQL_INTENT_JSON_SCHEMA = strict_intent_schema(GRAPHQL_INTENT_JSON_SCHEMA)
SQL_INTENT_JSON_SCHEMA = strict_intent_schema(SQL_INTENT_JSON_SCHEMA)


def build_graphql_prompts(
    text: str,
    config: NormalizedSchemaConfig,
    template: str | None = None,
    language: str = "english",
) -> tuple[str, str]:
    resolved_language = resolve_language(language)
    config = select_prompt_schema(text, config, _MAX_PROMPT_ENTITIES)
    entities = config.entities or ["user", "customer", "order", "product", "items"]
    fields = config.fields or ["id", "name", "title", "email", "status", "price"]
    field_aliases = config.field_aliases
    filter_aliases = config.filter_key_aliases
    # Include relation targets and join keys with their names.
    relations_by_entity = getattr(config, "relations_by_entity", {})
    relations: dict[str, dict[str, dict[str, str]]] = {
        ent: {name: {"target": rel.target, "on": rel.on} for name, rel in rel_map.items()}
        for ent, rel_map in relations_by_entity.items()
        if rel_map
    }

    user_template = template or ENGLISH_GRAPHQL_USER_TEMPLATE
    user_prompt = user_template.format(
        text=text.strip(),
        entities=json.dumps(entities),
        fields=json.dumps(config.fields_by_entity or fields),
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
    config = select_prompt_schema(text, config, _MAX_PROMPT_ENTITIES)
    entities = config.entities or ["users", "customers", "orders", "products", "items"]
    fields = config.fields or ["id", "name", "createdAt", "status", "price", "amount"]
    field_aliases = config.field_aliases
    filter_aliases = config.filter_key_aliases
    # Include relation targets and join keys with their names.
    relations_by_entity = getattr(config, "relations_by_entity", {})
    relations: dict[str, dict[str, dict[str, str]]] = {
        tbl: {name: {"target": rel.target, "on": rel.on} for name, rel in rel_map.items()}
        for tbl, rel_map in relations_by_entity.items()
        if rel_map
    }

    user_template = template or ENGLISH_SQL_USER_TEMPLATE
    user_prompt = user_template.format(
        text=text.strip(),
        entities=json.dumps(entities),
        fields=json.dumps(config.fields_by_entity or fields),
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
    "Given a natural language question and a database schema, write a single valid SQL SELECT query.\n"
    "Rules:\n"
    "- Output ONLY the SQL query — no explanation, no markdown fences, no comments.\n"
    "- Use standard SQL syntax compatible with SQLite.\n"
    "- Use identifier quoting appropriate for the SQL dialect when names require it; "
    "prefer bare identifiers (e.g. SELECT id FROM orders) for simple names.\n"
    "- SELECT ONLY the columns explicitly requested.  Do NOT add extra columns or AS aliases "
    "unless the question asks for a renamed output.\n"
    "- Use the minimum number of tables and JOINs needed to answer the question.  "
    "If the answer can be derived from a single table, do NOT JOIN other tables.\n"
    "- Prefer direct literal value filters (WHERE charter = 1) over subqueries when the value is "
    "known from the question or the domain hints below.\n"
    "- Use subqueries (NOT IN, EXISTS) only when the question genuinely requires exclusion or "
    "correlated logic that cannot be expressed as a simple literal filter.\n"
    "- Do NOT add DISTINCT unless the question explicitly asks for unique/distinct values.\n"
    "- Use HAVING for post-aggregation filters (e.g. HAVING COUNT(*) > 2).\n"
    "- For 'best', 'highest', 'most', 'largest' use MAX/DESC; "
    "for 'worst', 'lowest', 'least', 'smallest' use MIN/ASC.\n"
    "- Use exact column values as they are stored in the database.  "
    "Domain hints (if provided) tell you the exact stored representation — follow them strictly.\n"
    "- Do not end the query with a semicolon."
)

ENGLISH_SQL_DIRECT_USER_TEMPLATE = """Write a SQL query for the following request.

Request: {text}

Database schema:
Tables: {tables}

Columns per table:
{columns}

Foreign key relations:
{relations}
{evidence_block}
SQL query:"""

ENGLISH_GRAPHQL_DIRECT_SYSTEM_PROMPT = (
    "You are an expert GraphQL query writer. "
    "Given a natural language question and a GraphQL schema, write a single valid GraphQL query.\n"
    "Rules:\n"
    "- Output ONLY the GraphQL query — no explanation, no markdown fences, no comments.\n"
    "- Use standard GraphQL syntax.\n"
    "- SELECT ONLY the fields explicitly requested.  Do NOT add extra fields or field aliases "
    "unless the question asks for a renamed output.\n"
    "- Use the minimum number of types and nested selections needed to answer the question.  "
    "If the answer can be derived from a single type, do NOT nest into related types.\n"
    "- Use nested selections only when the question genuinely requires data from a related type.\n"
    "- Use filter arguments with exact stored values.  "
    "Domain hints (if provided) tell you the exact stored representation — follow them strictly.\n"
    "- For 'best', 'highest', 'most', 'largest' use orderBy DESC; "
    "for 'worst', 'lowest', 'least', 'smallest' use orderBy ASC.\n"
    "- Use aliases for aggregated fields only when a rename is requested "
    "(e.g. totalCount: count).\n"
    "- Do NOT add __typename or metadata fields unless requested."
)

ENGLISH_GRAPHQL_DIRECT_USER_TEMPLATE = """Write a GraphQL query for the following request.

Request: {text}

Available types: {entities}

Fields per type:
{fields}

Relations:
{relations}
{evidence_block}
GraphQL query:"""

NONE_LABEL = "  (none)"


def build_sql_direct_prompts(
    text: str,
    config: NormalizedSchemaConfig,
    language: str = "english",
    evidence: str | None = None,
    dialect: str = "sqlite",
) -> tuple[str, str]:
    """Build prompts for direct SQL generation (mode='llm').

    The LLM is asked to write the full SQL query rather than a structured
    intent JSON.  This enables subqueries, HAVING, DISTINCT, and any other
    SQL construct that the compiler does not support.

    Parameters
    ----------
    evidence:
        Optional domain hint string injected into the user prompt.  Used by
        BIRD-style benchmarks to convey exact stored values (e.g.
        "carcinogenic means label = '+'").
    """
    resolve_language(language)  # validate
    config = select_prompt_schema(text, config, _MAX_PROMPT_ENTITIES)

    tables = config.entities or []
    columns_by_table: dict[str, list[str]] = {}
    for entity in tables:
        cols = config.fields_by_entity.get(entity, config.fields)
        if not cols and hasattr(config, "args_by_entity"):
            cols = config.args_by_entity.get(entity, [])
        columns_by_table[entity] = cols

    relations_by_entity = getattr(config, "relations_by_entity", {})
    relations_text_parts: list[str] = []
    for tbl, rel_map in relations_by_entity.items():
        for rel_name, rel in rel_map.items():
            on = getattr(rel, "on", None) or f"{tbl}.? = {rel.target}.?"
            relations_text_parts.append(f"  {tbl} → {rel.target} (via {on})")
    relations_text = "\n".join(relations_text_parts) if relations_text_parts else NONE_LABEL

    columns_text = "\n".join(
        f"  {tbl}: {', '.join(cols)}" for tbl, cols in columns_by_table.items()
    ) or NONE_LABEL

    evidence_block = ""
    if evidence and evidence.strip():
        evidence_block = f"\nDomain hints (use these for exact column values):\n{evidence.strip()}\n"

    user_prompt = ENGLISH_SQL_DIRECT_USER_TEMPLATE.format(
        text=text.strip(),
        tables=", ".join(tables),
        columns=columns_text,
        relations=relations_text,
        evidence_block=evidence_block,
    )
    from text2ql.query_validation import sql_dialect
    dialect = sql_dialect(dialect)
    system_prompt = ENGLISH_SQL_DIRECT_SYSTEM_PROMPT.replace("compatible with SQLite", f"for the {dialect} dialect")
    return system_prompt, user_prompt


def build_graphql_direct_prompts(
    text: str,
    config: NormalizedSchemaConfig,
    language: str = "english",
    evidence: str | None = None,
) -> tuple[str, str]:
    """Build prompts for direct GraphQL generation (mode='llm').

    The LLM writes the full GraphQL query rather than a structured intent JSON,
    enabling nested selections and complex filter expressions.

    Parameters
    ----------
    evidence:
        Optional domain hint string injected into the user prompt.
    """
    resolve_language(language)  # validate
    config = select_prompt_schema(text, config, _MAX_PROMPT_ENTITIES)

    entities = config.entities or []
    fields_text_parts = [_graphql_entity_fields_line(config, entity) for entity in entities]
    relations_text = _graphql_relations_text(config)

    evidence_block = ""
    if evidence and evidence.strip():
        evidence_block = f"\nDomain hints (use these for exact field values):\n{evidence.strip()}\n"

    user_prompt = ENGLISH_GRAPHQL_DIRECT_USER_TEMPLATE.format(
        text=text.strip(),
        entities=", ".join(entities),
        fields="\n".join(fields_text_parts) or NONE_LABEL,
        relations=relations_text,
        evidence_block=evidence_block,
    )
    user_prompt += "\nArguments per root field:\n" + json.dumps(config.introspection_query_args or config.args_by_entity)
    return ENGLISH_GRAPHQL_DIRECT_SYSTEM_PROMPT, user_prompt


def _graphql_entity_fields_line(config: NormalizedSchemaConfig, entity: str) -> str:
    cols: list[str] = []
    if hasattr(config, "fields_by_entity"):
        cols = config.fields_by_entity.get(entity, config.fields)
    if not cols and hasattr(config, "args_by_entity"):
        cols = config.args_by_entity.get(entity, [])
    return f"  {entity}: {', '.join(cols) if cols else '(none)'}"


def _graphql_relations_text(config: NormalizedSchemaConfig) -> str:
    relations_by_entity = getattr(config, "relations_by_entity", {})
    relations_text_parts: list[str] = []
    for entity, rel_map in relations_by_entity.items():
        for rel_name, relation in rel_map.items():
            relations_text_parts.append(f"  {entity}.{rel_name} → {relation.target}")
    return "\n".join(relations_text_parts) if relations_text_parts else NONE_LABEL


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
