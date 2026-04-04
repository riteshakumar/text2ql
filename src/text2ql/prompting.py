from __future__ import annotations

import json
from typing import Any

from text2ql.schema_config import NormalizedSchemaConfig

DEFAULT_GRAPHQL_SYSTEM_PROMPT = (
    "You are a GraphQL intent extractor. Return only valid JSON with keys: "
    "entity (string), fields (array of strings), filters (object), "
    "explanation (string), confidence (number in [0,1])."
)

DEFAULT_GRAPHQL_USER_TEMPLATE = """Convert this request into GraphQL intent JSON.

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


def build_graphql_prompts(
    text: str, config: NormalizedSchemaConfig, template: str | None = None
) -> tuple[str, str]:
    entities = config.entities or ["user", "customer", "order", "product", "items"]
    fields = config.fields or ["id", "name", "title", "email", "status", "price"]
    user_template = template or DEFAULT_GRAPHQL_USER_TEMPLATE
    user_prompt = user_template.format(
        text=text.strip(),
        entities=json.dumps(entities),
        fields=json.dumps(fields),
        field_aliases=json.dumps(config.field_aliases),
        filter_aliases=json.dumps(config.filter_key_aliases),
    )
    return DEFAULT_GRAPHQL_SYSTEM_PROMPT, user_prompt


def resolve_prompt_template(context: dict[str, Any]) -> str | None:
    template = context.get("prompt_template")
    if isinstance(template, str) and template.strip():
        return template
    return None
