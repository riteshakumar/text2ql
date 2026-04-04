from __future__ import annotations

import re
from textwrap import dedent

from text2ql.types import QueryRequest, QueryResult

from .base import QueryEngine


class GraphQLEngine(QueryEngine):
    """GraphQL-first engine with simple intent extraction.

    Design intent:
    - Keep the engine deterministic for testability.
    - Let LLM providers be optional, pluggable upgrades.
    """

    def generate(self, request: QueryRequest) -> QueryResult:
        prompt = request.text.strip()
        entity = self._detect_entity(prompt, request.schema or {})
        fields = self._detect_fields(prompt, request.schema or {})
        filters = self._detect_filters(prompt)

        query = self._build_query(entity, fields, filters)

        explanation = (
            f"Mapped text to GraphQL selection on '{entity}' with fields {fields}."
        )
        if filters:
            explanation += f" Added filters: {filters}."

        confidence = 0.62 if filters else 0.56

        return QueryResult(
            query=query,
            target="graphql",
            confidence=confidence,
            explanation=explanation,
            metadata={"entity": entity, "fields": fields, "filters": filters},
        )

    def _detect_entity(self, text: str, schema: dict) -> str:
        if "entities" in schema:
            candidates = [e.lower() for e in schema["entities"]]
            for candidate in candidates:
                if candidate in text.lower():
                    return candidate

        for entity in ["user", "customer", "order", "product", "movie", "person"]:
            if entity in text.lower():
                return entity

        return "items"

    def _detect_fields(self, text: str, schema: dict) -> list[str]:
        lowered = text.lower()
        common = ["id", "name", "title", "email", "createdAt", "status", "price"]

        schema_fields = schema.get("fields", []) if schema else []
        if schema_fields:
            selected = [f for f in schema_fields if f.lower() in lowered]
            return selected or schema_fields[:3]

        selected = [f for f in common if f.lower() in lowered.lower()]
        return selected or ["id", "name"]

    def _detect_filters(self, text: str) -> dict[str, str]:
        filters: dict[str, str] = {}

        limit_match = re.search(r"(?:top|first|limit)\s+(\d+)", text.lower())
        if limit_match:
            filters["limit"] = limit_match.group(1)

        status_match = re.search(r"status\s+(?:is\s+)?([a-zA-Z_]+)", text.lower())
        if status_match:
            filters["status"] = status_match.group(1)

        return filters

    def _build_query(self, entity: str, fields: list[str], filters: dict[str, str]) -> str:
        args = ""
        if filters:
            args_str = ", ".join(f"{k}: {self._format_arg(v)}" for k, v in filters.items())
            args = f"({args_str})"

        selection = "\n    ".join(fields)
        return dedent(
            f"""
            query GeneratedQuery {{
              {entity}{args} {{
                {selection}
              }}
            }}
            """
        ).strip()

    @staticmethod
    def _format_arg(value: str) -> str:
        return value if value.isdigit() else f'"{value}"'
