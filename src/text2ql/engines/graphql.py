from __future__ import annotations

import re
from textwrap import dedent

from text2ql.schema_config import NormalizedSchemaConfig, normalize_schema_config
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
        config = normalize_schema_config(request.schema, request.mapping)
        entity = self._detect_entity(prompt, config)
        fields = self._detect_fields(prompt, config, entity)
        filters = self._detect_filters(prompt, config)

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

    def _detect_entity(self, text: str, config: NormalizedSchemaConfig) -> str:
        lowered = text.lower()
        for alias, canonical in self._sorted_alias_pairs(config.entity_aliases):
            if self._contains_entity_token(lowered, alias):
                return canonical

        for entity in config.entities:
            if self._contains_entity_token(lowered, entity.lower()):
                return entity

        if config.default_entity:
            return config.default_entity

        for entity in ["user", "customer", "order", "product", "movie", "person"]:
            if self._contains_entity_token(lowered, entity):
                return entity

        return "items"

    def _detect_fields(
        self, text: str, config: NormalizedSchemaConfig, entity: str
    ) -> list[str]:
        lowered = text.lower()
        common = ["id", "name", "title", "email", "createdAt", "status", "price"]

        schema_fields = config.fields_by_entity.get(entity, config.fields)
        selected: list[str] = []
        for field in schema_fields:
            if self._contains_token(lowered, field.lower()):
                selected.append(field)

        for alias, canonical in self._sorted_alias_pairs(config.field_aliases):
            if canonical not in schema_fields:
                continue
            if self._contains_token(lowered, alias):
                selected.append(canonical)

        if schema_fields:
            unique_selected = self._unique_in_order(selected)
            if unique_selected:
                return unique_selected
            if config.default_fields:
                return config.default_fields
            return schema_fields[:3]

        selected = [f for f in common if self._contains_token(lowered, f.lower())]
        return selected or ["id", "name"]

    def _detect_filters(self, text: str, config: NormalizedSchemaConfig) -> dict[str, str]:
        filters: dict[str, str] = {}
        lowered = text.lower()

        limit_match = re.search(r"(?:top|first|limit)\s+(\d+)", lowered)
        if limit_match:
            filters["limit"] = limit_match.group(1)

        filter_key_aliases = {"status": "status"}
        filter_key_aliases.update(config.filter_key_aliases)

        for alias, canonical in self._sorted_alias_pairs(filter_key_aliases):
            key_pattern = re.escape(alias)
            match = re.search(rf"\b{key_pattern}\b\s+(?:is\s+)?([a-zA-Z_]+)", lowered)
            if not match:
                continue
            value = match.group(1)
            mapped_value = (
                config.filter_value_aliases.get(canonical.lower(), {}).get(value.lower(), value)
            )
            filters[canonical] = mapped_value

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

    @staticmethod
    def _contains_token(text: str, token: str) -> bool:
        return re.search(rf"\b{re.escape(token)}\b", text) is not None

    @staticmethod
    def _contains_entity_token(text: str, token: str) -> bool:
        """Match entity words with simple singular/plural tolerance."""
        if GraphQLEngine._contains_token(text, token):
            return True
        if token.endswith("s"):
            return False
        return GraphQLEngine._contains_token(text, f"{token}s")

    @staticmethod
    def _sorted_alias_pairs(alias_map: dict[str, str]) -> list[tuple[str, str]]:
        return sorted(alias_map.items(), key=lambda pair: len(pair[0]), reverse=True)

    @staticmethod
    def _unique_in_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            ordered.append(item)
            seen.add(item)
        return ordered
