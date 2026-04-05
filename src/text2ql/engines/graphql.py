from __future__ import annotations

import re
from textwrap import dedent
from typing import Any

from text2ql.constrained import ConstrainedOutputError, parse_graphql_intent
from text2ql.prompting import build_graphql_prompts, resolve_language, resolve_prompt_template
from text2ql.providers.base import LLMProvider
from text2ql.schema_config import (
    NormalizedRelation,
    NormalizedSchemaConfig,
    normalize_schema_config,
)
from text2ql.types import QueryRequest, QueryResult

from .base import QueryEngine


class GraphQLEngine(QueryEngine):
    """GraphQL-first engine with simple intent extraction.

    Design intent:
    - Keep the engine deterministic for testability.
    - Let LLM providers be optional, pluggable upgrades.
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    def generate(self, request: QueryRequest) -> QueryResult:
        prompt = request.text.strip()
        config = normalize_schema_config(request.schema, request.mapping)
        mode = str(request.context.get("mode", "deterministic")).strip().lower()
        llm_error: str | None = None

        if mode == "llm" and self.provider is not None:
            llm_result = self._generate_with_llm(prompt, config, request.context)
            if llm_result is not None:
                return llm_result
            llm_error = "LLM mode fallback to deterministic mode."

        entity = self._detect_entity(prompt, config)
        fields = self._detect_fields(prompt, config, entity)
        filters = self._detect_filters(prompt, config, entity)
        aggregations = self._detect_aggregations(prompt, config, entity)
        nested = self._detect_nested(prompt, config, entity)
        entity, fields, filters, aggregations, nested, validation_notes = self._validate_components(
            entity, fields, filters, aggregations, nested, config
        )

        query = self._build_query(entity, fields, filters, aggregations, nested)
        validation_notes.extend(self._validate_generated_query_against_introspection(query, config))

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
            metadata={
                "entity": entity,
                "fields": fields,
                "filters": filters,
                "aggregations": aggregations,
                "nested": nested,
                "mode": "deterministic",
                "llm_error": llm_error,
                "validation_notes": validation_notes,
            },
        )

    def _generate_with_llm(
        self,
        prompt: str,
        config: NormalizedSchemaConfig,
        context: dict,
    ) -> QueryResult | None:
        template = resolve_prompt_template(context)
        language = str(context.get("language", "english"))
        try:
            resolved_language = resolve_language(language)
        except ValueError:
            return None

        system_prompt, user_prompt = build_graphql_prompts(
            prompt,
            config,
            template,
            language=resolved_language,
        )
        system_prompt = self._apply_system_context(system_prompt, context)
        try:
            raw = self.provider.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        except (RuntimeError, ValueError, TypeError):
            return None
        try:
            intent = parse_graphql_intent(raw, config, language=resolved_language)
        except ConstrainedOutputError:
            return None

        nested: list[dict[str, Any]] = []
        aggregations: list[dict[str, str]] = []
        entity, fields, filters, aggregations, nested, validation_notes = self._validate_components(
            intent.entity,
            intent.fields,
            intent.filters,
            aggregations,
            nested,
            config,
        )
        query = self._build_query(entity, fields, filters, aggregations, nested)
        validation_notes.extend(self._validate_generated_query_against_introspection(query, config))
        return QueryResult(
            query=query,
            target="graphql",
            confidence=intent.confidence,
            explanation=intent.explanation,
            metadata={
                "entity": entity,
                "fields": fields,
                "filters": filters,
                "aggregations": aggregations,
                "nested": nested,
                "mode": "llm",
                "language": resolved_language,
                "raw_completion": raw,
                "validation_notes": validation_notes,
            },
        )

    @staticmethod
    def _apply_system_context(system_prompt: str, context: dict[str, Any]) -> str:
        extra = context.get("system_context")
        if not isinstance(extra, str):
            return system_prompt
        cleaned = extra.strip()
        if not cleaned:
            return system_prompt
        return f"{system_prompt}\n\nAdditional system context:\n{cleaned}"

    def _detect_entity(self, text: str, config: NormalizedSchemaConfig) -> str:
        lowered = text.lower()
        owned_asset = self._detect_owned_asset(lowered)
        holdings_entity = self._resolve_holdings_entity(config)
        if owned_asset and holdings_entity:
            return holdings_entity
        special_entity = self._resolve_special_entity(lowered, config)
        if special_entity is not None:
            return special_entity

        alias_or_name_entity = self._resolve_entity_by_alias_or_name(lowered, config)
        if alias_or_name_entity is not None:
            return alias_or_name_entity

        semantic_entity = self._resolve_entity_by_semantic_field_match(lowered, config)
        if semantic_entity is not None:
            return semantic_entity

        if config.default_entity:
            return config.default_entity

        for entity in ["user", "customer", "order", "product", "movie", "person"]:
            if self._contains_entity_token(lowered, entity):
                return entity

        return "items"

    def _resolve_special_entity(self, lowered: str, config: NormalizedSchemaConfig) -> str | None:
        if ("transaction" in lowered or "transactions" in lowered) and "as of date" in lowered:
            return self._find_entity_with_field(
                config,
                candidate_fields=["asOfDate", "asOfDateTime"],
                preferred_entity_names=["transactionsSummary"],
            )
        if "dividend" in lowered:
            return self._find_entity_by_name(config, "transactions")
        if "net worth" in lowered:
            return self._find_entity_with_field(
                config,
                candidate_fields=["netWorth", "regulatoryNetWorth"],
            )
        if "available" in lowered and "withdraw" in lowered:
            return self._find_entity_with_field(
                config,
                candidate_fields=["cashOnly", "cashWithMargin", "availBorr"],
                preferred_entity_names=["availableToWithdrawDetail"],
            )
        if "buying power" in lowered:
            return self._find_entity_with_field(
                config,
                candidate_fields=["cash", "margin", "withoutMarginImpact"],
                preferred_entity_names=["buyingPowerDetail"],
            )
        return None

    def _resolve_entity_by_alias_or_name(
        self,
        lowered: str,
        config: NormalizedSchemaConfig,
    ) -> str | None:
        for alias, canonical in self._sorted_alias_pairs(config.entity_aliases):
            if self._contains_entity_token(lowered, alias):
                return canonical
        for entity in config.entities:
            if self._contains_entity_token(lowered, entity.lower()):
                return entity
        return None

    def _detect_fields(
        self, text: str, config: NormalizedSchemaConfig, entity: str
    ) -> list[str]:
        lowered = text.lower()
        common = ["id", "name", "title", "email", "createdAt", "status", "price"]

        schema_fields = self._fields_for_entity(config, entity)
        owned_asset = self._detect_owned_asset(lowered)
        if owned_asset and self._entity_looks_like_holdings(entity, schema_fields):
            owned_fields = self._resolve_holdings_fields(schema_fields)
            if owned_fields:
                return owned_fields

        if not schema_fields:
            return self._detect_common_fields(lowered, common)

        selected = self._select_fields_from_schema(lowered, schema_fields, config)
        if selected:
            return selected
        if self._entity_looks_like_holdings(entity, schema_fields):
            contextual = self._resolve_holdings_context_fields(lowered, schema_fields)
            if contextual:
                return contextual
        semantic_fields = self._resolve_fields_by_semantic_match(lowered, schema_fields)
        if semantic_fields:
            return semantic_fields
        return config.default_fields or schema_fields[:3]

    def _detect_common_fields(self, lowered: str, common_fields: list[str]) -> list[str]:
        selected = [field for field in common_fields if self._contains_token(lowered, field.lower())]
        return selected or ["id", "name"]

    def _select_fields_from_schema(
        self,
        lowered: str,
        schema_fields: list[str],
        config: NormalizedSchemaConfig,
    ) -> list[str]:
        selected: list[str] = []
        for field in schema_fields:
            if self._contains_token(lowered, field.lower()):
                selected.append(field)
        for alias, canonical in self._sorted_alias_pairs(config.field_aliases):
            if canonical in schema_fields and self._contains_token(lowered, alias):
                selected.append(canonical)
        return self._unique_in_order(selected)

    def _detect_filters(
        self,
        text: str,
        config: NormalizedSchemaConfig,
        entity: str,
    ) -> dict[str, Any]:
        lowered = text.lower()
        where_clause = self._extract_where_clause(lowered)
        filters = self._extract_limit_filters(lowered)

        filter_key_aliases = {"status": "status"}
        filter_key_aliases.update(config.filter_key_aliases)

        self._apply_alias_key_filters(
            filters=filters,
            lowered=lowered,
            where_clause=where_clause,
            config=config,
            entity=entity,
            filter_key_aliases=filter_key_aliases,
        )
        self._apply_alias_value_filters(
            filters=filters,
            lowered=lowered,
            config=config,
            entity=entity,
        )
        self._apply_owned_asset_filter(
            filters=filters,
            lowered=lowered,
            config=config,
            entity=entity,
            filter_key_aliases=filter_key_aliases,
        )
        self._apply_advanced_filters(filters=filters, lowered=lowered)
        return filters

    @staticmethod
    def _extract_limit_filters(lowered: str) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if "most recent" in lowered:
            filters["limit"] = "1"
        limit_match = re.search(r"(?:top|first|limit)\s+(\d+)", lowered)
        if limit_match:
            filters["limit"] = limit_match.group(1)
        return filters

    def _apply_alias_key_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        where_clause: str | None,
        config: NormalizedSchemaConfig,
        entity: str,
        filter_key_aliases: dict[str, str],
    ) -> None:
        for alias, canonical in self._sorted_alias_pairs(filter_key_aliases):
            value = self._extract_filter_value(alias=alias, text=where_clause or lowered)
            if value is None and where_clause is not None:
                value = self._extract_filter_value(alias=alias, text=lowered)
            if value is None:
                continue
            resolved_canonical = self._resolve_filter_key_for_entity(config, entity, canonical)
            mapped_value = (
                config.filter_value_aliases.get(str(resolved_canonical).lower(), {}).get(value.lower(), value)
            )
            filters[str(resolved_canonical)] = mapped_value

    def _apply_alias_value_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        config: NormalizedSchemaConfig,
        entity: str,
    ) -> None:
        for canonical, alias_map in config.filter_value_aliases.items():
            resolved_canonical = self._resolve_filter_key_for_entity(config, entity, canonical)
            if str(resolved_canonical) in filters or not isinstance(alias_map, dict):
                continue
            for alias, mapped_value in alias_map.items():
                if self._contains_token(lowered, str(alias).lower()):
                    filters[str(resolved_canonical)] = mapped_value
                    break

    def _apply_owned_asset_filter(
        self,
        filters: dict[str, Any],
        lowered: str,
        config: NormalizedSchemaConfig,
        entity: str,
        filter_key_aliases: dict[str, str],
    ) -> None:
        owned_asset = self._detect_owned_asset(lowered)
        if owned_asset is not None:
            identifier_key = self._resolve_identifier_filter_key(
                config=config,
                entity=entity,
                filter_key_aliases=filter_key_aliases,
            )
            if identifier_key is not None:
                mapped_value = (
                    config.filter_value_aliases.get(identifier_key.lower(), {}).get(
                        owned_asset.lower(),
                        owned_asset.upper(),
                    )
                )
                filters[identifier_key] = mapped_value

    def _apply_advanced_filters(self, filters: dict[str, Any], lowered: str) -> None:
        range_filters = self._detect_between_filters(lowered)
        in_filters = self._detect_in_filters(lowered)
        grouped_filters = self._detect_grouped_filters(lowered, range_filters, in_filters)

        filters.update(range_filters)
        filters.update(in_filters)
        if grouped_filters:
            filters.update(grouped_filters)

    def _detect_between_filters(self, lowered: str) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        for match in re.finditer(r"\b([a-zA-Z_]+)\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", lowered):
            field = match.group(1)
            start = match.group(2)
            end = match.group(3)
            filters[f"{field}_gte"] = start
            filters[f"{field}_lte"] = end
        return filters

    def _detect_in_filters(self, lowered: str) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        for match in re.finditer(r"\b([a-zA-Z_]+)\s+in\s+([a-zA-Z0-9_,\s]+)", lowered):
            field = match.group(1)
            values_blob = match.group(2)
            values = [
                token.strip()
                for token in re.split(r",|\s+or\s+|\s+and\s+", values_blob)
                if token.strip()
            ]
            if values:
                filters[f"{field}_in"] = values
        return filters

    def _detect_grouped_filters(
        self,
        lowered: str,
        range_filters: dict[str, Any],
        in_filters: dict[str, Any],
    ) -> dict[str, Any]:
        simple_conditions: list[dict[str, Any]] = []
        for key, value in {**range_filters, **in_filters}.items():
            simple_conditions.append({key: value})

        status_match = re.search(r"status\s+(?:is\s+)?([a-zA-Z_]+)", lowered)
        if status_match:
            simple_conditions.append({"status": status_match.group(1)})

        if len(simple_conditions) < 2:
            return {}
        if " or " in lowered:
            return {"or": simple_conditions}
        if " and " in lowered:
            return {"and": simple_conditions}
        return {}

    @staticmethod
    def _extract_where_clause(lowered: str) -> str | None:
        parts = lowered.split(" where ", maxsplit=1)
        if len(parts) == 2:
            return parts[1].strip()
        return None

    @staticmethod
    def _extract_filter_value(alias: str, text: str) -> str | None:
        key_pattern = re.escape(alias)
        matches = list(re.finditer(rf"\b{key_pattern}\b\s+(?:is\s+)?([a-zA-Z0-9_]+)", text))
        if not matches:
            return None
        return matches[-1].group(1)

    @staticmethod
    def _detect_owned_asset(lowered: str) -> str | None:
        match = re.search(r"\bhow many\s+([a-z0-9_]+)\s+do i own\b", lowered)
        if match is not None:
            return match.group(1)
        match = re.search(r"\bhow many\s+([a-z0-9_]+)\s+i own\b", lowered)
        if match is not None:
            return match.group(1)
        return None

    def _resolve_identifier_filter_key(
        self,
        config: NormalizedSchemaConfig,
        entity: str,
        filter_key_aliases: dict[str, str],
    ) -> str | None:
        for candidate in self._identifier_field_candidates():
            canonical = filter_key_aliases.get(candidate)
            if isinstance(canonical, str) and canonical:
                return canonical
        schema_fields = self._fields_for_entity(config, entity)
        for field in schema_fields:
            if field.lower() in self._identifier_field_candidates():
                return field
        return None

    def _resolve_filter_key_for_entity(
        self,
        config: NormalizedSchemaConfig,
        entity: str,
        candidate_key: str,
    ) -> str:
        entity_args = config.args_by_entity.get(entity, [])
        for arg in entity_args:
            if arg.lower() == candidate_key.lower():
                return arg
        schema_fields = self._fields_for_entity(config, entity)
        for field in schema_fields:
            if field.lower() == candidate_key.lower():
                return field
        return candidate_key

    @staticmethod
    def _find_entity_by_name(config: NormalizedSchemaConfig, expected: str) -> str | None:
        lowered_expected = expected.lower()
        for entity in config.entities:
            if entity.lower() == lowered_expected:
                return entity
        return None

    def _find_entity_with_field(
        self,
        config: NormalizedSchemaConfig,
        candidate_fields: list[str],
        preferred_entity_names: list[str] | None = None,
    ) -> str | None:
        lowered_candidates = {field.lower() for field in candidate_fields}
        preferred = {name.lower() for name in (preferred_entity_names or [])}

        best_entity: str | None = None
        best_score = -1
        for entity in config.entities:
            fields = self._fields_for_entity(config, entity)
            if not fields:
                continue
            lowered_fields = {field.lower() for field in fields}
            score = len(lowered_candidates.intersection(lowered_fields))
            if score == 0:
                continue
            if preferred and entity.lower() in preferred:
                score += 2
            if score > best_score:
                best_score = score
                best_entity = entity
        return best_entity

    def _resolve_holdings_entity(self, config: NormalizedSchemaConfig) -> str | None:
        best_entity: str | None = None
        best_score = 0
        for entity in config.entities:
            fields = self._fields_for_entity(config, entity)
            if not fields:
                continue
            score = self._score_holdings_entity(entity, fields)
            if score > best_score:
                best_score = score
                best_entity = entity
        return best_entity if best_score > 0 else None

    def _score_holdings_entity(self, entity: str, fields: list[str]) -> int:
        lowered_entity = entity.lower()
        lowered_fields = {field.lower() for field in fields}
        has_identifier = any(candidate in lowered_fields for candidate in self._identifier_field_candidates())
        has_quantity = any(candidate in lowered_fields for candidate in self._quantity_field_candidates())
        if not (has_identifier and has_quantity):
            return 0
        score = 1
        if lowered_entity in {"positions", "holdings", "assets"}:
            score += 4
        if {"symbol", "securitytype"}.intersection(lowered_fields):
            score += 1
        if {"acctnum", "acctname", "accountpositioncount"}.intersection(lowered_fields):
            score -= 3
        return score

    def _resolve_entity_by_semantic_field_match(
        self,
        lowered: str,
        config: NormalizedSchemaConfig,
    ) -> str | None:
        best_entity: str | None = None
        best_score = 0.0
        for entity in config.entities:
            fields = self._fields_for_entity(config, entity)
            if not fields:
                continue
            score = (1.6 * self._score_fields_for_prompt(lowered, fields)) + (
                1.0 * self._score_entity_name_for_prompt(lowered, entity)
            )
            if score > best_score:
                best_score = score
                best_entity = entity
        return best_entity if best_score >= 0.6 else None

    def _entity_looks_like_holdings(self, entity: str, fields: list[str]) -> bool:
        return self._score_holdings_entity(entity, fields) > 0

    def _resolve_holdings_fields(self, fields: list[str]) -> list[str]:
        lowered_to_original = {field.lower(): field for field in fields}
        selection: list[str] = []
        for candidate in self._quantity_field_candidates():
            field = lowered_to_original.get(candidate)
            if field is not None:
                selection.append(field)
                break
        for candidate in self._identifier_field_candidates():
            field = lowered_to_original.get(candidate)
            if field is not None and field not in selection:
                selection.append(field)
                break
        return selection

    def _resolve_holdings_context_fields(self, lowered: str, fields: list[str]) -> list[str]:
        selected = self._resolve_holdings_fields(fields)
        lowered_to_original = {field.lower(): field for field in fields}
        optional_candidates = [
            ("traded", "tradeMarket"),
            ("market", "tradeMarket"),
            ("core", "securityType"),
            ("type", "securityType"),
        ]
        for token, candidate_field in optional_candidates:
            if token not in lowered:
                continue
            match = lowered_to_original.get(candidate_field.lower())
            if match is not None and match not in selected:
                selected.append(match)
        return selected

    @staticmethod
    def _quantity_field_candidates() -> tuple[str, ...]:
        return ("quantity", "qty", "shares", "units", "amount", "holding")

    @staticmethod
    def _identifier_field_candidates() -> tuple[str, ...]:
        return ("symbol", "ticker", "stock", "asset", "security", "code", "name")

    def _resolve_fields_by_semantic_match(
        self,
        lowered: str,
        fields: list[str],
        max_fields: int = 3,
    ) -> list[str]:
        scored: list[tuple[float, str]] = []
        for field in fields:
            score = self._score_field_for_prompt(lowered, field)
            if score > 0:
                scored.append((score, field))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [field for _, field in scored[:max_fields]]

    def _score_fields_for_prompt(self, lowered: str, fields: list[str]) -> float:
        if not fields:
            return 0.0
        return max(self._score_field_for_prompt(lowered, field) for field in fields)

    def _score_entity_name_for_prompt(self, lowered: str, entity: str) -> float:
        prompt_tokens = self._expanded_tokens(self._tokenize(lowered))
        entity_tokens = self._expanded_tokens(self._tokenize(entity))
        if not entity_tokens:
            return 0.0
        overlap = len(prompt_tokens.intersection(entity_tokens))
        if overlap == 0:
            return 0.0
        return overlap / max(1, len(entity_tokens))

    def _score_field_for_prompt(self, lowered: str, field: str) -> float:
        prompt_tokens = self._expanded_tokens(self._tokenize(lowered))
        field_tokens = self._expanded_tokens(self._tokenize(field))
        if not field_tokens:
            return 0.0
        overlap = len(prompt_tokens.intersection(field_tokens))
        if overlap == 0:
            return 0.0
        return overlap / max(1, len(field_tokens))

    def _fields_for_entity(self, config: NormalizedSchemaConfig, entity: str) -> list[str]:
        if config.fields_by_entity:
            return config.fields_by_entity.get(entity, [])
        return config.fields

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        with_spaces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        with_spaces = with_spaces.replace("_", " ")
        return {token for token in re.findall(r"[a-z0-9]+", with_spaces.lower()) if token}

    @staticmethod
    def _expanded_tokens(tokens: set[str]) -> set[str]:
        expanded = set(tokens)
        synonyms = {
            "val": "value",
            "value": "val",
            "mkt": "market",
            "market": "mkt",
            "pct": "percent",
            "percent": "pct",
            "chg": "change",
            "change": "chg",
            "txn": "transaction",
            "transaction": "txn",
            "acct": "account",
            "account": "acct",
            "todays": "today",
            "today": "todays",
        }
        for token in tokens:
            mapped = synonyms.get(token)
            if mapped:
                expanded.add(mapped)
        return expanded

    def _detect_aggregations(
        self,
        text: str,
        config: NormalizedSchemaConfig,
        entity: str,
    ) -> list[dict[str, str]]:
        lowered = text.lower()
        aggregations: list[dict[str, str]] = []
        candidate_fields = self._fields_for_entity(config, entity)

        if re.search(r"\bcount\b", lowered) or (
            re.search(r"\bhow many\b", lowered) and self._detect_owned_asset(lowered) is None
        ):
            aggregations.append({"function": "count", "field": ""})

        for fn_name in ["sum", "avg", "min", "max"]:
            if not re.search(rf"\b{fn_name}\b", lowered):
                continue
            metric_field = self._detect_metric_field(lowered, candidate_fields)
            aggregations.append({"function": fn_name, "field": metric_field})

        return aggregations

    def _detect_metric_field(self, lowered: str, candidate_fields: list[str]) -> str:
        preferred_metrics = ["total", "amount", "price", "cost", "score"]
        for metric in preferred_metrics:
            if self._contains_token(lowered, metric):
                return metric
        for field in candidate_fields:
            if self._contains_token(lowered, field.lower()):
                return field
        return "value"

    def _detect_nested(
        self,
        text: str,
        config: NormalizedSchemaConfig,
        entity: str,
    ) -> list[dict[str, Any]]:
        lowered = text.lower()
        relation_map = config.relations_by_entity.get(entity, {})
        nested: list[dict[str, Any]] = []

        for relation in relation_map.values():
            aliases = [relation.name, relation.target, *relation.aliases]
            relation_mentioned = any(
                self._contains_entity_token(lowered, alias.lower()) for alias in aliases
            )
            if not relation_mentioned:
                continue

            relation_fields = relation.fields or ["id"]
            selected_fields = [
                field for field in relation_fields if self._contains_token(lowered, field.lower())
            ]
            if not selected_fields:
                selected_fields = relation_fields[:1]

            relation_filters: dict[str, str] = {}
            if "latest" in lowered and "limit" in relation.args:
                relation_filters["limit"] = "1"

            nested.append(
                {
                    "relation": relation.name,
                    "target": relation.target,
                    "fields": selected_fields,
                    "filters": relation_filters,
                }
            )
        return nested

    def _build_query(
        self,
        entity: str,
        fields: list[str],
        filters: dict[str, Any],
        aggregations: list[dict[str, str]],
        nested: list[dict[str, Any]],
    ) -> str:
        args = ""
        if filters:
            ordered_filters = self._order_filters(filters)
            args_str = ", ".join(f"{k}: {self._format_arg(v)}" for k, v in ordered_filters)
            args = f"({args_str})"

        selection_lines = list(fields)
        for agg in aggregations:
            fn_name = agg.get("function", "")
            field = agg.get("field", "")
            if not fn_name:
                continue
            if fn_name == "count":
                selection_lines.append("count")
            else:
                selection_lines.append(f'{fn_name}(field: "{field}")')

        for nested_node in nested:
            relation = nested_node["relation"]
            nested_args = ""
            nested_filters = nested_node.get("filters", {})
            if nested_filters:
                ordered_nested_filters = self._order_filters(nested_filters)
                nested_args_str = ", ".join(
                    f"{k}: {self._format_arg(v)}" for k, v in ordered_nested_filters
                )
                nested_args = f"({nested_args_str})"
            nested_fields = nested_node.get("fields", ["id"])
            nested_selection = "\n      ".join(nested_fields)
            selection_lines.append(
                dedent(
                    f"""
                    {relation}{nested_args} {{
                      {nested_selection}
                    }}
                    """
                ).strip()
            )

        selection = "\n    ".join(selection_lines)
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
    def _format_arg(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value if value.isdigit() else f'"{value}"'
        if isinstance(value, list):
            return "[" + ", ".join(GraphQLEngine._format_arg(item) for item in value) + "]"
        if isinstance(value, dict):
            parts = [f"{k}: {GraphQLEngine._format_arg(v)}" for k, v in value.items()]
            return "{ " + ", ".join(parts) + " }"
        return f'"{str(value)}"'

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

    def _validate_components(
        self,
        entity: str,
        fields: list[str],
        filters: dict[str, Any],
        aggregations: list[dict[str, str]],
        nested: list[dict[str, Any]],
        config: NormalizedSchemaConfig,
    ) -> tuple[str, list[str], dict[str, Any], list[dict[str, str]], list[dict[str, Any]], list[str]]:
        notes: list[str] = []
        validated_entity = self._resolve_entity_for_schema(entity, config, notes)
        allowed_fields = self._resolve_allowed_fields(validated_entity, config)
        validated_fields = self._validate_fields(
            fields=fields,
            allowed_fields=allowed_fields,
            default_fields=config.default_fields,
            entity=validated_entity,
            notes=notes,
        )
        allowed_args = self._resolve_allowed_args(validated_entity, config)
        validated_filters = self._validate_filters(
            filters=filters,
            allowed_args=allowed_args,
            entity=validated_entity,
            notes=notes,
        )
        validated_aggregations = self._validate_aggregations(
            aggregations=aggregations,
            allowed_fields=allowed_fields,
            entity=validated_entity,
            notes=notes,
        )
        validated_nested = self._validate_nested_nodes(
            nested=nested,
            entity=validated_entity,
            config=config,
            notes=notes,
        )

        return (
            validated_entity,
            validated_fields,
            validated_filters,
            validated_aggregations,
            validated_nested,
            notes,
        )

    def _resolve_entity_for_schema(
        self,
        entity: str,
        config: NormalizedSchemaConfig,
        notes: list[str],
    ) -> str:
        known_entities = config.entities or list(config.introspection_query_args.keys())
        if not known_entities or entity in known_entities:
            return entity
        if config.default_entity:
            notes.append(f"entity '{entity}' is not in schema; using default '{config.default_entity}'")
            return config.default_entity
        notes.append(f"entity '{entity}' is not in schema; using '{known_entities[0]}'")
        return known_entities[0]

    def _resolve_allowed_fields(
        self,
        entity: str,
        config: NormalizedSchemaConfig,
    ) -> list[str]:
        allowed_fields = config.fields_by_entity.get(entity, config.fields)
        introspection_fields = config.introspection_entity_fields.get(entity, set())
        if not introspection_fields:
            return allowed_fields
        if not allowed_fields:
            return sorted(introspection_fields)
        return sorted(set(allowed_fields) | introspection_fields)

    def _validate_fields(
        self,
        fields: list[str],
        allowed_fields: list[str],
        default_fields: list[str],
        entity: str,
        notes: list[str],
    ) -> list[str]:
        if not allowed_fields:
            return fields
        filtered_fields = [field for field in fields if field in allowed_fields]
        dropped = [field for field in fields if field not in allowed_fields]
        if dropped:
            notes.append(f"dropped invalid fields for '{entity}': {dropped}")
        if filtered_fields:
            return filtered_fields
        defaults = [field for field in default_fields if field in allowed_fields]
        if defaults:
            notes.append("no valid requested fields remained; applied schema defaults")
            return defaults
        notes.append("no valid requested fields remained; applied schema defaults")
        return allowed_fields[:2]

    def _resolve_allowed_args(self, entity: str, config: NormalizedSchemaConfig) -> list[str]:
        allowed_args = config.args_by_entity.get(entity, [])
        introspection_args = config.introspection_query_args.get(entity, {})
        if not introspection_args:
            return allowed_args
        return sorted(set(allowed_args) | set(introspection_args.keys()))

    def _validate_filters(
        self,
        filters: dict[str, Any],
        allowed_args: list[str],
        entity: str,
        notes: list[str],
    ) -> dict[str, Any]:
        validated_filters = dict(filters)
        self._drop_invalid_filter_keys(validated_filters, allowed_args, entity, notes)
        self._validate_grouped_filters(validated_filters, allowed_args, notes)
        return validated_filters

    def _drop_invalid_filter_keys(
        self,
        filters: dict[str, Any],
        allowed_args: list[str],
        entity: str,
        notes: list[str],
    ) -> None:
        if not allowed_args:
            return
        invalid_keys = [key for key in filters if key not in allowed_args and key not in {"and", "or"}]
        for key in invalid_keys:
            filters.pop(key, None)
        if invalid_keys:
            notes.append(f"dropped invalid args for '{entity}': {invalid_keys}")

    def _validate_grouped_filters(
        self,
        filters: dict[str, Any],
        allowed_args: list[str],
        notes: list[str],
    ) -> None:
        for group_key in {"and", "or"}:
            group_conditions = filters.get(group_key)
            if not isinstance(group_conditions, list):
                continue
            valid_group_conditions: list[dict[str, Any]] = []
            for condition in group_conditions:
                if not isinstance(condition, dict):
                    continue
                filtered_condition = {
                    key: value for key, value in condition.items() if not allowed_args or key in allowed_args
                }
                if filtered_condition:
                    valid_group_conditions.append(filtered_condition)
            if valid_group_conditions:
                filters[group_key] = valid_group_conditions
                continue
            filters.pop(group_key, None)
            notes.append(f"dropped empty '{group_key}' group after arg validation")

    def _validate_aggregations(
        self,
        aggregations: list[dict[str, str]],
        allowed_fields: list[str],
        entity: str,
        notes: list[str],
    ) -> list[dict[str, str]]:
        allowed_agg_fields = set(allowed_fields)
        validated_aggregations: list[dict[str, str]] = []
        for agg in aggregations:
            fn_name = agg.get("function", "")
            field = agg.get("field", "")
            if fn_name not in {"count", "sum", "avg", "min", "max"}:
                notes.append(f"dropped unsupported aggregation '{fn_name}'")
                continue
            if fn_name != "count" and allowed_agg_fields and field not in allowed_agg_fields:
                notes.append(f"dropped aggregation field '{field}' not in schema for '{entity}'")
                continue
            validated_aggregations.append({"function": fn_name, "field": field})
        return validated_aggregations

    def _validate_nested_nodes(
        self,
        nested: list[dict[str, Any]],
        entity: str,
        config: NormalizedSchemaConfig,
        notes: list[str],
    ) -> list[dict[str, Any]]:
        validated_nested: list[dict[str, Any]] = []
        for node in nested:
            validated_node = self._validate_nested_node(node, entity, config, notes)
            if validated_node is not None:
                validated_nested.append(validated_node)
        return validated_nested

    def _validate_nested_node(
        self,
        node: dict[str, Any],
        entity: str,
        config: NormalizedSchemaConfig,
        notes: list[str],
    ) -> dict[str, Any] | None:
        relation_name = str(node.get("relation", ""))
        relation = self._resolve_relation(entity, relation_name, config)
        if relation is None:
            notes.append(f"dropped unknown relation '{relation_name}' for '{entity}'")
            return None

        relation_fields = node.get("fields", [])
        filtered_nested_fields = [field for field in relation_fields if field in relation.fields]
        dropped_nested_fields = [field for field in relation_fields if field not in relation.fields]
        if dropped_nested_fields:
            notes.append(
                f"dropped invalid nested fields for relation '{relation_name}': {dropped_nested_fields}"
            )
        if not filtered_nested_fields:
            filtered_nested_fields = relation.fields[:1] or ["id"]
            notes.append(
                f"no valid nested fields remained for '{relation_name}'; applied relation defaults"
            )

        nested_filters = dict(node.get("filters", {}))
        if relation.args:
            invalid_nested_args = [key for key in nested_filters if key not in relation.args]
            for key in invalid_nested_args:
                nested_filters.pop(key, None)
            if invalid_nested_args:
                notes.append(
                    f"dropped invalid nested args for relation '{relation_name}': {invalid_nested_args}"
                )

        return {
            "relation": relation.name,
            "target": relation.target,
            "fields": filtered_nested_fields,
            "filters": nested_filters,
        }

    def _resolve_relation(
        self,
        entity: str,
        relation_name: str,
        config: NormalizedSchemaConfig,
    ) -> NormalizedRelation | None:
        relation_map = config.relations_by_entity.get(entity, {})
        relation = relation_map.get(relation_name)
        if relation is not None:
            return relation
        intro_relations = config.introspection_relation_targets.get(entity, {})
        if relation_name not in intro_relations:
            return None
        target_type = intro_relations[relation_name]
        return NormalizedRelation(
            name=relation_name,
            target=target_type,
            fields=sorted(config.introspection_entity_fields.get(target_type, {"id"})),
            args=[],
        )

    @staticmethod
    def _order_filters(filters: dict[str, Any]) -> list[tuple[str, Any]]:
        preferred_order = {"limit": 0, "status": 1}
        return sorted(
            filters.items(),
            key=lambda kv: (preferred_order.get(kv[0], 100), kv[0]),
        )

    def _validate_generated_query_against_introspection(
        self,
        query: str,
        config: NormalizedSchemaConfig,
    ) -> list[str]:
        notes: list[str] = []
        if not config.introspection_query_args and not config.introspection_entity_fields:
            return notes

        root_match = re.search(r"\{\s*([A-Za-z_]\w*)\s*(\([^)]*\))?\s*\{", query)
        if not root_match:
            return notes
        entity = root_match.group(1)
        args_blob = root_match.group(2) or ""

        notes.extend(self._validate_introspection_args(entity, args_blob, config))
        notes.extend(self._validate_introspection_fields(entity, query, config))
        return notes

    def _validate_introspection_args(
        self,
        entity: str,
        args_blob: str,
        config: NormalizedSchemaConfig,
    ) -> list[str]:
        notes: list[str] = []
        allowed_args = set(config.introspection_query_args.get(entity, {}).keys())
        if not (allowed_args and args_blob):
            return notes
        for arg_key in self._extract_arg_keys(args_blob):
            if arg_key in allowed_args:
                continue
            notes.append(
                f"post-validation: arg '{arg_key}' is not declared in introspection for '{entity}'"
            )
        return notes

    def _validate_introspection_fields(
        self,
        entity: str,
        query: str,
        config: NormalizedSchemaConfig,
    ) -> list[str]:
        notes: list[str] = []
        allowed_fields = config.introspection_entity_fields.get(entity, set())
        if not allowed_fields:
            return notes
        root_body_match = re.search(
            r"\{\s*[A-Za-z_]\w*\s*(?:\([^)]*\))?\s*\{(.*?)\}\s*\}",
            query,
            re.S,
        )
        if not root_body_match:
            return notes
        for field_name in self._extract_selection_field_names(root_body_match.group(1)):
            if field_name in allowed_fields or field_name in {"count", "sum", "avg", "min", "max"}:
                continue
            notes.append(
                f"post-validation: field '{field_name}' is not declared for '{entity}' in introspection"
            )
        return notes

    @staticmethod
    def _extract_arg_keys(args_blob: str) -> list[str]:
        return re.findall(r"([A-Za-z_]\w*)\s*:", args_blob)

    @staticmethod
    def _extract_selection_field_names(selection_block: str) -> list[str]:
        names: list[str] = []
        for line in selection_block.splitlines():
            token = line.strip()
            if not token:
                continue
            token = token.split("(", 1)[0].strip()
            if token.endswith("{"):
                token = token[:-1].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", token):
                names.append(token)
        return names
