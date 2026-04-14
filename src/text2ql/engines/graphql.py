from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from text2ql.constrained import ConstrainedOutputError, extract_raw_graphql, parse_graphql_intent
from text2ql.renderers import GraphQLIRRenderer
from text2ql.filters import (
    AND_TOKEN as _AND_TOKEN,
    FILTER_VALUE as _FILTER_VALUE,
    SPURIOUS_FILTER_VALUES as _SPURIOUS_FILTER_VALUES,
    WORD_IDENTIFIER as _WORD_IDENTIFIER,
    detect_between_filters,
    detect_comparison_filters,
    detect_date_range_filters,
    detect_in_filters,
    detect_negation_filters,
    detect_owned_asset,
)
from text2ql.prompting import (
    GRAPHQL_INTENT_JSON_SCHEMA,
    build_graphql_direct_prompts,
    build_graphql_prompts,
    resolve_language,
    resolve_prompt_template,
)
from text2ql.providers.base import LLMProvider
from text2ql.schema_config import (
    NormalizedRelation,
    NormalizedSchemaConfig,
    normalize_schema_config,
)
from text2ql.types import QueryRequest, QueryResult, ValidationError

from .base import QueryEngine, compute_deterministic_confidence
from .graphql_detection import (
    detect_aggregations as _detect_aggregations_stage,
    detect_entity as _detect_entity_stage,
    detect_fields as _detect_fields_stage,
)
from .graphql_filter_parsing import detect_filters as _detect_filters_stage
from .graphql_validation import validate_components as _validate_components_stage
from .holdings_utils import (
    identifier_candidates as _identifier_candidates,
    quantity_candidates as _quantity_candidates,
    resolve_holdings_container as _resolve_holdings_container,
    resolve_holdings_projection as _resolve_holdings_projection,
    resolve_identifier_filter_key as _resolve_identifier_filter_key,
    score_holdings_container as _score_holdings_container,
)
from .text_utils import (
    contains_entity_token as _contains_entity_token,
    contains_token as _contains_token,
    extract_filter_value as _extract_filter_value,
    extract_where_clause as _extract_where_clause,
    parse_grouped_boolean_filters as _parse_grouped_boolean_filters,
    sorted_alias_pairs as _sorted_alias_pairs,
    split_top_level as _split_top_level,
    strip_outer_parentheses as _strip_outer_parentheses,
    unique_in_order as _unique_in_order,
)

logger = logging.getLogger(__name__)

# Module-level renderer singleton — re-used across calls to avoid allocation.
_GRAPHQL_RENDERER = GraphQLIRRenderer()


class GraphQLEngine(QueryEngine):
    """GraphQL-first engine with simple intent extraction.

    Design intent:
    - Keep the engine deterministic for testability.
    - Let LLM providers be optional, pluggable upgrades.

    Parameters
    ----------
    provider:
        Optional LLM provider for ``mode="llm"`` or ``mode="function_calling"``.
    strict_validation:
        When ``True``, raise :class:`~text2ql.types.ValidationError` on
        contradictory filters instead of silently noting them.  Defaults to
        ``False`` for backwards-compatible graceful degradation.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        strict_validation: bool = False,
    ) -> None:
        self.provider = provider
        self.strict_validation = strict_validation

    def generate(self, request: QueryRequest) -> QueryResult:
        prompt = request.text.strip()
        config = normalize_schema_config(request.schema, request.mapping)
        mode = str(request.context.get("mode", "deterministic")).strip().lower()
        llm_error: str | None = None

        if mode in {"llm", "function_calling"} and self.provider is not None:
            llm_result, llm_error = self._generate_with_llm(prompt, config, request.context, mode=mode)
            if llm_result is not None:
                return llm_result
            llm_error = llm_error or "LLM mode fallback to deterministic mode."

        logger.debug("GraphQLEngine.generate: prompt=%r", prompt)
        entity = self._detect_entity(prompt, config)
        logger.debug("GraphQLEngine: detected entity=%r", entity)
        fields = self._detect_fields(prompt, config, entity)
        logger.debug("GraphQLEngine: detected fields=%r", fields)
        filters = self._detect_filters(prompt, config, entity)
        logger.debug("GraphQLEngine: detected filters=%r", filters)
        aggregations = self._detect_aggregations(prompt, config, entity)
        logger.debug("GraphQLEngine: detected aggregations=%r", aggregations)
        nested = self._detect_nested(prompt, config, entity)
        logger.debug("GraphQLEngine: detected nested=%r", nested)
        entity, fields, filters, aggregations, nested, validation_notes = self._validate_components(
            entity, fields, filters, aggregations, nested, config
        )
        logger.debug("GraphQLEngine: after validation entity=%r fields=%r notes=%r", entity, fields, validation_notes)

        query = self._build_query(entity, fields, filters, aggregations, nested)
        logger.debug("GraphQLEngine: built query=%r", query)
        validation_notes.extend(self._validate_generated_query_against_introspection(query, config))

        explanation = (
            f"Mapped text to GraphQL selection on '{entity}' with fields {fields}."
        )
        if filters:
            explanation += f" Added filters: {filters}."

        confidence = compute_deterministic_confidence(
            entity=entity,
            fields=fields,
            filters=filters,
            validation_notes=validation_notes,
            config=config,
            extra_signals={"aggregations": aggregations, "nested": nested},
        )

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

    def _prepare_llm_prompts(
        self,
        prompt: str,
        config: NormalizedSchemaConfig,
        context: dict,
    ) -> tuple[str, str, str] | None:
        """Build (system_prompt, user_prompt, resolved_language) or return None on error."""
        template = resolve_prompt_template(context)
        language = str(context.get("language", "english"))
        try:
            resolved_language = resolve_language(language)
        except ValueError:
            logger.warning(
                "GraphQLEngine: unsupported language %r; falling back to deterministic mode. "
                "Supported languages: english",
                language,
            )
            return None
        system_prompt, user_prompt = build_graphql_prompts(
            prompt, config, template, language=resolved_language,
        )
        return self._apply_system_context(system_prompt, context), user_prompt, resolved_language

    def _build_llm_result(
        self,
        raw: str,
        prompt: str,
        config: NormalizedSchemaConfig,
        resolved_language: str,
    ) -> tuple[QueryResult | None, str | None]:
        """Parse raw LLM output and assemble a QueryResult, or return (None, error) on failure."""
        try:
            intent = parse_graphql_intent(raw, config, language=resolved_language)
        except ConstrainedOutputError as exc:
            error = f"LLM output parse error: {exc}"
            logger.warning("GraphQLEngine: %s", error)
            return None, error

        # Capture whether LLM intended empty fields (pure aggregation — no extra field padding)
        llm_fields_empty = len(intent.fields) == 0

        reconciled_entity, reconciled_fields, reconciled_filters = self._reconcile_owned_asset_intent(
            prompt=prompt,
            entity=intent.entity,
            fields=list(intent.fields),
            filters=dict(intent.filters),
            config=config,
        )
        # Prefer LLM-provided aggregations; fall back to text detection.
        if intent.aggregations:
            aggregations = [
                {
                    "function": str(a.get("function", "COUNT")).upper(),
                    "field": str(a.get("field", "*")),
                    "alias": a.get("alias"),
                }
                for a in intent.aggregations
            ]
        else:
            aggregations = self._detect_aggregations(prompt, config, reconciled_entity)

        # When the LLM explicitly returned fields=[] with aggregations, respect that
        # intent: a pure aggregation needs no scalar fields in the selection set.
        if llm_fields_empty and aggregations:
            reconciled_fields = []

        # Prefer LLM-provided nested relations; fall back to text detection.
        if intent.nested:
            nested = intent.nested
        else:
            nested = self._detect_nested(prompt, config, reconciled_entity)
        entity, fields, filters, aggregations, nested, validation_notes = self._validate_components(
            reconciled_entity, reconciled_fields, reconciled_filters, aggregations, nested, config,
        )
        query = self._build_query(
            entity, fields, filters, aggregations, nested,
            distinct=intent.distinct,
            having=intent.having,
        )
        validation_notes.extend(self._validate_generated_query_against_introspection(query, config))
        # Use calibrated schema-aware confidence instead of the LLM's self-report,
        # which is an uncalibrated guess that produces incomparable scores across modes.
        confidence = compute_deterministic_confidence(
            entity=entity,
            fields=fields,
            filters=filters,
            validation_notes=validation_notes,
            config=config,
            extra_signals={"aggregations": aggregations, "nested": nested},
        )
        return QueryResult(
            query=query,
            target="graphql",
            confidence=confidence,
            explanation=intent.explanation,
            metadata={
                "entity": entity,
                "fields": fields,
                "filters": filters,
                "aggregations": aggregations,
                "nested": nested,
                "distinct": intent.distinct,
                "having": intent.having,
                "mode": "llm",
                "language": resolved_language,
                "raw_completion": raw,
                "llm_confidence": intent.confidence,
                "validation_notes": validation_notes,
            },
        ), None

    def _generate_direct_graphql(
        self,
        prompt: str,
        config: NormalizedSchemaConfig,
        context: dict,
    ) -> tuple[QueryResult | None, str | None]:
        """Generate GraphQL by having the LLM write the raw query directly (mode='llm')."""
        language = str(context.get("language", "english"))
        try:
            resolved_language = resolve_language(language)
        except ValueError:
            return None, f"Unsupported language '{language}'"
        evidence = context.get("evidence") or None
        try:
            system_prompt, user_prompt = build_graphql_direct_prompts(
                prompt, config, language=resolved_language, evidence=evidence
            )
            system_prompt = self._apply_system_context(system_prompt, context)
        except Exception as exc:
            return None, f"Failed to build direct GraphQL prompts: {exc}"
        logger.debug("GraphQLEngine: calling LLM for direct GraphQL generation")
        try:
            raw = self.provider.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        except (RuntimeError, ValueError, TypeError) as exc:
            return None, f"LLM provider error: {exc}"
        query = extract_raw_graphql(raw)
        if not query:
            return None, "LLM returned empty GraphQL query"
        return QueryResult(
            query=query,
            target="graphql",
            confidence=0.8,
            explanation=f"Direct GraphQL generated by LLM for: {prompt[:80]}",
            metadata={
                "mode": "llm_direct",
                "language": resolved_language,
                "raw_completion": raw,
            },
        ), None

    async def _agenerate_direct_graphql(
        self,
        prompt: str,
        config: NormalizedSchemaConfig,
        context: dict,
    ) -> tuple[QueryResult | None, str | None]:
        """Async version of _generate_direct_graphql."""
        language = str(context.get("language", "english"))
        try:
            resolved_language = resolve_language(language)
        except ValueError:
            return None, f"Unsupported language '{language}'"
        evidence = context.get("evidence") or None
        try:
            system_prompt, user_prompt = build_graphql_direct_prompts(
                prompt, config, language=resolved_language, evidence=evidence
            )
            system_prompt = self._apply_system_context(system_prompt, context)
        except Exception as exc:
            return None, f"Failed to build direct GraphQL prompts: {exc}"
        logger.debug("GraphQLEngine: calling LLM for direct GraphQL generation (async)")
        try:
            raw = await self.provider.acomplete(system_prompt=system_prompt, user_prompt=user_prompt)
        except (RuntimeError, ValueError, TypeError) as exc:
            return None, f"LLM provider error: {exc}"
        query = extract_raw_graphql(raw)
        if not query:
            return None, "LLM returned empty GraphQL query"
        return QueryResult(
            query=query,
            target="graphql",
            confidence=0.8,
            explanation=f"Direct GraphQL generated by LLM for: {prompt[:80]}",
            metadata={
                "mode": "llm_direct",
                "language": resolved_language,
                "raw_completion": raw,
            },
        ), None

    def _generate_with_llm(
        self,
        prompt: str,
        config: NormalizedSchemaConfig,
        context: dict,
        mode: str = "llm",
    ) -> tuple[QueryResult | None, str | None]:
        if mode == "llm":
            return self._generate_direct_graphql(prompt, config, context)
        prepared = self._prepare_llm_prompts(prompt, config, context)
        if prepared is None:
            return None, "Failed to prepare LLM prompts (invalid language or template)."
        system_prompt, user_prompt, resolved_language = prepared
        logger.debug("GraphQLEngine: calling LLM provider (sync, function_calling=True)")
        try:
            raw = self.provider.complete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=GRAPHQL_INTENT_JSON_SCHEMA,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            error = f"LLM provider error: {exc}"
            logger.warning("GraphQLEngine: %s", error)
            return None, error
        return self._build_llm_result(raw, prompt, config, resolved_language)

    async def _agenerate_with_llm(
        self,
        prompt: str,
        config: NormalizedSchemaConfig,
        context: dict,
        mode: str = "llm",
    ) -> tuple[QueryResult | None, str | None]:
        if mode == "llm":
            return await self._agenerate_direct_graphql(prompt, config, context)
        prepared = self._prepare_llm_prompts(prompt, config, context)
        if prepared is None:
            return None, "Failed to prepare LLM prompts (invalid language or template)."
        system_prompt, user_prompt, resolved_language = prepared
        logger.debug("GraphQLEngine: calling LLM provider (async, function_calling=True)")
        try:
            raw = await self.provider.acomplete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=GRAPHQL_INTENT_JSON_SCHEMA,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            error = f"LLM provider error: {exc}"
            logger.warning("GraphQLEngine: async LLM provider error: %s", error)
            return None, error
        return self._build_llm_result(raw, prompt, config, resolved_language)

    async def agenerate(self, request: QueryRequest) -> QueryResult:
        """Async generate — LLM I/O is truly async; deterministic path runs inline."""
        prompt = request.text.strip()
        config = normalize_schema_config(request.schema, request.mapping)
        mode = str(request.context.get("mode", "deterministic")).strip().lower()

        llm_error: str | None = None
        if mode in {"llm", "function_calling"} and self.provider is not None:
            llm_result, llm_error = await self._agenerate_with_llm(prompt, config, request.context, mode=mode)
            if llm_result is not None:
                return llm_result
            llm_error = llm_error or "LLM mode fallback to deterministic mode."

        # Deterministic path is pure CPU — safe to run inline in async context
        det_request = QueryRequest(
            text=request.text,
            target=request.target,
            schema=request.schema,
            mapping=request.mapping,
            context={**request.context, "mode": "deterministic"},
        )
        det_result = self.generate(det_request)
        if llm_error:
            det_result.metadata["llm_error"] = llm_error
        return det_result

    def _reconcile_owned_asset_intent(
        self,
        prompt: str,
        entity: str,
        fields: list[str],
        filters: dict[str, Any],
        config: NormalizedSchemaConfig,
    ) -> tuple[str, list[str], dict[str, Any]]:
        lowered = prompt.lower()
        owned_asset = self._detect_owned_asset(lowered)
        if owned_asset is None:
            return entity, fields, filters

        holdings_entity = self._resolve_holdings_entity(config)
        resolved_entity = holdings_entity or entity
        schema_fields = self._fields_for_entity(config, resolved_entity)
        if not schema_fields:
            return resolved_entity, fields, filters

        identifier_key = self._resolve_identifier_filter_key(
            config=config,
            entity=resolved_entity,
            filter_key_aliases={"status": "status", **config.filter_key_aliases},
        )
        resolved_filters = dict(filters)
        if identifier_key and identifier_key not in resolved_filters:
            mapped_value = config.filter_value_aliases.get(identifier_key.lower(), {}).get(
                owned_asset.lower(),
                owned_asset.upper(),
            )
            resolved_filters[identifier_key] = mapped_value

        resolved_fields = list(fields)
        owned_fields = self._resolve_holdings_fields(schema_fields)
        for field in owned_fields:
            if field not in resolved_fields:
                resolved_fields.append(field)
        if not resolved_fields:
            resolved_fields = owned_fields or schema_fields[:2]
        return resolved_entity, resolved_fields, resolved_filters

    def _detect_entity(self, text: str, config: NormalizedSchemaConfig) -> str:
        return _detect_entity_stage(self, text, config)

    def _resolve_special_entity(self, lowered: str, config: NormalizedSchemaConfig) -> str | None:
        """Route compound-keyword intents to their schema entities.

        Rules are supplied via ``schema["keyword_intents"]`` — a list of dicts:

        .. code-block:: json

            {
              "keywords": ["net", "worth"],
              "find_entity_with_fields": ["netWorth", "regulatoryNetWorth"]
            }

        Each rule requires **all** ``keywords`` to appear in the lowered query.
        The engine then looks up the best-matching schema entity using either
        ``find_entity_by_name`` (exact name match) or ``find_entity_with_fields``
        (field-presence score).  An optional ``preferred_entity_names`` list
        breaks ties in favour of named entities.

        Domain-specific rules (dividends, net worth, buying power, …) are **not**
        hardcoded here — they belong in the calling schema's ``keyword_intents``
        config so that the engine stays domain-agnostic.
        """
        for intent in config.keyword_intents:
            if not self._intent_matches_keywords(lowered, intent):
                continue
            by_name = self._resolve_special_entity_by_name(config, intent)
            if by_name:
                return by_name
            by_fields = self._resolve_special_entity_by_fields(config, intent)
            if by_fields:
                return by_fields
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
        return _detect_fields_stage(self, text, config, entity)

    def _detect_common_fields(self, lowered: str, common_fields: list[str]) -> list[str]:
        selected = [field for field in common_fields if self._contains_field_token(lowered, field.lower())]
        return selected or ["id", "name"]

    def _select_fields_from_schema(
        self,
        lowered: str,
        schema_fields: list[str],
        config: NormalizedSchemaConfig,
    ) -> list[str]:
        selected: list[str] = []
        for field in schema_fields:
            if self._contains_field_token(lowered, field.lower()):
                selected.append(field)
        for alias, canonical in self._sorted_alias_pairs(config.field_aliases):
            if canonical in schema_fields and self._contains_field_token(lowered, alias):
                selected.append(canonical)
        return self._unique_in_order(selected)

    @staticmethod
    def _contains_field_token(text: str, token: str) -> bool:
        if GraphQLEngine._contains_token(text, token):
            return True
        if token.endswith("s") and len(token) > 3 and GraphQLEngine._contains_token(text, token[:-1]):
            return True
        if not token.endswith("s") and GraphQLEngine._contains_token(text, f"{token}s"):
            return True
        return False

    def _detect_filters(
        self,
        text: str,
        config: NormalizedSchemaConfig,
        entity: str,
    ) -> dict[str, Any]:
        return _detect_filters_stage(self, text, config, entity)

    @staticmethod
    def _extract_limit_filters(lowered: str) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if "most recent" in lowered:
            filters["limit"] = "1"
        limit_match = re.search(r"(?:top|first|limit)\s+(\d+)", lowered)
        if limit_match:
            filters["limit"] = limit_match.group(1)
        offset_match = re.search(r"(?:offset|skip)\s+(\d+)", lowered)
        if offset_match:
            filters["offset"] = offset_match.group(1)
        first_match = re.search(r"\bfirst\s+(\d+)\b", lowered)
        if first_match:
            filters["first"] = first_match.group(1)
        after_match = re.search(r"\bafter\s+([a-zA-Z0-9_\-]+)\b", lowered)
        if after_match:
            filters["after"] = after_match.group(1)
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
            value = self._extract_alias_filter_value(alias, lowered, where_clause)
            if value is None:
                continue
            resolved_canonical = self._resolve_filter_key_for_entity(config, entity, canonical)
            mapped_value = (
                config.filter_value_aliases.get(str(resolved_canonical).lower(), {}).get(value.lower(), value)
            )
            if not self._is_valid_alias_filter_value(
                entity=entity,
                resolved_canonical=resolved_canonical,
                raw_value=value,
                mapped_value=mapped_value,
            ):
                continue
            filters[str(resolved_canonical)] = mapped_value

    def _apply_alias_value_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        config: NormalizedSchemaConfig,
        entity: str,
    ) -> None:
        value_alias_candidates: list[tuple[float, int, int, str, Any]] = []
        for canonical, alias_map in config.filter_value_aliases.items():
            resolved_canonical = self._resolve_filter_key_for_entity(config, entity, canonical)
            if str(resolved_canonical) in filters or not isinstance(alias_map, dict):
                continue
            best_match = None
            for alias, mapped_value in alias_map.items():
                score = self._alias_match_score(lowered, str(alias))
                if score <= 0:
                    continue
                candidate = (
                    score,
                    self._canonical_filter_priority(str(resolved_canonical)),
                    self._alias_specificity(str(alias)),
                    str(resolved_canonical),
                    mapped_value,
                )
                if best_match is None or candidate[:3] > best_match[:3]:
                    best_match = candidate
            if best_match is not None:
                value_alias_candidates.append(best_match)

        single_value_focus = "how many" in lowered and " where " not in lowered
        if single_value_focus and value_alias_candidates:
            _, _, _, resolved, mapped_value = max(value_alias_candidates, key=lambda item: item[:3])
            filters[str(resolved)] = mapped_value
            return
        for _, _, _, resolved, mapped_value in value_alias_candidates:
            filters[str(resolved)] = mapped_value

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
                # Guard against spurious lexical extraction like `quantity: "of"`
                # from prompts such as "what quantity of QQQ do I own".
                for quantity_key in self._quantity_field_candidates():
                    if self._is_spurious_quantity_value(filters.get(quantity_key)):
                        del filters[quantity_key]

    @staticmethod
    def _is_spurious_quantity_value(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return value.strip().lower() in {"of", "for"}

    @staticmethod
    def _is_spurious_filter_value(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return value.strip().lower() in _SPURIOUS_FILTER_VALUES

    def _apply_advanced_filters(self, filters: dict[str, Any], lowered: str) -> None:
        range_filters = self._detect_between_filters(lowered)
        in_filters = self._detect_in_filters(lowered)
        comparison_filters = self._detect_comparison_filters(lowered)
        negation_filters = self._detect_negation_filters(lowered)
        date_range_filters = self._detect_date_range_filters(lowered)
        ordering_filters = self._detect_ordering_filters(lowered, filters)
        grouped_filters = self._detect_grouped_filters(lowered, range_filters, in_filters)

        filters.update(range_filters)
        filters.update(in_filters)
        filters.update(comparison_filters)
        filters.update(negation_filters)
        filters.update(date_range_filters)
        filters.update(ordering_filters)
        if grouped_filters:
            filters.update(grouped_filters)

    def _detect_comparison_filters(self, lowered: str) -> dict[str, Any]:
        return detect_comparison_filters(lowered)

    def _detect_negation_filters(self, lowered: str) -> dict[str, Any]:
        return detect_negation_filters(lowered)

    def _detect_date_range_filters(self, lowered: str) -> dict[str, Any]:
        return detect_date_range_filters(lowered)

    def _detect_ordering_filters(self, lowered: str, existing_filters: dict[str, Any]) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if "latest order" in lowered:
            return filters
        if any(token in lowered for token in ("latest", "newest", "most recent")):
            order_field = self._detect_order_field(lowered)
            filters["orderBy"] = order_field
            filters["orderDirection"] = "DESC"
            if "limit" not in existing_filters and "first" not in existing_filters:
                filters["limit"] = "1"
        highest = re.search(rf"\bhighest\s+{_WORD_IDENTIFIER}\b", lowered)
        if highest:
            filters["orderBy"] = highest.group(1)
            filters["orderDirection"] = "DESC"
        lowest = re.search(rf"\blowest\s+{_WORD_IDENTIFIER}\b", lowered)
        if lowest:
            filters["orderBy"] = lowest.group(1)
            filters["orderDirection"] = "ASC"
        return filters

    @staticmethod
    def _detect_order_field(lowered: str) -> str:
        for candidate in ("createdAt", "updatedAt", "date", "timestamp", "asOfDate", "asOfDateTime"):
            if candidate.lower() in lowered:
                return candidate
        return "createdAt"

    def _detect_between_filters(self, lowered: str) -> dict[str, Any]:
        return detect_between_filters(lowered)

    def _detect_in_filters(self, lowered: str) -> dict[str, Any]:
        return detect_in_filters(lowered)

    def _detect_grouped_filters(
        self,
        lowered: str,
        range_filters: dict[str, Any],
        in_filters: dict[str, Any],
    ) -> dict[str, Any]:
        precedence_group = self._parse_grouped_precedence_filters(lowered)
        if precedence_group:
            return precedence_group
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
        if _AND_TOKEN in lowered:
            return {"and": simple_conditions}
        return {}

    def _parse_grouped_precedence_filters(self, lowered: str) -> dict[str, Any]:
        return _parse_grouped_boolean_filters(lowered, self._parse_and_conditions)

    def _parse_and_conditions(self, text: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        normalized = self._strip_outer_parentheses(text)
        for part in self._split_top_level(normalized, "and"):
            if part.startswith("where "):
                part = part[6:].strip()
            part = self._strip_outer_parentheses(part)
            condition = self._parse_atomic_filter_condition(part)
            if condition:
                out.append(condition)
        return out

    @staticmethod
    def _split_top_level(text: str, operator: str) -> list[str]:
        return _split_top_level(text, operator)

    @staticmethod
    def _strip_outer_parentheses(text: str) -> str:
        return _strip_outer_parentheses(text)

    def _parse_atomic_filter_condition(self, text: str) -> dict[str, Any] | None:
        in_match = re.match(rf"^{_WORD_IDENTIFIER}\s+in\s+([A-Za-z0-9_,\s-]+)$", text)
        if in_match:
            field = in_match.group(1)
            values_blob = in_match.group(2)
            values = [
                token.strip()
                for token in re.split(r",|\s+or\s+|\s+and\s+", values_blob)
                if token.strip()
            ]
            if values:
                return {f"{field}_in": values}
        for pattern, suffix in [
            (rf"^{_WORD_IDENTIFIER}\s*(>=|<=|>|<)\s*{_FILTER_VALUE}$", None),
            (rf"^{_WORD_IDENTIFIER}\s*(?:!=|is not|not)\s*{_FILTER_VALUE}$", "_ne"),
            (rf"^{_WORD_IDENTIFIER}\s*(?:is\s+)?{_FILTER_VALUE}$", ""),
        ]:
            match = re.match(pattern, text)
            if not match:
                continue
            if suffix is None:
                field, operator, value = match.group(1), match.group(2), match.group(3)
                op_suffix = {">=": "_gte", "<=": "_lte", ">": "_gt", "<": "_lt"}[operator]
                return {f"{field}{op_suffix}": value}
            if suffix == "_ne":
                return {f"{match.group(1)}_ne": match.group(2)}
            return {match.group(1): match.group(2)}
        return None

    @staticmethod
    def _extract_where_clause(lowered: str) -> str | None:
        return _extract_where_clause(lowered)

    @staticmethod
    def _extract_filter_value(alias: str, text: str) -> str | None:
        return _extract_filter_value(alias, text, spurious_values=_SPURIOUS_FILTER_VALUES)

    @staticmethod
    def _detect_owned_asset(lowered: str) -> str | None:
        return detect_owned_asset(lowered)

    def _resolve_identifier_filter_key(
        self,
        config: NormalizedSchemaConfig,
        entity: str,
        filter_key_aliases: dict[str, str],
    ) -> str | None:
        return _resolve_identifier_filter_key(
            args=config.args_by_entity.get(entity, []),
            fields=self._fields_for_entity(config, entity),
            candidate_aliases=filter_key_aliases,
            identifier_keys=self._identifier_field_candidates(),
        )

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
        return _resolve_holdings_container(
            config.entities,
            fields_for_container=lambda entity: self._fields_for_entity(config, entity),
            quantity_keys=self._quantity_field_candidates(),
            identifier_keys=self._identifier_field_candidates(),
        )

    def _score_holdings_entity(self, entity: str, fields: list[str]) -> int:
        return _score_holdings_container(
            entity,
            fields,
            quantity_keys=self._quantity_field_candidates(),
            identifier_keys=self._identifier_field_candidates(),
        )

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
        return _resolve_holdings_projection(
            fields,
            quantity_keys=self._quantity_field_candidates(),
            identifier_keys=self._identifier_field_candidates(),
        )

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
        return _quantity_candidates()

    @staticmethod
    def _identifier_field_candidates() -> tuple[str, ...]:
        return _identifier_candidates()

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
        return _detect_aggregations_stage(self, text, config, entity)

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
        *,
        max_depth: int = 3,
        _visited: frozenset[str] | None = None,
        _depth: int = 0,
    ) -> list[dict[str, Any]]:
        """Detect nested relation selections, recursing up to *max_depth* hops.

        Parameters
        ----------
        text:
            Original (un-lowered) query text.
        config:
            Normalised schema configuration for the current engine instance.
        entity:
            The entity whose outgoing relations we are inspecting at this depth.
        max_depth:
            Maximum number of relation hops to follow (default 3).
        _visited:
            Internal set of entity names already present in the current path —
            prevents infinite loops in schemas with cycles (e.g. ``user → posts
            → user``).
        _depth:
            Internal recursion counter.
        """
        if _depth >= max_depth:
            return []

        lowered = text.lower()
        visited: frozenset[str] = (_visited or frozenset()) | {entity}
        relation_map = config.relations_by_entity.get(entity, {})
        nested: list[dict[str, Any]] = []

        for relation in relation_map.values():
            # Skip back-edges to avoid cycles in the output tree.
            if relation.target in visited:
                continue

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

            relation_filters: dict[str, Any] = {}
            if "latest" in lowered and "limit" in relation.args:
                relation_filters["limit"] = 1
            relation_filters.update(self._detect_relation_local_filters(lowered, relation))

            # Recurse into the target entity's own relations.
            child_nested = self._detect_nested(
                text,
                config,
                relation.target,
                max_depth=max_depth,
                _visited=visited | {relation.target},
                _depth=_depth + 1,
            )

            node: dict[str, Any] = {
                "relation": relation.name,
                "target": relation.target,
                "fields": selected_fields,
                "filters": relation_filters,
            }
            if child_nested:
                node["nested"] = child_nested

            nested.append(node)
        return nested

    def _detect_relation_local_filters(
        self,
        lowered: str,
        relation: NormalizedRelation,
    ) -> dict[str, str]:
        if not relation.args:
            return {}
        # Capture only local mentions near relation aliases to avoid parent-child leakage.
        aliases = [relation.name, relation.target, *relation.aliases]
        windows: list[str] = []
        for alias in aliases:
            pattern = rf"\b{re.escape(alias.lower())}\b(.{{0,80}})"
            for match in re.finditer(pattern, lowered):
                windows.append(match.group(1))
        if not windows:
            return {}

        local_filters: dict[str, str] = {}
        for window in windows:
            for arg in relation.args:
                value = self._extract_filter_value(alias=arg, text=window)
                if value is not None:
                    local_filters[arg] = value
        return local_filters

    def _build_query(
        self,
        entity: str,
        fields: list[str],
        filters: dict[str, Any],
        aggregations: list[dict[str, str]],
        nested: list[dict[str, Any]],
        distinct: bool = False,
        having: list[dict] | None = None,
    ) -> str:
        """Build a GraphQL query string via :class:`~text2ql.renderers.GraphQLIRRenderer`.

        The engine detects components; the renderer assembles the final string.
        ``IRRenderer.render()`` is now the production path for query generation.
        """
        from text2ql.ir import QueryIR
        ir = QueryIR.from_components(
            entity=entity,
            fields=fields,
            filters=filters,
            aggregations=aggregations,
            nested=nested,
            target="graphql",
            distinct=distinct,
            having=having or [],
        )
        return _GRAPHQL_RENDERER.render(ir)

    def _build_nested_selection(self, node: dict[str, Any], indent: int) -> str:
        """Delegate nested-node rendering to :class:`~text2ql.renderers.GraphQLIRRenderer`."""
        return _GRAPHQL_RENDERER._render_nested(node, indent)

    @staticmethod
    def _format_arg(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, list):
            return "[" + ", ".join(GraphQLEngine._format_arg(item) for item in value) + "]"
        if isinstance(value, dict):
            parts = [f"{k}: {GraphQLEngine._format_arg(v)}" for k, v in value.items()]
            return "{ " + ", ".join(parts) + " }"
        return f'"{str(value)}"'

    @staticmethod
    def _contains_token(text: str, token: str) -> bool:
        return _contains_token(text, token)

    def _alias_match_score(self, text: str, alias: str) -> float:
        alias_text = str(alias).strip().lower()
        if not alias_text:
            return 0.0
        if self._contains_token(text, alias_text):
            return 3.0
        alias_tokens = [token for token in re.findall(r"[a-z0-9]+", alias_text) if len(token) >= 4]
        if not alias_tokens:
            return 0.0
        text_tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
        if len(alias_tokens) <= 3:
            return 2.0 if all(token in text_tokens for token in alias_tokens) else 0.0
        overlap = sum(1 for token in alias_tokens if token in text_tokens)
        return 1.0 if overlap >= 2 else 0.0

    @staticmethod
    def _matches_filter_value_alias(lowered: str, alias: str) -> bool:
        alias_text = str(alias).strip().lower()
        if not alias_text:
            return False
        if GraphQLEngine._contains_token(lowered, alias_text):
            return True
        alias_tokens = [token for token in re.findall(r"[a-z0-9]+", alias_text) if len(token) >= 4]
        if not alias_tokens:
            return False
        lowered_tokens = set(re.findall(r"[a-z0-9]+", lowered))
        if len(alias_tokens) <= 3:
            return all(token in lowered_tokens for token in alias_tokens)
        overlap = sum(1 for token in alias_tokens if token in lowered_tokens)
        return overlap >= 2

    @staticmethod
    def _canonical_filter_priority(key: str) -> int:
        lowered = str(key).lower()
        if lowered.endswith("typedesc"):
            return 4
        if lowered.endswith("status") or lowered == "status":
            return 3
        if lowered.endswith("subcatdesc"):
            return 2
        if lowered.endswith("desc"):
            return 1
        return 0

    @staticmethod
    def _alias_specificity(alias: str) -> int:
        return len(re.findall(r"[a-z0-9]+", str(alias).lower()))

    @staticmethod
    def _contains_entity_token(text: str, token: str) -> bool:
        """Match entity words with simple singular/plural tolerance."""
        return _contains_entity_token(text, token)

    @staticmethod
    def _sorted_alias_pairs(alias_map: dict[str, str]) -> list[tuple[str, str]]:
        return _sorted_alias_pairs(alias_map)

    @staticmethod
    def _unique_in_order(items: list[str]) -> list[str]:
        return _unique_in_order(items)

    def _validate_components(
        self,
        entity: str,
        fields: list[str],
        filters: dict[str, Any],
        aggregations: list[dict[str, str]],
        nested: list[dict[str, Any]],
        config: NormalizedSchemaConfig,
    ) -> tuple[str, list[str], dict[str, Any], list[dict[str, str]], list[dict[str, Any]], list[str]]:
        return _validate_components_stage(
            self,
            entity,
            fields,
            filters,
            aggregations,
            nested,
            config,
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
        intersection = sorted(set(allowed_fields).intersection(introspection_fields))
        return intersection or sorted(introspection_fields)

    def _validate_fields(
        self,
        fields: list[str],
        allowed_fields: list[str],
        default_fields: list[str],
        entity: str,
        notes: list[str],
        aggregation_only: bool = False,
    ) -> list[str]:
        # Pure aggregation query — no scalar fields needed in the selection set.
        if aggregation_only:
            return []
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
        if not allowed_args:
            return sorted(introspection_args.keys())
        intersection = sorted(set(allowed_args).intersection(set(introspection_args.keys())))
        return intersection or sorted(introspection_args.keys())

    def _validate_filters(
        self,
        filters: dict[str, Any],
        allowed_args: list[str],
        entity: str,
        config: NormalizedSchemaConfig,
        notes: list[str],
    ) -> dict[str, Any]:
        validated_filters = dict(filters)
        self._drop_invalid_filter_keys(validated_filters, allowed_args, entity, notes)
        self._validate_grouped_filters(validated_filters, allowed_args, notes)
        self._coerce_and_validate_filter_values(validated_filters, entity, config, notes)
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
        invalid_keys = [
            key
            for key in filters
            if not self._is_allowed_filter_key(key, allowed_args) and key not in {"and", "or", "not"}
        ]
        for key in invalid_keys:
            filters.pop(key, None)
        if invalid_keys:
            notes.append(f"dropped invalid args for '{entity}': {invalid_keys}")

    @staticmethod
    def _is_allowed_filter_key(key: str, allowed_args: list[str]) -> bool:
        if key in allowed_args:
            return True
        suffixes = ("_gte", "_lte", "_gt", "_lt", "_in", "_nin", "_ne")
        for suffix in suffixes:
            if key.endswith(suffix) and key[: -len(suffix)] in allowed_args:
                return True
        return False

    def _validate_grouped_filters(
        self,
        filters: dict[str, Any],
        allowed_args: list[str],
        notes: list[str],
    ) -> None:
        for group_key in {"and", "or", "not"}:
            group_conditions = filters.get(group_key)
            if not isinstance(group_conditions, list):
                continue
            valid_group_conditions: list[dict[str, Any]] = []
            for condition in group_conditions:
                if not isinstance(condition, dict):
                    continue
                filtered_condition = {
                    key: value
                    for key, value in condition.items()
                    if (not allowed_args or self._is_allowed_filter_key(key, allowed_args))
                    or key in {"and", "or", "not"}
                }
                if filtered_condition:
                    valid_group_conditions.append(filtered_condition)
            if valid_group_conditions:
                filters[group_key] = valid_group_conditions
                continue
            filters.pop(group_key, None)
            notes.append(f"dropped empty '{group_key}' group after arg validation")

    def _coerce_and_validate_filter_values(
        self,
        filters: dict[str, Any],
        entity: str,
        config: NormalizedSchemaConfig,
        notes: list[str],
    ) -> None:
        for key, value in list(filters.items()):
            if self._coerce_group_filter_value(filters, key, value, entity, config, notes):
                continue
            coerced = self._coerce_filter_value(entity, key, value, config, notes)
            if coerced is None and value is not None and not self._is_explicit_null_literal(value):
                filters.pop(key, None)
            else:
                filters[key] = coerced

    def _coerce_group_filter_value(
        self,
        filters: dict[str, Any],
        key: str,
        value: Any,
        entity: str,
        config: NormalizedSchemaConfig,
        notes: list[str],
    ) -> bool:
        if key not in {"and", "or", "not"} or not isinstance(value, list):
            return False
        compact_children: list[dict[str, Any]] = []
        for child in value:
            if not isinstance(child, dict):
                continue
            self._coerce_and_validate_filter_values(child, entity, config, notes)
            if child:
                compact_children.append(child)
        if compact_children:
            filters[key] = compact_children
        else:
            filters.pop(key, None)
        return True

    @staticmethod
    def _is_explicit_null_literal(value: Any) -> bool:
        return isinstance(value, str) and value.strip().lower() == "null"

    def _coerce_filter_value(
        self,
        entity: str,
        key: str,
        value: Any,
        config: NormalizedSchemaConfig,
        notes: list[str],
    ) -> Any:
        base_key = self._base_filter_key(key)
        arg_types = config.introspection_query_args.get(entity, {})
        arg_type = arg_types.get(base_key)
        enum_values: set[str] = self._enum_values_for_type(arg_type, config) if arg_type else set()
        if isinstance(value, list):
            return self._coerce_list_filter_values(value, arg_type, enum_values, base_key, notes)
        coerced_scalar = self._coerce_scalar_value(value, arg_type)
        if enum_values and isinstance(coerced_scalar, str):
            canonical = self._coerce_enum_value(coerced_scalar, enum_values)
            if canonical is None:
                notes.append(f"dropped invalid enum value '{coerced_scalar}' for '{base_key}'")
                return None
            coerced_scalar = canonical
        return coerced_scalar

    @staticmethod
    def _intent_keywords(intent: dict[str, Any]) -> list[str]:
        keywords = intent.get("keywords", [])
        if isinstance(keywords, str):
            return [keywords]
        if isinstance(keywords, list):
            return [str(item) for item in keywords if str(item).strip()]
        return []

    def _intent_matches_keywords(self, lowered: str, intent: dict[str, Any]) -> bool:
        keywords = self._intent_keywords(intent)
        return bool(keywords) and all(keyword in lowered for keyword in keywords)

    def _resolve_special_entity_by_name(
        self,
        config: NormalizedSchemaConfig,
        intent: dict[str, Any],
    ) -> str | None:
        if "find_entity_by_name" not in intent:
            return None
        return self._find_entity_by_name(config, str(intent["find_entity_by_name"]))

    def _resolve_special_entity_by_fields(
        self,
        config: NormalizedSchemaConfig,
        intent: dict[str, Any],
    ) -> str | None:
        if "find_entity_with_fields" not in intent:
            return None
        fields = intent["find_entity_with_fields"]
        preferred = intent.get("preferred_entity_names")
        return self._find_entity_with_field(
            config,
            candidate_fields=list(fields) if isinstance(fields, list) else [str(fields)],
            preferred_entity_names=list(preferred) if isinstance(preferred, list) else None,
        )

    def _extract_alias_filter_value(
        self,
        alias: str,
        lowered: str,
        where_clause: str | None,
    ) -> str | None:
        value = self._extract_filter_value(alias=alias, text=where_clause or lowered)
        if value is None and where_clause is not None:
            value = self._extract_filter_value(alias=alias, text=lowered)
        return value

    def _is_valid_alias_filter_value(
        self,
        entity: str,
        resolved_canonical: Any,
        raw_value: Any,
        mapped_value: Any,
    ) -> bool:
        if self._is_spurious_filter_value(raw_value):
            return False
        if str(resolved_canonical).strip().lower() == entity.strip().lower():
            return False
        if self._is_spurious_filter_value(mapped_value):
            return False
        return not (
            str(resolved_canonical).lower() in self._quantity_field_candidates()
            and self._is_spurious_quantity_value(mapped_value)
        )

    def _coerce_list_filter_values(
        self,
        values: list[Any],
        arg_type: str | None,
        enum_values: set[str],
        base_key: str,
        notes: list[str],
    ) -> list[Any]:
        out: list[Any] = []
        for item in values:
            coerced = self._coerce_scalar_value(item, arg_type)
            if coerced is None:
                continue
            if enum_values and isinstance(coerced, str):
                canonical = self._coerce_enum_value(coerced, enum_values)
                if canonical is None:
                    notes.append(f"dropped invalid enum value '{coerced}' for '{base_key}'")
                    continue
                coerced = canonical
            out.append(coerced)
        return out

    @staticmethod
    def _coerce_enum_value(value: str, enum_values: set[str]) -> str | None:
        return next((enum_value for enum_value in enum_values if enum_value.lower() == value.lower()), None)

    @staticmethod
    def _base_filter_key(key: str) -> str:
        for suffix in ("_gte", "_lte", "_gt", "_lt", "_in", "_nin", "_ne"):
            if key.endswith(suffix):
                return key[: -len(suffix)]
        return key

    @staticmethod
    def _coerce_scalar_value(value: Any, arg_type: str | None) -> Any:
        if not isinstance(value, str):
            return value
        raw = value.strip()
        lowered = raw.lower()
        if lowered == "null":
            return None
        if lowered in {"true", "false"}:
            return lowered == "true"
        if re.fullmatch(r"-?\d+", raw):
            return int(raw)
        if re.fullmatch(r"-?\d+\.\d+", raw):
            return float(raw)
        if GraphQLEngine._is_iso_date_or_datetime(raw):
            return raw
        if arg_type and "enum" in arg_type.lower():
            return raw.upper()
        return raw

    @staticmethod
    def _is_iso_date_or_datetime(raw: str) -> bool:
        try:
            datetime.strptime(raw, "%Y-%m-%d")
            return True
        except ValueError:
            pass
        datetime_raw = raw.replace("Z", "+00:00").replace("z", "+00:00")
        try:
            datetime.fromisoformat(datetime_raw)
            return True
        except ValueError:
            return False

    @staticmethod
    def _enum_values_for_type(arg_type: str, config: NormalizedSchemaConfig) -> set[str]:
        cleaned = arg_type.replace("[", "").replace("]", "").replace("!", "").strip()
        return config.introspection_enum_values.get(cleaned, set())

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

        validated: dict[str, Any] = {
            "relation": relation.name,
            "target": relation.target,
            "fields": filtered_nested_fields,
            "filters": nested_filters,
        }
        # Recursively validate and carry through any deeper nested children.
        raw_children: list[dict[str, Any]] = node.get("nested", [])
        if raw_children:
            validated_children = self._validate_nested_nodes(
                raw_children, relation.target, config, notes
            )
            if validated_children:
                validated["nested"] = validated_children

        return validated

    def _resolve_relation(
        self,
        entity: str,
        relation_name: str,
        config: NormalizedSchemaConfig,
    ) -> NormalizedRelation | None:
        intro_relations = config.introspection_relation_targets.get(entity, {})
        strict_intro_relations = bool(intro_relations)
        relation_map = config.relations_by_entity.get(entity, {})
        relation = relation_map.get(relation_name)
        if relation is not None:
            if strict_intro_relations and relation_name not in intro_relations:
                return None
            if relation_name in intro_relations:
                target_type = intro_relations[relation_name]
                intro_fields = sorted(config.introspection_entity_fields.get(target_type, {"id"}))
                filtered_fields = (
                    sorted(set(relation.fields).intersection(set(intro_fields)))
                    if relation.fields
                    else intro_fields
                )
                return NormalizedRelation(
                    name=relation.name,
                    target=target_type,
                    fields=filtered_fields or intro_fields,
                    args=relation.args,
                    aliases=relation.aliases,
                )
            return relation
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
        preferred_order = {
            "limit": 0,
            "offset": 1,
            "first": 2,
            "after": 3,
            "orderBy": 4,
            "orderDirection": 5,
            "orderDir": 6,
            "status": 10,
            "and": 90,
            "or": 91,
            "not": 92,
        }
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
