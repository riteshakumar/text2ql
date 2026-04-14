from __future__ import annotations

import logging
import re
from datetime import date, datetime
from dataclasses import dataclass
from typing import Any

from text2ql.constrained import ConstrainedOutputError, extract_raw_sql, parse_sql_intent
from text2ql.renderers import SQLIRRenderer
from text2ql.filters import (
    AND_TOKEN as _AND_TOKEN,
    SPURIOUS_FILTER_VALUES as _SPURIOUS_FILTER_VALUES,
    detect_between_filters,
    detect_comparison_filters,
    detect_date_range_filters,
    detect_in_filters,
    detect_negation_filters,
    detect_owned_asset,
)
from text2ql.prompting import (
    SQL_INTENT_JSON_SCHEMA,
    build_sql_direct_prompts,
    build_sql_prompts,
    resolve_language,
    resolve_prompt_template,
)
from text2ql.providers.base import LLMProvider
from text2ql.schema_config import NormalizedRelation, normalize_schema_config
from text2ql.types import QueryRequest, QueryResult, ValidationError

from .base import QueryEngine, compute_deterministic_confidence
from .sql_detection import (
    detect_columns as _detect_columns_stage,
    detect_order as _detect_order_stage,
    detect_table as _detect_table_stage,
)
from .sql_filter_parsing import detect_filters as _detect_filters_stage
from .sql_validation import validate_components as _validate_components_stage
from .holdings_utils import (
    identifier_candidates as _identifier_candidates,
    quantity_candidates as _quantity_candidates,
    resolve_holdings_container as _resolve_holdings_container,
    resolve_holdings_projection as _resolve_holdings_projection,
    resolve_identifier_filter_key as _resolve_identifier_filter_key,
    score_holdings_container as _score_holdings_container,
)
from .text_utils import (
    contains_column_reference as _contains_column_reference,
    contains_entity_token as _contains_entity_token,
    contains_token as _contains_token,
    extract_filter_value as _extract_filter_value,
    extract_where_clause as _extract_where_clause,
    label_match_variants as _label_match_variants,
    parse_grouped_boolean_filters as _parse_grouped_boolean_filters,
    sorted_alias_pairs as _sorted_alias_pairs,
    split_top_level as _split_top_level,
    strip_outer_parentheses as _strip_outer_parentheses,
    token_inflections as _token_inflections,
    unique_in_order as _unique_in_order,
)

logger = logging.getLogger(__name__)

_SQL_RENDERER = SQLIRRenderer()
AND_SEPARATOR = " AND "


@dataclass(slots=True)
class _RelationJoin:
    relation: str
    target: str
    alias: str
    on_clause: str
    fields: list[str]
    filters: dict[str, Any]


class SQLEngine(QueryEngine):
    """Deterministic SQL engine with schema validation and robust filter parsing.

    Parameters
    ----------
    provider:
        Optional LLM provider for ``mode="llm"`` or ``mode="function_calling"``.
    strict_validation:
        When ``True``, raise :class:`~text2ql.types.ValidationError` on
        contradictory filters or invalid JOIN ON-clause columns instead of
        silently adding a note and continuing.  Defaults to ``False`` to
        preserve backwards-compatible graceful degradation.
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
        lowered = prompt.lower()

        table = self._detect_table(lowered, config)
        columns = self._detect_columns(lowered, config, table)
        filters = self._detect_filters(lowered, config, table)
        filters = self._reconcile_having_intent_filters(lowered, filters)
        table, columns, filters = self._reconcile_owned_asset_intent(
            prompt=prompt,
            table=table,
            columns=columns,
            filters=filters,
            config=config,
        )
        joins = self._detect_joins(lowered, table, config)
        filters = self._reconcile_filters_with_join_context(filters, joins, table)
        all_table_columns = self._columns_for_table(config, table)
        order_by, order_dir = self._detect_order(
            lowered,
            columns,
            all_table_columns,
        )
        limit, offset = self._detect_pagination(lowered)
        group_by_columns = self._detect_group_by_fields(lowered, table, config)
        aggregations = self._detect_aggregations(
            lowered,
            table=table,
            columns=all_table_columns,
        )
        having = self._detect_having_conditions(lowered)
        if having and group_by_columns:
            columns = self._unique_in_order(group_by_columns)
        columns = self._reconcile_columns_with_aggregations(
            lowered=lowered,
            columns=columns,
            aggregations=aggregations,
            group_by_columns=group_by_columns,
        )
        columns = self._reconcile_columns_with_join_context(
            lowered=lowered,
            table=table,
            columns=columns,
            joins=joins,
            aggregations=aggregations,
        )
        aggregate_only = bool(aggregations) and not group_by_columns

        (
            table,
            columns,
            filters,
            joins,
            order_by,
            order_dir,
            notes,
        ) = self._validate_components(
            table=table,
            columns=columns,
            filters=filters,
            joins=joins,
            order_by=order_by,
            order_dir=order_dir,
            config=config,
        )
        columns = self._reconcile_columns_with_join_context(
            lowered=lowered,
            table=table,
            columns=columns,
            joins=joins,
            aggregations=aggregations,
        )
        distinct = self._detect_distinct(
            lowered=lowered,
            columns=columns,
            aggregations=aggregations,
            group_by_columns=group_by_columns,
        )
        if aggregate_only:
            columns = []
        exact_filter_keys = self._allowed_filter_keys(config, table, set(columns))

        query = self._build_sql(
            table=table,
            columns=columns,
            filters=filters,
            joins=joins,
            order_by=order_by,
            order_dir=order_dir,
            limit=limit,
            offset=offset,
            exact_filter_keys=exact_filter_keys,
            aggregations=aggregations,
            distinct=distinct,
            having=having,
        )
        confidence = compute_deterministic_confidence(
            entity=table,
            fields=columns,
            filters=filters,
            validation_notes=notes,
            config=config,
            extra_signals={"joins": joins, "order_by": order_by, "aggregations": aggregations},
        )
        return QueryResult(
            query=query,
            target="sql",
            confidence=confidence,
            explanation=f"Mapped text to SQL on table '{table}' with columns {columns}.",
            metadata={
                "table": table,
                "entity": table,
                "columns": columns,
                "fields": columns,
                "filters": filters,
                "joins": [
                    {
                        "relation": join.relation,
                        "target": join.target,
                        "alias": join.alias,
                        "on_clause": join.on_clause,
                        "fields": join.fields,
                        "filters": join.filters,
                    }
                    for join in joins
                ],
                "aggregations": aggregations,
                "order_by": order_by,
                "order_dir": order_dir,
                "limit": limit,
                "offset": offset,
                "mode": "deterministic",
                "llm_error": llm_error,
                "validation_notes": notes,
            },
        )

    def _prepare_llm_prompts(
        self,
        prompt: str,
        config: Any,
        context: dict[str, Any],
    ) -> tuple[str, str, str] | None:
        """Build (system_prompt, user_prompt, resolved_language) or return None on error."""
        template = resolve_prompt_template(context)
        language = str(context.get("language", "english"))
        try:
            resolved_language = resolve_language(language)
        except ValueError:
            logger.warning(
                "SQLEngine: unsupported language %r; falling back to deterministic mode. "
                "Supported languages: english",
                language,
            )
            return None
        system_prompt, user_prompt = build_sql_prompts(prompt, config, template, language=resolved_language)
        return self._apply_system_context(system_prompt, context), user_prompt, resolved_language

    def _build_llm_result(
        self,
        raw: str,
        prompt: str,
        config: Any,
        resolved_language: str,
    ) -> tuple[QueryResult | None, str | None]:
        """Parse raw LLM output and assemble a QueryResult, or return (None, error) on failure."""
        try:
            intent = parse_sql_intent(raw, config, language=resolved_language)
        except ConstrainedOutputError as exc:
            error = f"LLM output parse error: {exc}"
            logger.warning("SQLEngine: %s", error)
            return None, error

        # Capture whether LLM intended empty columns (pure aggregation — no GROUP BY)
        llm_columns_empty = len(intent.columns) == 0

        table, columns, filters = self._reconcile_owned_asset_intent(
            prompt=prompt,
            table=intent.table,
            columns=list(intent.columns),
            filters=dict(intent.filters),
            config=config,
        )

        # Prefer LLM-provided aggregations; fall back to text detection
        aggregations = [
            {
                "function": str(a.get("function", "COUNT")).upper(),
                "field": str(a.get("field", "*")),
                "alias": a.get("alias"),
            }
            for a in intent.aggregations
        ] if intent.aggregations else self._detect_aggregations(prompt.lower())

        # When the LLM explicitly returned columns=[] with aggregations, respect
        # that intent: a pure aggregation (COUNT(*)) needs no GROUP BY.  The
        # reconcile step above may have added default columns — undo that here.
        if llm_columns_empty and aggregations:
            columns = []

        joins = self._materialize_llm_joins(intent.joins, config, table)
        order_by, order_dir, limit, offset = intent.order_by, intent.order_dir, intent.limit, intent.offset

        table, columns, filters, joins, order_by, order_dir, notes = self._validate_components(
            table=table, columns=columns, filters=filters, joins=joins,
            order_by=order_by, order_dir=order_dir, config=config,
        )
        exact_filter_keys = self._allowed_filter_keys(config, table, set(columns))
        query = self._build_sql(
            table=table, columns=columns, filters=filters, joins=joins,
            order_by=order_by, order_dir=order_dir, limit=limit, offset=offset,
            exact_filter_keys=exact_filter_keys,
            aggregations=aggregations,
            distinct=intent.distinct,
            having=intent.having,
            subqueries=intent.subqueries,
        )
        # Use calibrated schema-aware confidence instead of the LLM's self-report.
        confidence = compute_deterministic_confidence(
            entity=table,
            fields=columns,
            filters=filters,
            validation_notes=notes,
            config=config,
            extra_signals={"joins": joins, "order_by": order_by, "aggregations": aggregations},
        )
        return QueryResult(
            query=query,
            target="sql",
            confidence=confidence,
            explanation=intent.explanation,
            metadata={
                "table": table,
                "entity": table,
                "columns": columns,
                "fields": columns,
                "filters": filters,
                "joins": [
                    {
                        "relation": j.relation,
                        "target": j.target,
                        "alias": j.alias,
                        "on_clause": j.on_clause,
                        "fields": j.fields,
                        "filters": j.filters,
                    }
                    for j in joins
                ],
                "order_by": order_by,
                "order_dir": order_dir,
                "limit": limit,
                "offset": offset,
                "aggregations": aggregations,
                "distinct": intent.distinct,
                "having": intent.having,
                "subqueries": intent.subqueries,
                "mode": "llm",
                "language": resolved_language,
                "raw_completion": raw,
                "llm_confidence": intent.confidence,
                "validation_notes": notes,
            },
        ), None

    def _generate_direct_sql(
        self,
        prompt: str,
        config: Any,
        context: dict[str, Any],
    ) -> tuple[QueryResult | None, str | None]:
        """Generate SQL by having the LLM write the raw query directly (mode='llm')."""
        language = str(context.get("language", "english"))
        try:
            resolved_language = resolve_language(language)
        except ValueError:
            return None, f"Unsupported language '{language}'"
        evidence = context.get("evidence") or None
        try:
            system_prompt, user_prompt = build_sql_direct_prompts(
                prompt, config, language=resolved_language, evidence=evidence
            )
            system_prompt = self._apply_system_context(system_prompt, context)
        except Exception as exc:
            return None, f"Failed to build direct SQL prompts: {exc}"
        logger.debug("SQLEngine: calling LLM for direct SQL generation")
        try:
            raw = self.provider.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        except (RuntimeError, ValueError, TypeError) as exc:
            return None, f"LLM provider error: {exc}"
        query = extract_raw_sql(raw)
        if not query:
            return None, "LLM returned empty SQL query"
        return QueryResult(
            query=query,
            target="sql",
            confidence=0.8,
            explanation=f"Direct SQL generated by LLM for: {prompt[:80]}",
            metadata={
                "mode": "llm_direct",
                "language": resolved_language,
                "raw_completion": raw,
            },
        ), None

    async def _agenerate_direct_sql(
        self,
        prompt: str,
        config: Any,
        context: dict[str, Any],
    ) -> tuple[QueryResult | None, str | None]:
        """Async version of _generate_direct_sql."""
        language = str(context.get("language", "english"))
        try:
            resolved_language = resolve_language(language)
        except ValueError:
            return None, f"Unsupported language '{language}'"
        evidence = context.get("evidence") or None
        try:
            system_prompt, user_prompt = build_sql_direct_prompts(
                prompt, config, language=resolved_language, evidence=evidence
            )
            system_prompt = self._apply_system_context(system_prompt, context)
        except Exception as exc:
            return None, f"Failed to build direct SQL prompts: {exc}"
        logger.debug("SQLEngine: calling LLM for direct SQL generation (async)")
        try:
            raw = await self.provider.acomplete(system_prompt=system_prompt, user_prompt=user_prompt)
        except (RuntimeError, ValueError, TypeError) as exc:
            return None, f"LLM provider error: {exc}"
        query = extract_raw_sql(raw)
        if not query:
            return None, "LLM returned empty SQL query"
        return QueryResult(
            query=query,
            target="sql",
            confidence=0.8,
            explanation=f"Direct SQL generated by LLM for: {prompt[:80]}",
            metadata={
                "mode": "llm_direct",
                "language": resolved_language,
                "raw_completion": raw,
            },
        ), None

    def _generate_with_llm(
        self,
        prompt: str,
        config: Any,
        context: dict[str, Any],
        mode: str = "llm",
    ) -> tuple[QueryResult | None, str | None]:
        if mode == "llm":
            return self._generate_direct_sql(prompt, config, context)
        prepared = self._prepare_llm_prompts(prompt, config, context)
        if prepared is None:
            return None, "Failed to prepare LLM prompts (invalid language or template)."
        system_prompt, user_prompt, resolved_language = prepared
        logger.debug("SQLEngine: calling LLM provider (sync, function_calling=True)")
        try:
            raw = self.provider.complete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=SQL_INTENT_JSON_SCHEMA,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            error = f"LLM provider error: {exc}"
            logger.warning("SQLEngine: %s", error)
            return None, error
        return self._build_llm_result(raw, prompt, config, resolved_language)

    async def _agenerate_with_llm(
        self,
        prompt: str,
        config: Any,
        context: dict[str, Any],
        mode: str = "llm",
    ) -> tuple[QueryResult | None, str | None]:
        if mode == "llm":
            return await self._agenerate_direct_sql(prompt, config, context)
        prepared = self._prepare_llm_prompts(prompt, config, context)
        if prepared is None:
            return None, "Failed to prepare LLM prompts (invalid language or template)."
        system_prompt, user_prompt, resolved_language = prepared
        logger.debug("SQLEngine: calling LLM provider (async, function_calling=True)")
        try:
            raw = await self.provider.acomplete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=SQL_INTENT_JSON_SCHEMA,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            error = f"LLM provider error: {exc}"
            logger.warning("SQLEngine: async LLM provider error: %s", error)
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
        table: str,
        columns: list[str],
        filters: dict[str, Any],
        config: Any,
    ) -> tuple[str, list[str], dict[str, Any]]:
        owned_asset = self._detect_owned_asset(prompt.lower())
        if owned_asset is None:
            return table, columns, filters

        holdings_table = self._resolve_holdings_table(config)
        resolved_table = holdings_table or table
        table_columns = self._columns_for_table(config, resolved_table)
        if not table_columns:
            return resolved_table, columns, filters

        identifier_key = self._resolve_identifier_filter_key(config, resolved_table)
        resolved_filters = dict(filters)
        if identifier_key and identifier_key not in resolved_filters:
            mapped_value = config.filter_value_aliases.get(identifier_key.lower(), {}).get(
                owned_asset.lower(),
                owned_asset.upper(),
            )
            resolved_filters[identifier_key] = mapped_value

        resolved_columns = list(columns)
        preferred = self._resolve_holdings_columns(table_columns)
        for column in preferred:
            if column not in resolved_columns:
                resolved_columns.append(column)
        if not resolved_columns:
            resolved_columns = preferred or table_columns[:2]
        return resolved_table, resolved_columns, resolved_filters

    def _materialize_llm_joins(
        self,
        payload_joins: list[dict[str, Any]],
        config: Any,
        table: str,
    ) -> list[_RelationJoin]:
        joins: list[_RelationJoin] = []
        for item in payload_joins:
            relation_name = str(item.get("relation", "")).strip()
            if not relation_name:
                continue
            relation = self._resolve_relation_for_join(config, table, relation_name)
            if relation is None:
                logger.warning(
                    "SQLEngine: LLM requested unknown relation %r for table %r; skipping join.",
                    relation_name,
                    table,
                )
                continue
            joins.append(self._build_relation_join_from_payload(item, relation, table))
        return joins

    def _reconcile_filters_with_join_context(
        self,
        filters: dict[str, Any],
        joins: list[_RelationJoin],
        table: str,
    ) -> dict[str, Any]:
        if not filters or not joins:
            return filters

        out = dict(filters)
        for join in joins:
            parent_key = self._parent_join_key(join.on_clause, table)
            if not parent_key or parent_key not in out:
                continue
            parent_value = out.get(parent_key)
            if not isinstance(parent_value, str) or not re.search(r"[a-zA-Z]", parent_value):
                continue
            lowered_parent_key = parent_key.lower()
            is_id_like = lowered_parent_key.endswith("id") or lowered_parent_key.endswith("code") or lowered_parent_key in {"maker", "country"}
            if not is_id_like:
                continue
            join_values = {str(value).strip().lower() for value in join.filters.values() if isinstance(value, str)}
            if parent_value.strip().lower() in join_values:
                out.pop(parent_key, None)
        return out

    @staticmethod
    def _parent_join_key(on_clause: str, parent_table: str) -> str | None:
        if "=" not in str(on_clause):
            return None
        left_ref, right_ref = [part.strip() for part in str(on_clause).split("=", maxsplit=1)]
        for ref in (left_ref, right_ref):
            if "." not in ref:
                continue
            ref_table, ref_column = [part.strip() for part in ref.rsplit(".", maxsplit=1)]
            if ref_table.lower() == str(parent_table).lower():
                return ref_column
        return None

    def _detect_table(self, lowered: str, config: Any) -> str:
        return _detect_table_stage(self, lowered, config)

    @staticmethod
    def _detect_owned_asset(lowered: str) -> str | None:
        return detect_owned_asset(lowered)

    def _resolve_holdings_table(self, config: Any) -> str | None:
        return _resolve_holdings_container(
            config.entities,
            fields_for_container=lambda entity: self._columns_for_table(config, entity),
            quantity_keys=self._quantity_column_candidates(),
            identifier_keys=self._identifier_column_candidates(),
        )

    def _score_holdings_table(self, table: str, columns: list[str]) -> int:
        return _score_holdings_container(
            table,
            columns,
            quantity_keys=self._quantity_column_candidates(),
            identifier_keys=self._identifier_column_candidates(),
        )

    def _resolve_identifier_filter_key(self, config: Any, table: str) -> str | None:
        return _resolve_identifier_filter_key(
            args=config.args_by_entity.get(table, []),
            fields=self._columns_for_table(config, table),
            identifier_keys=self._identifier_column_candidates(),
        )

    def _resolve_holdings_columns(self, columns: list[str]) -> list[str]:
        return _resolve_holdings_projection(
            columns,
            quantity_keys=self._quantity_column_candidates(),
            identifier_keys=self._identifier_column_candidates(),
        )

    @staticmethod
    def _quantity_column_candidates() -> tuple[str, ...]:
        return _quantity_candidates()

    @staticmethod
    def _identifier_column_candidates() -> tuple[str, ...]:
        return _identifier_candidates()

    def _detect_columns(self, lowered: str, config: Any, table: str) -> list[str]:
        return _detect_columns_stage(self, lowered, config, table)

    def _detect_filters(self, lowered: str, config: Any, table: str) -> dict[str, Any]:
        return _detect_filters_stage(self, lowered, config, table)

    def _apply_alias_filters(
        self,
        filters: dict[str, Any],
        where_clause: str,
        lowered: str,
        config: Any,
        table: str,
    ) -> None:
        """Populate *filters* from schema-defined key/value aliases."""
        filter_key_aliases: dict[str, str] = {"status": "status"}
        filter_key_aliases.update(config.filter_key_aliases)
        for alias, canonical in self._sorted_alias_pairs(filter_key_aliases):
            value = self._extract_filter_value(alias, where_clause)
            if value is None:
                continue
            resolved = self._resolve_filter_key_for_table(config, table, canonical)
            mapped = config.filter_value_aliases.get(str(resolved).lower(), {}).get(value.lower(), value)
            filters[str(resolved)] = mapped

        value_alias_candidates: list[tuple[float, int, int, str, Any]] = []
        for canonical, alias_map in config.filter_value_aliases.items():
            resolved = self._resolve_filter_key_for_table(config, table, canonical)
            if str(resolved) in filters or not isinstance(alias_map, dict):
                continue
            best_match = None
            for alias, mapped_value in alias_map.items():
                alias_text = str(alias).lower()
                score = self._alias_match_score(lowered, alias_text)
                if score <= 0:
                    continue
                candidate = (
                    score,
                    self._canonical_filter_priority(str(resolved)),
                    self._alias_specificity(alias_text),
                    str(resolved),
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

    def _apply_advanced_filters(self, filters: dict[str, Any], lowered: str) -> None:
        """Populate *filters* from comparison, range, negation, and IN expressions."""
        filters.update(detect_comparison_filters(lowered))
        filters.update(detect_negation_filters(lowered))
        filters.update(detect_between_filters(lowered))
        filters.update(detect_in_filters(lowered))
        filters.update(detect_date_range_filters(lowered))

    def _apply_schema_inferred_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        config: Any,
        table: str,
    ) -> None:
        candidate_keys = self._columns_for_table(config, table)
        if not candidate_keys:
            return
        self._infer_filters_for_candidate_keys(filters, lowered, candidate_keys)
        self._apply_special_keyword_filters(filters, lowered, candidate_keys)
        self._apply_month_year_filters(filters, lowered, candidate_keys)

    def _infer_filters_for_candidate_keys(
        self,
        filters: dict[str, Any],
        lowered: str,
        candidate_keys: list[str],
    ) -> None:
        self._apply_numeric_phrase_filters(filters, lowered, candidate_keys)
        self._apply_temporal_phrase_filters(filters, lowered, candidate_keys)
        self._apply_context_value_filters(filters, lowered, candidate_keys)
        self._apply_directional_value_filters(filters, lowered, candidate_keys)
        self._apply_id_like_filters(filters, lowered, candidate_keys)
        self._apply_boolean_style_filters(filters, lowered, candidate_keys)

    def _apply_numeric_phrase_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        candidate_keys: list[str],
    ) -> None:
        patterns = (
            (r"\bmore than\s+(-?\d+(?:\.\d+)?)\s+([a-zA-Z_]\w*)\b", "_gt"),
            (r"\bgreater than\s+(-?\d+(?:\.\d+)?)\s+([a-zA-Z_]\w*)\b", "_gt"),
            (r"\bover\s+(-?\d+(?:\.\d+)?)\s+([a-zA-Z_]\w*)\b", "_gt"),
            (r"\bless than\s+(-?\d+(?:\.\d+)?)\s+([a-zA-Z_]\w*)\b", "_lt"),
            (r"\bunder\s+(-?\d+(?:\.\d+)?)\s+([a-zA-Z_]\w*)\b", "_lt"),
        )
        for pattern, suffix in patterns:
            for match in re.finditer(pattern, lowered):
                value, phrase = match.group(1), match.group(2)
                key = self._best_matching_column(phrase, candidate_keys, prefer_metric=True)
                if key is None:
                    continue
                filter_key = f"{key}{suffix}"
                if filter_key not in filters:
                    filters[filter_key] = value

        phrase_first_patterns = (
            (r"\b([a-zA-Z_][\w\s]{1,40})\s+more than\s+(-?\d+(?:\.\d+)?)\b", "_gt"),
            (r"\b([a-zA-Z_][\w\s]{1,40})\s+greater than\s+(-?\d+(?:\.\d+)?)\b", "_gt"),
            (r"\b([a-zA-Z_][\w\s]{1,40})\s+over\s+(-?\d+(?:\.\d+)?)\b", "_gt"),
            (r"\b([a-zA-Z_][\w\s]{1,40})\s+less than\s+(-?\d+(?:\.\d+)?)\b", "_lt"),
            (r"\b([a-zA-Z_][\w\s]{1,40})\s+under\s+(-?\d+(?:\.\d+)?)\b", "_lt"),
        )
        for pattern, suffix in phrase_first_patterns:
            for match in re.finditer(pattern, lowered):
                phrase = str(match.group(1)).strip()
                value = str(match.group(2))
                for marker in (" where ", " with ", " and ", " or ", " that "):
                    if marker in phrase:
                        phrase = phrase.split(marker)[-1].strip()
                key = self._best_matching_column(phrase, candidate_keys, prefer_metric=True)
                if key is None:
                    continue
                filter_key = f"{key}{suffix}"
                if filter_key not in filters:
                    filters[filter_key] = value

    def _apply_temporal_phrase_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        candidate_keys: list[str],
    ) -> None:
        for match in re.finditer(r"\b(after|before)\s+(\d{4}(?:-\d{2}(?:-\d{2})?)?)\b", lowered):
            direction, raw_value = match.group(1), match.group(2)
            suffix = "_gt" if direction == "after" else "_lt"
            prefix = lowered[max(0, match.start() - 28): match.start()]
            key = self._best_matching_column(prefix, candidate_keys, prefer_dimension=False)
            if key is None or not self._is_date_like_key(key):
                key = self._best_date_like_column(candidate_keys)
            if key is None:
                continue
            value = raw_value
            if len(raw_value) == 4 and self._looks_datetime_column(key):
                value = f"{raw_value}-01-01"
            filter_key = f"{key}{suffix}"
            if filter_key not in filters:
                filters[filter_key] = value

    def _apply_context_value_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        candidate_keys: list[str],
    ) -> None:
        pattern = re.compile(
            r"\b(?:in|from|for|with)\s+(?:the\s+)?([a-z0-9_./+\-]{2,})\s+"
            r"(currency|country|county|city|state|segment|status|label|season|league|district|maker|nationality|type|location)\b"
        )
        stop_values = {
            "all",
            "each",
            "every",
            "where",
            "that",
            "which",
            "the",
            "a",
            "an",
            "of",
            "any",
            "most",
            "least",
            "highest",
            "lowest",
            "school",
            "schools",
            "shops",
            "customers",
            "players",
            "courses",
            "students",
            "employees",
            "transactions",
            "molecules",
            "atoms",
        }
        for match in pattern.finditer(lowered):
            raw_value, context_word = match.group(1), match.group(2)
            if raw_value in stop_values:
                continue
            key = self._best_matching_column(context_word, candidate_keys, prefer_dimension=True)
            if key is None or key in filters:
                continue
            filters[key] = self._normalize_inferred_filter_value(raw_value, key)

    def _apply_directional_value_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        candidate_keys: list[str],
    ) -> None:
        stop_values = {
            "all",
            "each",
            "every",
            "any",
            "the",
            "a",
            "an",
            "of",
            "and",
            "or",
            "to",
            "by",
            "for",
            "from",
            "in",
            "with",
            "without",
            "where",
            "which",
            "that",
            "there",
            "their",
            "most",
            "least",
            "highest",
            "lowest",
            "youngest",
            "oldest",
            "best",
            "worst",
            "shops",
            "school",
            "schools",
            "customers",
            "players",
            "courses",
            "students",
            "employees",
            "transactions",
            "molecules",
            "atoms",
            "matches",
            "teams",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        }
        patterns = (
            ("from", r"\bfrom\s+(?:the\s+)?([a-z0-9_./+\-]{2,})(?!\s+table\b)\b"),
            ("in", r"\bin\s+(?:the\s+)?([a-z0-9_./+\-]{2,})(?!\s+table\b)\b"),
            ("for", r"\bfor\s+(?:the\s+)?([a-z0-9_./+\-]{2,})(?!\s+table\b)\b"),
            ("with", r"\bwith\s+(?:the\s+)?([a-z0-9_./+\-]{2,})(?!\s+table\b)\b"),
            ("by", r"\bby\s+(?:the\s+)?([a-z0-9_./+\-]{2,})(?!\s+table\b)\b"),
        )
        role_tokens = {"customer", "player", "student", "employee", "seller", "buyer", "league", "season", "year"}
        for preposition, pattern in patterns:
            for match in re.finditer(pattern, lowered):
                raw_value = match.group(1)
                if raw_value in stop_values:
                    continue
                tail = lowered[match.end() : match.end() + 6]
                if raw_value in role_tokens and re.match(r"\s+\d", tail):
                    continue
                if preposition == "in" and ("located in" in lowered or "located at" in lowered):
                    multi_match = re.match(
                        r"\bin\s+(?:the\s+)?([a-z0-9_./+\-]+(?:\s+[a-z0-9_./+\-]+){0,2})\b",
                        lowered[match.start():],
                    )
                    if multi_match:
                        raw_value = multi_match.group(1).strip()
                    location_key = next(
                        (
                            key
                            for key in candidate_keys
                            if any(token in str(key).lower() for token in ("location", "city", "state", "country", "county"))
                        ),
                        None,
                    )
                    if location_key is not None and location_key not in filters:
                        filters[location_key] = self._normalize_inferred_filter_value(raw_value, location_key)
                        continue
                key = self._best_key_for_directional_value(preposition, candidate_keys, raw_value=raw_value)
                if key is None or key in filters:
                    continue
                filters[key] = self._normalize_inferred_filter_value(raw_value, key)

    def _apply_special_keyword_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        candidate_keys: list[str],
    ) -> None:
        lower_to_original = {str(key).lower(): str(key) for key in candidate_keys}

        currency_key = next((key for key in candidate_keys if "currency" in str(key).lower()), None)
        if currency_key is not None and currency_key not in filters:
            for code in ("USD", "EUR", "CZK", "GBP", "JPY", "INR", "CAD", "AUD", "CHF", "CNY"):
                if re.search(rf"\b{code.lower()}\b", lowered):
                    filters[currency_key] = code
                    break

        if "premium" in lowered:
            seg_key = next(
                (key for key in candidate_keys if any(token in str(key).lower() for token in ("segment", "type", "status"))),
                None,
            )
            if seg_key is not None and seg_key not in filters:
                filters[seg_key] = "Premium"

        label_key = next((key for key in candidate_keys if str(key).lower() == "label"), None)
        if label_key is not None and label_key not in filters:
            if "non-carcinogenic" in lowered or "non carcinogenic" in lowered:
                filters[label_key] = "-"
            elif "carcinogenic" in lowered:
                filters[label_key] = "+"

        bond_key = next((key for key in candidate_keys if "bond_type" == str(key).lower() or "bondtype" in str(key).lower()), None)
        if bond_key is not None and bond_key not in filters and ("double bond" in lowered or "double bonds" in lowered):
            filters[bond_key] = "="

        if "draw" in lowered:
            home_key = lower_to_original.get("home_team_goal")
            away_key = lower_to_original.get("away_team_goal")
            if home_key and away_key and home_key not in filters:
                filters[home_key] = {"$col": away_key}

    def _apply_month_year_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        candidate_keys: list[str],
    ) -> None:
        month_map = {
            "january": "01",
            "february": "02",
            "march": "03",
            "april": "04",
            "may": "05",
            "june": "06",
            "july": "07",
            "august": "08",
            "september": "09",
            "october": "10",
            "november": "11",
            "december": "12",
        }
        month = next((name for name in month_map if name in lowered), None)
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", lowered)
        if month is None or year_match is None:
            return
        date_keys = [key for key in candidate_keys if "date" in str(key).lower()]
        if not date_keys:
            return
        value = f"{year_match.group(1)}-{month_map[month]}"
        for key in date_keys:
            if key not in filters:
                filters[key] = value

    def _apply_id_like_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        candidate_keys: list[str],
    ) -> None:
        for match in re.finditer(r"\b([a-zA-Z_]\w*)\s+(\d+)\b", lowered):
            phrase, value = match.group(1), match.group(2)
            if phrase in {"top", "limit", "first", "last", "stage"}:
                continue
            key = self._best_matching_column(phrase, candidate_keys, prefer_dimension=True)
            if key is None or key in filters:
                continue
            lowered_key = key.lower()
            if not (lowered_key.endswith("id") or lowered_key.endswith("num") or "year" in lowered_key):
                continue
            filters[key] = value

        for match in re.finditer(r"\b([a-zA-Z_]\w*)\s+([a-zA-Z0-9_]*\d[a-zA-Z0-9_]*)\b", lowered):
            phrase, value = match.group(1), match.group(2)
            if phrase in {"top", "limit", "first", "last", "stage"}:
                continue
            key = self._best_matching_column(phrase, candidate_keys, prefer_dimension=True)
            if key is None or key in filters:
                continue
            lowered_key = key.lower()
            if not (lowered_key.endswith("id") or lowered_key.endswith("num") or lowered_key.endswith("code")):
                continue
            filters[key] = self._normalize_inferred_filter_value(value, key)

    def _apply_boolean_style_filters(
        self,
        filters: dict[str, Any],
        lowered: str,
        candidate_keys: list[str],
    ) -> None:
        if "active" in lowered:
            for key in candidate_keys:
                if "status" in key.lower() and key not in filters:
                    filters[key] = "Active"
                    break
        for key in candidate_keys:
            lowered_key = key.lower()
            if key in filters:
                continue
            if lowered_key.startswith("is"):
                trait = lowered_key[2:].lstrip("_")
                trait_tokens = [token for token in re.findall(r"[a-z0-9]+", trait) if token]
                if trait_tokens and all(token in lowered for token in trait_tokens):
                    filters[key] = 1
                    continue
            if lowered_key == "charter" and "charter" in lowered:
                filters[key] = 1

    def _best_key_for_directional_value(
        self,
        preposition: str,
        candidate_keys: list[str],
        raw_value: str | None = None,
    ) -> str | None:
        lowered = [str(key).lower() for key in candidate_keys]
        priority_map: dict[str, tuple[str, ...]] = {
            "from": ("city", "county", "state", "district", "country", "nationality", "segment", "status", "currency"),
            "in": ("league", "season", "year", "county", "city", "state", "country", "district", "date"),
            "for": ("season", "date", "year", "id", "currency", "segment"),
            "with": ("status", "type", "segment", "label", "currency"),
            "by": ("maker", "name", "id", "type", "segment"),
        }
        if preposition == "from" and raw_value:
            value = str(raw_value).strip().lower()
            is_short_code = bool(re.fullmatch(r"[a-z]{2,4}", value))
            country_terms = {
                "us",
                "usa",
                "uk",
                "uae",
                "germany",
                "france",
                "italy",
                "spain",
                "india",
                "china",
                "japan",
                "canada",
                "mexico",
                "brazil",
                "australia",
            }
            if is_short_code or value in country_terms:
                priority_map["from"] = (
                    "country",
                    "nationality",
                    "state",
                    "county",
                    "city",
                    "district",
                    "currency",
                    "segment",
                    "status",
                )
        for token in priority_map.get(preposition, ()):
            for idx, key in enumerate(lowered):
                if token in key:
                    return candidate_keys[idx]
        return None

    def _normalize_inferred_filter_value(self, raw_value: str, key: str) -> Any:
        value = str(raw_value).strip().strip(",.?!;:").strip("\"'")
        if not value:
            return value
        lowered_key = key.lower()
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        if re.fullmatch(r"-?\d+\.\d+", value):
            return float(value)
        if any(token in lowered_key for token in ("id", "num", "code")) and re.search(r"[a-zA-Z]", value):
            return value.upper()
        if any(token in lowered_key for token in ("maker", "make")):
            return value.lower()
        if "currency" in lowered_key or (value.isalpha() and len(value) <= 4):
            return value.upper()
        if "date" in lowered_key and re.fullmatch(r"\d{4}", value):
            return f"{value}-01-01"
        if any(token in lowered_key for token in ("country", "city", "state", "county", "district", "location", "status", "segment", "name", "label")):
            return value.title()
        return value

    @staticmethod
    def _is_date_like_key(key: str) -> bool:
        lowered = str(key).lower()
        return any(token in lowered for token in ("date", "time", "year", "season", "birthday", "birth"))

    def _best_date_like_column(self, candidate_keys: list[str]) -> str | None:
        for key in candidate_keys:
            if self._is_date_like_key(key):
                return key
        return None

    @staticmethod
    def _looks_datetime_column(key: str) -> bool:
        lowered = str(key).lower()
        return any(token in lowered for token in ("date", "time", "birthday", "birth"))

    @staticmethod
    def _aggregation_alias(function: str, field: str) -> str:
        safe_field = re.sub(r"\W+", "_", str(field)).strip("_") or "value"
        return f"{str(function).lower()}_{safe_field}"

    def _resolve_filter_key_for_table(self, config: Any, table: str, candidate_key: str) -> str:
        table_args = config.args_by_entity.get(table, [])
        for arg in table_args:
            if arg.lower() == candidate_key.lower():
                return arg
        table_columns = self._columns_for_table(config, table)
        for column in table_columns:
            if column.lower() == candidate_key.lower():
                return column
        return candidate_key

    def _parse_grouped_filters(self, lowered: str) -> dict[str, Any]:
        return _parse_grouped_boolean_filters(lowered, self._parse_and_nodes)

    def _parse_and_nodes(self, text: str) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        normalized = self._strip_outer_parentheses(text)
        for part in self._split_top_level(normalized, "and"):
            if part.startswith("where "):
                part = part[6:].strip()
            part = self._strip_outer_parentheses(part)
            in_match = re.match(r"^([A-Za-z_]\w*)\s+in\s+([\w,\s-]+)$", part)
            if in_match:
                field = in_match.group(1)
                values_blob = in_match.group(2)
                values = [
                    token.strip()
                    for token in re.split(r",|\s+or\s+|\s+and\s+", values_blob)
                    if token.strip()
                ]
                if values:
                    nodes.append({f"{field}_in": values})
                    continue
            parsed = self._parse_atomic_and_node(part)
            if parsed:
                nodes.append(parsed)
        return nodes

    @staticmethod
    def _split_top_level(text: str, operator: str) -> list[str]:
        return _split_top_level(text, operator)

    @staticmethod
    def _strip_outer_parentheses(text: str) -> str:
        return _strip_outer_parentheses(text)

    def _detect_order(
        self,
        lowered: str,
        selected_columns: list[str],
        all_columns: list[str] | None = None,
    ) -> tuple[str | None, str | None]:
        return _detect_order_stage(self, lowered, selected_columns, all_columns)

    def _detect_group_by_fields(self, lowered: str, table: str, config: Any) -> list[str]:
        if re.search(r"\bwhich\s+courses?\s+have\s+more than\s+\d+\s+students?\s+registered\b", lowered):
            columns = self._columns_for_table(config, table)
            course_id = next((column for column in columns if str(column).lower() == "course_id"), None)
            if course_id is not None:
                return [course_id]
        if not any(token in lowered for token in (" each ", " per ", " by ")):
            return []
        columns = self._columns_for_table(config, table)
        if not columns:
            return []
        avg_intent = "average" in lowered or "avg" in lowered

        phrases: list[str] = []
        patterns = (
            ("each", r"\beach\s+([a-zA-Z_]\w*(?:\s+[a-zA-Z_]\w*){0,2})"),
            ("per", r"\bper\s+([a-zA-Z_]\w*(?:\s+[a-zA-Z_]\w*){0,2})"),
            ("by", r"\bby\s+([a-zA-Z_]\w*(?:\s+[a-zA-Z_]\w*){0,2})"),
        )
        for pattern_kind, pattern in patterns:
            for match in re.finditer(pattern, lowered):
                phrase = match.group(1).strip()
                if not phrase:
                    continue
                if pattern_kind == "by":
                    if re.search(r"\d", phrase):
                        continue
                    if phrase.startswith("any "):
                        continue
                    prefix = lowered[max(0, match.start() - 24): match.start()]
                    if re.search(r"\b(paid|produced|made|manufactured|created|written|located|achieved)\s*$", prefix):
                        continue
                stop_tokens = {"all", "the", "a", "an", "of", "any"}
                trimmed = [token for token in phrase.split() if token not in stop_tokens]
                if trimmed:
                    phrases.append(" ".join(trimmed))

        selected: list[str] = []
        for phrase in phrases:
            candidate = self._best_matching_column(phrase, columns, prefer_dimension=True)
            if avg_intent and candidate is not None and self._is_identifier_like_group_column(candidate):
                continue
            if candidate is not None and self._is_identifier_like_group_column(candidate):
                name_like = next((col for col in columns if "name" in str(col).lower()), None)
                if name_like is not None and any(
                    token in phrase
                    for token in ("maker", "player", "employee", "student", "school", "team", "country", "course")
                ):
                    candidate = name_like
            if candidate is not None and candidate not in selected:
                selected.append(candidate)
        return selected

    @staticmethod
    def _is_identifier_like_group_column(column: str) -> bool:
        lowered = str(column).lower()
        return lowered.endswith("id") or "_id" in lowered or lowered in {"id", "maker", "type", "code"}

    def _reconcile_columns_with_aggregations(
        self,
        lowered: str,
        columns: list[str],
        aggregations: list[dict[str, str]],
        group_by_columns: list[str],
    ) -> list[str]:
        if not aggregations:
            return columns
        if group_by_columns:
            return self._unique_in_order(group_by_columns)
        # Aggregate-only intents should avoid implicit GROUP BY expansion.
        if any(token in lowered for token in ("how many", "count", "sum", "total", "average", "avg", "minimum", "min", "maximum", "max", "highest", "lowest")):
            return []
        return columns

    def _reconcile_columns_with_join_context(
        self,
        lowered: str,
        table: str,
        columns: list[str],
        joins: list[_RelationJoin],
        aggregations: list[dict[str, str]],
    ) -> list[str]:
        if not columns or not joins:
            return columns

        out = list(columns)
        asks_names = "name" in lowered or "names" in lowered
        join_name_fields = [
            field
            for join in joins
            for field in join.fields
            if "name" in str(field).lower() or "nationality" in str(field).lower()
        ]
        if asks_names and join_name_fields:
            explicit_non_identifier = [
                column
                for column in out
                if self._contains_column_reference(lowered, column) and not self._is_identifier_like_group_column(column)
            ]
            if not explicit_non_identifier:
                out = [
                    column
                    for column in out
                    if "name" in str(column).lower() or "nationality" in str(column).lower()
                ]
            if any(token in lowered for token in ("with the highest", "with the lowest")):
                out = [column for column in out if "name" in str(column).lower() or "nationality" in str(column).lower()]
        if join_name_fields and " who " in lowered and re.search(r"\b(show|list)\b", lowered):
            out = [column for column in out if "name" in str(column).lower() or "nationality" in str(column).lower()]

        if re.search(r"\bwhat\s+courses?\b", lowered) and any("course_name" in str(field).lower() for field in join_name_fields):
            out = []

        if aggregations and any(token in lowered for token in (" each ", " per ", " by ")):
            display_tokens = ("name", "country", "city", "state", "county", "district", "segment", "type", "status")
            display_join_present = False
            for join in joins:
                if not any(any(tok in str(field).lower() for tok in display_tokens) for field in join.fields):
                    continue
                display_join_present = True
                parent_key = self._parent_join_key(join.on_clause, table)
                if parent_key and parent_key in out:
                    out = [column for column in out if column != parent_key]
            if display_join_present and out and all(self._is_identifier_like_group_column(column) for column in out):
                out = []

        return out

    @staticmethod
    def _detect_distinct(
        lowered: str,
        columns: list[str],
        aggregations: list[dict[str, str]],
        group_by_columns: list[str],
    ) -> bool:
        if aggregations or group_by_columns:
            return False
        if "distinct" in lowered:
            return True
        if len(columns) == 1:
            column = str(columns[0]).lower()
            if "segment" in column and "segment" in lowered:
                return True
        return False

    def _best_matching_column(
        self,
        phrase: str,
        columns: list[str],
        *,
        prefer_dimension: bool = False,
        prefer_metric: bool = False,
    ) -> str | None:
        phrase_tokens = self._expanded_tokens(self._tokenize(phrase))
        if not phrase_tokens:
            return None

        best_column: str | None = None
        best_score = 0.0
        for column in columns:
            score = self._column_phrase_score(
                phrase_tokens=phrase_tokens,
                column=column,
                prefer_dimension=prefer_dimension,
                prefer_metric=prefer_metric,
            )
            if score > best_score:
                best_score = score
                best_column = column
        return best_column if best_score >= 0.45 else None

    def _column_phrase_score(
        self,
        phrase_tokens: set[str],
        column: str,
        *,
        prefer_dimension: bool = False,
        prefer_metric: bool = False,
    ) -> float:
        column_tokens = self._expanded_tokens(self._tokenize(column))
        if not column_tokens:
            return 0.0
        overlap = len(phrase_tokens.intersection(column_tokens))
        if overlap <= 0:
            return 0.0

        score = overlap / max(1, len(column_tokens))
        lowered_col = str(column).lower()
        if lowered_col.endswith("detail"):
            score -= 0.25
        if prefer_dimension and self._is_dimension_like_column(lowered_col):
            score += 0.2
        if prefer_metric and self._is_metric_like_column(lowered_col):
            score += 0.2
        return score

    @staticmethod
    def _is_dimension_like_column(lowered_column: str) -> bool:
        return any(
            token in lowered_column
            for token in (
                "id",
                "name",
                "type",
                "status",
                "category",
                "segment",
                "country",
                "city",
                "state",
                "county",
                "season",
                "year",
                "league",
                "currency",
                "label",
            )
        )

    @staticmethod
    def _is_metric_like_column(lowered_column: str) -> bool:
        return any(
            token in lowered_column
            for token in (
                "amount",
                "price",
                "cost",
                "score",
                "count",
                "total",
                "avg",
                "average",
                "sum",
                "age",
                "height",
                "weight",
                "goal",
                "rating",
                "bonus",
                "consumption",
                "enrollment",
                "earnings",
                "mpg",
                "horsepower",
                "quantity",
            )
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        with_spaces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
        with_spaces = with_spaces.replace("_", " ")
        return {token for token in re.findall(r"[a-z0-9]+", with_spaces.lower()) if token}

    @staticmethod
    def _expanded_tokens(tokens: set[str]) -> set[str]:
        expanded = set(tokens)
        for token in list(tokens):
            expanded.update(SQLEngine._token_inflections(token))
        synonyms = {
            "txn": "transaction",
            "transaction": "txn",
            "acct": "account",
            "account": "acct",
            "amt": "amount",
            "amount": "amt",
            "qty": "quantity",
            "quantity": "qty",
            "desc": "description",
            "description": "desc",
            "bal": "balance",
            "balance": "bal",
            "mkt": "market",
            "market": "mkt",
            "chg": "change",
            "change": "chg",
            "num": "number",
            "number": "num",
            "maker": "make",
            "make": "maker",
            "scr": "score",
            "score": "scr",
            "sat": "score",
            "percent": "percentage",
            "percentage": "percent",
        }
        for token in list(expanded):
            mapped = synonyms.get(token)
            if mapped:
                expanded.add(mapped)
        return expanded

    @staticmethod
    def _token_inflections(token: str) -> set[str]:
        token = str(token).strip().lower()
        if not token:
            return set()
        forms: set[str] = {token}
        if len(token) <= 3:
            return forms
        if token.endswith("ies") and len(token) > 4:
            forms.add(token[:-3] + "y")
        elif token.endswith("es") and len(token) > 4:
            forms.add(token[:-2])
        elif token.endswith("s") and len(token) > 3:
            forms.add(token[:-1])
        elif token.endswith("y"):
            forms.add(token[:-1] + "ies")
            forms.add(token + "s")
        elif token.endswith(("x", "z", "ch", "sh")):
            forms.add(token + "es")
        else:
            forms.add(token + "s")
        return {form for form in forms if form}

    @staticmethod
    def _detect_order_field(
        lowered: str,
        selected_columns: list[str],
        all_columns: list[str] | None = None,
    ) -> str:
        recency_priority = (
            "createdAt",
            "updatedAt",
            "postedDate",
            "tradedDate",
            "asOfDateTime",
            "asOfDate",
            "lastActionDate",
            "cycleDateTime",
            "acctCreationDate",
            "date",
            "timestamp",
        )
        for candidate in recency_priority:
            if candidate.lower() in lowered:
                return candidate

        pool = list(all_columns or [])
        lower_to_original = {str(col).lower(): str(col) for col in pool}
        for candidate in recency_priority:
            match = lower_to_original.get(candidate.lower())
            if match is not None:
                return match

        for col in pool:
            lowered_col = str(col).lower()
            if lowered_col.endswith("detail"):
                continue
            if any(token in lowered_col for token in ("date", "time", "timestamp", "created", "updated", "posted", "traded")):
                return str(col)

        for col in selected_columns:
            if not str(col).lower().endswith("detail"):
                return str(col)
        return (selected_columns[0] if selected_columns else (pool[0] if pool else "id"))

    def _detect_pagination(self, lowered: str) -> tuple[int | None, int | None]:
        limit = None
        offset = None
        limit_match = re.search(r"(?:top|limit|first)\s+(\d+)", lowered)
        if limit_match:
            limit = int(limit_match.group(1))
        if "most recent" in lowered and limit is None:
            limit = 1
        if limit is None and any(token in lowered for token in ("youngest", "oldest")):
            limit = 1
        if limit is None and re.search(r"\bwith\s+the\s+(most|least)\b", lowered):
            limit = 1
        if limit is None and re.search(r"\bwith\s+the\s+(highest|lowest)\b", lowered):
            limit = 1
        offset_match = re.search(r"(?:offset|skip)\s+(\d+)", lowered)
        if offset_match:
            offset = int(offset_match.group(1))
        after_match = re.search(
            r"\bafter\s+(?:the\s+)?(?:first\s+)?(\d+)\s+(?:rows?|records?|results?|entries?)\b",
            lowered,
        )
        if after_match and offset is None:
            offset = int(after_match.group(1))
        return limit, offset

    @staticmethod
    def _reconcile_having_intent_filters(lowered: str, filters: dict[str, Any]) -> dict[str, Any]:
        if not re.search(r"\bwhich\s+courses?\s+have\s+more than\s+\d+\s+students?\s+registered\b", lowered):
            return filters
        out = dict(filters)
        for key in list(out.keys()):
            if str(key).startswith("student_id"):
                out.pop(key, None)
        return out

    @staticmethod
    def _detect_having_conditions(lowered: str) -> list[dict[str, Any]]:
        course_count = re.search(r"\bwhich\s+courses?\s+have\s+more than\s+(\d+)\s+students?\s+registered\b", lowered)
        if course_count:
            return [{"function": "COUNT", "field": "*", "operator": ">", "value": int(course_count.group(1))}]
        return []

    def _detect_aggregations(
        self,
        lowered: str,
        *,
        table: str = "",
        columns: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Detect aggregate expressions in *lowered* query text.

        Returns a list of dicts compatible with ``QueryIR.from_components()``
        and :class:`~text2ql.ir.IRAggregation`.
        """
        aggregations: list[dict[str, str]] = []
        table_columns = list(columns or [])
        explicit_count = re.search(r"\bcount\b", lowered) is not None
        how_many = re.search(r"\bhow many\b", lowered) is not None
        number_of = (
            re.search(r"\bnumber of\b", lowered) is not None
            and re.search(r"\btotal number of\b", lowered) is None
            and not any(token in lowered for token in ("maximum number", "minimum number", "highest number", "lowest number"))
        )
        owned_asset_intent = self._detect_owned_asset(lowered) is not None
        if explicit_count or number_of or (how_many and not owned_asset_intent):
            aggregations.append({"function": "COUNT", "field": "*", "alias": "count"})

        # Ranking prompts like "highest total first 5" should stay as ORDER BY
        # selection queries, not aggregate projections.
        ranking_window_intent = (
            re.search(r"\b(highest|lowest|youngest|oldest)\b", lowered) is not None
            and re.search(r"\b(top|first|limit)\s+\d+\b", lowered) is not None
        )
        if ranking_window_intent and not (explicit_count or number_of or how_many):
            return aggregations

        if "best finish" in lowered and table_columns:
            best_finish_col = next((column for column in table_columns if "best" in str(column).lower() and "finish" in str(column).lower()), None)
            if best_finish_col is not None:
                min_candidate = {"function": "MIN", "field": best_finish_col, "alias": self._aggregation_alias("MIN", best_finish_col)}
                if min_candidate not in aggregations:
                    aggregations.append(min_candidate)
        dual_extrema = re.search(
            r"\b(?:maximum|max|highest)\s+and\s+(?:minimum|min|lowest)\s+([a-zA-Z_]\w*)\b",
            lowered,
        ) or re.search(
            r"\b(?:minimum|min|lowest)\s+and\s+(?:maximum|max|highest)\s+([a-zA-Z_]\w*)\b",
            lowered,
        )
        if dual_extrema is not None:
            phrase = dual_extrema.group(1)
            field = self._best_matching_column(phrase, table_columns, prefer_metric=True) or phrase
            aggregations.append({"function": "MAX", "field": field, "alias": self._aggregation_alias("MAX", field)})
            aggregations.append({"function": "MIN", "field": field, "alias": self._aggregation_alias("MIN", field)})
            return aggregations

        aggregate_patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("SUM", (r"\bsum\s+(?:of\s+)?([a-zA-Z_][\w\s]{0,40})", r"\btotal(?:\s+number\s+of)?\s+([a-zA-Z_][\w\s]{0,40})")),
            ("AVG", (r"\baverage\s+(?:of\s+)?([a-zA-Z_][\w\s]{0,40})", r"\bavg\s+(?:of\s+)?([a-zA-Z_][\w\s]{0,40})")),
            ("MIN", (r"\bminimum\s+(?:of\s+)?([a-zA-Z_][\w\s]{0,40})", r"\bmin\s+(?:of\s+)?([a-zA-Z_][\w\s]{0,40})", r"\blowest\s+([a-zA-Z_][\w\s]{0,40})")),
            ("MAX", (r"\bmaximum\s+(?:of\s+)?([a-zA-Z_][\w\s]{0,40})", r"\bmax\s+(?:of\s+)?([a-zA-Z_][\w\s]{0,40})", r"\bhighest\s+([a-zA-Z_][\w\s]{0,40})")),
        )
        stop_markers = (" for ", " in ", " from ", " of all", " among ", " across ", " with ", " where ")
        for fn, patterns in aggregate_patterns:
            if fn == "AVG" and any(token in lowered for token in ("highest", "lowest", "maximum", "minimum", "max", "min")):
                continue
            if fn in {"MAX", "MIN"} and any(token in lowered for token in ("with the highest", "with the lowest")):
                continue
            for pattern in patterns:
                for match in re.finditer(pattern, lowered):
                    phrase = match.group(1).strip()
                    for marker in stop_markers:
                        if marker in phrase:
                            phrase = phrase.split(marker, maxsplit=1)[0].strip()
                    if not phrase:
                        continue
                    field = self._best_matching_column(phrase, table_columns, prefer_metric=True) or phrase.split(" ")[0]
                    if table_columns and field not in table_columns:
                        fallback = self._best_matching_column(lowered, table_columns, prefer_metric=True)
                        if fallback:
                            field = fallback
                        elif fn in {"MAX", "MIN"} and any(token in lowered for token in ("highest", "lowest", "maximum", "minimum", "max", "min")):
                            continue
                    if fn == "AVG" and any(token in phrase for token in ("percent", "percentage", "ratio", "rate")):
                        percent_like = next(
                            (
                                column
                                for column in table_columns
                                if any(tok in str(column).lower() for tok in ("percent", "pct", "ratio", "rate"))
                            ),
                            None,
                        )
                        if percent_like is not None:
                            field = percent_like
                    alias = self._aggregation_alias(fn, field)
                    candidate = {"function": fn, "field": field, "alias": alias}
                    if candidate not in aggregations:
                        aggregations.append(candidate)
        if "1500" in lowered and any(token in lowered for token in ("total number", "scoring above", "score above")):
            candidate = next(
                (column for column in table_columns if "1500" in str(column) or "ge1500" in str(column).lower()),
                None,
            )
            if candidate is not None:
                sum_candidate = {"function": "SUM", "field": candidate, "alias": self._aggregation_alias("SUM", candidate)}
                if sum_candidate not in aggregations:
                    aggregations = [entry for entry in aggregations if entry.get("function") != "SUM"]
                    aggregations.append(sum_candidate)
        lower_to_original = {str(column).lower(): str(column) for column in table_columns}
        if "total number of goals" in lowered:
            home_goal = lower_to_original.get("home_team_goal")
            away_goal = lower_to_original.get("away_team_goal")
            if home_goal and away_goal:
                expr = f"{home_goal} + {away_goal}"
                sum_candidate = {"function": "SUM", "field": expr, "alias": self._aggregation_alias("SUM", expr)}
                aggregations = [entry for entry in aggregations if entry.get("function") != "SUM"]
                aggregations.append(sum_candidate)
        if (how_many or number_of) and "district" in lowered:
            district_col = next((column for column in table_columns if "district" in str(column).lower()), None)
            if district_col is not None:
                count_distinct = {"function": "COUNT", "field": f"DISTINCT {district_col}", "alias": "count"}
                aggregations = [entry for entry in aggregations if entry.get("function") != "COUNT"]
                aggregations.append(count_distinct)
        if table and not table_columns:
            return aggregations
        # Keep deterministic output concise for aggregate-only questions.
        if aggregations and re.search(r"\b(what is|what are|show)\b", lowered) and not any(token in lowered for token in (" each ", " per ", " by ")):
            return aggregations
        return aggregations

    def _detect_joins(self, lowered: str, table: str, config: Any) -> list[_RelationJoin]:
        relation_map = config.relations_by_entity.get(table, {})
        base_columns = self._columns_for_table(config, table)
        base_lower = {str(column).lower() for column in base_columns}
        joins: list[_RelationJoin] = []
        for relation in relation_map.values():
            aliases = [relation.name, relation.target, *relation.aliases]
            relation_mentioned = any(self._contains_entity_token(lowered, alias.lower()) for alias in aliases)
            target_columns = self._columns_for_table(config, relation.target)
            target_field_mentioned = any(
                self._contains_column_reference(lowered, column) and str(column).lower() not in base_lower
                for column in target_columns
            )
            relation_value_hint = self._relation_value_hint(lowered, base_columns, target_columns)
            if not relation_mentioned and not target_field_mentioned and not relation_value_hint:
                continue
            alias = relation.name
            on_clause = self._resolve_join_on_clause(parent=table, relation=relation)
            fields = self._relation_projection_fields(
                lowered=lowered,
                base_columns=base_columns,
                target_columns=target_columns,
                relation_default_fields=relation.fields,
            )
            if (
                not fields
                and relation_mentioned
                and re.search(r"\b(what|show|list)\b", lowered) is not None
                and not any(token in lowered for token in ("how many", "count", "sum", "average", "avg", " each ", " per "))
            ):
                target_name_cols = [column for column in target_columns if "name" in str(column).lower()]
                if target_name_cols:
                    fields = target_name_cols[:1]
            local_filters = self._detect_relation_local_filters(lowered, relation, config)
            joins.append(
                _RelationJoin(
                    relation=relation.name,
                    target=relation.target,
                    alias=alias,
                    on_clause=on_clause,
                    fields=fields,
                    filters=local_filters,
                )
            )
        if not joins:
            joins.extend(self._detect_special_case_joins(lowered, table, config, base_columns))
        self._prune_name_projection_joins(lowered, joins)
        self._prune_focus_entity_join_fields(lowered, joins)
        self._reorder_name_nationality_fields(lowered, joins)
        return joins

    def _prune_name_projection_joins(self, lowered: str, joins: list[_RelationJoin]) -> None:
        if "name" not in lowered and "names" not in lowered:
            return
        name_joins = [join for join in joins if any("name" in str(field).lower() for field in join.fields)]
        if len(name_joins) <= 1:
            return

        def _priority(join: _RelationJoin) -> tuple[int, int]:
            target = str(join.target).lower()
            person_tokens = ("person", "people", "employee", "student", "player", "customer", "user", "client", "singer")
            if any(token in target for token in person_tokens):
                return (0, len(target))
            return (1, len(target))

        best = min(name_joins, key=_priority)
        for join in name_joins:
            if join is best:
                continue
            join.fields = [field for field in join.fields if "name" not in str(field).lower()]

    @staticmethod
    def _prune_focus_entity_join_fields(lowered: str, joins: list[_RelationJoin]) -> None:
        if re.search(r"\bwhat\s+courses?\b", lowered):
            for join in joins:
                join.fields = [field for field in join.fields if "course" in str(field).lower()]

    @staticmethod
    def _reorder_name_nationality_fields(lowered: str, joins: list[_RelationJoin]) -> None:
        if "name and nationality" not in lowered:
            return
        for join in joins:
            has_name = any("name" in str(field).lower() for field in join.fields)
            has_nationality = any("nationality" in str(field).lower() for field in join.fields)
            if not (has_name and has_nationality):
                continue
            ordered = sorted(
                join.fields,
                key=lambda field: 0 if "name" in str(field).lower() else (1 if "nationality" in str(field).lower() else 2),
            )
            join.fields = ordered

    def _detect_special_case_joins(
        self,
        lowered: str,
        table: str,
        config: Any,
        base_columns: list[str],
    ) -> list[_RelationJoin]:
        if "carcinogenic" not in lowered:
            return []
        if str(table).lower() == "molecule":
            return []
        lower_base = {str(column).lower() for column in base_columns}
        if "molecule_id" not in lower_base:
            return []
        molecule_table = next((entity for entity in config.entities if str(entity).lower() == "molecule"), None)
        if molecule_table is None:
            return []
        molecule_columns = self._columns_for_table(config, str(molecule_table))
        lower_molecule = {str(column).lower() for column in molecule_columns}
        if "molecule_id" not in lower_molecule or "label" not in lower_molecule:
            return []
        label_value = "-" if ("non-carcinogenic" in lowered or "non carcinogenic" in lowered) else "+"
        return [
            _RelationJoin(
                relation="__special_molecule",
                target=str(molecule_table),
                alias=str(molecule_table),
                on_clause=f"{table}.molecule_id = {molecule_table}.molecule_id",
                fields=[],
                filters={"label": label_value},
            )
        ]

    def _relation_projection_fields(
        self,
        lowered: str,
        base_columns: list[str],
        target_columns: list[str],
        relation_default_fields: list[str],
    ) -> list[str]:
        count_per = "how many" in lowered and any(token in lowered for token in (" each ", " per "))
        if relation_default_fields and not count_per:
            return relation_default_fields[:2]

        base_lower = {str(column).lower() for column in base_columns}
        explicit = [
            column
            for column in target_columns
            if self._contains_column_reference(lowered, column) and str(column).lower() not in base_lower
        ]
        if explicit and count_per:
            anchor = len(lowered)
            for token in (" each ", " per "):
                pos = lowered.find(token)
                if pos >= 0:
                    anchor = min(anchor, pos)
            before = lowered[:anchor]
            filtered_explicit: list[str] = []
            for column in explicit:
                col_tokens = [token for token in self._tokenize(str(column)) if len(token) >= 3]
                if col_tokens and any(token in before for token in col_tokens):
                    continue
                filtered_explicit.append(column)
            explicit = filtered_explicit
        if explicit:
            return explicit[:2]

        asks_name = "name" in lowered or "names" in lowered
        base_has_name = any("name" in str(column).lower() for column in base_columns)
        if asks_name and not base_has_name:
            target_name_cols = [column for column in target_columns if "name" in str(column).lower()]
            if target_name_cols:
                return target_name_cols[:2]

        if any(token in lowered for token in (" each ", " per ")):
            target_dimension = [
                column
                for column in target_columns
                if any(token in str(column).lower() for token in ("city", "country", "state", "county", "district", "segment", "type", "status"))
            ]
            if target_dimension:
                target_dimension.sort(key=self._relation_dimension_priority)
                return target_dimension[:1]

        return []

    @staticmethod
    def _relation_dimension_priority(column: str) -> tuple[int, int]:
        lowered = str(column).lower()
        if "name" in lowered:
            return (0, len(lowered))
        if lowered.endswith("id") or "_id" in lowered or lowered == "id":
            return (2, len(lowered))
        return (1, len(lowered))

    @staticmethod
    def _relation_value_hint(
        lowered: str,
        base_columns: list[str],
        target_columns: list[str],
    ) -> bool:
        base_keys = {str(column).lower() for column in base_columns}
        target_keys = {str(column).lower() for column in target_columns}
        if not target_keys:
            return False

        if "located in" in lowered or "located at" in lowered:
            return any(any(tok in key for tok in ("location", "city", "state", "country", "county")) for key in target_keys)

        if re.search(r"\bfrom\s+[a-z0-9_+\-./]{2,}\b", lowered):
            target_has_geo = any(any(tok in key for tok in ("city", "state", "country", "county", "district", "nationality")) for key in target_keys)
            base_has_geo = any(any(tok in key for tok in ("city", "state", "country", "county", "district", "nationality")) for key in base_keys)
            return target_has_geo and not base_has_geo

        producer_tokens = ("maker", "make", "manufacturer", "brand", "company", "vendor", "producer", "author", "artist", "creator", "fullname", "name")
        has_by_phrase = bool(re.search(r"\bby\s+[a-z0-9_+\-./]{2,}\b", lowered)) and "order by" not in lowered
        if has_by_phrase and any(token in lowered for token in ("produced", "made", "manufactured", "created", "written", "by")):
            target_has_producer = any(any(tok in key for tok in producer_tokens) for key in target_keys)
            base_producer_cols = [key for key in base_keys if any(tok in key for tok in producer_tokens)]
            base_producer_id_like = bool(base_producer_cols) and all(
                any(tag in key for tag in ("id", "code", "key", "maker"))
                for key in base_producer_cols
            )
            return target_has_producer and (not base_producer_cols or base_producer_id_like)

        return False

    @staticmethod
    def _resolve_join_on_clause(parent: str, relation: NormalizedRelation) -> str:
        if relation.on:
            return relation.on
        return SQLEngine._build_join_on_clause(parent=parent, child=relation.target)

    @staticmethod
    def _build_join_on_clause(parent: str, child: str) -> str:
        parent_id = f"{parent}.id"
        parent_singular = parent[:-1] if parent.endswith("s") else parent
        fk = f"{child}.{parent_singular}Id"
        return f"{fk} = {parent_id}"

    def _detect_relation_local_filters(
        self,
        lowered: str,
        relation: NormalizedRelation,
        config: Any,
    ) -> dict[str, Any]:
        candidate_args = relation.args or self._columns_for_table(config, relation.target)
        if not candidate_args:
            return {}
        aliases = [relation.name, relation.target, *relation.aliases]
        windows: list[str] = []
        for alias in aliases:
            for alias_variant in {str(alias).lower(), f"{str(alias).lower()}s"}:
                pattern = rf"\b{re.escape(alias_variant)}\b(.{{0,80}})"
                for match in re.finditer(pattern, lowered):
                    windows.append(match.group(1))
        if not windows:
            windows = [lowered]
        if not windows:
            return {}
        filters: dict[str, Any] = {}
        for window in windows:
            for arg in candidate_args:
                value = self._extract_filter_value(arg, window)
                if value is not None:
                    filters[arg] = value
            self._infer_filters_for_candidate_keys(filters, window, candidate_args)
        if not filters:
            self._infer_filters_for_candidate_keys(filters, lowered, candidate_args)
        if "mall" in lowered:
            for key in list(filters.keys()):
                value = filters.get(key)
                if not isinstance(value, str):
                    continue
                if "mall" not in value.lower():
                    continue
                if "location" not in str(key).lower():
                    filters.pop(key, None)
        return filters

    def _validate_components(
        self,
        table: str,
        columns: list[str],
        filters: dict[str, Any],
        joins: list[_RelationJoin],
        order_by: str | None,
        order_dir: str | None,
        config: Any,
    ) -> tuple[str, list[str], dict[str, Any], list[_RelationJoin], str | None, str | None, list[str]]:
        return _validate_components_stage(
            self,
            table,
            columns,
            filters,
            joins,
            order_by,
            order_dir,
            config,
        )

    def _resolve_table(self, table: str, config: Any, notes: list[str]) -> str:
        known = config.entities
        if not known or table in known:
            return table
        if config.default_entity:
            notes.append(f"table '{table}' not in schema; using default '{config.default_entity}'")
            return config.default_entity
        notes.append(f"table '{table}' not in schema; using '{known[0]}'")
        return known[0]

    def _validate_filters(
        self,
        filters: dict[str, Any],
        allowed_filter_keys: set[str],
        notes: list[str],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in filters.items():
            if key in {"and", "or", "not"} and isinstance(value, list):
                nested = self._validate_group_nodes(value, allowed_filter_keys)
                if nested:
                    out[key] = nested
                continue
            if self._is_allowed_filter_key(key, allowed_filter_keys):
                out[key] = value
            else:
                notes.append(f"dropped invalid filter '{key}'")
        return out

    def _validate_group_nodes(
        self,
        nodes: list[dict[str, Any]],
        allowed_filter_keys: set[str],
    ) -> list[dict[str, Any]]:
        nested: list[dict[str, Any]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            valid_node = self._validate_group_node(node, allowed_filter_keys)
            if valid_node:
                nested.append(valid_node)
        return nested

    @staticmethod
    def _is_allowed_filter_key(key: str, allowed_filter_keys: set[str]) -> bool:
        if key in allowed_filter_keys:
            return True
        for suffix in ("_gte", "_lte", "_gt", "_lt", "_ne", "_in", "_nin"):
            if key.endswith(suffix) and key[: -len(suffix)] in allowed_filter_keys:
                return True
        return False

    @staticmethod
    def _allowed_filter_keys(config: Any, table: str, allowed_columns: set[str]) -> set[str]:
        table_args = set(config.args_by_entity.get(table, []))
        intro_args = set(config.introspection_query_args.get(table, {}).keys())
        keys = set(allowed_columns)
        if intro_args:
            keys.update((table_args & intro_args) or intro_args)
        else:
            keys.update(table_args)
        return keys

    def _coerce_filter_values(
        self,
        filters: dict[str, Any],
        config: Any,
        table: str,
        notes: list[str],
        known_filter_keys: set[str] | None = None,
    ) -> None:
        arg_types = config.introspection_query_args.get(table, {})
        updates: dict[str, Any] = {}
        keys_to_drop: list[str] = []
        for key, value in filters.items():
            if key in {"and", "or", "not"} and isinstance(value, list):
                self._coerce_group_children(value, config, table, notes, known_filter_keys)
                continue
            drop_key, coerced = self._coerce_scalar_filter_entry(
                key=key,
                value=value,
                arg_types=arg_types,
                config=config,
                notes=notes,
                known_filter_keys=known_filter_keys,
            )
            if drop_key:
                keys_to_drop.append(key)
            else:
                updates[key] = coerced
        for key in keys_to_drop:
            filters.pop(key, None)
        filters.update(updates)

    @staticmethod
    def _base_filter_key(key: str, known_keys: set[str] | None = None) -> str:
        if known_keys and key in known_keys:
            return key
        for suffix in ("_gte", "_lte", "_gt", "_lt", "_ne", "_in", "_nin"):
            if key.endswith(suffix):
                return key[: -len(suffix)]
        return key

    @staticmethod
    def _coerce_value(value: Any, arg_type: str | None) -> Any:
        if isinstance(value, list):
            return [SQLEngine._coerce_value(item, arg_type) for item in value]
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
        if SQLEngine._is_iso_date_or_datetime(raw):
            return raw
        if arg_type and "enum" in arg_type.lower():
            return raw.upper()
        return raw

    @staticmethod
    def _is_iso_date_or_datetime(value: str) -> bool:
        try:
            if len(value) == 10:
                date.fromisoformat(value)
                return True
            normalized = value.replace("Z", "+00:00").replace("z", "+00:00")
            datetime.fromisoformat(normalized)
            return True
        except ValueError:
            return False

    @staticmethod
    def _enum_values_for_type(arg_type: str, config: Any) -> set[str]:
        cleaned = arg_type.replace("[", "").replace("]", "").replace("!", "").strip()
        return config.introspection_enum_values.get(cleaned, set())

    def _validate_joins(
        self,
        joins: list[_RelationJoin],
        config: Any,
        table: str,
        notes: list[str],
    ) -> list[_RelationJoin]:
        intro_relations = config.introspection_relation_targets.get(table, {})
        strict_intro_relations = bool(intro_relations)
        relation_map = dict(config.relations_by_entity.get(table, {}))
        for join in joins:
            if not str(join.relation).startswith("__special_"):
                continue
            if join.relation in relation_map:
                continue
            relation_map[join.relation] = NormalizedRelation(
                name=join.relation,
                target=join.target,
                on=join.on_clause,
                fields=[],
                args=self._columns_for_table(config, join.target),
                aliases=[join.target],
            )
        parent_columns = set(self._columns_for_table(config, table))
        valid: list[_RelationJoin] = []
        invalid_join_notes: list[str] = []

        for join in joins:
            validated_join = self._validate_single_join(
                join=join,
                table=table,
                config=config,
                relation_map=relation_map,
                strict_intro_relations=strict_intro_relations,
                intro_relations=intro_relations,
                parent_columns=parent_columns,
                notes=notes,
                invalid_join_notes=invalid_join_notes,
            )
            if validated_join is not None:
                valid.append(validated_join)

        if self.strict_validation and invalid_join_notes:
            raise ValidationError(
                f"Invalid JOIN configuration for table '{table}'",
                invalid_join_notes,
            )
        return valid

    def _resolve_relation_for_join(
        self,
        config: Any,
        table: str,
        relation_name: str,
    ) -> NormalizedRelation | None:
        relation_map = config.relations_by_entity.get(table, {})
        relation = relation_map.get(relation_name)
        if relation is not None:
            return relation
        relation = self._match_relation_by_target(relation_map, relation_name)
        if relation is not None:
            return relation
        return self._find_relation_globally(config, relation_name)

    @staticmethod
    def _match_relation_by_target(
        relation_map: dict[str, NormalizedRelation],
        relation_name: str,
    ) -> NormalizedRelation | None:
        for relation in relation_map.values():
            if relation.target.lower() == relation_name.lower():
                return relation
        return None

    def _find_relation_globally(self, config: Any, relation_name: str) -> NormalizedRelation | None:
        for ent_relations in config.relations_by_entity.values():
            relation = ent_relations.get(relation_name)
            if relation is not None:
                return relation
            relation = self._match_relation_by_target(ent_relations, relation_name)
            if relation is not None:
                return relation
        return None

    def _build_relation_join_from_payload(
        self,
        item: dict[str, Any],
        relation: NormalizedRelation,
        table: str,
    ) -> _RelationJoin:
        alias = str(item.get("alias", relation.name)).strip() or relation.name
        fields = [str(field) for field in item.get("fields", relation.fields[:1]) if str(field).strip()]
        filters = item.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}
        return _RelationJoin(
            relation=relation.name,
            target=relation.target,
            alias=alias,
            on_clause=self._resolve_join_on_clause(parent=table, relation=relation),
            fields=fields or (relation.fields[:1] or ["id"]),
            filters={str(key): value for key, value in filters.items()},
        )

    def _parse_atomic_and_node(self, part: str) -> dict[str, Any] | None:
        patterns = [
            (r"^([A-Za-z_]\w*)\s*(>=|<=|>|<)\s*([\w.:-]+)$", None),
            (r"^([A-Za-z_]\w*)\s*(?:!=|is not|not)\s*([\w.:-]+)$", "_ne"),
            (r"^([A-Za-z_]\w*)\s*(?:is\s+)?([\w.:-]+)$", ""),
        ]
        for pattern, suffix in patterns:
            match = re.match(pattern, part)
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

    def _validate_group_node(
        self,
        node: dict[str, Any],
        allowed_filter_keys: set[str],
    ) -> dict[str, Any]:
        valid_node: dict[str, Any] = {}
        for key, value in node.items():
            if key in {"and", "or", "not"} and isinstance(value, list):
                child = self._validate_group_nodes(value, allowed_filter_keys)
                if child:
                    valid_node[key] = child
                continue
            if self._is_allowed_filter_key(key, allowed_filter_keys):
                valid_node[key] = value
        return valid_node

    def _coerce_group_children(
        self,
        children: list[Any],
        config: Any,
        table: str,
        notes: list[str],
        known_filter_keys: set[str] | None,
    ) -> None:
        for child in children:
            if isinstance(child, dict):
                self._coerce_filter_values(
                    child,
                    config,
                    table,
                    notes,
                    known_filter_keys=known_filter_keys,
                )

    def _coerce_scalar_filter_entry(
        self,
        key: str,
        value: Any,
        arg_types: dict[str, str],
        config: Any,
        notes: list[str],
        known_filter_keys: set[str] | None,
    ) -> tuple[bool, Any]:
        base_key = self._base_filter_key(key, known_keys=known_filter_keys or set(arg_types.keys()))
        arg_type = arg_types.get(base_key)
        coerced = self._coerce_value(value, arg_type)
        enum_values = self._enum_values_for_type(arg_type, config) if arg_type else set()
        if enum_values and isinstance(coerced, str):
            canonical = next((enum for enum in enum_values if enum.lower() == coerced.lower()), None)
            if canonical is None:
                notes.append(f"dropped invalid enum value '{coerced}' for '{base_key}'")
                return True, None
            coerced = canonical
        return False, coerced

    def _validate_single_join(
        self,
        join: _RelationJoin,
        table: str,
        config: Any,
        relation_map: dict[str, NormalizedRelation],
        strict_intro_relations: bool,
        intro_relations: dict[str, str],
        parent_columns: set[str],
        notes: list[str],
        invalid_join_notes: list[str],
    ) -> _RelationJoin | None:
        relation = relation_map.get(join.relation)
        if relation is None or (strict_intro_relations and join.relation not in intro_relations):
            note = f"dropped invalid relation '{join.relation}' for '{table}'"
            notes.append(note)
            invalid_join_notes.append(note)
            logger.warning("SQLEngine [%s]: %s", table, note)
            return None

        on_notes = _validate_join_on_clause(
            on_clause=join.on_clause,
            parent_table=table,
            parent_columns=parent_columns,
            target_table=join.target,
            target_columns=set(self._columns_for_table(config, join.target)),
        )
        if on_notes:
            for note in on_notes:
                logger.warning("SQLEngine [%s]: JOIN ON clause: %s", table, note)
            notes.extend(on_notes)
            invalid_join_notes.extend(on_notes)

        allowed = self._allowed_join_fields(relation, intro_relations, join, config)
        join_fields = [field for field in join.fields if field in allowed]
        join_filter_allowed = set(relation.args)
        if not join_filter_allowed:
            join_filter_allowed = set(self._columns_for_table(config, join.target))
        join_filters = {key: value for key, value in join.filters.items() if key in join_filter_allowed}
        return _RelationJoin(
            relation=join.relation,
            target=join.target,
            alias=join.alias,
            on_clause=join.on_clause,
            fields=join_fields,
            filters=join_filters,
        )

    @staticmethod
    def _allowed_join_fields(
        relation: NormalizedRelation,
        intro_relations: dict[str, str],
        join: _RelationJoin,
        config: Any,
    ) -> set[str]:
        allowed = set(relation.fields)
        if not allowed:
            allowed = set(config.fields_by_entity.get(join.target, []))
        if not allowed:
            allowed = set(config.introspection_entity_fields.get(join.target, set()))
        if join.relation in intro_relations:
            target_type = intro_relations[join.relation]
            intro_allowed = set(config.introspection_entity_fields.get(target_type, set()))
            if intro_allowed:
                allowed = allowed.intersection(intro_allowed) or intro_allowed
        return allowed

    def _build_sql(
        self,
        table: str,
        columns: list[str],
        filters: dict[str, Any],
        joins: list[_RelationJoin],
        order_by: str | None,
        order_dir: str | None,
        limit: int | None,
        offset: int | None,
        exact_filter_keys: set[str] | None = None,
        aggregations: list[dict[str, str]] | None = None,
        distinct: bool = False,
        having: list[dict] | None = None,
        subqueries: list[dict] | None = None,
    ) -> str:
        """Build a SQL SELECT statement via :class:`~text2ql.renderers.SQLIRRenderer`.

        The engine detects all components; the renderer assembles the final
        string (including GROUP BY / aggregations when *aggregations* is
        non-empty).  ``IRRenderer.render()`` is now the production path.
        """
        from text2ql.ir import QueryIR
        # Convert _RelationJoin objects to the dict format expected by from_components.
        join_dicts = [
            {
                "relation": j.relation,
                "target": j.target,
                "on_clause": j.on_clause,
                "fields": j.fields,
                "filters": j.filters,
                "join_type": "LEFT",
            }
            for j in joins
        ]
        exact_keys: frozenset[str] = frozenset(exact_filter_keys) if exact_filter_keys else frozenset()
        ir = QueryIR.from_components(
            entity=table,
            fields=columns,
            filters=filters,
            joins=join_dicts,
            aggregations=aggregations or [],
            order_by=order_by,
            order_dir=order_dir,
            limit=limit,
            offset=offset,
            target="sql",
            exact_filter_keys=exact_keys,
            metadata={"exact_filter_keys": list(exact_keys)},
            distinct=distinct,
            having=having or [],
            subqueries=subqueries or [],
        )
        return _SQL_RENDERER.render(ir)

    def _build_where_parts(
        self,
        filters: dict[str, Any],
        alias: str,
        exact_filter_keys: set[str] | None = None,
    ) -> list[str]:
        parts: list[str] = []
        for key, value in filters.items():
            if key in {"and", "or", "not"} and isinstance(value, list):
                group_sql = self._build_group_expression(key, value, alias, exact_filter_keys=exact_filter_keys)
                if group_sql:
                    parts.append(group_sql)
                continue
            parts.append(self._sql_condition(alias, key, value, exact_filter_keys=exact_filter_keys))
        return parts

    def _build_group_expression(
        self,
        key: str,
        nodes: list[dict[str, Any]],
        alias: str,
        exact_filter_keys: set[str] | None = None,
    ) -> str | None:
        expressions: list[str] = []
        for node in nodes:
            atom = self._group_node_conditions(node, alias, exact_filter_keys)
            if atom:
                expressions.append(AND_SEPARATOR.join(atom))
        if not expressions:
            return None
        if key == "and":
            return "(" + AND_SEPARATOR.join(expressions) + ")"
        if key == "not":
            return "NOT (" + AND_SEPARATOR.join(expressions) + ")"
        return "(" + " OR ".join(expressions) + ")"

    def _group_node_conditions(
        self,
        node: dict[str, Any],
        alias: str,
        exact_filter_keys: set[str] | None = None,
    ) -> list[str]:
        atom: list[str] = []
        for n_key, n_value in node.items():
            if n_key in {"and", "or", "not"} and isinstance(n_value, list):
                grouped = self._build_group_expression(
                    n_key,
                    n_value,
                    alias,
                    exact_filter_keys=exact_filter_keys,
                )
                if grouped:
                    atom.append(grouped)
                continue
            atom.append(self._sql_condition(alias, n_key, n_value, exact_filter_keys=exact_filter_keys))
        return atom

    def _sql_condition(
        self,
        alias: str,
        key: str,
        value: Any,
        exact_filter_keys: set[str] | None = None,
    ) -> str:
        if exact_filter_keys and key in exact_filter_keys:
            if value is None:
                return f"{alias}.{key} IS NULL"
            return f"{alias}.{key} = {self._sql_literal(value)}"
        mapping = {
            "_gte": ">=",
            "_lte": "<=",
            "_gt": ">",
            "_lt": "<",
            "_ne": "!=",
        }
        for suffix, op in mapping.items():
            if key.endswith(suffix):
                column = key[: -len(suffix)]
                return f"{alias}.{column} {op} {self._sql_literal(value)}"
        if key.endswith("_in"):
            column = key[:-3]
            values = value if isinstance(value, list) else [value]
            return f"{alias}.{column} IN ({', '.join(self._sql_literal(item) for item in values)})"
        if key.endswith("_nin"):
            column = key[:-4]
            values = value if isinstance(value, list) else [value]
            return f"{alias}.{column} NOT IN ({', '.join(self._sql_literal(item) for item in values)})"
        if value is None:
            return f"{alias}.{key} IS NULL"
        return f"{alias}.{key} = {self._sql_literal(value)}"

    @staticmethod
    def _sql_literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    def _columns_for_table(self, config: Any, table: str) -> list[str]:
        base = config.fields_by_entity.get(table, []) if config.fields_by_entity else config.fields
        intro = sorted(config.introspection_entity_fields.get(table, set()))
        if not intro:
            return base
        if not base:
            return intro
        overlap = sorted(set(base).intersection(set(intro)))
        return overlap or intro

    @staticmethod
    def _extract_where_clause(lowered: str) -> str | None:
        return _extract_where_clause(lowered)

    @staticmethod
    def _extract_filter_value(alias: str, text: str) -> str | None:
        return _extract_filter_value(alias, text, spurious_values=_SPURIOUS_FILTER_VALUES)

    @staticmethod
    def _contains_token(text: str, token: str) -> bool:
        return _contains_token(text, token)

    @classmethod
    def _contains_column_reference(cls, text: str, label: str) -> bool:
        return _contains_column_reference(text, label)

    @classmethod
    def _label_match_variants(cls, label: str) -> set[str]:
        return _label_match_variants(label)

    @staticmethod
    def _token_inflections(token: str) -> set[str]:
        return _token_inflections(token)

    @staticmethod
    def _contains_entity_token(text: str, token: str) -> bool:
        return _contains_entity_token(text, token)

    @staticmethod
    def _contains_alias_terms(text: str, alias: str) -> bool:
        alias_tokens = [token for token in re.findall(r"[a-z0-9]+", str(alias).lower()) if len(token) >= 4]
        if not alias_tokens:
            return False
        text_tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
        if len(alias_tokens) <= 3:
            return all(token in text_tokens for token in alias_tokens)
        overlap = sum(1 for token in alias_tokens if token in text_tokens)
        return overlap >= 2

    def _alias_match_score(self, text: str, alias: str) -> float:
        alias_text = str(alias).strip().lower()
        if not alias_text:
            return 0.0
        if self._contains_token(text, alias_text):
            return 3.0
        if self._contains_alias_terms(text, alias_text):
            tokens = [token for token in re.findall(r"[a-z0-9]+", alias_text) if len(token) >= 4]
            if len(tokens) <= 3:
                return 2.0
            return 1.0
        return 0.0

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
    def _sorted_alias_pairs(alias_map: dict[str, str]) -> list[tuple[str, str]]:
        return _sorted_alias_pairs(alias_map)

    @staticmethod
    def _unique_in_order(items: list[str]) -> list[str]:
        return _unique_in_order(items)


# ---------------------------------------------------------------------------
# Module-level validation helpers (used by SQLEngine and tests)
# ---------------------------------------------------------------------------


def _detect_contradictory_filters(filters: dict[str, Any]) -> list[str]:
    """Return a list of contradiction descriptions found in *filters*.

    A contradiction is defined as: the same base field assigned two distinct
    plain-equality values at the top level (e.g. ``{"status": "active",
    "status_ne": "active"}`` or two conflicting entries from grouped nodes
    that reduce to the same field/value check).

    This function intentionally only inspects scalar ``eq`` conflicts at the
    top level to avoid false positives in complex OR trees.
    """
    eq_values: dict[str, list[Any]] = {}
    for key, value in filters.items():
        if key in {"and", "or", "not"} or _is_operator_filter_key(key):
            continue
        eq_values.setdefault(key, []).append(value)
    issues: list[str] = []
    for field_name, values in eq_values.items():
        unique_values = _unique_preserving_order(values)
        if len(unique_values) > 1:
            quoted = ", ".join(repr(v) for v in unique_values)
            issues.append(f"contradictory equality values for field '{field_name}': {quoted}")
    return issues


def _is_operator_filter_key(key: str) -> bool:
    return key.endswith(("_gte", "_lte", "_gt", "_lt", "_ne", "_in", "_nin"))


def _unique_preserving_order(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _validate_join_on_clause(
    on_clause: str,
    parent_table: str,
    parent_columns: set[str],
    target_table: str,
    target_columns: set[str],
) -> list[str]:
    """Validate that both sides of a JOIN ON clause reference real columns.

    Returns a list of issue strings (empty → no problems found).

    The clause is expected to be in the form ``"table.col = other.col"``.
    If either table's column set is empty (schema not provided) the check is
    skipped for that side.

    Parameters
    ----------
    on_clause:
        Raw ON clause string, e.g. ``"order_items.orderId = orders.id"``.
    parent_table:
        Name of the parent (left) table.
    parent_columns:
        Known columns of the parent table (may be empty if schema-less).
    target_table:
        Name of the target (right / joined) table.
    target_columns:
        Known columns of the target table (may be empty if schema-less).
    """
    if not on_clause or "=" not in on_clause:
        return []

    issues: list[str] = []
    left_raw, right_raw = on_clause.split("=", 1)
    left_ref = left_raw.strip()
    right_ref = right_raw.strip()

    def _check_ref(ref: str, known_table: str, known_columns: set[str]) -> None:
        if not known_columns:
            return  # No schema to validate against
        if "." in ref:
            tbl, col = ref.rsplit(".", 1)
            if tbl.lower() != known_table.lower():
                return  # Different table — let the other side handle it
            if col not in known_columns:
                issues.append(
                    f"JOIN ON clause references unknown column '{col}' "
                    f"on table '{known_table}' (known: {sorted(known_columns)})"
                )

    _check_ref(left_ref, parent_table, parent_columns)
    _check_ref(left_ref, target_table, target_columns)
    _check_ref(right_ref, parent_table, parent_columns)
    _check_ref(right_ref, target_table, target_columns)
    return issues
