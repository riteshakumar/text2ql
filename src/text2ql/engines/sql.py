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
        table, columns, filters = self._reconcile_owned_asset_intent(
            prompt=prompt,
            table=table,
            columns=columns,
            filters=filters,
            config=config,
        )
        joins = self._detect_joins(lowered, table, config)
        order_by, order_dir = self._detect_order(
            lowered,
            columns,
            self._columns_for_table(config, table),
        )
        limit, offset = self._detect_pagination(lowered)
        aggregations = self._detect_aggregations(lowered)
        count_only_intent = any(str(agg.get("function", "")).upper() == "COUNT" for agg in aggregations) and "how many" in lowered
        if count_only_intent:
            columns = []

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
        if count_only_intent:
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
        offset_match = re.search(r"(?:offset|skip)\s+(\d+)", lowered)
        if offset_match:
            offset = int(offset_match.group(1))
        after_match = re.search(r"\bafter\s+(\d+)\b", lowered)
        if after_match and offset is None:
            offset = int(after_match.group(1))
        return limit, offset

    def _detect_aggregations(self, lowered: str) -> list[dict[str, str]]:
        """Detect aggregate expressions in *lowered* query text.

        Returns a list of dicts compatible with ``QueryIR.from_components()``
        and :class:`~text2ql.ir.IRAggregation`.
        """
        aggregations: list[dict[str, str]] = []
        explicit_count = re.search(r"\bcount\b", lowered) is not None
        how_many = re.search(r"\bhow many\b", lowered) is not None
        owned_asset_intent = self._detect_owned_asset(lowered) is not None
        if explicit_count or (how_many and not owned_asset_intent):
            aggregations.append({"function": "COUNT", "field": "*", "alias": "count"})
            return aggregations  # count overrides other aggs — keep it simple
        agg_patterns = [
            (r"\bsum\s+(?:of\s+)?([a-zA-Z_]\w*)\b", "SUM"),
            (r"\bavg(?:erage)?\s+(?:of\s+)?([a-zA-Z_]\w*)\b", "AVG"),
            (r"\bmin(?:imum)?\s+(?:of\s+)?([a-zA-Z_]\w*)\b", "MIN"),
            (r"\bmax(?:imum)?\s+(?:of\s+)?([a-zA-Z_]\w*)\b", "MAX"),
        ]
        for pattern, fn in agg_patterns:
            m = re.search(pattern, lowered)
            if m:
                field = m.group(1)
                aggregations.append({"function": fn, "field": field, "alias": f"{fn.lower()}_{field}"})
        return aggregations

    def _detect_joins(self, lowered: str, table: str, config: Any) -> list[_RelationJoin]:
        relation_map = config.relations_by_entity.get(table, {})
        joins: list[_RelationJoin] = []
        for relation in relation_map.values():
            aliases = [relation.name, relation.target, *relation.aliases]
            if not any(self._contains_entity_token(lowered, alias.lower()) for alias in aliases):
                continue
            alias = relation.name
            on_clause = self._build_join_on_clause(parent=table, child=relation.target)
            fields = relation.fields[:2] if relation.fields else ["id"]
            local_filters = self._detect_relation_local_filters(lowered, relation)
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
        return joins

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
    ) -> dict[str, Any]:
        if not relation.args:
            return {}
        aliases = [relation.name, relation.target, *relation.aliases]
        windows: list[str] = []
        for alias in aliases:
            pattern = rf"\b{re.escape(alias.lower())}\b(.{{0,80}})"
            for match in re.finditer(pattern, lowered):
                windows.append(match.group(1))
        if not windows:
            return {}
        filters: dict[str, Any] = {}
        for window in windows:
            for arg in relation.args:
                value = self._extract_filter_value(arg, window)
                if value is not None:
                    filters[arg] = value
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
        relation_map = config.relations_by_entity.get(table, {})
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
            on_clause=self._build_join_on_clause(table, relation.target),
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
        join_fields = [field for field in join.fields if field in allowed] or (relation.fields[:1] or ["id"])
        join_filters = {key: value for key, value in join.filters.items() if key in set(relation.args)}
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
