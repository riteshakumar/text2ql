from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from text2ql.constrained import ConstrainedOutputError, parse_sql_intent
from text2ql.prompting import build_sql_prompts, resolve_language, resolve_prompt_template
from text2ql.providers.base import LLMProvider
from text2ql.schema_config import NormalizedRelation, normalize_schema_config
from text2ql.types import QueryRequest, QueryResult

from .base import QueryEngine

_SPURIOUS_FILTER_VALUES = {"where", "with", "and", "or", "for", "of", "in", "is"}


@dataclass(slots=True)
class _RelationJoin:
    relation: str
    target: str
    alias: str
    on_clause: str
    fields: list[str]
    filters: dict[str, Any]


class SQLEngine(QueryEngine):
    """Deterministic SQL engine with schema validation and robust filter parsing."""

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
        lowered = prompt.lower()

        table = self._detect_table(lowered, config)
        columns = self._detect_columns(lowered, config, table)
        filters = self._detect_filters(lowered, config, table)
        joins = self._detect_joins(lowered, table, config)
        order_by, order_dir = self._detect_order(lowered, columns)
        limit, offset = self._detect_pagination(lowered)

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
        )
        return QueryResult(
            query=query,
            target="sql",
            confidence=0.62 if filters else 0.56,
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
                "order_by": order_by,
                "order_dir": order_dir,
                "limit": limit,
                "offset": offset,
                "mode": "deterministic",
                "llm_error": llm_error,
                "validation_notes": notes,
            },
        )

    def _generate_with_llm(
        self,
        prompt: str,
        config: Any,
        context: dict[str, Any],
    ) -> QueryResult | None:
        template = resolve_prompt_template(context)
        language = str(context.get("language", "english"))
        try:
            resolved_language = resolve_language(language)
        except ValueError:
            return None

        system_prompt, user_prompt = build_sql_prompts(
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
            intent = parse_sql_intent(raw, config, language=resolved_language)
        except ConstrainedOutputError:
            return None

        table = intent.table
        columns = intent.columns
        filters = dict(intent.filters)
        joins = self._materialize_llm_joins(intent.joins, config, table)
        order_by = intent.order_by
        order_dir = intent.order_dir
        limit = intent.limit
        offset = intent.offset

        table, columns, filters, joins, order_by, order_dir, notes = self._validate_components(
            table=table,
            columns=columns,
            filters=filters,
            joins=joins,
            order_by=order_by,
            order_dir=order_dir,
            config=config,
        )
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
        )
        return QueryResult(
            query=query,
            target="sql",
            confidence=intent.confidence,
            explanation=intent.explanation,
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
                "order_by": order_by,
                "order_dir": order_dir,
                "limit": limit,
                "offset": offset,
                "mode": "llm",
                "language": resolved_language,
                "raw_completion": raw,
                "validation_notes": notes,
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

    def _materialize_llm_joins(
        self,
        payload_joins: list[dict[str, Any]],
        config: Any,
        table: str,
    ) -> list[_RelationJoin]:
        relation_map = config.relations_by_entity.get(table, {})
        joins: list[_RelationJoin] = []
        for item in payload_joins:
            relation_name = str(item.get("relation", "")).strip()
            relation = relation_map.get(relation_name)
            if relation is None:
                continue
            alias = str(item.get("alias", relation.name)).strip() or relation.name
            fields = [str(field) for field in item.get("fields", relation.fields[:1]) if str(field).strip()]
            filters = item.get("filters", {})
            if not isinstance(filters, dict):
                filters = {}
            joins.append(
                _RelationJoin(
                    relation=relation.name,
                    target=relation.target,
                    alias=alias,
                    on_clause=self._build_join_on_clause(table, relation.target),
                    fields=fields or (relation.fields[:1] or ["id"]),
                    filters={str(key): value for key, value in filters.items()},
                )
            )
        return joins

    def _detect_table(self, lowered: str, config: Any) -> str:
        for alias, canonical in self._sorted_alias_pairs(config.entity_aliases):
            if self._contains_token(lowered, alias):
                return canonical
        for entity in config.entities:
            if self._contains_entity_token(lowered, entity.lower()):
                return entity
        if config.default_entity:
            return config.default_entity
        return (config.entities[0] if config.entities else "items")

    def _detect_columns(self, lowered: str, config: Any, table: str) -> list[str]:
        allowed = self._columns_for_table(config, table)
        selected: list[str] = []
        for column in allowed:
            if self._contains_token(lowered, column.lower()):
                selected.append(column)
        for alias, canonical in self._sorted_alias_pairs(config.field_aliases):
            if canonical in allowed and self._contains_token(lowered, alias):
                selected.append(canonical)
        selected = self._unique_in_order(selected)
        if selected:
            return selected
        if config.default_fields:
            defaults = [field for field in config.default_fields if field in allowed]
            if defaults:
                return defaults
        return allowed[:3] if allowed else ["id"]

    def _detect_filters(self, lowered: str, config: Any, table: str) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        where_clause = self._extract_where_clause(lowered) or lowered

        filter_key_aliases = {"status": "status"}
        filter_key_aliases.update(config.filter_key_aliases)
        for alias, canonical in self._sorted_alias_pairs(filter_key_aliases):
            value = self._extract_filter_value(alias, where_clause)
            if value is None:
                continue
            resolved = self._resolve_filter_key_for_table(config, table, canonical)
            mapped = config.filter_value_aliases.get(str(resolved).lower(), {}).get(value.lower(), value)
            filters[str(resolved)] = mapped

        for canonical, alias_map in config.filter_value_aliases.items():
            resolved = self._resolve_filter_key_for_table(config, table, canonical)
            if str(resolved) in filters or not isinstance(alias_map, dict):
                continue
            for alias, mapped_value in alias_map.items():
                if self._contains_token(lowered, str(alias).lower()):
                    filters[str(resolved)] = mapped_value
                    break

        for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*>=\s*([a-zA-Z0-9_.:-]+)\b", lowered):
            filters[f"{match.group(1)}_gte"] = match.group(2)
        for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*<=\s*([a-zA-Z0-9_.:-]+)\b", lowered):
            filters[f"{match.group(1)}_lte"] = match.group(2)
        for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*>\s*([a-zA-Z0-9_.:-]+)\b", lowered):
            filters[f"{match.group(1)}_gt"] = match.group(2)
        for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*<\s*([a-zA-Z0-9_.:-]+)\b", lowered):
            filters[f"{match.group(1)}_lt"] = match.group(2)
        for match in re.finditer(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:!=|is not|not)\s*([a-zA-Z0-9_.:-]+)\b",
            lowered,
        ):
            filters[f"{match.group(1)}_ne"] = match.group(2)

        for match in re.finditer(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+between\s+([0-9]{1,4}(?:-[0-9]{2}-[0-9]{2})?)\s+and\s+([0-9]{1,4}(?:-[0-9]{2}-[0-9]{2})?)\b",
            lowered,
        ):
            field = match.group(1)
            filters[f"{field}_gte"] = match.group(2)
            filters[f"{field}_lte"] = match.group(3)

        for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+([a-zA-Z0-9_,\s]+)", lowered):
            field = match.group(1)
            values = [
                token.strip()
                for token in re.split(r",|\s+or\s+|\s+and\s+", match.group(2))
                if token.strip()
            ]
            if values:
                filters[f"{field}_in"] = values

        grouped = self._parse_grouped_filters(lowered)
        if grouped:
            filters.update(grouped)
        return filters

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
        if " and " not in lowered and " or " not in lowered:
            return {}
        where_clause = self._strip_outer_parentheses(self._extract_where_clause(lowered) or lowered)
        or_parts = self._split_top_level(where_clause, "or")
        if len(or_parts) <= 1:
            and_nodes = self._parse_and_nodes(where_clause)
            return {"and": and_nodes} if len(and_nodes) > 1 else {}
        nodes: list[dict[str, Any]] = []
        for part in or_parts:
            and_nodes = self._parse_and_nodes(part)
            if len(and_nodes) == 1:
                nodes.append(and_nodes[0])
            elif len(and_nodes) > 1:
                nodes.append({"and": and_nodes})
        return {"or": nodes} if len(nodes) > 1 else {}

    def _parse_and_nodes(self, text: str) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        normalized = self._strip_outer_parentheses(text)
        for part in self._split_top_level(normalized, "and"):
            if part.startswith("where "):
                part = part[6:].strip()
            part = self._strip_outer_parentheses(part)
            in_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+([a-zA-Z0-9_,\s-]+)$", part)
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
            for pattern, suffix in [
                (r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(>=|<=|>|<)\s*([a-zA-Z0-9_.:-]+)$", None),
                (r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:!=|is not|not)\s*([a-zA-Z0-9_.:-]+)$", "_ne"),
                (r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:is\s+)?([a-zA-Z0-9_.:-]+)$", ""),
            ]:
                match = re.match(pattern, part)
                if not match:
                    continue
                if suffix is None:
                    field, operator, value = match.group(1), match.group(2), match.group(3)
                    op_suffix = {">=": "_gte", "<=": "_lte", ">": "_gt", "<": "_lt"}[operator]
                    nodes.append({f"{field}{op_suffix}": value})
                elif suffix == "_ne":
                    nodes.append({f"{match.group(1)}_ne": match.group(2)})
                else:
                    nodes.append({match.group(1): match.group(2)})
                break
        return nodes

    @staticmethod
    def _split_top_level(text: str, operator: str) -> list[str]:
        if not text:
            return []
        parts: list[str] = []
        depth = 0
        start = 0
        i = 0
        token = f" {operator} "
        token_len = len(token)
        while i < len(text):
            ch = text[i]
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if depth == 0 and text[i : i + token_len] == token:
                part = text[start:i].strip()
                if part:
                    parts.append(part)
                i += token_len
                start = i
                continue
            i += 1
        tail = text[start:].strip()
        if tail:
            parts.append(tail)
        return parts

    @staticmethod
    def _strip_outer_parentheses(text: str) -> str:
        candidate = text.strip()
        while candidate.startswith("(") and candidate.endswith(")"):
            depth = 0
            balanced = True
            for idx, ch in enumerate(candidate):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        balanced = False
                        break
                    if depth == 0 and idx != len(candidate) - 1:
                        balanced = False
                        break
            if not balanced or depth != 0:
                break
            candidate = candidate[1:-1].strip()
        return candidate

    def _detect_order(self, lowered: str, selected_columns: list[str]) -> tuple[str | None, str | None]:
        if "latest order" in lowered:
            return None, None
        if any(token in lowered for token in ("latest", "newest", "most recent")):
            return self._detect_order_field(lowered, selected_columns), "DESC"
        highest = re.search(r"\bhighest\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", lowered)
        if highest:
            return highest.group(1), "DESC"
        lowest = re.search(r"\blowest\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", lowered)
        if lowest:
            return lowest.group(1), "ASC"
        return None, None

    @staticmethod
    def _detect_order_field(lowered: str, selected_columns: list[str]) -> str:
        for candidate in ("createdAt", "updatedAt", "date", "timestamp"):
            if candidate.lower() in lowered:
                return candidate
        return (selected_columns[0] if selected_columns else "id")

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
        notes: list[str] = []
        table = self._resolve_table(table, config, notes)
        allowed_columns = set(self._columns_for_table(config, table))
        columns = [column for column in columns if column in allowed_columns] or list(allowed_columns)[:2] or ["id"]
        allowed_filter_keys = self._allowed_filter_keys(config, table, allowed_columns)
        filters = self._validate_filters(filters, allowed_filter_keys, notes)
        self._coerce_filter_values(filters, config, table, notes, known_filter_keys=allowed_filter_keys)

        if order_by and order_by not in allowed_columns:
            notes.append(f"dropped invalid orderBy '{order_by}' for '{table}'")
            order_by = None
            order_dir = None

        joins = self._validate_joins(joins, config, table, notes)
        return table, columns, filters, joins, order_by, order_dir, notes

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
                nested = []
                for node in value:
                    if not isinstance(node, dict):
                        continue
                    valid_node = {}
                    for n_key, n_val in node.items():
                        if self._is_allowed_filter_key(n_key, allowed_filter_keys):
                            valid_node[n_key] = n_val
                    if valid_node:
                        nested.append(valid_node)
                if nested:
                    out[key] = nested
                continue
            if self._is_allowed_filter_key(key, allowed_filter_keys):
                out[key] = value
            else:
                notes.append(f"dropped invalid filter '{key}'")
        return out

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
        for key, value in list(filters.items()):
            if key in {"and", "or", "not"} and isinstance(value, list):
                for child in value:
                    if isinstance(child, dict):
                        self._coerce_filter_values(
                            child,
                            config,
                            table,
                            notes,
                            known_filter_keys=known_filter_keys,
                        )
                continue
            base_key = self._base_filter_key(key, known_keys=known_filter_keys or set(arg_types.keys()))
            arg_type = arg_types.get(base_key)
            coerced = self._coerce_value(value, arg_type)
            enum_values = self._enum_values_for_type(arg_type, config) if arg_type else set()
            if enum_values and isinstance(coerced, str):
                canonical = next((enum for enum in enum_values if enum.lower() == coerced.lower()), None)
                if canonical is None:
                    notes.append(f"dropped invalid enum value '{coerced}' for '{base_key}'")
                    filters.pop(key, None)
                    continue
                coerced = canonical
            filters[key] = coerced

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
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[tT]\d{2}:\d{2}(?::\d{2})?(?:z|[+\-]\d{2}:\d{2})?)?", raw):
            return raw
        if arg_type and "enum" in arg_type.lower():
            return raw.upper()
        return raw

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
        valid: list[_RelationJoin] = []
        for join in joins:
            relation = relation_map.get(join.relation)
            if relation is None:
                notes.append(f"dropped invalid relation '{join.relation}' for '{table}'")
                continue
            if strict_intro_relations and join.relation not in intro_relations:
                notes.append(f"dropped invalid relation '{join.relation}' for '{table}'")
                continue
            allowed = set(relation.fields)
            if join.relation in intro_relations:
                target_type = intro_relations[join.relation]
                intro_allowed = set(config.introspection_entity_fields.get(target_type, set()))
                if intro_allowed:
                    allowed = allowed.intersection(intro_allowed) or intro_allowed
            join_fields = [field for field in join.fields if field in allowed] or (relation.fields[:1] or ["id"])
            join_filters = {
                key: value
                for key, value in join.filters.items()
                if key in set(relation.args)
            }
            valid.append(
                _RelationJoin(
                    relation=join.relation,
                    target=join.target,
                    alias=join.alias,
                    on_clause=join.on_clause,
                    fields=join_fields,
                    filters=join_filters,
                )
            )
        return valid

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
    ) -> str:
        select_columns = [f"{table}.{column}" for column in columns]
        join_sql: list[str] = []
        where_parts = self._build_where_parts(filters, table, exact_filter_keys=exact_filter_keys)

        for join in joins:
            join_sql.append(f"LEFT JOIN {join.target} {join.alias} ON {join.on_clause}")
            select_columns.extend([f"{join.alias}.{field} AS {join.alias}_{field}" for field in join.fields])
            where_parts.extend(self._build_where_parts(join.filters, join.alias))

        sql = f"SELECT {', '.join(select_columns)} FROM {table}"
        if join_sql:
            sql += " " + " ".join(join_sql)
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        if order_by and order_dir:
            sql += f" ORDER BY {table}.{order_by} {order_dir}"
        if limit is not None:
            sql += f" LIMIT {limit}"
        if offset is not None:
            sql += f" OFFSET {offset}"
        return sql + ";"

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
            atom = [
                self._sql_condition(alias, n_key, n_value, exact_filter_keys=exact_filter_keys)
                for n_key, n_value in node.items()
            ]
            if atom:
                expressions.append(" AND ".join(atom))
        if not expressions:
            return None
        if key == "and":
            return "(" + " AND ".join(expressions) + ")"
        if key == "not":
            return "NOT (" + " AND ".join(expressions) + ")"
        return "(" + " OR ".join(expressions) + ")"

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
        parts = lowered.split(" where ", maxsplit=1)
        if len(parts) == 2:
            return parts[1].strip()
        return None

    @staticmethod
    def _extract_filter_value(alias: str, text: str) -> str | None:
        key_pattern = re.escape(alias)
        patterns = [
            rf"\b{key_pattern}\b\s*(?:=|:)\s*([^\s,)\]]+)",
            rf"\b{key_pattern}\b\s+(?:is\s+|equals\s+|equal to\s+)?([^\s,)\]]+)",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if not matches:
                continue
            candidate = matches[-1].group(1).strip().strip('"').strip("'")
            if candidate.lower() in _SPURIOUS_FILTER_VALUES:
                continue
            return candidate
        return None

    @staticmethod
    def _contains_token(text: str, token: str) -> bool:
        return re.search(rf"\b{re.escape(token)}\b", text) is not None

    @staticmethod
    def _contains_entity_token(text: str, token: str) -> bool:
        if SQLEngine._contains_token(text, token):
            return True
        if token.endswith("s"):
            return False
        return SQLEngine._contains_token(text, f"{token}s")

    @staticmethod
    def _sorted_alias_pairs(alias_map: dict[str, str]) -> list[tuple[str, str]]:
        return sorted(alias_map.items(), key=lambda pair: len(pair[0]), reverse=True)

    @staticmethod
    def _unique_in_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            out.append(item)
            seen.add(item)
        return out
