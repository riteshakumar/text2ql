"""Concrete ``IRRenderer`` implementations for GraphQL and SQL.

Both renderers receive a :class:`~text2ql.ir.QueryIR` and produce a query
string.  They are the *only* place in the codebase where query strings are
assembled from IR — the engines now build a ``QueryIR`` and delegate all
string construction here.

Adding a new target language (Cypher, SPARQL, JQ, …) means subclassing
:class:`~text2ql.ir.IRRenderer` and implementing ``render()``; no existing
engine code needs to change.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from text2ql.ir import IRAggregation, IRFilter, IRJoin, IRNested, IRRenderer, QueryIR

# ---------------------------------------------------------------------------
# SQL identifier quoting helper
# ---------------------------------------------------------------------------


def _q(name: str) -> str:
    """Wrap *name* in double-quotes so it is safe to use as a SQL identifier."""
    return f'"{name}"'


# ---------------------------------------------------------------------------
# GraphQL renderer
# ---------------------------------------------------------------------------

_GRAPHQL_FILTER_ORDER: dict[str, int] = {
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


class GraphQLIRRenderer(IRRenderer):
    """Render a :class:`~text2ql.ir.QueryIR` as a GraphQL selection-set string.

    Example
    -------
    .. code-block:: python

        from text2ql import Text2QL
        from text2ql.renderers import GraphQLIRRenderer
        from text2ql.ir import QueryIR

        result = Text2QL().generate("show top 5 orders", target="graphql", ...)
        ir = QueryIR.from_query_result(result)
        print(GraphQLIRRenderer().render(ir))
    """

    def render(self, ir: QueryIR) -> str:
        """Produce a GraphQL query string from *ir*."""
        args = self._build_args(ir.filters, ir.group_filters)
        selection_lines: list[str] = list(ir.fields)

        for agg in ir.aggregations:
            selection_lines.append(self._render_aggregation(agg))

        for nested in ir.nested:
            selection_lines.append(self._render_nested(nested, indent=2))

        selection = "\n    ".join(selection_lines) if selection_lines else "id"
        return dedent(
            f"""
            {{
              {ir.entity}{args} {{
                {selection}
              }}
            }}
            """
        ).strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_args(
        self,
        filters: list[IRFilter],
        group_filters: dict[str, Any],
    ) -> str:
        parts: dict[str, Any] = {}
        for f in filters:
            key = _ir_filter_to_dict_key(f)
            parts[key] = f.value
        parts.update(group_filters)
        if not parts:
            return ""
        ordered = sorted(parts.items(), key=lambda kv: (_GRAPHQL_FILTER_ORDER.get(kv[0], 100), kv[0]))
        args_str = ", ".join(f"{k}: {self._format_arg(v)}" for k, v in ordered)
        return f"({args_str})"

    @staticmethod
    def _render_aggregation(agg: IRAggregation) -> str:
        fn = agg.function.lower()
        if fn == "count":
            return "count"
        return f'{fn}(field: "{agg.field}")'

    def _render_nested(self, nested: IRNested | dict[str, Any], indent: int) -> str:
        """Recursively render a nested selection node.

        *nested* may be an :class:`~text2ql.ir.IRNested` dataclass **or** the
        raw ``dict`` format used internally by the GraphQL engine (which may
        also carry a ``"nested"`` key for deeper levels).
        """
        if isinstance(nested, dict):
            relation = nested["relation"]
            node_filters: list[IRFilter] = []
            raw_filters = nested.get("filters", {})
            if isinstance(raw_filters, dict):
                from text2ql.ir import _split_filters  # local import avoids cycle
                node_filters, _ = _split_filters(raw_filters)
            node_args = self._build_args(node_filters, {})
            field_lines: list[str] = list(nested.get("fields", ["id"]))
            child_nodes: list[dict[str, Any]] = nested.get("nested", [])
        else:
            relation = nested.relation
            node_args = self._build_args(nested.filters, {})
            field_lines = list(nested.fields) or ["id"]
            child_nodes = list(nested.children)

        pad = " " * indent
        child_pad = " " * (indent + 2)

        for child in child_nodes:
            field_lines.append(self._render_nested(child, indent + 2))

        inner = f"\n{child_pad}".join(field_lines)
        return f"{relation}{node_args} {{\n{child_pad}{inner}\n{pad}}}"

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
            return "[" + ", ".join(GraphQLIRRenderer._format_arg(item) for item in value) + "]"
        if isinstance(value, dict):
            parts = [f"{k}: {GraphQLIRRenderer._format_arg(v)}" for k, v in value.items()]
            return "{ " + ", ".join(parts) + " }"
        return f'"{str(value)}"'


# ---------------------------------------------------------------------------
# SQL renderer
# ---------------------------------------------------------------------------


class SQLIRRenderer(IRRenderer):
    """Render a :class:`~text2ql.ir.QueryIR` as a SQL ``SELECT`` statement.

    Supports
    --------
    - Plain ``SELECT col1, col2``
    - Aggregations: ``SELECT COUNT(*), SUM(amount) AS total``
    - ``LEFT JOIN … ON …`` (from ``ir.joins``)
    - ``WHERE`` (flat and grouped AND/OR/NOT predicates)
    - ``GROUP BY`` (non-aggregated columns when aggregations are present)
    - ``ORDER BY … ASC|DESC``
    - ``LIMIT`` / ``OFFSET``

    Example
    -------
    .. code-block:: python

        from text2ql.renderers import SQLIRRenderer
        from text2ql.ir import QueryIR, IRAggregation

        ir = QueryIR.from_components(
            entity="orders",
            fields=["status"],
            filters={"status": "active"},
            aggregations=[{"function": "COUNT", "field": "*", "alias": "cnt"}],
            target="sql",
        )
        print(SQLIRRenderer().render(ir))
        # SELECT orders.status, COUNT(*) AS cnt
        # FROM orders WHERE orders.status = 'active'
        # GROUP BY orders.status;
    """

    def render(self, ir: QueryIR) -> str:
        """Produce a SQL SELECT statement from *ir*."""
        table = ir.entity
        exact_keys: frozenset[str] = frozenset(ir.metadata.get("exact_filter_keys", []))
        join_clauses: list[str] = []
        join_where_parts: list[str] = []
        join_select_cols: list[str] = []

        for join in ir.joins:
            join_clauses.append(
                f"{join.join_type} JOIN {_q(join.target)} {_q(join.target)} ON {join.on_left} = {join.on_right}"
            )
            join_select_cols.extend(
                f"{_q(join.target)}.{_q(f)} AS {join.target}_{f}" for f in join.fields
            )
            join_where_parts.extend(self._build_where_parts(join.filters, {}, join.target, exact_keys))

        base_cols = [f"{_q(table)}.{_q(col)}" for col in ir.fields]
        agg_cols = [self._render_agg_column(agg) for agg in ir.aggregations]

        select_cols = base_cols + join_select_cols + agg_cols or [f"{_q(table)}.*"]

        where_parts = self._build_where_parts(ir.filters, ir.group_filters, table, exact_keys)
        where_parts.extend(join_where_parts)

        sql = f"SELECT {', '.join(select_cols)} FROM {_q(table)}"
        if join_clauses:
            sql += " " + " ".join(join_clauses)
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        # GROUP BY is needed when mixing aggregate and non-aggregate columns.
        if ir.aggregations and ir.fields:
            group_cols = [f"{_q(table)}.{_q(col)}" for col in ir.fields]
            if join_select_cols:
                group_cols += join_select_cols
            sql += " GROUP BY " + ", ".join(group_cols)

        if ir.order_by and ir.order_dir:
            sql += f" ORDER BY {_q(table)}.{_q(ir.order_by)} {ir.order_dir}"
        if ir.limit is not None:
            sql += f" LIMIT {ir.limit}"
        if ir.offset is not None:
            sql += f" OFFSET {ir.offset}"

        return sql + ";"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_agg_column(agg: IRAggregation) -> str:
        field_expr = agg.field if agg.field == "*" else _q(agg.field)
        expr = f"{agg.function}({field_expr})"
        return f"{expr} AS {agg.alias}" if agg.alias else expr

    def _build_where_parts(
        self,
        filters: list[IRFilter],
        group_filters: dict[str, Any],
        alias: str,
        exact_keys: frozenset[str] = frozenset(),
    ) -> list[str]:
        parts: list[str] = []
        for f in filters:
            parts.append(self._ir_filter_condition(alias, f))
        for key, nodes in group_filters.items():
            if key in {"and", "or", "not"} and isinstance(nodes, list):
                group_expr = self._build_group_expression(key, nodes, alias, exact_keys)
                if group_expr:
                    parts.append(group_expr)
        return parts

    def _build_group_expression(
        self,
        key: str,
        nodes: list[dict[str, Any]],
        alias: str,
        exact_keys: frozenset[str] = frozenset(),
    ) -> str | None:
        expressions: list[str] = []
        for node in nodes:
            atom: list[str] = []
            for n_key, n_value in node.items():
                if n_key in {"and", "or", "not"} and isinstance(n_value, list):
                    child = self._build_group_expression(n_key, n_value, alias, exact_keys)
                    if child:
                        atom.append(child)
                else:
                    atom.append(self._dict_filter_condition(alias, n_key, n_value, exact_keys))
            if atom:
                expressions.append(" AND ".join(atom))
        if not expressions:
            return None
        if key == "and":
            return "(" + " AND ".join(expressions) + ")"
        if key == "not":
            return "NOT (" + " AND ".join(expressions) + ")"
        return "(" + " OR ".join(expressions) + ")"

    @staticmethod
    def _ir_filter_condition(alias: str, f: IRFilter) -> str:
        col = f"{_q(alias)}.{_q(f.key)}"
        if f.operator == "eq":
            if f.value is None:
                return f"{col} IS NULL"
            return f"{col} = {SQLIRRenderer._sql_literal(f.value)}"
        if f.operator == "ne":
            return f"{col} != {SQLIRRenderer._sql_literal(f.value)}"
        if f.operator == "gt":
            return f"{col} > {SQLIRRenderer._sql_literal(f.value)}"
        if f.operator == "gte":
            return f"{col} >= {SQLIRRenderer._sql_literal(f.value)}"
        if f.operator == "lt":
            return f"{col} < {SQLIRRenderer._sql_literal(f.value)}"
        if f.operator == "lte":
            return f"{col} <= {SQLIRRenderer._sql_literal(f.value)}"
        if f.operator == "in":
            vals = f.value if isinstance(f.value, list) else [f.value]
            return f"{col} IN ({', '.join(SQLIRRenderer._sql_literal(v) for v in vals)})"
        if f.operator == "nin":
            vals = f.value if isinstance(f.value, list) else [f.value]
            return f"{col} NOT IN ({', '.join(SQLIRRenderer._sql_literal(v) for v in vals)})"
        if f.operator == "is_null":
            return f"{col} IS NULL"
        # Fallback: equality
        return f"{col} = {SQLIRRenderer._sql_literal(f.value)}"

    @staticmethod
    def _dict_filter_condition(
        alias: str,
        key: str,
        value: Any,
        exact_keys: frozenset[str] = frozenset(),
    ) -> str:
        """Render a raw dict-style filter key (may carry ``_gte`` suffix etc.).

        When *key* is in *exact_keys* it is a real schema column name and its
        suffix must not be stripped (e.g. a column literally named
        ``shipped_ne`` should render as ``alias.shipped_ne = …``).
        """
        if key not in exact_keys:
            mapping = {"_gte": ">=", "_lte": "<=", "_gt": ">", "_lt": "<", "_ne": "!="}
            for suffix, op in mapping.items():
                if key.endswith(suffix):
                    col = key[: -len(suffix)]
                    return f"{_q(alias)}.{_q(col)} {op} {SQLIRRenderer._sql_literal(value)}"
            if key.endswith("_in"):
                col = key[:-3]
                vals = value if isinstance(value, list) else [value]
                return f"{_q(alias)}.{_q(col)} IN ({', '.join(SQLIRRenderer._sql_literal(v) for v in vals)})"
            if key.endswith("_nin"):
                col = key[:-4]
                vals = value if isinstance(value, list) else [value]
                return f"{_q(alias)}.{_q(col)} NOT IN ({', '.join(SQLIRRenderer._sql_literal(v) for v in vals)})"
        if value is None:
            return f"{_q(alias)}.{_q(key)} IS NULL"
        return f"{_q(alias)}.{_q(key)} = {SQLIRRenderer._sql_literal(value)}"

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ir_filter_to_dict_key(f: IRFilter) -> str:
    """Convert an ``IRFilter`` back to the ``key_suffix`` dict format for GraphQL args."""
    suffix_map = {
        "gte": "_gte",
        "lte": "_lte",
        "gt": "_gt",
        "lt": "_lt",
        "ne": "_ne",
        "in": "_in",
        "nin": "_nin",
    }
    suffix = suffix_map.get(f.operator, "")
    return f"{f.key}{suffix}"
