"""Abstract Intermediate Representation (IR) for text2ql query generation.

The IR sits between natural-language parsing and target-language rendering.
Both GraphQL and SQL engines can express their output as a ``QueryIR``, and new
engines (Cypher, SPARQL, JQ, Jsonata…) only need to implement ``IRRenderer``
rather than re-implementing the full parsing stack.

Typical flow
------------
1. A ``QueryEngine`` parses a ``QueryRequest`` into a ``QueryIR`` via
   ``QueryEngine.parse_to_ir()``.
2. The engine (or a third-party renderer) calls ``IRRenderer.render(ir)`` to
   produce the final query string.
3. ``QueryIR.from_query_result()`` lets you reconstruct an IR from a finished
   ``QueryResult`` for inspection, testing, or cross-target comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Filter operators
# ---------------------------------------------------------------------------

#: Canonical operator tokens shared across all render targets.
FILTER_OPS = frozenset(
    {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "is_null", "range"}
)


@dataclass(slots=True)
class IRFilter:
    """A single filter predicate on a field.

    Parameters
    ----------
    key:
        The canonical field/column name (after alias resolution).
    value:
        Scalar, list (for ``in``/``nin``), or ``None`` (for ``is_null``).
    operator:
        One of the tokens in ``FILTER_OPS``.  Defaults to ``"eq"``.
    """

    key: str
    value: Any
    operator: str = "eq"

    def __post_init__(self) -> None:
        if self.operator not in FILTER_OPS:
            raise ValueError(
                f"Unknown filter operator '{self.operator}'. "
                f"Must be one of: {', '.join(sorted(FILTER_OPS))}"
            )


@dataclass(slots=True)
class IRJoin:
    """A JOIN between two tables (SQL) or a nested relation fetch (GraphQL).

    Parameters
    ----------
    relation:
        The relation name as declared in the schema.
    target:
        The target table/entity name.
    on_left:
        Left side of the ON clause, e.g. ``"orders.id"``.
    on_right:
        Right side of the ON clause, e.g. ``"order_items.orderId"``.
    fields:
        Columns/fields to select from the target.
    filters:
        Additional predicates scoped to the joined table.
    join_type:
        SQL join type token (``"LEFT"``, ``"INNER"``, ``"RIGHT"``).
        Ignored for GraphQL renderers.
    """

    relation: str
    target: str
    on_left: str
    on_right: str
    fields: list[str] = field(default_factory=list)
    filters: list[IRFilter] = field(default_factory=list)
    join_type: str = "LEFT"


@dataclass(slots=True)
class IRNested:
    """A GraphQL nested selection (sub-query within a parent entity).

    Parameters
    ----------
    relation:
        The relation field name on the parent entity.
    target:
        The target entity/type name.
    fields:
        Fields to select from the nested type.
    filters:
        Arguments/filters applied to the nested selection.
    """

    relation: str
    target: str
    fields: list[str] = field(default_factory=list)
    filters: list[IRFilter] = field(default_factory=list)


@dataclass(slots=True)
class IRAggregation:
    """A single aggregation expression (``COUNT(*)``, ``SUM(amount)``, …).

    Parameters
    ----------
    function:
        Aggregation function name in upper-case: ``COUNT``, ``SUM``, ``AVG``,
        ``MIN``, ``MAX``.
    field:
        Target field/column.  For ``COUNT(*)`` use ``"*"``.
    alias:
        Optional output alias (e.g. ``"total"``).
    """

    function: str
    field: str
    alias: str | None = None


# ---------------------------------------------------------------------------
# Top-level IR
# ---------------------------------------------------------------------------


@dataclass
class QueryIR:
    """The canonical intermediate representation of a parsed NL query.

    Consumers
    ---------
    - ``IRRenderer`` subclasses translate this to a concrete query string.
    - ``QueryIR.from_query_result()`` reconstructs it from a finished result
      for diffing, testing, or cross-engine comparison.
    - Future engines (Cypher, SPARQL, …) receive a ``QueryIR`` and only need
      to implement ``IRRenderer.render()``.

    Parameters
    ----------
    entity:
        Primary table/entity name.
    fields:
        Selected columns/fields.
    filters:
        Top-level filter predicates.
    joins:
        SQL-style JOIN descriptors.
    nested:
        GraphQL-style nested selections.
    aggregations:
        Aggregate expressions.
    group_filters:
        Grouped (AND/OR/NOT) filter trees represented as raw dicts.  These
        mirror the existing dict-based grouped filter format so that engines
        can handle them without a full IR migration.
    order_by:
        Field to order by.
    order_dir:
        ``"ASC"`` or ``"DESC"``.
    limit:
        Row limit.
    offset:
        Row offset.
    target:
        Intended render target (``"graphql"``, ``"sql"``, etc.).
    source_text:
        Original natural-language input.
    metadata:
        Arbitrary pass-through metadata from the originating engine.
    """

    entity: str
    fields: list[str] = field(default_factory=list)
    filters: list[IRFilter] = field(default_factory=list)
    joins: list[IRJoin] = field(default_factory=list)
    nested: list[IRNested] = field(default_factory=list)
    aggregations: list[IRAggregation] = field(default_factory=list)
    group_filters: dict[str, Any] = field(default_factory=dict)
    order_by: str | None = None
    order_dir: str | None = None
    limit: int | None = None
    offset: int | None = None
    target: str = "graphql"
    source_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_query_result(cls, result: "Any", source_text: str = "") -> "QueryIR":
        """Reconstruct a ``QueryIR`` from a finished ``QueryResult``.

        This is lossy for complex queries — grouped filters and nested
        selections are preserved as raw metadata rather than re-parsed.
        """
        meta = result.metadata if isinstance(result.metadata, dict) else {}
        entity = meta.get("entity") or meta.get("table") or ""

        raw_filters = meta.get("filters", {})
        flat_filters, group_filters = _split_filters(raw_filters)

        raw_joins: list[dict[str, Any]] = meta.get("joins", [])
        joins = [_join_from_dict(j) for j in raw_joins if isinstance(j, dict)]

        raw_nested: list[dict[str, Any]] = meta.get("nested", [])
        nested = [_nested_from_dict(n) for n in raw_nested if isinstance(n, dict)]

        raw_aggs: list[dict[str, Any]] = meta.get("aggregations", [])
        aggregations = [_agg_from_dict(a) for a in raw_aggs if isinstance(a, dict)]

        return cls(
            entity=entity,
            fields=list(meta.get("fields") or meta.get("columns") or []),
            filters=flat_filters,
            joins=joins,
            nested=nested,
            aggregations=aggregations,
            group_filters=group_filters,
            order_by=meta.get("order_by"),
            order_dir=meta.get("order_dir"),
            limit=meta.get("limit"),
            offset=meta.get("offset"),
            target=result.target,
            source_text=source_text,
            metadata=meta,
        )

    @classmethod
    def from_components(
        cls,
        *,
        entity: str,
        fields: list[str],
        filters: dict[str, Any],
        joins: list[dict[str, Any]] | None = None,
        nested: list[dict[str, Any]] | None = None,
        aggregations: list[dict[str, Any]] | None = None,
        order_by: str | None = None,
        order_dir: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        target: str = "graphql",
        source_text: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "QueryIR":
        """Build a ``QueryIR`` directly from engine component dicts.

        This is the bridge that lets existing engines create an IR without
        being fully refactored.  Engines call this at the end of
        ``generate()`` after they've resolved all their internal state.
        """
        flat_filters, group_filters = _split_filters(filters)
        ir_joins = [_join_from_dict(j) for j in (joins or []) if isinstance(j, dict)]
        ir_nested = [_nested_from_dict(n) for n in (nested or []) if isinstance(n, dict)]
        ir_aggs = [_agg_from_dict(a) for a in (aggregations or []) if isinstance(a, dict)]
        return cls(
            entity=entity,
            fields=list(fields),
            filters=flat_filters,
            joins=ir_joins,
            nested=ir_nested,
            aggregations=ir_aggs,
            group_filters=group_filters,
            order_by=order_by,
            order_dir=order_dir,
            limit=limit,
            offset=offset,
            target=target,
            source_text=source_text,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Renderer base class
# ---------------------------------------------------------------------------


class IRRenderer:
    """Base class for target-language renderers.

    Subclass this to add a new query language without touching any existing
    engine code.  Only ``render()`` is mandatory.

    Example
    -------
    .. code-block:: python

        class CypherRenderer(IRRenderer):
            def render(self, ir: QueryIR) -> str:
                label = ir.entity.capitalize()
                fields = ", ".join(f"n.{f}" for f in ir.fields)
                where_clauses = self._build_where(ir.filters)
                where = f" WHERE {where_clauses}" if where_clauses else ""
                return f"MATCH (n:{label}){where} RETURN {fields or 'n'}"

            def _build_where(self, filters):
                parts = []
                for f in filters:
                    if f.operator == "eq":
                        parts.append(f"n.{f.key} = {self._lit(f.value)}")
                    ...
                return " AND ".join(parts)
    """

    def render(self, ir: QueryIR) -> str:
        """Render ``ir`` to a query string in the target language."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement render(ir: QueryIR) -> str"
        )

    def render_many(self, irs: list[QueryIR]) -> list[str]:
        """Convenience: render a batch of IR objects."""
        return [self.render(ir) for ir in irs]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _split_filters(
    raw: dict[str, Any],
) -> tuple[list[IRFilter], dict[str, Any]]:
    """Separate flat scalar filters from grouped AND/OR/NOT trees.

    Returns
    -------
    flat_filters:
        ``IRFilter`` objects for simple key-op-value predicates.
    group_filters:
        Raw dict with ``"and"``/``"or"``/``"not"`` keys preserved for
        engines that handle them natively.
    """
    flat: list[IRFilter] = []
    groups: dict[str, Any] = {}

    _SUFFIX_OP = {
        "_gte": "gte",
        "_lte": "lte",
        "_gt": "gt",
        "_lt": "lt",
        "_ne": "ne",
        "_in": "in",
        "_nin": "nin",
    }

    for key, value in raw.items():
        if key in {"and", "or", "not"} and isinstance(value, list):
            groups[key] = value
            continue
        if value is None:
            flat.append(IRFilter(key=key, value=None, operator="is_null"))
            continue
        op = "eq"
        canonical_key = key
        for suffix, suffix_op in _SUFFIX_OP.items():
            if key.endswith(suffix):
                canonical_key = key[: -len(suffix)]
                op = suffix_op
                break
        flat.append(IRFilter(key=canonical_key, value=value, operator=op))

    return flat, groups


def _join_from_dict(d: dict[str, Any]) -> IRJoin:
    on_clause = d.get("on_clause", "")
    on_left, on_right = _parse_on_clause(on_clause)
    raw_filters = d.get("filters", {})
    flat_filters, _ = _split_filters(raw_filters if isinstance(raw_filters, dict) else {})
    return IRJoin(
        relation=str(d.get("relation", "")),
        target=str(d.get("target", "")),
        on_left=on_left,
        on_right=on_right,
        fields=list(d.get("fields") or []),
        filters=flat_filters,
        join_type=str(d.get("join_type", "LEFT")).upper(),
    )


def _nested_from_dict(d: dict[str, Any]) -> IRNested:
    raw_filters = d.get("filters", d.get("args", {}))
    flat_filters, _ = _split_filters(raw_filters if isinstance(raw_filters, dict) else {})
    return IRNested(
        relation=str(d.get("relation", d.get("name", ""))),
        target=str(d.get("target", d.get("entity", ""))),
        fields=list(d.get("fields") or []),
        filters=flat_filters,
    )


def _agg_from_dict(d: dict[str, Any]) -> IRAggregation:
    return IRAggregation(
        function=str(d.get("function", "COUNT")).upper(),
        field=str(d.get("field", "*")),
        alias=d.get("alias"),
    )


def _parse_on_clause(on_clause: str) -> tuple[str, str]:
    """Split ``"table.col = other.col"`` into ``(left, right)``."""
    if "=" in on_clause:
        parts = on_clause.split("=", 1)
        return parts[0].strip(), parts[1].strip()
    return on_clause.strip(), ""
