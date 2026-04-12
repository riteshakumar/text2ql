"""End-to-end tests for the concrete IRRenderer implementations.

These tests verify the full pipeline from IR construction through rendering,
covering aggregations, JOIN, nested selections with multi-hop children, and
the ``QueryIR.from_query_result()`` round-trip helper.
"""

import pytest

from text2ql import (
    GraphQLIRRenderer,
    IRAggregation,
    IRFilter,
    IRNested,
    QueryIR,
    SQLIRRenderer,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# SQLIRRenderer
# ---------------------------------------------------------------------------


def test_sql_ir_renderer_basic_select_e2e() -> None:
    ir = QueryIR.from_components(
        entity="orders",
        fields=["id", "status"],
        filters={"status": "active"},
        target="sql",
    )
    sql = SQLIRRenderer().render(ir)

    assert sql.startswith("SELECT")
    assert '"orders"."id"' in sql
    assert '"orders"."status"' in sql
    assert 'FROM "orders"' in sql
    assert '"orders"."status" = \'active\'' in sql
    assert sql.endswith(";")


def test_sql_ir_renderer_aggregation_count_e2e() -> None:
    ir = QueryIR.from_components(
        entity="orders",
        fields=["status"],
        filters={},
        aggregations=[{"function": "COUNT", "field": "*", "alias": "total"}],
        target="sql",
    )
    sql = SQLIRRenderer().render(ir)

    assert "COUNT(*) AS total" in sql
    assert 'FROM "orders"' in sql
    assert 'GROUP BY "orders"."status"' in sql


def test_sql_ir_renderer_aggregation_sum_e2e() -> None:
    ir = QueryIR.from_components(
        entity="orders",
        fields=["status"],
        filters={"status": "shipped"},
        aggregations=[{"function": "SUM", "field": "amount", "alias": "total_amount"}],
        target="sql",
    )
    sql = SQLIRRenderer().render(ir)

    assert "SUM(" in sql and "amount" in sql and "AS total_amount" in sql
    assert '"orders"."status" = \'shipped\'' in sql
    assert "GROUP BY" in sql


def test_sql_ir_renderer_order_limit_offset_e2e() -> None:
    ir = QueryIR.from_components(
        entity="products",
        fields=["name", "price"],
        filters={},
        order_by="price",
        order_dir="DESC",
        limit=10,
        offset=20,
        target="sql",
    )
    sql = SQLIRRenderer().render(ir)

    assert 'ORDER BY "products"."price" DESC' in sql
    assert "LIMIT 10" in sql
    assert "OFFSET 20" in sql


def test_sql_ir_renderer_join_e2e() -> None:
    ir = QueryIR.from_components(
        entity="orders",
        fields=["id", "status"],
        filters={},
        joins=[
            {
                "relation": "customers",
                "target": "customers",
                "on_clause": "customers.orderId = orders.id",
                "fields": ["name", "email"],
                "filters": {},
            }
        ],
        target="sql",
    )
    sql = SQLIRRenderer().render(ir)

    assert 'LEFT JOIN "customers"' in sql
    assert '"customers"."name" AS customers_name' in sql
    assert '"customers"."email" AS customers_email' in sql


# ---------------------------------------------------------------------------
# GraphQLIRRenderer
# ---------------------------------------------------------------------------


def test_graphql_ir_renderer_basic_e2e() -> None:
    ir = QueryIR.from_components(
        entity="users",
        fields=["id", "name", "email"],
        filters={"limit": 5},
        target="graphql",
    )
    gql = GraphQLIRRenderer().render(ir)

    assert "users(limit: 5)" in gql
    assert "id" in gql
    assert "name" in gql
    assert "email" in gql


def test_graphql_ir_renderer_nested_children_e2e() -> None:
    """IRNested.children are rendered as nested sub-selections."""
    child = IRNested(
        relation="variants",
        target="productVariants",
        fields=["id", "sku"],
    )
    parent = IRNested(
        relation="items",
        target="orderItems",
        fields=["id", "quantity"],
        children=[child],
    )
    ir = QueryIR(
        entity="orders",
        fields=["id", "status"],
        nested=[parent],
        target="graphql",
    )
    gql = GraphQLIRRenderer().render(ir)

    assert "items" in gql
    assert "variants" in gql
    assert "sku" in gql
    # variants block must appear after items block (nested inside it).
    assert gql.index("items") < gql.index("variants")


def test_graphql_ir_renderer_aggregation_e2e() -> None:
    ir = QueryIR.from_components(
        entity="orders",
        fields=["id"],
        filters={"limit": 100},
        aggregations=[{"function": "COUNT", "field": "", "alias": ""}],
        target="graphql",
    )
    gql = GraphQLIRRenderer().render(ir)

    assert "orders" in gql
    assert "count" in gql


# ---------------------------------------------------------------------------
# Round-trip: QueryIR.from_query_result
# ---------------------------------------------------------------------------


def test_query_ir_round_trip_from_result_e2e() -> None:
    """QueryIR reconstructed from a QueryResult retains entity, filters, and fields."""
    from text2ql import QueryResult

    result = QueryResult(
        query="SELECT orders.status FROM orders WHERE orders.status = 'active';",
        target="sql",
        confidence=0.9,
        explanation="show active orders",
        metadata={
            "entity": "orders",
            "fields": ["status"],
            "filters": {"status": "active"},
            "order_by": None,
            "order_dir": None,
            "limit": None,
            "offset": None,
        },
    )
    ir = QueryIR.from_query_result(result, source_text="show active orders")

    assert ir.entity == "orders"
    assert ir.fields == ["status"]
    assert len(ir.filters) == 1
    assert ir.filters[0].key == "status"
    assert ir.filters[0].value == "active"
    assert ir.target == "sql"
    assert ir.source_text == "show active orders"

    # Re-render should produce a valid SQL statement.
    sql = SQLIRRenderer().render(ir)
    assert 'FROM "orders"' in sql
    assert '"orders"."status" = \'active\'' in sql
