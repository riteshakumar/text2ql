import pytest

from text2ql.engines.sql import SQLEngine
from text2ql.types import QueryRequest

pytestmark = pytest.mark.unit


def test_sql_engine_generates_basic_select_with_alias_mapping() -> None:
    engine = SQLEngine()
    request = QueryRequest(
        text="show top 5 client records with mail state enabled",
        target="sql",
        schema={"entities": ["customers"], "fields": {"customers": ["id", "email", "status"]}},
        mapping={
            "entities": {"client": "customers"},
            "fields": {"mail": "email"},
            "filters": {"state": "status"},
            "filter_values": {"status": {"enabled": "active"}},
        },
    )

    result = engine.generate(request)

    assert result.target == "sql"
    assert "FROM customers" in result.query
    assert "customers.email" in result.query
    assert "LIMIT 5" in result.query


def test_sql_engine_supports_ordering_and_pagination() -> None:
    engine = SQLEngine()
    request = QueryRequest(
        text="show customers highest total first 5 offset 10",
        target="sql",
        schema={"entities": ["customers"], "fields": {"customers": ["id", "total"]}},
    )

    result = engine.generate(request)

    assert "ORDER BY customers.total DESC" in result.query
    assert "LIMIT 5" in result.query
    assert "OFFSET 10" in result.query


def test_sql_engine_supports_negation_comparison_and_grouped_filters() -> None:
    engine = SQLEngine()
    request = QueryRequest(
        text="show products where price >= 10 and status != inactive or category in retail, wholesale",
        target="sql",
        schema={"entities": ["products"], "fields": {"products": ["price", "status", "category"]}},
    )

    result = engine.generate(request)

    assert "products.price >= 10" in result.query
    assert "products.status !=" in result.query
    assert "IN ('retail', 'wholesale')" in result.query
    assert "WHERE" in result.query


def test_sql_engine_coerces_enum_boolean_and_null() -> None:
    engine = SQLEngine()
    request = QueryRequest(
        text="show orders where status active and shipped_ne false and cursor null",
        target="sql",
        schema={
            "entities": ["orders"],
            "fields": {"orders": ["id", "status", "shipped_ne", "cursor"]},
            "introspection": {
                "query": {
                    "orders": {
                        "type": "[Order]",
                        "args": {
                            "status": "OrderStatus",
                            "shipped_ne": "Boolean",
                            "cursor": "String",
                        },
                    }
                },
                "types": {
                    "Order": {"fields": {"id": "ID"}},
                    "OrderStatus": {"enumValues": ["ACTIVE", "CANCELLED"]},
                },
            },
        },
    )

    result = engine.generate(request)

    assert "orders.status = 'ACTIVE'" in result.query
    assert "orders.shipped_ne = FALSE" in result.query
    assert "orders.cursor IS NULL" in result.query


def test_sql_engine_supports_relation_join_with_local_filters() -> None:
    engine = SQLEngine()
    request = QueryRequest(
        text="show customers with orders status shipped",
        target="sql",
        schema={
            "entities": ["customers"],
            "fields": {"customers": ["id", "email"]},
            "relations": {
                "customers": {
                    "orders": {
                        "target": "orders",
                        "fields": ["id", "status", "total"],
                        "args": ["status"],
                        "aliases": ["order"],
                    }
                }
            },
        },
    )

    result = engine.generate(request)

    assert "LEFT JOIN orders orders ON orders.customerId = customers.id" in result.query
    assert "orders.status = 'shipped'" in result.query


def test_sql_engine_parses_grouped_filters_with_parentheses_precedence() -> None:
    engine = SQLEngine()
    request = QueryRequest(
        text="show products where (status active and price >= 10) or category in retail, wholesale",
        target="sql",
        schema={"entities": ["products"], "fields": {"products": ["price", "status", "category"]}},
    )

    result = engine.generate(request)

    assert "products.status = 'active'" in result.query
    assert "products.price >= 10" in result.query
    assert "products.category IN ('retail', 'wholesale')" in result.query
    assert " OR " in result.query


def test_sql_engine_rejects_schema_relation_not_in_introspection() -> None:
    engine = SQLEngine()
    request = QueryRequest(
        text="show customers with orders status shipped",
        target="sql",
        schema={
            "entities": ["customers"],
            "fields": {"customers": ["id", "email"]},
            "relations": {
                "customers": {
                    "orders": {
                        "target": "orders",
                        "fields": ["id", "status", "total"],
                        "args": ["status"],
                        "aliases": ["order"],
                    }
                }
            },
            "introspection": {
                "query": {"customers": {"type": "[Customer]", "args": {}}},
                "types": {
                    "Customer": {"fields": {"id": "ID", "email": "String", "profile": "Profile"}},
                    "Order": {"fields": {"id": "ID", "status": "String", "total": "Float"}},
                    "Profile": {"fields": {"id": "ID"}},
                },
            },
        },
    )

    result = engine.generate(request)

    assert "LEFT JOIN orders" not in result.query
    notes = result.metadata.get("validation_notes", [])
    assert any("dropped invalid relation 'orders'" in note for note in notes)


def test_sql_engine_extract_filter_value_ignores_spurious_token() -> None:
    assert SQLEngine._extract_filter_value("status", "show account where status is") is None
