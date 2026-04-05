import pytest

from text2ql.engines.graphql import GraphQLEngine
from text2ql.types import QueryRequest

pytestmark = pytest.mark.unit


def test_engine_generates_default_query_without_schema() -> None:
    engine = GraphQLEngine()

    result = engine.generate(QueryRequest(text="list users", target="graphql"))

    assert result.target == "graphql"
    assert "query GeneratedQuery" in result.query
    assert "user" in result.query


def test_engine_applies_dynamic_mapping_aliases() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show top 3 client records with mail state enabled",
        target="graphql",
        schema={"entities": ["customers"], "fields": ["id", "email", "status"]},
        mapping={
            "entities": {"client": "customers"},
            "fields": {"mail": "email"},
            "filters": {"state": "status"},
            "filter_values": {"status": {"enabled": "active"}},
        },
    )

    result = engine.generate(request)

    assert "customers(limit: 3, status: \"active\")" in result.query
    assert "email" in result.query


def test_engine_supports_legacy_schema_with_limit_and_status() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show top 5 customers with email status active",
        target="graphql",
        schema={"entities": ["customers"], "fields": ["id", "email", "status"]},
    )

    result = engine.generate(request)

    assert "customers(limit: 5, status: \"active\")" in result.query
    assert "email" in result.query


def test_engine_uses_default_fields_when_none_mentioned() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show clients",
        target="graphql",
        schema={
            "entities": [{"name": "customers", "aliases": ["clients"]}],
            "fields": ["id", "email", "status"],
            "default_entity": "customers",
            "default_fields": ["id", "email", "status"],
        },
    )

    result = engine.generate(request)

    assert "customers" in result.query
    assert "id" in result.query
    assert "email" in result.query
    assert "status" in result.query


def test_engine_generates_nested_query_for_latest_order_total() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show customers with latest order total",
        target="graphql",
        schema={
            "entities": ["customers"],
            "fields": {"customers": ["id", "email"]},
            "args": {"customers": ["limit", "status"]},
            "relations": {
                "customers": {
                    "orders": {
                        "target": "orders",
                        "fields": ["id", "total", "createdAt"],
                        "args": ["limit"],
                        "aliases": ["order"],
                    }
                }
            },
        },
    )

    result = engine.generate(request)

    assert "customers {" in result.query
    assert "orders(limit: 1)" in result.query
    assert "total" in result.query


def test_engine_validates_and_drops_invalid_fields_against_schema() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show customers with email and password",
        target="graphql",
        schema={
            "entities": ["customers"],
            "fields": {"customers": ["id", "email"]},
            "default_fields": ["id"],
        },
    )

    result = engine.generate(request)

    assert "email" in result.query
    assert "password" not in result.query
    notes = result.metadata.get("validation_notes", [])
    assert any("dropped invalid fields" in note for note in notes)


def test_engine_validates_and_drops_invalid_args_against_schema() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show top 3 customers with email status active",
        target="graphql",
        schema={
            "entities": ["customers"],
            "fields": {"customers": ["id", "email"]},
            "args": {"customers": ["limit"]},
        },
    )

    result = engine.generate(request)

    assert "customers(limit: 3)" in result.query
    assert "status:" not in result.query
    notes = result.metadata.get("validation_notes", [])
    assert any("dropped invalid args" in note for note in notes)


def test_engine_generates_aggregation_intents() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show customers with count and sum total",
        target="graphql",
        schema={
            "entities": ["customers"],
            "fields": {"customers": ["id", "email", "total"]},
        },
    )

    result = engine.generate(request)

    assert "count" in result.query
    assert 'sum(field: "total")' in result.query
    assert any(agg.get("function") == "count" for agg in result.metadata.get("aggregations", []))


def test_engine_generates_advanced_filters_between_in_and_group() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show customers price between 10 and 20 and status active and category in retail, wholesale",
        target="graphql",
        schema={
            "entities": ["customers"],
            "fields": {"customers": ["id", "email"]},
            "args": {
                "customers": ["price_gte", "price_lte", "status", "category_in", "and"]
            },
        },
    )

    result = engine.generate(request)

    assert "price_gte: 10" in result.query
    assert "price_lte: 20" in result.query
    assert 'category_in: ["retail", "wholesale"]' in result.query
    assert "and:" in result.query


def test_engine_uses_introspection_for_post_generation_validation() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show top 3 customers with email status active",
        target="graphql",
        schema={
            "introspection": {
                "query": {
                    "customers": {
                        "type": "[Customer]",
                        "args": {"limit": "Int"},
                    }
                },
                "types": {
                    "Customer": {
                        "fields": {"id": "ID", "email": "String"},
                    }
                },
            }
        },
    )

    result = engine.generate(request)

    assert "customers(limit: 3)" in result.query
    assert "status:" not in result.query
    notes = result.metadata.get("validation_notes", [])
    assert any("post-validation" in note for note in notes) or any(
        "dropped invalid args" in note for note in notes
    )
