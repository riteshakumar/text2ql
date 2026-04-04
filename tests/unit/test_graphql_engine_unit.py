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
