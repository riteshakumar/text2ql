import pytest

from text2ql.json_execution import execute_query_result_on_json
from text2ql.types import QueryResult

pytestmark = pytest.mark.unit


def test_execute_query_result_on_json_applies_entity_fields_and_filters() -> None:
    payload = {
        "accounts": [
            {
                "positions": [
                    {"symbol": "QQQ", "quantity": 100.104},
                    {"symbol": "BAC", "quantity": 202.146},
                ]
            }
        ]
    }
    result = QueryResult(
        query="",
        target="graphql",
        confidence=1.0,
        explanation="",
        metadata={
            "entity": "positions",
            "fields": ["symbol", "quantity"],
            "filters": {"symbol": "QQQ"},
        },
    )

    rows, note = execute_query_result_on_json(result, payload)

    assert note == ""
    assert rows == [{"symbol": "QQQ", "quantity": 100.104}]


def test_execute_query_result_on_json_supports_range_filters() -> None:
    payload = {
        "items": [
            {"price": 10.5, "name": "A"},
            {"price": 25.0, "name": "B"},
            {"price": 31.0, "name": "C"},
        ]
    }
    result = QueryResult(
        query="",
        target="graphql",
        confidence=1.0,
        explanation="",
        metadata={
            "entity": "items",
            "fields": ["name", "price"],
            "filters": {"price_gte": "20", "price_lte": "30"},
        },
    )

    rows, _ = execute_query_result_on_json(result, payload)

    assert rows == [{"name": "B", "price": 25.0}]


def test_execute_query_result_on_json_evaluates_count_aggregation() -> None:
    payload = {
        "positions": [
            {"symbol": "QQQ", "quantity": 100.104},
            {"symbol": "BAC", "quantity": 202.146},
        ]
    }
    result = QueryResult(
        query="",
        target="graphql",
        confidence=1.0,
        explanation="",
        metadata={
            "entity": "positions",
            "fields": ["symbol"],
            "filters": {},
            "aggregations": [{"function": "count", "field": ""}],
        },
    )

    rows, note = execute_query_result_on_json(result, payload)

    assert note == ""
    assert rows == [{"count": 2}]


def test_execute_query_result_on_json_supports_sql_metadata_and_operators() -> None:
    payload = {
        "portfolio_data": {
            "positions": [
                {"symbol": "QQQ", "quantity": 100.104, "status": "active"},
                {"symbol": "BAC", "quantity": 50.0, "status": "inactive"},
                {"symbol": "AAPL", "quantity": 10.0, "status": "active"},
            ]
        }
    }
    result = QueryResult(
        query="SELECT ...",
        target="sql",
        confidence=1.0,
        explanation="",
        metadata={
            "table": "positions",
            "columns": ["symbol", "quantity"],
            "filters": {"quantity_gt": 20, "status_ne": "inactive"},
            "limit": 1,
            "offset": 0,
        },
    )

    rows, note = execute_query_result_on_json(result, payload, root_key="portfolio_data")

    assert note == ""
    assert rows == [{"symbol": "QQQ", "quantity": 100.104}]
