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


def test_engine_handles_how_many_ticker_owned_intent() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="how many qqq do I own",
        target="graphql",
        schema={
            "entities": ["positions"],
            "fields": {"positions": ["symbol", "quantity"]},
            "args": {"positions": ["symbol"]},
            "default_entity": "positions",
        },
        mapping={
            "filters": {"ticker": "symbol", "symbol": "symbol"},
            "filter_values": {"symbol": {"qqq": "QQQ"}},
        },
    )

    result = engine.generate(request)

    assert "positions(symbol: \"QQQ\")" in result.query
    assert "quantity" in result.query
    assert result.metadata.get("entity") == "positions"
    assert result.metadata.get("filters", {}).get("symbol") == "QQQ"


def test_engine_prefers_where_clause_filter_match_over_earlier_mentions() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show positions with symbol quantity where symbol qqq",
        target="graphql",
        schema={
            "entities": ["positions"],
            "fields": {"positions": ["symbol", "quantity"]},
            "args": {"positions": ["symbol"]},
        },
        mapping={
            "filters": {"symbol": "symbol"},
            "filter_values": {"symbol": {"qqq": "QQQ"}},
        },
    )

    result = engine.generate(request)

    assert "positions(symbol: \"QQQ\")" in result.query
    assert result.metadata.get("filters", {}).get("symbol") == "QQQ"


def test_engine_does_not_infer_quantity_filter_from_list_symbols_phrase() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="list symbols and quantity for all positions",
        target="graphql",
        schema={
            "entities": ["positions"],
            "fields": {"positions": ["symbol", "quantity"]},
            "args": {"positions": ["quantity"]},
            "default_entity": "positions",
            "default_fields": ["symbol", "quantity"],
        },
    )

    result = engine.generate(request)

    assert "positions(quantity:" not in result.query
    assert "symbol" in result.query
    assert "quantity" in result.query
    assert result.metadata.get("filters", {}).get("quantity") is None


def test_engine_drops_unknown_filter_when_args_are_auto_discovered() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show accountSummary status active",
        target="graphql",
        schema={
            "entities": ["accountSummary"],
            "fields": {"accountSummary": ["isPartialBalance", "asOfDateTime"]},
            "default_entity": "accountSummary",
            "default_fields": ["isPartialBalance"],
        },
    )

    result = engine.generate(request)

    assert "accountSummary(" not in result.query
    assert "status:" not in result.query
    assert "status" not in result.metadata.get("filters", {})


def test_engine_drops_spurious_entity_alias_filter_value_where() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show accountSummary where status is X21985452",
        target="graphql",
        schema={
            "entities": ["accountSummary"],
            "fields": {"accountSummary": ["isPartialBalance", "acctNum"]},
            "default_entity": "accountSummary",
            "default_fields": ["isPartialBalance"],
        },
        mapping={"filters": {"acctNum": "acctNum"}},
    )

    result = engine.generate(request)

    assert "accountSummary(" not in result.query
    assert "status:" not in result.query
    assert "accountSummary:" not in result.query
    assert result.metadata.get("filters") == {}


def test_engine_generalizes_how_many_owned_for_ticker_field() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="how many tsla do I own",
        target="graphql",
        schema={
            "entities": ["holdings"],
            "fields": {"holdings": ["ticker", "shares"]},
            "args": {"holdings": ["ticker"]},
        },
    )

    result = engine.generate(request)

    assert "holdings(ticker: \"TSLA\")" in result.query
    assert "shares" in result.query


def test_engine_maps_how_many_positions_to_count_aggregation() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="how many positions do i have",
        target="graphql",
        schema={
            "entities": ["positions"],
            "fields": {"positions": ["symbol", "quantity"]},
            "args": {"positions": ["limit"]},
        },
    )

    result = engine.generate(request)

    assert any(agg.get("function") == "count" for agg in result.metadata.get("aggregations", []))


def test_engine_supports_ordering_and_pagination_primitives() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show customers highest total first 5 offset 10",
        target="graphql",
        schema={
            "entities": ["customers"],
            "fields": {"customers": ["id", "total"]},
            "args": {"customers": ["first", "offset", "orderBy", "orderDirection"]},
        },
    )

    result = engine.generate(request)

    assert "first: 5" in result.query
    assert "offset: 10" in result.query
    assert 'orderBy: "total"' in result.query
    assert 'orderDirection: "DESC"' in result.query


def test_engine_supports_negation_and_comparison_filters() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show products where price >= 10 and status != inactive",
        target="graphql",
        schema={
            "entities": ["products"],
            "fields": {"products": ["id", "price", "status"]},
            "args": {"products": ["price_gte", "status_ne", "and"]},
        },
    )

    result = engine.generate(request)

    assert "price_gte: 10" in result.query
    assert 'status_ne: "inactive"' in result.query


def test_engine_coerces_typed_filter_values_and_validates_enum() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="show orders where status active and shipped_ne false and cursor null",
        target="graphql",
        schema={
            "entities": ["orders"],
            "fields": {"orders": ["id"]},
            "args": {"orders": ["status", "shipped_ne", "cursor"]},
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

    assert 'status: "ACTIVE"' in result.query
    assert "shipped_ne: false" in result.query
    assert "cursor: null" in result.query


def test_engine_uses_semantic_field_match_for_metric_prompt() -> None:
    engine = GraphQLEngine()
    request = QueryRequest(
        text="what is my total market value",
        target="graphql",
        schema={
            "entities": ["accounts", "gainLossBalanceDetail"],
            "fields": {
                "accounts": ["acctNum", "acctName"],
                "gainLossBalanceDetail": ["totalMarketVal", "totalGainLoss"],
            },
            "default_entity": "accounts",
            "default_fields": ["acctNum"],
        },
    )

    result = engine.generate(request)

    assert result.metadata.get("entity") == "gainLossBalanceDetail"
    assert "totalMarketVal" in result.query
