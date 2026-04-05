import pytest

from text2ql.mapping import generate_hybrid_mapping

pytestmark = pytest.mark.unit


def test_generate_hybrid_mapping_builds_schema_and_data_driven_aliases() -> None:
    schema = {
        "entities": ["positions"],
        "fields": {"positions": ["symbol", "quantity", "tradeMarket"]},
    }
    payload = {
        "positions": [
            {"symbol": "QQQ", "quantity": 100.104, "tradeMarket": "NASDAQ"},
            {"symbol": "BAC", "quantity": 200.0, "tradeMarket": "NYSE"},
        ]
    }

    mapping = generate_hybrid_mapping(schema_payload=schema, data_payload=payload)

    assert mapping["filters"]["ticker"] == "symbol"
    assert mapping["filters"]["symbol"] == "symbol"
    assert mapping["filter_values"]["symbol"]["qqq"] == "QQQ"
    assert mapping["filter_values"]["tradeMarket"]["nasdaq"] == "NASDAQ"
    assert mapping["metadata"]["overrides_applied"] is False


def test_generate_hybrid_mapping_applies_overrides_with_provenance() -> None:
    schema = {"entities": ["positions"], "fields": {"positions": ["symbol", "securityType"]}}
    payload = {"positions": [{"symbol": "QQQ", "securityType": "Equity"}]}
    overrides = {
        "filters": {"asset": "symbol"},
        "filter_values": {"securityType": {"core": "Core"}},
    }

    mapping = generate_hybrid_mapping(schema_payload=schema, data_payload=payload, overrides=overrides)

    assert mapping["filters"]["asset"] == "symbol"
    assert mapping["filter_values"]["securityType"]["core"] == "Core"
    assert mapping["metadata"]["provenance"]["filters"]["asset"] == "override"
    assert (
        mapping["metadata"]["provenance"]["filter_values"]["securityType"]["core"]
        == "override"
    )
