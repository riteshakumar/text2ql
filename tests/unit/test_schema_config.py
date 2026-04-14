import pytest

from text2ql.schema_config import infer_schema_from_json_payload, normalize_schema_config

pytestmark = pytest.mark.unit


def test_normalize_schema_config_from_structured_schema() -> None:
    schema = {
        "entities": [{"name": "customers", "aliases": ["client", "clients"]}],
        "fields": [
            {"name": "id", "aliases": ["identifier"]},
            {"name": "email", "aliases": ["mail"]},
        ],
        "default_entity": "customers",
        "default_fields": ["id", "email"],
        "default_fields_by_entity": {"customers": ["id", "email"]},
    }

    config = normalize_schema_config(schema)

    assert config.entities == ["customers"]
    assert config.entity_aliases["client"] == "customers"
    assert config.field_aliases["mail"] == "email"
    assert config.default_entity == "customers"
    assert config.default_fields == ["id", "email"]
    assert config.default_fields_by_entity["customers"] == ["id", "email"]


def test_normalize_schema_config_merges_mapping_and_value_aliases() -> None:
    schema = {
        "entities": ["customers"],
        "fields": ["id", "email", "status"],
        "mapping": {
            "entities": {"buyer": "customers"},
            "field_aliases": {"mail": "email"},
        },
    }
    mapping = {
        "entities": {"client": "customers"},
        "filters": {"state": "status"},
        "filter_values": {"status": {"enabled": "active"}},
    }

    config = normalize_schema_config(schema, mapping)

    assert config.entity_aliases["buyer"] == "customers"
    assert config.entity_aliases["client"] == "customers"
    assert config.field_aliases["mail"] == "email"
    assert config.filter_key_aliases["state"] == "status"
    assert config.filter_value_aliases["status"]["enabled"] == "active"


def test_normalize_schema_config_supports_fields_by_entity() -> None:
    schema = {
        "entities": ["customers", "orders"],
        "fields": {
            "customers": ["id", "email"],
            "orders": ["id", "status"],
        },
    }

    config = normalize_schema_config(schema)

    assert config.fields_by_entity["customers"] == ["id", "email"]
    assert config.fields_by_entity["orders"] == ["id", "status"]
    assert "status" in config.fields


def test_normalize_schema_config_infers_structure_from_arbitrary_nested_json() -> None:
    schema = {
        "portfolio_data": {
            "accounts": [
                {
                    "acctNum": "X123",
                    "positions": [{"symbol": "QQQ", "quantity": 100.104}],
                }
            ],
            "positionsSummary": {"portfolioPositionCount": 1},
        }
    }

    config = normalize_schema_config(schema)

    assert "accounts" in config.entities
    assert "positions" in config.entities
    assert "acctNum" in config.fields_by_entity["accounts"]
    assert "symbol" in config.fields_by_entity["positions"]
    assert "quantity" in config.fields_by_entity["positions"]
    assert config.default_entity in {"accounts", "portfolio_data"}


def test_infer_schema_from_json_payload_returns_text2ql_shape() -> None:
    payload = {
        "accounts": [
            {
                "acctNum": "X123",
                "positions": [{"symbol": "QQQ", "quantity": 100.104}],
            }
        ]
    }

    inferred = infer_schema_from_json_payload(payload)

    assert "entities" in inferred
    assert "fields" in inferred
    assert "args" in inferred
    assert "default_entity" in inferred
    assert "default_fields_by_entity" in inferred


def test_normalize_schema_config_auto_discovers_args_from_fields_when_missing() -> None:
    schema = {
        "entities": ["accountSummary"],
        "fields": {"accountSummary": ["isPartialBalance", "asOfDateTime"]},
    }

    config = normalize_schema_config(schema)

    assert "accountSummary" in config.args_by_entity
    assert "isPartialBalance" in config.args_by_entity["accountSummary"]
    assert "asOfDateTime" in config.args_by_entity["accountSummary"]
    assert "limit" in config.args_by_entity["accountSummary"]


def test_normalize_schema_config_does_not_bleed_global_fields_into_empty_entity_args() -> None:
    payload = {
        "accounts": [{"acctNum": "X123", "transactions": [{"txnTypeDesc": "Dividend Received"}]}],
        "summary": {"balanceSummary": {}},
    }

    config = normalize_schema_config(payload)

    assert "balanceSummary" in config.entities
    assert config.fields_by_entity.get("balanceSummary") in (None, [])
    assert "txnTypeDesc" not in config.args_by_entity.get("balanceSummary", [])
