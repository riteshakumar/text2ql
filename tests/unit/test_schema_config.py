import pytest

from text2ql.schema_config import normalize_schema_config

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
    }

    config = normalize_schema_config(schema)

    assert config.entities == ["customers"]
    assert config.entity_aliases["client"] == "customers"
    assert config.field_aliases["mail"] == "email"
    assert config.default_entity == "customers"
    assert config.default_fields == ["id", "email"]


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
