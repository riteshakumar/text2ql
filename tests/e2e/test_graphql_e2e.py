import json
import sys
from pathlib import Path

import pytest

from text2ql import Text2QL
from text2ql.cli import main

pytestmark = pytest.mark.e2e


def test_graphql_nested_relation_e2e() -> None:
    """Nested relation selection appears in the GraphQL query when the query
    text mentions the relation name and the schema declares it."""
    service = Text2QL()

    result = service.generate(
        text="show orders with their items",
        target="graphql",
        schema={
            "entities": ["orders"],
            "fields": {"orders": ["id", "status"]},
            "relations": {
                "orders": {
                    "items": {
                        "target": "orderItems",
                        "fields": ["id", "quantity", "price"],
                    }
                }
            },
        },
    )

    assert result.target == "graphql"
    assert "orders" in result.query
    assert "items" in result.query
    # At least one item field should appear inside the nested selection.
    assert any(f in result.query for f in ("id", "quantity", "price"))


def test_graphql_nested_multihop_e2e() -> None:
    """Two-hop nested relation (orders → items → variants) is rendered
    correctly, with variants nested inside items."""
    service = Text2QL()

    result = service.generate(
        text="show orders with items and their variants",
        target="graphql",
        schema={
            "entities": ["orders"],
            "fields": {"orders": ["id", "status"]},
            "relations": {
                "orders": {
                    "items": {
                        "target": "orderItems",
                        "fields": ["id", "quantity"],
                    }
                },
                "orderItems": {
                    "variants": {
                        "target": "productVariants",
                        "fields": ["id", "sku"],
                    }
                },
            },
        },
    )

    assert result.target == "graphql"
    assert "items" in result.query
    assert "variants" in result.query


def test_graphql_aggregation_count_e2e() -> None:
    """COUNT aggregation appears in the GraphQL query when the query asks to
    count records."""
    service = Text2QL()

    result = service.generate(
        text="count orders",
        target="graphql",
        schema={"entities": ["orders"], "fields": {"orders": ["id", "status", "total"]}},
    )

    assert result.target == "graphql"
    assert "orders" in result.query
    assert "count" in result.query


def test_graphql_generate_with_schema_mapping_e2e() -> None:
    service = Text2QL()

    result = service.generate(
        text="show top 5 client records with mail state enabled",
        target="graphql",
        schema={"entities": ["customers"], "fields": ["id", "email", "status"]},
        mapping={
            "entities": {"client": "customers"},
            "fields": {"mail": "email"},
            "filters": {"state": "status"},
            "filter_values": {"status": {"enabled": "active"}},
        },
    )

    assert result.target == "graphql"
    assert "customers(limit: 5, status: \"active\")" in result.query
    assert result.metadata["entity"] == "customers"


def test_graphql_cli_supports_schema_and_mapping_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema_path = tmp_path / "schema.json"
    mapping_path = tmp_path / "mapping.json"

    schema_path.write_text(
        json.dumps({"entities": ["customers"], "fields": ["id", "email", "status"]}),
        encoding="utf-8",
    )
    mapping_path.write_text(
        json.dumps(
            {
                "entities": {"client": "customers"},
                "fields": {"mail": "email"},
                "filters": {"state": "status"},
                "filter_values": {"status": {"enabled": "active"}},
            }
        ),
        encoding="utf-8",
    )

    original_argv = sys.argv
    try:
        sys.argv = [
            "text2ql",
            "show top 2 client records with mail state enabled",
            "--schema-file",
            str(schema_path),
            "--mapping-file",
            str(mapping_path),
        ]
        main()
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "customers(limit: 2, status: \"active\")" in captured.out
