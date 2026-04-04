import json
import sys
from pathlib import Path

import pytest

from text2ql import Text2QL
from text2ql.cli import main

pytestmark = pytest.mark.e2e


def test_text2ql_generate_with_schema_mapping_e2e() -> None:
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


def test_text2ql_generate_unsupported_target_raises() -> None:
    service = Text2QL()

    with pytest.raises(ValueError, match="Unsupported target"):
        service.generate("show customers", target="sql")


def test_cli_main_supports_schema_and_mapping_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
