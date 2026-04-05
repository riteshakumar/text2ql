import json
import sys
from pathlib import Path

import pytest

from text2ql import Text2QL
from text2ql.cli import main

pytestmark = pytest.mark.e2e


def test_sql_generate_with_schema_mapping_e2e() -> None:
    service = Text2QL()

    result = service.generate(
        text="show customers highest total first 5 offset 10",
        target="sql",
        schema={"entities": ["customers"], "fields": {"customers": ["id", "total", "status"]}},
    )

    assert result.target == "sql"
    assert "FROM customers" in result.query
    assert "ORDER BY customers.total DESC" in result.query
    assert "LIMIT 5" in result.query
    assert "OFFSET 10" in result.query


def test_sql_cli_supports_target_sql(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps({"entities": ["customers"], "fields": {"customers": ["id", "total", "status"]}}),
        encoding="utf-8",
    )

    original_argv = sys.argv
    try:
        sys.argv = [
            "text2ql",
            "show customers highest total first 5 offset 10",
            "--target",
            "sql",
            "--schema-file",
            str(schema_path),
        ]
        main()
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "SELECT" in captured.out
    assert "FROM customers" in captured.out
    assert "ORDER BY customers.total DESC" in captured.out
