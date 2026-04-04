from pathlib import Path

import pytest

from text2ql.cli import _load_json_object

pytestmark = pytest.mark.unit


def test_load_json_object_from_inline_only() -> None:
    payload = _load_json_object('{"entities": ["customers"]}', "")

    assert payload == {"entities": ["customers"]}


def test_load_json_object_merges_file_then_inline(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.json"
    schema_file.write_text('{"entities": ["customers"], "fields": ["id"]}', encoding="utf-8")

    payload = _load_json_object('{"fields": ["id", "email"]}', str(schema_file))

    assert payload == {"entities": ["customers"], "fields": ["id", "email"]}


def test_load_json_object_requires_top_level_object(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.json"
    schema_file.write_text('["customers"]', encoding="utf-8")

    with pytest.raises(ValueError, match="top level"):
        _load_json_object("", str(schema_file))
