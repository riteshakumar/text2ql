import argparse
from pathlib import Path

import pytest

from text2ql.cli import _build_hybrid_mapping_from_args, _load_json_object

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


def test_build_hybrid_mapping_from_args_supports_schema_data_and_overrides(tmp_path: Path) -> None:
    data_file = tmp_path / "data.json"
    data_file.write_text(
        '{"positions":[{"symbol":"QQQ","quantity":100.104,"tradeMarket":"NASDAQ"}]}',
        encoding="utf-8",
    )
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(
        '{"entities":["positions"],"fields":{"positions":["symbol","quantity","tradeMarket"]}}',
        encoding="utf-8",
    )
    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_text(
        '{"filters":{"asset":"symbol"}}',
        encoding="utf-8",
    )

    args = argparse.Namespace(
        data_file=str(data_file),
        schema="",
        schema_file=str(schema_file),
        mapping_overrides="",
        mapping_overrides_file=str(overrides_file),
    )

    mapping = _build_hybrid_mapping_from_args(args)

    assert mapping["filters"]["asset"] == "symbol"
    assert mapping["filter_values"]["symbol"]["qqq"] == "QQQ"


def test_build_hybrid_mapping_from_args_requires_data_file() -> None:
    args = argparse.Namespace(
        data_file="",
        schema="",
        schema_file="",
        mapping_overrides="",
        mapping_overrides_file="",
    )

    with pytest.raises(ValueError, match="--data-file"):
        _build_hybrid_mapping_from_args(args)
