from __future__ import annotations

from text2ql.engines.text_utils import (
    contains_column_reference,
    extract_filter_value,
    parse_grouped_boolean_filters,
    split_top_level,
    strip_outer_parentheses,
)

import pytest

pytestmark = pytest.mark.unit


def _parse_atomic_nodes(text: str) -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []
    normalized = strip_outer_parentheses(text)
    for part in split_top_level(normalized, "and"):
        part = strip_outer_parentheses(part).strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            nodes.append({key.strip(): value.strip()})
    return nodes


def test_parse_grouped_boolean_filters_respects_or_precedence() -> None:
    grouped = parse_grouped_boolean_filters(
        "where (status = active and type = retail) or status = pending",
        _parse_atomic_nodes,
    )

    assert "or" in grouped
    assert len(grouped["or"]) == 2


def test_extract_filter_value_skips_spurious_tokens() -> None:
    assert extract_filter_value("status", "status is and", spurious_values={"and"}) is None
    assert extract_filter_value("status", "status is active", spurious_values={"and"}) == "active"


def test_contains_column_reference_handles_inflections() -> None:
    assert contains_column_reference("highest quantities first", "quantity") is True
    assert contains_column_reference("list symbols", "symbol") is True
    assert contains_column_reference("list users", "quantity") is False
