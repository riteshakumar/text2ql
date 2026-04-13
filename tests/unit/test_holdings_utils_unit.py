from __future__ import annotations

import pytest

from text2ql.engines.holdings_utils import (
    resolve_holdings_container,
    resolve_holdings_projection,
    resolve_identifier_filter_key,
    score_holdings_container,
)

pytestmark = pytest.mark.unit


def test_score_holdings_container_prefers_positions_like_entities() -> None:
    score = score_holdings_container("positions", ["symbol", "quantity", "status"])

    assert score > 0


def test_score_holdings_container_returns_zero_without_identifier_or_quantity() -> None:
    score = score_holdings_container("accounts", ["id", "status", "email"])

    assert score == 0


def test_resolve_holdings_container_selects_best_candidate() -> None:
    containers = ["accounts", "positions"]
    fields_by_container = {
        "accounts": ["id", "acctNum", "acctName"],
        "positions": ["symbol", "quantity", "status"],
    }

    resolved = resolve_holdings_container(
        containers,
        fields_for_container=lambda name: fields_by_container[name],
    )

    assert resolved == "positions"


def test_resolve_identifier_filter_key_supports_aliases() -> None:
    key = resolve_identifier_filter_key(
        args=["status", "symbol"],
        fields=["id", "quantity"],
        candidate_aliases={"ticker": "symbol"},
    )

    assert key == "symbol"


def test_resolve_holdings_projection_prefers_quantity_then_identifier() -> None:
    projection = resolve_holdings_projection(["id", "symbol", "quantity", "status"])

    assert projection == ["quantity", "symbol"]
