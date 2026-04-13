from __future__ import annotations

import pytest

import text2ql.sql_executor as sql_executor

pytestmark = pytest.mark.unit


def test_maybe_add_limit_appends_when_missing() -> None:
    query = "SELECT id FROM users ORDER BY id"

    limited = sql_executor._maybe_add_limit(query, 5)

    assert limited.endswith("LIMIT 5;")


def test_maybe_add_limit_preserves_existing_limit() -> None:
    query = "SELECT id FROM users LIMIT 3;"

    limited = sql_executor._maybe_add_limit(query, 10)

    assert limited == query


def test_require_sqlalchemy_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sql_executor, "_SQLALCHEMY_AVAILABLE", False)

    with pytest.raises(ImportError, match="SQLAlchemy is required"):
        sql_executor._require_sqlalchemy()


@pytest.mark.skipif(not getattr(sql_executor, "_SQLALCHEMY_AVAILABLE", False), reason="sqlalchemy not installed")
def test_create_sqlite_executor_executes_queries() -> None:
    executor = sql_executor.create_sqlite_executor(
        {
            "users": [
                {"id": 1, "name": "Alice", "status": "active"},
                {"id": 2, "name": "Bob", "status": "inactive"},
            ]
        },
        row_limit=10,
    )
    try:
        rows = executor.execute("SELECT id, name FROM users WHERE status = 'active' ORDER BY id")
    finally:
        executor.dispose()

    assert len(rows) == 1
    assert "id" in rows[0]
    assert "name" in rows[0]


@pytest.mark.skipif(not getattr(sql_executor, "_SQLALCHEMY_AVAILABLE", False), reason="sqlalchemy not installed")
def test_create_sqlite_executor_applies_row_limit() -> None:
    executor = sql_executor.create_sqlite_executor(
        {
            "orders": [
                {"id": 1},
                {"id": 2},
                {"id": 3},
            ]
        },
        row_limit=1,
    )
    try:
        rows = executor.execute("SELECT id FROM orders ORDER BY id")
    finally:
        executor.dispose()

    assert len(rows) == 1
