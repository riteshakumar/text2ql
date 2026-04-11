"""SQLAlchemy-backed SQL execution for text2ql evaluation and testing.

This module provides :class:`SQLAlchemyExecutor`, a callable that runs a
generated SQL string against a real database (or an in-memory SQLite database)
and returns the result rows as a list of dicts.  It is designed to plug
directly into the text2ql evaluation framework as an ``execution_backend``:

.. code-block:: python

    from text2ql import evaluate_examples
    from text2ql.sql_executor import SQLAlchemyExecutor

    executor = SQLAlchemyExecutor("sqlite:///mydb.sqlite")
    report = evaluate_examples(service, examples, execution_backend=executor)

SQLAlchemy is an **optional** dependency.  Install it with::

    pip install text2ql[sql]

or directly::

    pip install sqlalchemy>=2.0

A lightweight in-memory SQLite helper (:func:`create_sqlite_executor`) is also
provided for unit tests and CI pipelines — no external database required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text2ql.dataset import DatasetExample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SQLAlchemy import guard
# ---------------------------------------------------------------------------

_SQLALCHEMY_AVAILABLE = False
try:
    import sqlalchemy  # noqa: F401

    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    pass


def _require_sqlalchemy() -> None:
    if not _SQLALCHEMY_AVAILABLE:
        raise ImportError(
            "SQLAlchemy is required for SQL execution. "
            "Install it with: pip install text2ql[sql]  "
            "or: pip install 'sqlalchemy>=2.0'"
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class SQLAlchemyExecutor:
    """Execute SQL queries against a SQLAlchemy-compatible database.

    Parameters
    ----------
    engine_or_url:
        A SQLAlchemy engine instance *or* a connection URL string
        (e.g. ``"sqlite:///mydb.sqlite"``, ``"postgresql+psycopg2://user:pw@host/db"``).
    connect_args:
        Extra keyword arguments forwarded to ``sqlalchemy.create_engine()``
        via its ``connect_args`` parameter (e.g. ``{"check_same_thread": False}``
        for SQLite).
    row_limit:
        Safety cap: never return more than this many rows from a single
        query.  Defaults to 10 000.  Pass ``None`` to disable.

    Usage
    -----
    As a plain callable (``execution_backend`` signature)::

        executor = SQLAlchemyExecutor("sqlite:///sales.db")
        result = executor("SELECT id, name FROM orders WHERE status = 'active';", example)

    Or via :meth:`execute` directly (ignores the ``DatasetExample``)::

        rows = executor.execute("SELECT COUNT(*) AS cnt FROM users;")
        # [{"cnt": 42}]
    """

    def __init__(
        self,
        engine_or_url: Any,
        connect_args: dict[str, Any] | None = None,
        row_limit: int | None = 10_000,
    ) -> None:
        _require_sqlalchemy()
        import sqlalchemy as sa

        if isinstance(engine_or_url, str):
            kwargs: dict[str, Any] = {}
            if connect_args:
                kwargs["connect_args"] = connect_args
            self._engine = sa.create_engine(engine_or_url, **kwargs)
            logger.debug("Created SQLAlchemy engine from URL: %s", engine_or_url)
        else:
            self._engine = engine_or_url
            logger.debug("Using provided SQLAlchemy engine: %s", engine_or_url)

        self._row_limit = row_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, sql: str) -> list[dict[str, Any]]:
        """Run *sql* and return rows as a list of dicts.

        Parameters
        ----------
        sql:
            A SQL statement string.  If ``row_limit`` is set, a
            ``LIMIT`` clause is appended when the statement does not
            already contain one.

        Returns
        -------
        list[dict[str, Any]]
            Each dict maps column names to values.

        Raises
        ------
        sqlalchemy.exc.SQLAlchemyError
            Any database or driver error propagates unchanged so that the
            evaluation framework can record it in ``execution_backend_error``.
        """
        import sqlalchemy as sa

        statement = _maybe_add_limit(sql, self._row_limit)
        logger.debug("Executing SQL: %s", statement)
        with self._engine.connect() as conn:
            result = conn.execute(sa.text(statement))
            keys = list(result.keys())
            rows = [dict(zip(keys, row)) for row in result.fetchall()]
        logger.debug("SQL returned %d row(s)", len(rows))
        return rows

    async def aexecute(self, sql: str) -> list[dict[str, Any]]:
        """Async wrapper — offloads synchronous execution to a thread pool.

        For truly async execution use an async SQLAlchemy engine
        (``create_async_engine``) and override this method.
        """
        import asyncio

        return await asyncio.to_thread(self.execute, sql)

    def __call__(self, sql: str, example: "DatasetExample") -> list[dict[str, Any]]:
        """Callable interface matching ``execution_backend`` signature.

        The ``example`` argument is accepted but not used — the SQL is
        executed as-is against the configured database.
        """
        return self.execute(sql)

    def load_json_data(
        self,
        table_name: str,
        rows: list[dict[str, Any]],
        if_exists: str = "replace",
    ) -> None:
        """Bulk-load a list of dicts into a table (SQLite-friendly helper).

        This is useful for tests: create an in-memory SQLite engine, call
        ``load_json_data()`` to populate it with fixture data, then run
        evaluation against generated queries.

        Parameters
        ----------
        table_name:
            Destination table name.
        rows:
            Data rows — each dict maps column name to value.  All rows
            must have identical keys.
        if_exists:
            Passed to ``pandas.DataFrame.to_sql()`` if pandas is available;
            otherwise tables are created via raw DDL.
        """
        if not rows:
            logger.warning("load_json_data: no rows provided for table '%s'", table_name)
            return

        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_sql(table_name, con=self._engine, if_exists=if_exists, index=False)
            logger.debug("Loaded %d rows into '%s' via pandas", len(rows), table_name)
        except ImportError:
            _load_json_data_raw(self._engine, table_name, rows)
            logger.debug("Loaded %d rows into '%s' via raw DDL", len(rows), table_name)

    def dispose(self) -> None:
        """Dispose the underlying SQLAlchemy engine (releases connection pool)."""
        self._engine.dispose()
        logger.debug("Disposed SQLAlchemy engine")

    def __enter__(self) -> "SQLAlchemyExecutor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.dispose()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_sqlite_executor(
    rows_by_table: dict[str, list[dict[str, Any]]] | None = None,
    row_limit: int | None = 10_000,
) -> SQLAlchemyExecutor:
    """Create an in-memory SQLite executor pre-loaded with fixture data.

    Ideal for unit tests and CI — no external database required.

    Parameters
    ----------
    rows_by_table:
        Mapping of ``table_name -> list_of_row_dicts``.  If ``None``, an
        empty database is returned.
    row_limit:
        Forwarded to :class:`SQLAlchemyExecutor`.

    Example
    -------
    .. code-block:: python

        executor = create_sqlite_executor({
            "users": [
                {"id": 1, "name": "Alice", "status": "active"},
                {"id": 2, "name": "Bob",   "status": "inactive"},
            ]
        })
        rows = executor.execute("SELECT name FROM users WHERE status = 'active';")
        # [{"name": "Alice"}]
    """
    _require_sqlalchemy()
    import sqlalchemy as sa

    engine = sa.create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    executor = SQLAlchemyExecutor(engine, row_limit=row_limit)

    if rows_by_table:
        for table_name, rows in rows_by_table.items():
            executor.load_json_data(table_name, rows)

    return executor


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _maybe_add_limit(sql: str, row_limit: int | None) -> str:
    """Append a LIMIT clause if the query lacks one and row_limit is set."""
    if row_limit is None:
        return sql
    normalized = sql.strip().upper()
    if "LIMIT" in normalized:
        return sql
    stripped = sql.rstrip().rstrip(";")
    return f"{stripped} LIMIT {row_limit};"


def _load_json_data_raw(engine: Any, table_name: str, rows: list[dict[str, Any]]) -> None:
    """DDL-based bulk insert without pandas."""
    import sqlalchemy as sa

    columns = list(rows[0].keys())
    col_defs = ", ".join(f'"{col}" TEXT' for col in columns)
    with engine.begin() as conn:
        conn.execute(sa.text(f'DROP TABLE IF EXISTS "{table_name}"'))
        conn.execute(sa.text(f'CREATE TABLE "{table_name}" ({col_defs})'))
        placeholders = ", ".join(f":{col}" for col in columns)
        insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
        for row in rows:
            conn.execute(sa.text(insert_sql), row)
