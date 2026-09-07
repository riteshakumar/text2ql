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

import importlib
import importlib.util
import logging
import math
import threading
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Mapping

from text2ql.query_validation import validate_sql, sql_dialect, sqlalchemy_sql
from text2ql.sqlite_guard import sqlite_read_guard
from text2ql.types import QueryResult, ValidationError

if TYPE_CHECKING:
    from text2ql.dataset import DatasetExample

logger = logging.getLogger(__name__)

_SQLALCHEMY_AVAILABLE = importlib.util.find_spec("sqlalchemy") is not None


def _require_sqlalchemy() -> None:
    if not _SQLALCHEMY_AVAILABLE:
        raise ImportError(
            "SQLAlchemy is required for SQL execution. "
            "Install it with: pip install text2ql[sql]  "
            "or: pip install 'sqlalchemy>=2.0'"
        )


def _sqlalchemy_module() -> Any:
    """Load SQLAlchemy lazily so optional dependency checks stay runtime-only."""
    _require_sqlalchemy()
    try:
        return importlib.import_module("sqlalchemy")
    except ImportError as exc:
        raise ImportError(
            "SQLAlchemy is required for SQL execution. "
            "Install it with: pip install text2ql[sql]  "
            "or: pip install 'sqlalchemy>=2.0'"
        ) from exc


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
        timeout_seconds: float | None = 30.0,
        before_execute: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        sa = _sqlalchemy_module()

        if isinstance(engine_or_url, str):
            kwargs: dict[str, Any] = {}
            if connect_args:
                kwargs["connect_args"] = connect_args
            self._engine = sa.create_engine(engine_or_url, **kwargs)
            logger.debug("Created SQLAlchemy engine")
        else:
            self._engine = engine_or_url
            logger.debug("Using provided SQLAlchemy engine")

        self._row_limit = row_limit
        if row_limit is not None and (isinstance(row_limit, bool) or not isinstance(row_limit, int) or row_limit < 0):
            raise ValueError("row_limit must be a non-negative integer or None")
        if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be positive or None")
        self._timeout_seconds = timeout_seconds
        self._before_execute = before_execute
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, sql: str | QueryResult, parameters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run *sql* and return rows as a list of dicts.

        Parameters
        ----------
        sql:
            A single read query or executable SQL QueryResult. Compiled results
            use named parameters. Row limits are enforced during fetching as
            well as by clamping the outer SQL limit.

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
        sa = _sqlalchemy_module()

        dialect = sql_dialect(self._engine.dialect.name)
        if isinstance(sql, QueryResult):
            if sql.target != "sql" or not sql.executable:
                raise ValidationError("Query result is not executable", [sql.status, sql.explanation])
            if sql.ir is not None and parameters is None:
                from text2ql.renderers import SQLIRRenderer
                sql, parameters = SQLIRRenderer(dialect).render_parameterized(sql.ir)
            else:
                sql = sql.query
        validate_sql(sql, dialect=dialect)
        statement = _maybe_add_limit(sql, self._row_limit, dialect=dialect)
        parameters = dict(parameters or {})
        if self._before_execute is not None:
            self._before_execute(statement, parameters)
        if self._timeout_seconds is not None and dialect not in {"sqlite", "postgres"}:
            raise ValueError("Built-in execution deadlines support SQLite and PostgreSQL; configure driver/server limits and set timeout_seconds=None for other backends")
        with self._lock, self._engine.connect() as conn:
            if dialect == "postgres":
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                if self._timeout_seconds is not None:
                    conn.execute(sa.text("SELECT set_config('statement_timeout', :timeout, true)"),
                                 {"timeout": str(max(1, int(self._timeout_seconds * 1000)))})
            raw = conn.connection.driver_connection
            guard = sqlite_read_guard(raw, self._timeout_seconds) if dialect == "sqlite" else nullcontext()
            with guard:
                result = conn.execution_options(stream_results=True).execute(sa.text(statement), parameters)
                try:
                    keys = list(result.keys())
                    records = result.fetchall() if self._row_limit is None else (result.fetchmany(self._row_limit) if self._row_limit else [])
                    rows = [dict(zip(keys, row)) for row in records]
                finally:
                    result.close()
        logger.debug("SQL returned %d row(s)", len(rows))
        return rows

    async def aexecute(self, sql: str | QueryResult, parameters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Async wrapper — offloads synchronous execution to a thread pool.

        For truly async execution use an async SQLAlchemy engine
        (``create_async_engine``) and override this method.
        """
        import asyncio

        return await asyncio.to_thread(self.execute, sql, parameters)

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
            One of ``replace``, ``append`` or ``fail``. Fixtures preserve numeric
            types and use the same SQLAlchemy path on every installation.
        """
        if not rows:
            logger.warning("load_json_data: no rows provided for table '%s'", table_name)
            return

        with self._lock:
            _load_json_data_raw(self._engine, table_name, rows, if_exists)

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
    sa = _sqlalchemy_module()

    engine = sa.create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=sa.pool.StaticPool
    )
    executor = SQLAlchemyExecutor(engine, row_limit=row_limit)

    if rows_by_table:
        for table_name, rows in rows_by_table.items():
            executor.load_json_data(table_name, rows)

    return executor


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _maybe_add_limit(sql: str, row_limit: int | None, *, dialect: str = "sqlite") -> str:
    """Clamp the outer query's literal limit, independently of comments/strings."""
    from sqlglot import exp
    tree = validate_sql(sql, dialect=dialect)
    if row_limit is None:
        return sql
    existing = tree.args.get("limit")
    if existing is not None:
        expression = existing.expression
        if isinstance(expression, exp.Literal) and expression.is_int:
            if 0 <= int(expression.this) <= row_limit:
                return sql
        elif expression is not None:
            # Preserve parameterized/dynamic limits; fetchmany still enforces
            # the independent output cap and the deadline bounds execution.
            return sql
    return sqlalchemy_sql(tree.limit(row_limit), dialect) + ";"


def _load_json_data_raw(engine: Any, table_name: str, rows: list[dict[str, Any]], if_exists: str = "replace") -> None:
    """Load typed fixtures consistently, whether or not pandas is installed."""
    sa = _sqlalchemy_module()
    if if_exists not in {"replace", "append", "fail"}:
        raise ValueError("if_exists must be replace, append or fail")
    columns = list(rows[0].keys())
    if any(set(row) != set(columns) for row in rows):
        raise ValueError("All fixture rows must have identical keys")
    def infer(column):
        values = [row[column] for row in rows if row[column] is not None]
        if values and all(isinstance(v, bool) for v in values):
            return sa.Boolean()
        if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return sa.BigInteger()
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return sa.Float()
        return sa.Text()
    with engine.begin() as conn:
        exists = sa.inspect(conn).has_table(table_name)
        if exists and if_exists == "fail":
            raise ValueError(f"Table '{table_name}' already exists")
        if exists:
            table = sa.Table(table_name, sa.MetaData(), autoload_with=conn)
            if if_exists == "replace":
                table.drop(conn)
                exists = False
        if not exists:
            table = sa.Table(table_name, sa.MetaData(), *(sa.Column(col, infer(col), quote=True) for col in columns), quote=True)
            table.create(conn)
        conn.execute(table.insert(), rows)
