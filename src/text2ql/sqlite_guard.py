"""SQLite read-only execution with a wall-clock deadline."""
from __future__ import annotations

import sqlite3
import math
import time
from contextlib import contextmanager


@contextmanager
def sqlite_read_guard(connection: sqlite3.Connection, timeout: float | None):
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive")
    deadline = None if timeout is None else time.monotonic() + timeout
    previous = connection.execute("PRAGMA query_only").fetchone()[0]
    previous_busy = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    connection.execute("PRAGMA query_only = ON")
    if timeout is not None:
        connection.execute(f"PRAGMA busy_timeout = {max(1, int(timeout * 1000))}")
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def authorize(action, arg1, arg2, database, source):
        if action not in allowed:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in {"load_extension", "writefile", "readfile"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorize)
    connection.set_progress_handler(lambda: int(deadline is not None and time.monotonic() >= deadline), 1000)
    try:
        yield
    finally:
        connection.set_progress_handler(None, 0)
        connection.set_authorizer(None)
        connection.execute(f"PRAGMA query_only = {int(previous)}")
        connection.execute(f"PRAGMA busy_timeout = {previous_busy}")
