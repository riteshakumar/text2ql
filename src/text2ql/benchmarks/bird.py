"""BIRD benchmark adapter.

Loads the `BIRD <https://bird-bench.github.io/>`_ text-to-SQL dataset and
converts each example into a :class:`~text2ql.dataset.DatasetExample`.

Expected directory layout::

    bird/
    ├── dev.json                      # dev-split examples
    ├── train.json                    # training-split examples (optional)
    └── dev_databases/                # (or train_databases/)
        └── <db_id>/
            ├── <db_id>.sqlite        # per-database SQLite file
            └── database_description/ # CSV column descriptions (optional)

BIRD uses the same core schema format as Spider (``column_names_original``,
``table_names_original``, foreign keys) but adds an ``evidence`` field per
example that provides domain hints.

Usage
-----
.. code-block:: python

    from text2ql.benchmarks.bird import load_bird

    examples = load_bird("/path/to/bird", split="dev")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from text2ql.dataset import DatasetExample

logger = logging.getLogger(__name__)


def load_bird(
    root: str | Path,
    split: str = "dev",
    *,
    limit: int | None = None,
    db_filter: str | None = None,
) -> list[DatasetExample]:
    """Load BIRD examples as text2ql ``DatasetExample`` objects.

    Parameters
    ----------
    root:
        Path to the BIRD dataset root directory.
    split:
        Which split to load — ``"dev"`` or ``"train"``.
    limit:
        Cap the number of examples.
    db_filter:
        Only load examples for this ``db_id``.

    Returns
    -------
    list[DatasetExample]
    """
    root = Path(root)
    split_file = _resolve_split_file(root, split)
    raw_examples = json.loads(split_file.read_text(encoding="utf-8"))

    db_dir = _resolve_db_dir(root, split)
    schema_cache: dict[str, dict[str, Any]] = {}

    examples: list[DatasetExample] = []
    for entry in raw_examples:
        db_id = entry.get("db_id", "")
        if db_filter and db_id != db_filter:
            continue

        question = entry.get("question", "")
        gold_sql = entry.get("SQL", entry.get("query", ""))
        if not question or not gold_sql:
            continue

        if db_id not in schema_cache:
            db_path = db_dir / db_id / f"{db_id}.sqlite"
            if db_path.exists():
                schema_cache[db_id] = _introspect_sqlite(db_path)
            else:
                logger.warning("No SQLite file for db_id=%r at %s", db_id, db_path)
                schema_cache[db_id] = {}

        raw_schema = schema_cache[db_id]
        if not raw_schema:
            continue

        text2ql_schema = bird_schema_to_text2ql(raw_schema)
        evidence = entry.get("evidence", "")
        db_path = db_dir / db_id / f"{db_id}.sqlite"

        examples.append(
            DatasetExample(
                text=question,
                target="sql",
                expected_query=gold_sql,
                schema=text2ql_schema,
                context={
                    "mode": "deterministic",
                    "evidence": evidence,
                },
                metadata={
                    "benchmark": "bird",
                    "split": split,
                    "db_id": db_id,
                    "db_path": str(db_path) if db_path.exists() else None,
                    "evidence": evidence,
                    "difficulty": entry.get("difficulty", "unknown"),
                    "question_id": entry.get("question_id"),
                },
            )
        )

        if limit and len(examples) >= limit:
            break

    logger.info("Loaded %d BIRD examples (split=%s)", len(examples), split)
    return examples


def bird_schema_to_text2ql(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an introspected BIRD database schema to text2ql format.

    Parameters
    ----------
    schema:
        Schema dict as returned by :func:`_introspect_sqlite`.

    Returns
    -------
    dict
        A text2ql-compatible schema.
    """
    tables: list[str] = schema.get("tables", [])
    columns_by_table: dict[str, list[dict[str, str]]] = schema.get(
        "columns_by_table", {}
    )
    foreign_keys: list[dict[str, str]] = schema.get("foreign_keys", [])

    entities = [{"name": t} for t in tables]

    fields: dict[str, list[str]] = {}
    args: dict[str, list[str]] = {}
    for table in tables:
        cols = columns_by_table.get(table, [])
        col_names = [c["name"] for c in cols]
        fields[table] = col_names
        args[table] = col_names

    relations: dict[str, list[dict[str, Any]]] = {}
    for fk in foreign_keys:
        from_table = fk["from_table"]
        to_table = fk["to_table"]
        from_col = fk["from_column"]
        to_col = fk["to_column"]

        relations.setdefault(from_table, []).append({
            "name": to_table.lower(),
            "target": to_table,
            "on": f"{from_table}.{from_col} = {to_table}.{to_col}",
        })
        relations.setdefault(to_table, []).append({
            "name": from_table.lower(),
            "target": from_table,
            "on": f"{to_table}.{to_col} = {from_table}.{from_col}",
        })

    return {
        "entities": entities,
        "fields": fields,
        "relations": relations,
        "args": args,
        "metadata": {"source": "bird"},
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _introspect_sqlite(db_path: Path) -> dict[str, Any]:
    """Read schema metadata from a SQLite database file."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get table names
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Get columns per table
        columns_by_table: dict[str, list[dict[str, str]]] = {}
        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
            cols = cursor.fetchall()
            columns_by_table[table] = [
                {"name": col[1], "type": col[2] or "TEXT"} for col in cols
            ]

        # Get foreign keys
        foreign_keys: list[dict[str, str]] = []
        for table in tables:
            cursor.execute(f'PRAGMA foreign_key_list("{table}")')
            fks = cursor.fetchall()
            for fk in fks:
                foreign_keys.append({
                    "from_table": table,
                    "to_table": fk[2],
                    "from_column": fk[3],
                    "to_column": fk[4],
                })

        conn.close()
        return {
            "tables": tables,
            "columns_by_table": columns_by_table,
            "foreign_keys": foreign_keys,
        }
    except Exception as exc:
        logger.warning("Failed to introspect %s: %s", db_path, exc)
        return {}


def _resolve_split_file(root: Path, split: str) -> Path:
    """Locate the JSON file for the requested split."""
    candidates = {
        "dev": [
            "dev.json",
            "mini_dev_sqlite.json",
            "mini_dev.json",
        ],
        "train": ["train.json"],
    }
    filenames = candidates.get(split, [f"{split}.json"])
    for name in filenames:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No split file found for '{split}' in {root}. "
        f"Tried: {', '.join(filenames)}"
    )


def _resolve_db_dir(root: Path, split: str) -> Path:
    """Find the database directory for a given split."""
    candidates = [
        root / f"{split}_databases",
        root / "databases",
        root / "database",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    # Fall back to root itself (some layouts put db dirs at root level)
    return root
