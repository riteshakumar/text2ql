"""Spider benchmark adapter.

Loads the `Spider <https://yale-lily.github.io/spider>`_ text-to-SQL dataset
and converts each example into a :class:`~text2ql.dataset.DatasetExample` that
the text2ql evaluation framework can consume directly.

Expected directory layout::

    spider/
    ├── tables.json          # schema definitions for all databases
    ├── dev.json             # dev-split examples
    ├── train_spider.json    # training-split examples
    └── database/
        └── <db_id>/
            └── <db_id>.sqlite   # per-database SQLite files

Usage
-----
.. code-block:: python

    from text2ql.benchmarks.spider import load_spider

    examples = load_spider("/path/to/spider", split="dev")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from text2ql.dataset import DatasetExample

logger = logging.getLogger(__name__)

# Spider column type tokens → rough SQL types for schema hints
_TYPE_MAP = {
    "text": "TEXT",
    "number": "REAL",
    "time": "TEXT",
    "boolean": "INTEGER",
    "others": "TEXT",
}


def load_spider(
    root: str | Path,
    split: str = "dev",
    *,
    limit: int | None = None,
    db_filter: str | None = None,
) -> list[DatasetExample]:
    """Load Spider examples as text2ql ``DatasetExample`` objects.

    Parameters
    ----------
    root:
        Path to the Spider dataset root directory.
    split:
        Which split to load — ``"dev"`` or ``"train"``.
    limit:
        Cap the number of examples (useful for quick smoke tests).
    db_filter:
        If set, only load examples whose ``db_id`` matches this value.

    Returns
    -------
    list[DatasetExample]
        Each example has ``target="sql"`` and a schema derived from
        Spider's ``tables.json``.
    """
    root = Path(root)
    tables_path = root / "tables.json"
    if not tables_path.exists():
        raise FileNotFoundError(
            f"tables.json not found at {tables_path}. "
            "Ensure the Spider dataset is extracted correctly."
        )

    schemas_by_db = _load_schemas(tables_path)

    split_file = _resolve_split_file(root, split)
    raw_examples = json.loads(split_file.read_text(encoding="utf-8"))

    examples: list[DatasetExample] = []
    for entry in raw_examples:
        db_id = entry.get("db_id", "")
        if db_filter and db_id != db_filter:
            continue

        question = entry.get("question", "")
        gold_sql = entry.get("query", "")
        if not question or not gold_sql:
            continue

        db_schema = schemas_by_db.get(db_id)
        if db_schema is None:
            logger.warning("No schema found for db_id=%r, skipping", db_id)
            continue

        text2ql_schema = spider_schema_to_text2ql(db_schema)
        db_path = root / "database" / db_id / f"{db_id}.sqlite"

        examples.append(
            DatasetExample(
                text=question,
                target="sql",
                expected_query=gold_sql,
                schema=text2ql_schema,
                context={"mode": "deterministic"},
                metadata={
                    "benchmark": "spider",
                    "split": split,
                    "db_id": db_id,
                    "db_path": str(db_path) if db_path.exists() else None,
                    "difficulty": entry.get("difficulty", "unknown"),
                    "question_id": entry.get("question_id"),
                },
            )
        )

        if limit and len(examples) >= limit:
            break

    logger.info("Loaded %d Spider examples (split=%s)", len(examples), split)
    return examples


def spider_schema_to_text2ql(db_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Spider ``tables.json`` entry to text2ql schema format.

    Parameters
    ----------
    db_schema:
        A single database entry from Spider's ``tables.json``.

    Returns
    -------
    dict
        A text2ql-compatible schema dict with ``entities``, ``fields``,
        ``relations``, and ``args`` keys.
    """
    table_names: list[str] = db_schema.get("table_names_original", [])
    column_names: list[list[Any]] = db_schema.get("column_names_original", [])
    column_types: list[str] = db_schema.get("column_types", [])
    foreign_keys: list[list[int]] = db_schema.get("foreign_keys", [])
    primary_keys: list[int] = db_schema.get("primary_keys", [])

    # Also grab the human-friendly names for aliases
    table_names_friendly: list[str] = db_schema.get("table_names", [])

    # Build entities list with aliases from friendly names
    entities: list[dict[str, Any]] = []
    for idx, tname in enumerate(table_names):
        entity: dict[str, Any] = {"name": tname}
        if idx < len(table_names_friendly):
            friendly = table_names_friendly[idx]
            if friendly.lower().replace(" ", "_") != tname.lower():
                entity["aliases"] = [friendly, friendly.lower().replace(" ", "_")]
        entities.append(entity)

    # Build fields by entity
    fields_by_entity: dict[str, list[dict[str, str]]] = {t: [] for t in table_names}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx < 0:
            continue  # skip the * column
        if table_idx >= len(table_names):
            continue
        table = table_names[table_idx]
        col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
        fields_by_entity[table].append({
            "name": col_name,
            "type": _TYPE_MAP.get(col_type, "TEXT"),
        })

    # Build fields structure (per-entity dict format)
    fields: dict[str, list[str]] = {}
    for table, cols in fields_by_entity.items():
        fields[table] = [c["name"] for c in cols]

    # Build relations from foreign keys
    relations: dict[str, list[dict[str, Any]]] = {}
    for fk_from_col_idx, fk_to_col_idx in foreign_keys:
        if fk_from_col_idx >= len(column_names) or fk_to_col_idx >= len(column_names):
            continue
        from_table_idx, from_col = column_names[fk_from_col_idx]
        to_table_idx, to_col = column_names[fk_to_col_idx]
        if from_table_idx < 0 or to_table_idx < 0:
            continue
        if from_table_idx >= len(table_names) or to_table_idx >= len(table_names):
            continue

        from_table = table_names[from_table_idx]
        to_table = table_names[to_table_idx]

        if from_table not in relations:
            relations[from_table] = []
        relations[from_table].append({
            "name": to_table.lower(),
            "target": to_table,
            "on": f"{from_table}.{from_col} = {to_table}.{to_col}",
        })

        # Add reverse relation
        if to_table not in relations:
            relations[to_table] = []
        relations[to_table].append({
            "name": from_table.lower(),
            "target": from_table,
            "on": f"{to_table}.{to_col} = {from_table}.{from_col}",
        })

    # Build args (filterable columns per entity)
    args: dict[str, list[str]] = {}
    for table, cols in fields_by_entity.items():
        args[table] = [c["name"] for c in cols]

    return {
        "entities": entities,
        "fields": fields,
        "relations": relations,
        "args": args,
        "metadata": {
            "source": "spider",
            "db_id": db_schema.get("db_id", ""),
            "primary_keys": primary_keys,
        },
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_schemas(tables_path: Path) -> dict[str, dict[str, Any]]:
    """Parse ``tables.json`` into a db_id → schema dict mapping."""
    raw = json.loads(tables_path.read_text(encoding="utf-8"))
    schemas: dict[str, dict[str, Any]] = {}
    for entry in raw:
        db_id = entry.get("db_id", "")
        if db_id:
            schemas[db_id] = entry
    logger.debug("Loaded schemas for %d databases from tables.json", len(schemas))
    return schemas


def _resolve_split_file(root: Path, split: str) -> Path:
    """Locate the JSON file for the requested split."""
    candidates = {
        "dev": ["dev.json"],
        "train": ["train_spider.json", "train.json"],
        "test": ["test.json"],
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
