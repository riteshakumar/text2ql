"""Shared utilities for the CLI and Streamlit app.

Both surfaces need the same helpers for SQL-on-JSON execution, novelty
scoring, and synthetic metadata assembly.  Keeping them here avoids
duplication while keeping ``cli.py`` and the Streamlit app thin.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def stable_json(value: Any) -> str:
    """Deterministic JSON string for equality comparison."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


# ---------------------------------------------------------------------------
# Synthetic scoring
# ---------------------------------------------------------------------------


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", str(text).lower()) if token}


def compute_novelty(seed_prompt: str, candidate_prompt: str) -> float:
    seed_tokens = tokenize(seed_prompt)
    cand_tokens = tokenize(candidate_prompt)
    if not seed_tokens and not cand_tokens:
        return 0.0
    union = seed_tokens | cand_tokens
    if not union:
        return 0.0
    overlap = seed_tokens & cand_tokens
    return max(0.0, min(1.0, 1.0 - (len(overlap) / len(union))))


def as_unit_float(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def dynamic_synthetic_meta(
    base_meta: dict[str, Any],
    seed_prompt: str,
    active_prompt: str,
    engine_confidence: float,
    rewrite_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(base_meta or {})
    novelty = compute_novelty(seed_prompt, active_prompt)
    fallback_conf = as_unit_float(engine_confidence, default=0.5)
    if rewrite_meta and isinstance(rewrite_meta, dict):
        confidence = as_unit_float(
            rewrite_meta.get("synthetic_rewrite_confidence", rewrite_meta.get("confidence", fallback_conf)),
            default=fallback_conf,
        )
    else:
        confidence = as_unit_float(
            meta.get("synthetic_rewrite_confidence", fallback_conf), default=fallback_conf
        )
    score = as_unit_float(0.65 * confidence + 0.35 * novelty)
    meta["synthetic_rewrite_confidence"] = confidence
    meta["synthetic_rewrite_novelty"] = novelty
    meta["synthetic_rewrite_score"] = score
    meta.setdefault("synthetic_rewrite_source", "seed" if novelty == 0 else "synthetic")
    return meta


# ---------------------------------------------------------------------------
# In-memory SQL execution on JSON payload
# ---------------------------------------------------------------------------


def collect_entity_rows(
    node: Any,
    out: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if out is None:
        out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict):
                out.setdefault(str(key), []).append(value)
                collect_entity_rows(value, out)
            elif isinstance(value, list):
                dict_items = [item for item in value if isinstance(item, dict)]
                if dict_items:
                    out.setdefault(str(key), []).extend(dict_items)
                    for item in dict_items:
                        collect_entity_rows(item, out)
                else:
                    for item in value:
                        collect_entity_rows(item, out)
    elif isinstance(node, list):
        for item in node:
            collect_entity_rows(item, out)
    return out


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _to_sql_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, default=str)
    return value


def execute_sql_on_json(
    query: str,
    data_payload: dict[str, Any],
    root_key: str = "portfolio_data",
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute *query* against an in-memory SQLite database built from *data_payload*.

    Returns ``(rows, error_note)`` where ``error_note`` is ``None`` on success.
    """
    root = data_payload.get(root_key, data_payload)
    if not isinstance(root, dict):
        return [], "SQL execution skipped: payload must be a JSON object."

    entity_rows = collect_entity_rows(root)
    if not entity_rows:
        return [], "SQL execution skipped: no tabular entities found in payload."

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        created_tables = 0
        for table_name, rows in entity_rows.items():
            if not rows:
                continue
            columns = sorted({str(key) for row in rows for key in row.keys()})
            if not columns:
                continue
            conn.execute(
                f"CREATE TABLE {_quote_ident(table_name)} "
                f"({', '.join(f'{_quote_ident(col)} TEXT' for col in columns)});"
            )
            insert_sql = (
                f"INSERT INTO {_quote_ident(table_name)} "
                f"({', '.join(_quote_ident(col) for col in columns)}) "
                f"VALUES ({', '.join(['?'] * len(columns))});"
            )
            values = [[_to_sql_scalar(row.get(column)) for column in columns] for row in rows]
            conn.executemany(insert_sql, values)
            created_tables += 1

        if created_tables == 0:
            return [], "SQL execution skipped: no usable tables were created."
        cursor = conn.execute(query)
        return [dict(row) for row in cursor.fetchall()], None
    except sqlite3.Error as exc:
        return [], f"SQL execution error: {exc}"
    finally:
        conn.close()
