"""Unified benchmark runner for Spider, BIRD, and custom text-to-SQL datasets.

Provides execution-accuracy evaluation (run both gold and predicted SQL against
the actual SQLite database and compare results) as well as exact-match and
structural-match metrics.

Usage
-----
.. code-block:: python

    from text2ql.benchmarks import load_spider, run_benchmark, format_report

    examples = load_spider("/path/to/spider", split="dev", limit=100)
    report = run_benchmark(examples, mode="execution")
    print(format_report(report))
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from text2ql.core import Text2QL
from text2ql.dataset import DatasetExample
from text2ql.evaluate import normalize_query, structural_execution_match

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Parameters
    ----------
    mode:
        Evaluation mode: ``"exact"`` (string match), ``"structural"``
        (parsed-signature match), or ``"execution"`` (run against SQLite
        and compare result sets).
    service:
        Pre-configured :class:`~text2ql.core.Text2QL` instance.
        If ``None``, a default rule-based service is created.
    concurrency:
        Max concurrent evaluations for async mode.
    timeout_per_query:
        Seconds before a single SQL execution is killed.
    ignore_order:
        When ``True``, sort result rows before comparing (execution mode).
    """

    mode: str = "execution"
    service: Text2QL | None = None
    concurrency: int = 10
    timeout_per_query: float = 30.0
    ignore_order: bool = True


@dataclass(slots=True)
class BenchmarkRow:
    """Result for a single benchmark example."""

    question: str
    db_id: str
    gold_sql: str
    predicted_sql: str
    exact_match: bool
    structural_match: bool
    execution_match: bool | None  # None if execution was not attempted / errored
    difficulty: str
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""

    benchmark: str
    split: str
    total: int
    exact_match_accuracy: float
    structural_accuracy: float
    execution_accuracy: float | None
    accuracy_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    accuracy_by_db: dict[str, dict[str, float]] = field(default_factory=dict)
    rows: list[BenchmarkRow] = field(default_factory=list)
    errors: int = 0
    elapsed_seconds: float = 0.0


def run_benchmark(
    examples: list[DatasetExample],
    *,
    config: BenchmarkConfig | None = None,
) -> BenchmarkReport:
    """Run a benchmark synchronously.

    Parameters
    ----------
    examples:
        Examples loaded via :func:`load_spider` or :func:`load_bird`.
    config:
        Evaluation configuration.  Defaults to execution mode.
    """
    cfg = config or BenchmarkConfig()
    service = cfg.service or Text2QL()

    start = time.monotonic()
    rows: list[BenchmarkRow] = []

    for example in examples:
        row = _evaluate_one(example, service, cfg)
        rows.append(row)

    elapsed = time.monotonic() - start
    return _build_report(examples, rows, elapsed)


async def arun_benchmark(
    examples: list[DatasetExample],
    *,
    config: BenchmarkConfig | None = None,
) -> BenchmarkReport:
    """Run a benchmark with concurrent evaluation.

    Parameters
    ----------
    examples:
        Examples loaded via :func:`load_spider` or :func:`load_bird`.
    config:
        Evaluation configuration.
    """
    cfg = config or BenchmarkConfig()
    service = cfg.service or Text2QL()
    sem = asyncio.Semaphore(cfg.concurrency)

    async def _eval(ex: DatasetExample) -> BenchmarkRow:
        async with sem:
            return await asyncio.to_thread(_evaluate_one, ex, service, cfg)

    start = time.monotonic()
    rows = list(await asyncio.gather(*[_eval(ex) for ex in examples]))
    elapsed = time.monotonic() - start
    return _build_report(examples, rows, elapsed)


def format_report(report: BenchmarkReport, *, verbose: bool = False) -> str:
    """Format a benchmark report as a human-readable string.

    Parameters
    ----------
    report:
        The report to format.
    verbose:
        If ``True``, include per-example details for failures.
    """
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append(f"  text2ql Benchmark Report — {report.benchmark.upper()} ({report.split})")
    lines.append("=" * 68)
    lines.append(f"  Total examples     : {report.total}")
    lines.append(f"  Errors             : {report.errors}")
    lines.append(f"  Elapsed            : {report.elapsed_seconds:.1f}s")
    lines.append("")
    lines.append(f"  Exact Match        : {report.exact_match_accuracy:.1%}")
    lines.append(f"  Structural Match   : {report.structural_accuracy:.1%}")
    if report.execution_accuracy is not None:
        lines.append(f"  Execution Accuracy : {report.execution_accuracy:.1%}")
    lines.append("")

    # Accuracy by difficulty
    if report.accuracy_by_difficulty:
        lines.append("  By Difficulty:")
        for diff, metrics in sorted(report.accuracy_by_difficulty.items()):
            n = int(metrics.get("count", 0))
            ex_acc = metrics.get("execution_accuracy")
            struct_acc = metrics.get("structural_accuracy", 0)
            label = f"    {diff:<12s} (n={n:>4d})"
            if ex_acc is not None:
                label += f"  exec={ex_acc:.1%}"
            label += f"  struct={struct_acc:.1%}"
            lines.append(label)
        lines.append("")

    # Top/bottom databases by execution accuracy
    if report.accuracy_by_db and report.execution_accuracy is not None:
        sorted_dbs = sorted(
            report.accuracy_by_db.items(),
            key=lambda kv: kv[1].get("execution_accuracy", 0),
        )
        if len(sorted_dbs) > 10:
            lines.append("  Bottom 5 databases (execution accuracy):")
            for db_id, metrics in sorted_dbs[:5]:
                n = int(metrics.get("count", 0))
                acc = metrics.get("execution_accuracy", 0)
                lines.append(f"    {db_id:<30s} {acc:.1%}  (n={n})")
            lines.append("")

    # Verbose: show failures
    if verbose:
        failures = [r for r in report.rows if not r.execution_match and r.execution_match is not None]
        if failures:
            lines.append(f"  Failed examples ({len(failures)}):")
            for row in failures[:50]:
                lines.append(f"    [{row.db_id}] {row.question[:80]}")
                lines.append(f"      Gold : {row.gold_sql[:100]}")
                lines.append(f"      Pred : {row.predicted_sql[:100]}")
                if row.error:
                    lines.append(f"      Error: {row.error[:100]}")
                lines.append("")

    lines.append("=" * 68)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _evaluate_one(
    example: DatasetExample,
    service: Text2QL,
    cfg: BenchmarkConfig,
) -> BenchmarkRow:
    """Evaluate a single benchmark example."""
    meta = example.metadata if isinstance(example.metadata, dict) else {}
    db_id = meta.get("db_id", "unknown")
    difficulty = meta.get("difficulty", "unknown")
    db_path = meta.get("db_path")

    error: str | None = None
    predicted_sql = ""
    exact_match = False
    structural_match = False
    execution_match: bool | None = None

    t0 = time.monotonic()
    try:
        result = service.generate(
            text=example.text,
            target="sql",
            schema=example.schema,
            mapping=example.mapping,
            context=example.context,
        )
        predicted_sql = result.query.strip()
    except Exception as exc:
        error = f"Generation error: {type(exc).__name__}: {exc}"
        predicted_sql = ""

    latency_ms = (time.monotonic() - t0) * 1000

    gold_sql = example.expected_query.strip()

    # Exact match (normalized whitespace)
    exact_match = normalize_query(predicted_sql) == normalize_query(gold_sql)

    # Structural match
    if predicted_sql:
        structural_match = structural_execution_match("sql", predicted_sql, gold_sql)

    # Execution match (run against SQLite)
    if cfg.mode == "execution" and db_path and predicted_sql:
        try:
            execution_match = _execution_match(
                db_path=db_path,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                timeout=cfg.timeout_per_query,
                ignore_order=cfg.ignore_order,
            )
        except Exception as exc:
            error = f"Execution error: {type(exc).__name__}: {exc}"
            execution_match = False

    return BenchmarkRow(
        question=example.text,
        db_id=db_id,
        gold_sql=gold_sql,
        predicted_sql=predicted_sql,
        exact_match=exact_match,
        structural_match=structural_match,
        execution_match=execution_match,
        difficulty=difficulty,
        error=error,
        latency_ms=latency_ms,
    )


def _execution_match(
    db_path: str,
    gold_sql: str,
    predicted_sql: str,
    timeout: float = 30.0,
    ignore_order: bool = True,
) -> bool:
    """Execute both queries against the SQLite database and compare results.

    This follows the standard Spider/BIRD evaluation protocol: two queries
    are considered equivalent if they produce the same result set (optionally
    ignoring row order).
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        gold_rows = _execute_sql(conn, gold_sql, timeout)
        pred_rows = _execute_sql(conn, predicted_sql, timeout)
    finally:
        conn.close()

    if ignore_order:
        gold_rows = sorted(gold_rows, key=_row_sort_key)
        pred_rows = sorted(pred_rows, key=_row_sort_key)

    return gold_rows == pred_rows


def _execute_sql(
    conn: sqlite3.Connection,
    sql: str,
    timeout: float,
) -> list[tuple[Any, ...]]:
    """Run a SQL statement with a timeout."""
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        return cursor.fetchall()
    except Exception:
        raise
    finally:
        cursor.close()


def _row_sort_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Create a sortable key from a result row, handling None values."""
    return tuple(
        ("", 0, str(v)) if v is None else (type(v).__name__, 0, str(v))
        for v in row
    )


def _build_report(
    examples: list[DatasetExample],
    rows: list[BenchmarkRow],
    elapsed: float,
) -> BenchmarkReport:
    """Aggregate individual rows into a BenchmarkReport."""
    total = len(rows)
    if total == 0:
        meta = examples[0].metadata if examples else {}
        return BenchmarkReport(
            benchmark=meta.get("benchmark", "unknown") if isinstance(meta, dict) else "unknown",
            split=meta.get("split", "unknown") if isinstance(meta, dict) else "unknown",
            total=0,
            exact_match_accuracy=0.0,
            structural_accuracy=0.0,
            execution_accuracy=0.0,
            rows=[],
            elapsed_seconds=elapsed,
        )

    meta = examples[0].metadata if isinstance(examples[0].metadata, dict) else {}
    benchmark = meta.get("benchmark", "unknown")
    split = meta.get("split", "unknown")

    exact_hits = sum(1 for r in rows if r.exact_match)
    struct_hits = sum(1 for r in rows if r.structural_match)
    errors = sum(1 for r in rows if r.error)

    exec_rows = [r for r in rows if r.execution_match is not None]
    exec_hits = sum(1 for r in exec_rows if r.execution_match)
    execution_accuracy = (exec_hits / len(exec_rows)) if exec_rows else None

    # By difficulty
    by_diff: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        by_diff[row.difficulty].append(row)

    accuracy_by_difficulty: dict[str, dict[str, float]] = {}
    for diff, diff_rows in by_diff.items():
        n = len(diff_rows)
        s_hits = sum(1 for r in diff_rows if r.structural_match)
        e_rows = [r for r in diff_rows if r.execution_match is not None]
        e_hits = sum(1 for r in e_rows if r.execution_match)
        entry: dict[str, float] = {
            "count": n,
            "structural_accuracy": s_hits / n if n else 0,
        }
        if e_rows:
            entry["execution_accuracy"] = e_hits / len(e_rows)
        accuracy_by_difficulty[diff] = entry

    # By database
    by_db: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        by_db[row.db_id].append(row)

    accuracy_by_db: dict[str, dict[str, float]] = {}
    for db_id, db_rows in by_db.items():
        n = len(db_rows)
        s_hits = sum(1 for r in db_rows if r.structural_match)
        e_rows = [r for r in db_rows if r.execution_match is not None]
        e_hits = sum(1 for r in e_rows if r.execution_match)
        entry = {
            "count": float(n),
            "structural_accuracy": s_hits / n if n else 0,
        }
        if e_rows:
            entry["execution_accuracy"] = e_hits / len(e_rows)
        accuracy_by_db[db_id] = entry

    return BenchmarkReport(
        benchmark=benchmark,
        split=split,
        total=total,
        exact_match_accuracy=exact_hits / total,
        structural_accuracy=struct_hits / total,
        execution_accuracy=execution_accuracy,
        accuracy_by_difficulty=accuracy_by_difficulty,
        accuracy_by_db=accuracy_by_db,
        rows=rows,
        errors=errors,
        elapsed_seconds=elapsed,
    )
