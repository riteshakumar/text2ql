from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable

from text2ql.core import Text2QL
from text2ql.dataset import DatasetExample

ExecutionBackend = Callable[[str, DatasetExample], Any]
AsyncExecutionBackend = Callable[[str, DatasetExample], Any]  # may be sync or async
ExecutionComparator = Callable[[Any, Any], bool]

@dataclass(slots=True)
class EvaluationRow:
    text: str
    expected_query: str
    predicted_query: str
    exact_match: bool
    execution_match: bool
    execution_mode: str = "structural"
    execution_backend_error: str | None = None


@dataclass(slots=True)
class EvaluationReport:
    total: int
    exact_match_accuracy: float
    execution_accuracy: float
    rows: list[EvaluationRow]


def evaluate_examples(
    service: Text2QL,
    examples: list[DatasetExample],
    execution_backend: ExecutionBackend | None = None,
    execution_comparator: ExecutionComparator | None = None,
) -> EvaluationReport:
    rows: list[EvaluationRow] = []
    exact_hits = 0
    exec_hits = 0
    comparator = execution_comparator or _default_execution_comparator

    for example in examples:
        try:
            result = service.generate(
                text=example.text,
                target=example.target,
                schema=example.schema,
                mapping=example.mapping,
                context=example.context,
            )
            if not result.executable:
                raise ValueError(f"{result.status}: {result.explanation}")
        except Exception as exc:
            rows.append(EvaluationRow(example.text, example.expected_query, "", False, False,
                                      "backend" if execution_backend else "structural", f"Generation error: {exc}"))
            continue
        predicted = result.query.strip()
        expected = example.expected_query.strip()

        exact_match = normalize_query(predicted, example.target) == normalize_query(expected, example.target)
        execution_match = False
        execution_mode = "structural"
        backend_error: str | None = None
        if execution_backend is None:
            execution_match = structural_execution_match(example.target, predicted, expected)
        else:
            execution_mode = "backend"
            try:
                predicted_result = execution_backend(predicted, example)
                expected_result = _resolve_expected_execution_result(
                    example=example,
                    expected_query=expected,
                    execution_backend=execution_backend,
                )
                execution_match = comparator(predicted_result, expected_result)
            except Exception as exc:  # noqa: BLE001
                backend_error = f"{type(exc).__name__}: {exc}"
                execution_match = False

        rows.append(
            EvaluationRow(
                text=example.text,
                expected_query=expected,
                predicted_query=predicted,
                exact_match=exact_match,
                execution_match=execution_match,
                execution_mode=execution_mode,
                execution_backend_error=backend_error,
            )
        )

        if exact_match:
            exact_hits += 1
        if execution_match:
            exec_hits += 1

    total = len(rows)
    if total == 0:
        return EvaluationReport(total=0, exact_match_accuracy=0.0, execution_accuracy=0.0, rows=[])

    return EvaluationReport(
        total=total,
        exact_match_accuracy=exact_hits / total,
        execution_accuracy=exec_hits / total,
        rows=rows,
    )


async def aevaluate_examples(
    service: Text2QL,
    examples: list[DatasetExample],
    execution_backend: ExecutionBackend | None = None,
    execution_comparator: ExecutionComparator | None = None,
    concurrency: int = 10,
) -> EvaluationReport:
    """Concurrent version of evaluate_examples.

    All examples are evaluated in parallel up to ``concurrency`` simultaneous
    in-flight requests. Throughput depends on provider latency and rate limits.
    """
    comparator = execution_comparator or _default_execution_comparator
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    sem = asyncio.Semaphore(concurrency)

    async def _eval_one(example: DatasetExample) -> EvaluationRow:
        try:
            async with sem:
                result = await service.agenerate(
                    text=example.text,
                    target=example.target,
                    schema=example.schema,
                    mapping=example.mapping,
                    context=example.context,
                )
            if not result.executable:
                raise ValueError(f"{result.status}: {result.explanation}")
        except Exception as exc:
            return EvaluationRow(example.text, example.expected_query, "", False, False,
                                 "backend" if execution_backend else "structural", f"Generation error: {exc}")
        predicted = result.query.strip()
        expected = example.expected_query.strip()
        exact_match = normalize_query(predicted, example.target) == normalize_query(expected, example.target)
        execution_match = False
        execution_mode = "structural"
        backend_error: str | None = None

        if execution_backend is None:
            execution_match = structural_execution_match(example.target, predicted, expected)
        else:
            execution_mode = "backend"
            try:
                if asyncio.iscoroutinefunction(execution_backend):
                    predicted_result = await execution_backend(predicted, example)
                else:
                    predicted_result = await asyncio.to_thread(execution_backend, predicted, example)
                expected_result = await _aresolve_expected_execution_result(
                    example=example,
                    expected_query=expected,
                    execution_backend=execution_backend,
                )
                execution_match = comparator(predicted_result, expected_result)
            except Exception as exc:  # noqa: BLE001
                backend_error = f"{type(exc).__name__}: {exc}"

        return EvaluationRow(
            text=example.text,
            expected_query=expected,
            predicted_query=predicted,
            exact_match=exact_match,
            execution_match=execution_match,
            execution_mode=execution_mode,
            execution_backend_error=backend_error,
        )

    rows = list(await asyncio.gather(*[_eval_one(ex) for ex in examples]))
    total = len(rows)
    if total == 0:
        return EvaluationReport(total=0, exact_match_accuracy=0.0, execution_accuracy=0.0, rows=[])

    exact_hits = sum(1 for r in rows if r.exact_match)
    exec_hits = sum(1 for r in rows if r.execution_match)
    return EvaluationReport(
        total=total,
        exact_match_accuracy=exact_hits / total,
        execution_accuracy=exec_hits / total,
        rows=rows,
    )


async def _aresolve_expected_execution_result(
    example: DatasetExample,
    expected_query: str,
    execution_backend: ExecutionBackend,
) -> Any:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    if "expected_execution_result" in metadata:
        return metadata["expected_execution_result"]
    if "expected_execution" in metadata:
        return metadata["expected_execution"]
    if asyncio.iscoroutinefunction(execution_backend):
        return await execution_backend(expected_query, example)
    return await asyncio.to_thread(execution_backend, expected_query, example)


def normalize_query(query: str, target: str | None = None) -> str:
    """Normalize syntax while preserving literals, case-sensitive names and order."""
    if not query.strip():
        return ""
    target = target or ("sql" if re.match(r"\s*(?:select|with|values)\b", query, re.I) else "graphql")
    try:
        if target == "sql":
            from sqlglot import exp, parse_one
            tree = parse_one(query, read="sqlite")
            for identifier in tree.find_all(exp.Identifier):
                if identifier.this == identifier.this.lower() and re.fullmatch(r"[a-z_]\w*", identifier.this):
                    identifier.set("quoted", False)
            return tree.sql(normalize=True, comments=False)
        from graphql import parse, print_ast
        from graphql.utilities import strip_ignored_characters
        return strip_ignored_characters(print_ast(parse(query)))
    except Exception:
        # Invalid input is never repaired into a false positive.
        return query.strip()


def _default_execution_comparator(left: Any, right: Any) -> bool:
    return _stable_serialize(left) == _stable_serialize(right)


def _stable_serialize(value: Any) -> str:
    try:
        import json

        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _resolve_expected_execution_result(
    example: DatasetExample,
    expected_query: str,
    execution_backend: ExecutionBackend,
) -> Any:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    if "expected_execution_result" in metadata:
        return metadata["expected_execution_result"]
    if "expected_execution" in metadata:
        return metadata["expected_execution"]
    return execution_backend(expected_query, example)


def graphql_execution_match(predicted_query: str, expected_query: str) -> bool:
    from graphql import parse
    try:
        parse(predicted_query)
        parse(expected_query)
    except Exception:
        return False
    return normalize_query(predicted_query, "graphql") == normalize_query(expected_query, "graphql")


def structural_execution_match(target: str, predicted_query: str, expected_query: str) -> bool:
    if target.lower() == "sql":
        return sql_execution_match(predicted_query, expected_query)
    return graphql_execution_match(predicted_query, expected_query)


def sql_execution_match(predicted_query: str, expected_query: str) -> bool:
    """Conservative AST comparison, not a claim of execution equivalence."""
    from sqlglot import exp
    from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
    from sqlglot.optimizer.qualify import qualify
    from text2ql.query_validation import validate_sql

    def signature(query: str) -> str:
        tree = normalize_identifiers(validate_sql(query), dialect="sqlite")
        for identifier in tree.find_all(exp.Identifier):
            identifier.set("quoted", False)
        tree = qualify(tree, dialect="sqlite", infer_schema=True, identify=False,
                       validate_qualify_columns=True)
        if isinstance(tree, exp.Select):
            # Result labels do not change row values, but labels referenced in
            # ORDER BY / GROUP BY / HAVING must retain their original meaning.
            referenced = {
                column.name for key, clause in tree.args.items() if key != "expressions" and isinstance(clause, exp.Expr)
                for column in clause.find_all(exp.Column) if not column.table
            }
            tree.set("expressions", [item.this if isinstance(item, exp.Alias) and item.alias not in referenced else item
                                     for item in tree.expressions])
        return tree.sql(dialect="sqlite", identify=True, normalize=True, comments=False)

    try:
        return signature(predicted_query) == signature(expected_query)
    except Exception:
        # Failed qualification remains a mismatch; never drop clauses to repair it.
        return False
