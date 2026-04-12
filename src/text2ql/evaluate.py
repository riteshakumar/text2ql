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
        result = service.generate(
            text=example.text,
            target=example.target,
            schema=example.schema,
            mapping=example.mapping,
            context=example.context,
        )
        predicted = result.query.strip()
        expected = example.expected_query.strip()

        exact_match = normalize_query(predicted) == normalize_query(expected)
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
    in-flight requests. On 100 examples at 500ms LLM latency this is ~50× faster
    than the serial version.
    """
    comparator = execution_comparator or _default_execution_comparator
    sem = asyncio.Semaphore(concurrency)

    async def _eval_one(example: DatasetExample) -> EvaluationRow:
        async with sem:
            result = await service.agenerate(
                text=example.text,
                target=example.target,
                schema=example.schema,
                mapping=example.mapping,
                context=example.context,
            )
        predicted = result.query.strip()
        expected = example.expected_query.strip()
        exact_match = normalize_query(predicted) == normalize_query(expected)
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


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


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
    pred = _parse_graphql_signature(predicted_query)
    exp = _parse_graphql_signature(expected_query)
    return pred == exp and pred is not None


def structural_execution_match(target: str, predicted_query: str, expected_query: str) -> bool:
    normalized_target = str(target).strip().lower()
    if normalized_target == "sql":
        return sql_execution_match(predicted_query, expected_query)
    return graphql_execution_match(predicted_query, expected_query)


def sql_execution_match(predicted_query: str, expected_query: str) -> bool:
    pred = _parse_sql_signature(predicted_query)
    exp = _parse_sql_signature(expected_query)
    return pred == exp and pred is not None


def _strip_sql_quotes(name: str) -> str:
    """Remove surrounding double-quotes from a quoted SQL identifier."""
    name = name.strip()
    if name.startswith('"') and name.endswith('"'):
        return name[1:-1]
    return name


def _parse_sql_signature(
    query: str,
) -> tuple[str, tuple[str, ...], tuple[tuple[str, str], ...], tuple[str, ...]] | None:
    # Support both quoted ("table") and unquoted (table) identifiers after FROM
    table_match = re.search(r'\bfrom\s+"?([A-Za-z_]\w*)"?\b', query, re.I)
    if not table_match:
        return None
    table = table_match.group(1).lower()

    select_match = re.search(r"\bselect\s+(.*?)\s+\bfrom\b", query, re.I | re.S)
    if not select_match:
        return None
    # Normalise field tokens: strip double-quotes and table prefix quotes
    raw_fields = [segment.strip() for segment in select_match.group(1).split(",") if segment.strip()]
    normalised_fields: list[str] = []
    for f in raw_fields:
        # Remove double-quote wrapping from each dot-separated part
        parts = f.split(".")
        unquoted = ".".join(_strip_sql_quotes(p) for p in parts)
        normalised_fields.append(unquoted.lower())
    fields = tuple(sorted(_dedupe(normalised_fields)))

    where_match = re.search(r"\bwhere\s+(.*?)(?:\border by\b|\blimit\b|\boffset\b|;|$)", query, re.I | re.S)
    filters: dict[str, str] = {}
    if where_match:
        for condition in re.split(r"\s+and\s+", where_match.group(1), flags=re.I):
            condition = condition.strip()
            if not condition:
                continue
            parts = re.split(r"\s*(>=|<=|!=|=|>|<|in|not in|is null|is not null)\s*", condition, maxsplit=1, flags=re.I)
            if len(parts) >= 2:
                # Strip quotes from key (column reference)
                raw_key = parts[0].strip()
                key_parts = raw_key.split(".")
                key = ".".join(_strip_sql_quotes(p) for p in key_parts).lower()
                op = parts[1].strip().lower()
                value = parts[2].strip().lower() if len(parts) > 2 else ""
                filters[f"{key} {op}"] = value

    # Support both quoted and unquoted identifiers in ORDER BY
    order_match = re.search(r'\border by\s+"?([A-Za-z_]\w*)"?\."?([A-Za-z_][\w]*)"?\s*(asc|desc)?\b', query, re.I)
    if not order_match:
        order_match = re.search(r'\border by\s+"?([A-Za-z_][\w.]*)"?\s*(asc|desc)?\b', query, re.I)
    ordering: list[str] = []
    if order_match:
        if order_match.lastindex == 3:
            col = f"{order_match.group(1).lower()}.{order_match.group(2).lower()}"
            direction = (order_match.group(3) or "asc").lower()
        else:
            col = order_match.group(1).lower()
            direction = (order_match.group(2) or "asc").lower()
        ordering.append(f"{col} {direction}")

    return (table, fields, tuple(sorted(filters.items())), tuple(ordering))


def _parse_graphql_signature(query: str) -> tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]] | None:
    entity_match = re.search(r"\{\s*([A-Za-z_]\w*)\s*(\([^)]*\))?\s*\{", query)
    if not entity_match:
        return None

    entity = entity_match.group(1)
    args_blob = entity_match.group(2) or ""

    block_match = re.search(r"\{\s*[A-Za-z_]\w*\s*(?:\([^)]*\))?\s*\{(.*?)\}\s*\}", query, re.S)
    if not block_match:
        return None

    fields = tuple(sorted(_extract_fields(block_match.group(1))))
    filters = tuple(sorted(_extract_filters(args_blob).items()))
    return (entity, fields, filters)


def _extract_fields(selection_block: str) -> list[str]:
    fields = re.findall(r"[A-Za-z_]\w*", selection_block)
    return _dedupe(fields)


def _extract_filters(args_blob: str) -> dict[str, str]:
    if not args_blob:
        return {}

    body = args_blob.strip()[1:-1].strip()
    if not body:
        return {}

    pairs: dict[str, str] = {}
    for item in body.split(","):
        if ":" not in item:
            continue
        key, raw_value = item.split(":", 1)
        key = key.strip()
        value = raw_value.strip().strip('"')
        if key:
            pairs[key] = value
    return pairs


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
