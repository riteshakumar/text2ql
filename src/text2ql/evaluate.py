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

IDENTIFIER_PATTERN = r"[A-Za-z_]\w*"
QUOTED_IDENTIFIER_RE = rf'"({IDENTIFIER_PATTERN})"'
FROM_IDENTIFIER_RE = rf'\bfrom\s+"?({IDENTIFIER_PATTERN})"?\b'
JOIN_IDENTIFIER_RE = rf'\bjoin\s+"?({IDENTIFIER_PATTERN})"?\b'
SELECT_CLAUSE_RE = r"\bselect\s+(.*?)\s+\bfrom\b"
AS_ALIAS_RE = r"\s+as\s+\w+"
ORDER_BY_KEYWORD = "order by"
WHERE_SPLIT_TERMINATORS = (ORDER_BY_KEYWORD, "limit", "offset", ";")
ORDER_BY_SPLIT_TERMINATORS = ("limit", "offset", ";")
JOIN_ON_END_TERMINATORS = ("join", "where", ORDER_BY_KEYWORD, "limit", "offset", ";")


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
    """Normalize a SQL/GraphQL query for comparison.

    Strips leading/trailing whitespace, collapses internal whitespace, removes
    surrounding double-quotes from identifiers (e.g. ``"table"."col"`` →
    ``table.col``), and lowercases the whole string so that minor formatting
    differences don't cause false negatives.
    """
    q = re.sub(r"\s+", " ", query.strip())
    # Strip double-quote wrapping from identifiers: "foo" → foo
    q = _normalize_quoted_identifiers(q)
    # Remove trailing semicolon
    q = q.rstrip(";").rstrip()
    return q.lower()


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
) -> tuple[
    str,
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
    tuple[str, ...],
] | None:
    table = _parse_from_table(query)
    if table is None:
        return None
    source_tables = _extract_source_tables(query)
    join_predicates = _extract_join_predicates(query)

    fields = _parse_select_fields(query)
    if fields is None:
        return None

    filters = _parse_where_filters(query)

    ordering = _extract_ordering_signature(query)

    return (
        table,
        source_tables,
        fields,
        tuple(sorted(filters.items())),
        tuple(ordering),
        join_predicates,
    )


def _extract_source_tables(query: str) -> tuple[str, ...]:
    tables: list[str] = []
    for pattern in (FROM_IDENTIFIER_RE, JOIN_IDENTIFIER_RE):
        for match in re.finditer(pattern, query, re.I):
            tables.append(match.group(1).lower())
    return tuple(sorted(_dedupe(tables)))


def _extract_join_predicates(query: str) -> tuple[str, ...]:
    """Collect normalized JOIN ON predicates from SQL query text."""
    predicates: list[str] = []
    for segment in re.split(r"\b(?:left|right|inner|outer|cross)?\s*join\b", query, flags=re.I):
        on_clause = _slice_clause(segment, "on", JOIN_ON_END_TERMINATORS)
        if on_clause:
            normalized = _normalize_quoted_identifiers(re.sub(r"\s+", " ", on_clause.strip())).lower()
            predicates.append(normalized)
    return tuple(sorted(_dedupe(predicates)))


def _extract_ordering_signature(query: str) -> list[str]:
    order_blob = _slice_clause(query, ORDER_BY_KEYWORD, ORDER_BY_SPLIT_TERMINATORS)
    if not order_blob:
        return []
    ordering: list[str] = []
    for clause in [part.strip() for part in order_blob.split(",") if part.strip()]:
        normalized_clause = _normalize_quoted_identifiers(clause).lower()
        pieces = normalized_clause.split()
        if not pieces:
            continue
        raw_col = pieces[0]
        col = raw_col.split(".")[-1] if "." in raw_col else raw_col
        direction = pieces[1] if len(pieces) > 1 and pieces[1] in {"asc", "desc"} else "asc"
        ordering.append(f"{col} {direction}")
    return ordering


def _parse_from_table(query: str) -> str | None:
    table_match = re.search(FROM_IDENTIFIER_RE, query, re.I)
    if table_match is None:
        return None
    return table_match.group(1).lower()


def _parse_select_fields(query: str) -> tuple[str, ...] | None:
    select_match = re.search(SELECT_CLAUSE_RE, query, re.I | re.S)
    if not select_match:
        return None
    raw_fields = [segment.strip() for segment in select_match.group(1).split(",") if segment.strip()]
    normalized = [_normalize_select_field(segment) for segment in raw_fields]
    return tuple(sorted(_dedupe([field for field in normalized if field])))


def _normalize_select_field(field: str) -> str:
    raw = re.sub(AS_ALIAS_RE, "", field, flags=re.I).strip()
    unquoted = ".".join(_strip_sql_quotes(part) for part in raw.split("."))
    if "." in unquoted:
        unquoted = unquoted.split(".")[-1]
    return unquoted.lower()


def _parse_where_filters(query: str) -> dict[str, str]:
    where_blob = _slice_clause(query, "where", WHERE_SPLIT_TERMINATORS)
    if not where_blob:
        return {}

    filters: dict[str, str] = {}
    for condition in re.split(r"\s+and\s+", where_blob, flags=re.I):
        condition = condition.strip()
        if not condition:
            continue
        key, op, value = _parse_filter_condition(condition)
        if key and op:
            filters[f"{key} {op}"] = value
    return filters


def _parse_filter_condition(condition: str) -> tuple[str, str, str]:
    op_match = re.search(
        r"\s*(>=|<=|!=|=|>|<|\bnot\s+in\b|\bin\b|\bis\s+not\s+null\b|\bis\s+null\b)\s*",
        condition,
        flags=re.I,
    )
    if op_match is None:
        return "", "", ""
    key = _normalize_filter_key(condition[: op_match.start()].strip())
    op = re.sub(r"\s+", " ", op_match.group(1).strip().lower())
    value = condition[op_match.end() :].strip().lower()
    return key, op, value


def _normalize_filter_key(raw_key: str) -> str:
    key = ".".join(_strip_sql_quotes(part) for part in raw_key.split(".")).lower()
    return key.split(".")[-1] if "." in key else key


def _normalize_quoted_identifiers(value: str) -> str:
    return re.sub(QUOTED_IDENTIFIER_RE, r"\1", value)


def _slice_clause(query: str, start_keyword: str, end_keywords: tuple[str, ...]) -> str:
    lowered = query.lower()
    start_marker = f"{start_keyword.lower()} "
    start = lowered.find(start_marker)
    if start < 0:
        return ""
    body_start = start + len(start_marker)
    body = query[body_start:]
    lowered_body = lowered[body_start:]
    end = len(body)
    for marker in end_keywords:
        idx = lowered_body.find(f" {marker}")
        if idx >= 0:
            end = min(end, idx)
    semicolon_index = lowered_body.find(";")
    if semicolon_index >= 0:
        end = min(end, semicolon_index)
    return body[:end].strip()


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
