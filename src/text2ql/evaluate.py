from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from text2ql.core import Text2QL
from text2ql.dataset import DatasetExample

ExecutionBackend = Callable[[str, DatasetExample], Any]
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
            execution_match = graphql_execution_match(predicted, expected)
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
