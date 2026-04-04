from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from text2ql.providers.base import LLMProvider


@dataclass(slots=True)
class DatasetExample:
    text: str
    target: str
    expected_query: str
    schema: dict[str, Any] | None = None
    mapping: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def ingest_dataset(path: str | Path) -> list[DatasetExample]:
    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".jsonl":
        return _ingest_jsonl(dataset_path)
    if dataset_path.suffix.lower() == ".json":
        return _ingest_json(dataset_path)
    raise ValueError("Unsupported dataset format. Use .json or .jsonl")


def generate_synthetic_examples(
    seed_examples: list[DatasetExample],
    variants_per_example: int = 1,
    provider: LLMProvider | None = None,
) -> list[DatasetExample]:
    synthetic: list[DatasetExample] = []
    for example in seed_examples:
        for i in range(max(0, variants_per_example)):
            rewritten_text = _rewrite_text(example.text, i)
            synthetic.append(
                DatasetExample(
                    text=rewritten_text,
                    target=example.target,
                    expected_query=example.expected_query,
                    schema=example.schema,
                    mapping=example.mapping,
                    context={**example.context, "synthetic": True},
                    metadata={**example.metadata, "synthetic_variant": i + 1},
                )
            )

            if provider is not None:
                hook_payload = {
                    "seed_text": example.text,
                    "synthetic_text": rewritten_text,
                    "target": example.target,
                }
                try:
                    provider.complete(
                        "Return a concise quality note for synthetic query generation.",
                        json.dumps(hook_payload),
                    )
                except Exception:
                    # Hook is optional and should never block generation.
                    pass
    return synthetic


def _ingest_jsonl(path: Path) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            examples.append(_parse_example(payload))
    return examples


def _ingest_json(path: Path) -> list[DatasetExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("JSON dataset must be a list of examples")
    return [_parse_example(item) for item in payload]


def _parse_example(payload: dict[str, Any]) -> DatasetExample:
    if not isinstance(payload, dict):
        raise ValueError("Each dataset item must be an object")

    text = payload.get("text")
    expected_query = payload.get("expected_query")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Example field 'text' is required and must be a string")
    if not isinstance(expected_query, str) or not expected_query.strip():
        raise ValueError("Example field 'expected_query' is required and must be a string")

    target = payload.get("target", "graphql")
    schema = payload.get("schema")
    mapping = payload.get("mapping")
    context = payload.get("context", {})
    metadata = payload.get("metadata", {})

    if not isinstance(target, str):
        raise ValueError("Example field 'target' must be a string")
    if schema is not None and not isinstance(schema, dict):
        raise ValueError("Example field 'schema' must be an object when provided")
    if mapping is not None and not isinstance(mapping, dict):
        raise ValueError("Example field 'mapping' must be an object when provided")
    if not isinstance(context, dict):
        raise ValueError("Example field 'context' must be an object")
    if not isinstance(metadata, dict):
        raise ValueError("Example field 'metadata' must be an object")

    return DatasetExample(
        text=text,
        target=target,
        expected_query=expected_query,
        schema=schema,
        mapping=mapping,
        context=context,
        metadata=metadata,
    )


def _rewrite_text(text: str, variant_index: int) -> str:
    rewrites = [
        text.replace("show", "list"),
        text.replace("list", "show"),
        text.replace("top", "first"),
        text,
    ]
    candidate = rewrites[variant_index % len(rewrites)]
    return candidate if candidate.strip() else text
