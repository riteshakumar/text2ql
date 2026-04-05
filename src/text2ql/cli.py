from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from text2ql.core import Text2QL
from text2ql.mapping import generate_hybrid_mapping
from text2ql.providers.base import LLMProvider
from text2ql.providers.openai_compatible import OpenAICompatibleProvider
from text2ql.providers.rule_based import RuleBasedProvider
from text2ql.schema_config import infer_schema_from_json_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text to Query Language CLI")
    parser.add_argument("text", nargs="?", default="", help="Natural language request")
    parser.add_argument("--target", default="graphql", help="Target query language")
    parser.add_argument(
        "--mode",
        default="deterministic",
        choices=["deterministic", "llm"],
        help="Generation mode",
    )
    parser.add_argument(
        "--language",
        default="english",
        help="Prompt/constraint language profile (currently: english, en)",
    )
    parser.add_argument(
        "--system-context",
        default="",
        help="Optional extra system context injected in llm mode.",
    )
    parser.add_argument(
        "--llm-provider",
        default="openai-compatible",
        choices=["openai-compatible", "rule-based"],
        help="LLM provider adapter when --mode llm",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model name for openai-compatible provider",
    )
    parser.add_argument(
        "--llm-base-url",
        default="https://api.openai.com/v1",
        help="Base URL for openai-compatible chat completions API",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=2,
        help="Max retry attempts for provider request failures (429/network)",
    )
    parser.add_argument(
        "--llm-retry-backoff",
        type=float,
        default=1.5,
        help="Base backoff seconds between retries (used when Retry-After is absent)",
    )
    parser.add_argument(
        "--schema",
        default="",
        help="Schema as JSON string, e.g. '{\"entities\":[\"users\"],\"fields\":[\"id\",\"name\"]}'",
    )
    parser.add_argument(
        "--schema-file",
        default="",
        help="Path to schema JSON file",
    )
    parser.add_argument(
        "--mapping",
        default="",
        help="Mapping as JSON string, e.g. '{\"fields\":{\"mail\":\"email\"}}'",
    )
    parser.add_argument(
        "--mapping-file",
        default="",
        help="Path to mapping JSON file",
    )
    parser.add_argument(
        "--generate-hybrid-mapping",
        action="store_true",
        help="Generate hybrid mapping (auto baseline + optional overrides) and exit.",
    )
    parser.add_argument(
        "--data-file",
        default="",
        help="JSON data file used for hybrid mapping generation.",
    )
    parser.add_argument(
        "--mapping-overrides",
        default="",
        help="Override mapping as JSON string for hybrid mapping generation.",
    )
    parser.add_argument(
        "--mapping-overrides-file",
        default="",
        help="Override mapping JSON file for hybrid mapping generation.",
    )
    parser.add_argument(
        "--mapping-output-file",
        default="",
        help="Output path for generated hybrid mapping JSON.",
    )
    return parser


def _read_json_file(path: str) -> dict[str, Any]:
    content = Path(path).read_text(encoding="utf-8")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file at '{path}' must contain an object at top level")
    return payload


def _load_json_object(inline_value: str, file_path: str) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if file_path:
        merged.update(_read_json_file(file_path))
    if inline_value:
        inline_payload = json.loads(inline_value)
        if not isinstance(inline_payload, dict):
            raise ValueError("Inline JSON value must contain an object at top level")
        merged.update(inline_payload)
    return merged or None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_hybrid_mapping:
        mapping = _build_hybrid_mapping_from_args(args)
        _emit_json_output(mapping, args.mapping_output_file)
        return

    if not args.text.strip():
        parser.error("text is required unless --generate-hybrid-mapping is used")

    schema = _load_json_object(args.schema, args.schema_file)
    mapping = _load_json_object(args.mapping, args.mapping_file)
    service = Text2QL(provider=_build_provider(args))
    result = service.generate(
        text=args.text,
        target=args.target,
        schema=schema,
        mapping=mapping,
        context={
            "mode": args.mode,
            "language": args.language,
            "system_context": args.system_context,
        },
    )

    print(result.query)


def _build_hybrid_mapping_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if not args.data_file:
        raise ValueError("--data-file is required with --generate-hybrid-mapping")

    data_payload = _read_json_file(args.data_file)
    schema_payload = _load_json_object(args.schema, args.schema_file)
    if schema_payload is None:
        schema_payload = infer_schema_from_json_payload(data_payload)
    overrides = _load_json_object(args.mapping_overrides, args.mapping_overrides_file)

    return generate_hybrid_mapping(
        schema_payload=schema_payload,
        data_payload=data_payload,
        overrides=overrides,
    )


def _emit_json_output(payload: dict[str, Any], output_path: str) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(str(path))
        return
    print(json.dumps(payload, indent=2))


def _build_provider(args: argparse.Namespace) -> LLMProvider:
    if args.mode != "llm":
        return RuleBasedProvider()
    if args.llm_provider == "rule-based":
        return RuleBasedProvider()
    return OpenAICompatibleProvider(
        model=args.llm_model,
        base_url=args.llm_base_url,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff,
    )


if __name__ == "__main__":
    main()
