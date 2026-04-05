from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from text2ql.core import Text2QL
from text2ql.dataset import DatasetExample, generate_synthetic_examples
from text2ql.json_execution import execute_query_result_on_json
from text2ql.mapping import generate_hybrid_mapping
from text2ql.providers.base import LLMProvider
from text2ql.providers.openai_compatible import OpenAICompatibleProvider
from text2ql.providers.rule_based import RuleBasedProvider
from text2ql.schema_config import infer_schema_from_json_payload
from text2ql.types import QueryResult


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
    parser.add_argument(
        "--variants-per-example",
        type=int,
        default=1,
        help="Synthetic variants per prompt when using rewrite plugins/domain.",
    )
    parser.add_argument(
        "--rewrite-plugins",
        default="",
        help="Comma-separated rewrite plugins (e.g. generic,portfolio).",
    )
    parser.add_argument(
        "--domain",
        default="",
        help="Optional domain hint for synthetic rewrites.",
    )
    parser.add_argument(
        "--expected-query",
        default="",
        help="Expected query for execution-match evaluation.",
    )
    parser.add_argument(
        "--expected-query-file",
        default="",
        help="Path to expected query file for execution-match evaluation.",
    )
    parser.add_argument(
        "--expected-execution-file",
        default="",
        help="Path to expected execution JSON payload (rows/object).",
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
    prompts, metadata, use_synthetic = _build_prompts_and_metadata(args, schema, mapping)
    expected_query, expected_execution, execution_eval_enabled, execution_data_payload = _load_execution_eval_inputs(
        args
    )
    results, execution_matches, execution_total = _generate_result_payloads(
        args=args,
        service=service,
        schema=schema,
        mapping=mapping,
        prompts=prompts,
        metadata=metadata,
        execution_eval_enabled=execution_eval_enabled,
        execution_data_payload=execution_data_payload,
        expected_query=expected_query,
        expected_execution=expected_execution,
    )

    if len(results) == 1 and not use_synthetic and not execution_eval_enabled:
        print(results[0]["query"])
        return

    summary = {
        "total_prompts": len(results),
        "execution_matches": execution_matches,
        "execution_total": execution_total,
        "execution_accuracy": (execution_matches / execution_total) if execution_total else 0.0,
    }
    print(json.dumps({"results": results, "summary": summary}, indent=2))


def _build_prompts_and_metadata(
    args: argparse.Namespace,
    schema: dict[str, Any] | None,
    mapping: dict[str, Any] | None,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    plugins = [token.strip() for token in args.rewrite_plugins.split(",") if token.strip()]
    use_synthetic = bool(plugins or args.domain or max(1, args.variants_per_example) > 1)
    if not use_synthetic:
        return [args.text], [{}], False
    seed = DatasetExample(
        text=args.text,
        target=args.target,
        expected_query="",
        schema=schema,
        mapping=mapping,
    )
    synthetic = generate_synthetic_examples(
        [seed],
        variants_per_example=max(1, args.variants_per_example),
        rewrite_plugins=plugins or None,
        domain=args.domain or None,
    )
    return [example.text for example in synthetic], [example.metadata for example in synthetic], True


def _load_execution_eval_inputs(
    args: argparse.Namespace,
) -> tuple[str, Any, bool, dict[str, Any] | None]:
    expected_query = args.expected_query.strip()
    if args.expected_query_file:
        expected_query = Path(args.expected_query_file).read_text(encoding="utf-8").strip()
    expected_execution = None
    if args.expected_execution_file:
        expected_execution = json.loads(Path(args.expected_execution_file).read_text(encoding="utf-8"))
    execution_eval_enabled = bool(expected_query or args.expected_execution_file)
    execution_data_payload = _read_json_file(args.data_file) if args.data_file else None
    return expected_query, expected_execution, execution_eval_enabled, execution_data_payload


def _generate_result_payloads(
    args: argparse.Namespace,
    service: Text2QL,
    schema: dict[str, Any] | None,
    mapping: dict[str, Any] | None,
    prompts: list[str],
    metadata: list[dict[str, Any]],
    execution_eval_enabled: bool,
    execution_data_payload: dict[str, Any] | None,
    expected_query: str,
    expected_execution: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    results: list[dict[str, Any]] = []
    execution_matches = 0
    execution_total = 0
    for idx, prompt in enumerate(prompts):
        result = service.generate(
            text=prompt,
            target=args.target,
            schema=schema,
            mapping=mapping,
            context={
                "mode": args.mode,
                "language": args.language,
                "system_context": args.system_context,
            },
        )
        payload: dict[str, Any] = {"prompt": prompt, "query": result.query, "metadata": metadata[idx]}
        if execution_eval_enabled:
            _apply_execution_evaluation(
                payload=payload,
                result=result,
                target=args.target,
                execution_data_payload=execution_data_payload,
                expected_query=expected_query,
                expected_execution=expected_execution,
            )
            if "execution_match" in payload:
                execution_total += 1
                if payload["execution_match"]:
                    execution_matches += 1
        results.append(payload)
    return results, execution_matches, execution_total


def _apply_execution_evaluation(
    payload: dict[str, Any],
    result: QueryResult,
    target: str,
    execution_data_payload: dict[str, Any] | None,
    expected_query: str,
    expected_execution: Any,
) -> None:
    if execution_data_payload is None:
        payload["execution_eval_warning"] = "execution evaluation requires --data-file JSON payload"
        return
    rows, note = execute_query_result_on_json(result, execution_data_payload, root_key="portfolio_data")
    payload["execution_rows"] = rows
    payload["execution_note"] = note

    expected_rows = expected_execution
    expected_note = None
    if expected_rows is None and expected_query:
        expected_result = QueryResult(
            query=expected_query,
            target=target,
            confidence=1.0,
            explanation="expected query",
        )
        expected_rows, expected_note = execute_query_result_on_json(
            expected_result, execution_data_payload, root_key="portfolio_data"
        )
    if expected_note:
        payload["execution_eval_warning"] = expected_note
        return
    payload["execution_match"] = _stable_json(rows) == _stable_json(expected_rows)


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


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


if __name__ == "__main__":
    main()
