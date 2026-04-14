from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from text2ql._cli_utils import (
    as_unit_float,
    dynamic_synthetic_meta,
    execute_sql_on_json,
    stable_json,
)
from text2ql.core import Text2QL
from text2ql.dataset import DatasetExample, generate_synthetic_examples
from text2ql.json_execution import execute_query_result_on_json
from text2ql.mapping import generate_hybrid_mapping
from text2ql.providers.base import LLMProvider
from text2ql.providers.openai_compatible import OpenAICompatibleProvider
from text2ql.providers.rule_based import RuleBasedProvider
from text2ql.rewrite import rewrite_user_utterance
from text2ql.schema_config import infer_schema_from_json_payload
from text2ql.types import QueryResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text to Query Language CLI")
    parser.add_argument("text", nargs="?", default="", help="Natural language request")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--target", default="graphql", help="Target query language")
    parser.add_argument(
        "--mode",
        default="deterministic",
        choices=["deterministic", "llm", "function_calling"],
        help="Generation mode (function_calling uses structured-output intent path).",
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
        help="LLM provider adapter when --mode llm or --mode function_calling",
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
        "--llm-api-key",
        default="",
        help="Optional API key for openai-compatible provider (falls back to OPENAI_API_KEY/TEXT2QL_API_KEY).",
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
        "--llm-rewrite",
        default="off",
        choices=["off", "on"],
        help="Enable schema-aware LLM utterance rewrite before query generation.",
    )
    parser.add_argument(
        "--benchmark",
        default="",
        choices=["", "spider", "bird"],
        help="Run a standard benchmark (spider or bird) and print report.",
    )
    parser.add_argument(
        "--benchmark-path",
        default="",
        help="Path to the benchmark dataset root directory.",
    )
    parser.add_argument(
        "--benchmark-split",
        default="dev",
        help="Benchmark split to evaluate (dev, train).",
    )
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=0,
        help="Limit the number of benchmark examples (0 = all).",
    )
    parser.add_argument(
        "--benchmark-db",
        default="",
        help="Only evaluate benchmark examples for this db_id.",
    )
    parser.add_argument(
        "--benchmark-mode",
        default="execution",
        choices=["exact", "structural", "execution"],
        help="Benchmark evaluation mode.",
    )
    parser.add_argument(
        "--benchmark-verbose",
        action="store_true",
        help="Show per-example failure details in benchmark report.",
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
    parser.add_argument(
        "--execute-on-payload",
        action="store_true",
        help="Execute generated query on --data-file payload without requiring expected-query comparison.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.version:
        try:
            from importlib.metadata import version
            print(f"text2ql {version('text2ql')}")
        except Exception:
            print("text2ql (version unavailable)")
        return

    if args.generate_hybrid_mapping:
        mapping = _build_hybrid_mapping_from_args(args)
        _emit_json_output(mapping, args.mapping_output_file)
        return

    if args.benchmark:
        _run_benchmark_from_args(args)
        return

    if not args.text.strip():
        parser.error("text is required unless --generate-hybrid-mapping or --benchmark is used")

    schema = _load_json_object(args.schema, args.schema_file)
    mapping = _load_json_object(args.mapping, args.mapping_file)
    schema, mapping = _resolve_generation_schema_mapping(args, schema, mapping)
    try:
        service = Text2QL(provider=_build_provider(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    prompts, metadata, use_synthetic = _build_prompts_and_metadata(args, schema, mapping)
    expected_query, expected_execution, execution_eval_enabled, execution_data_payload = _load_execution_eval_inputs(
        args
    )
    results, execution_matches, execution_total = _generate_result_payloads(
        args=args,
        service=service,
        rewrite_provider=_build_rewrite_provider(args),
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
        _warn_validation_notes(results[0].get("engine_metadata", {}))
        return

    summary = {
        "total_prompts": len(results),
        "execution_matches": execution_matches,
        "execution_total": execution_total,
        "execution_accuracy": (execution_matches / execution_total) if execution_total else 0.0,
    }
    print(json.dumps({"results": results, "summary": summary}, indent=2))


def _run_benchmark_from_args(args: argparse.Namespace) -> None:
    """Load and run a Spider or BIRD benchmark from CLI args."""
    from text2ql.benchmarks.runner import BenchmarkConfig, format_report, run_benchmark

    if not args.benchmark_path:
        print("error: --benchmark-path is required when using --benchmark", file=sys.stderr)
        sys.exit(1)

    limit = args.benchmark_limit if args.benchmark_limit > 0 else None
    db_filter = args.benchmark_db or None

    if args.benchmark == "spider":
        from text2ql.benchmarks.spider import load_spider

        examples = load_spider(
            args.benchmark_path,
            split=args.benchmark_split,
            limit=limit,
            db_filter=db_filter,
        )
    elif args.benchmark == "bird":
        from text2ql.benchmarks.bird import load_bird

        examples = load_bird(
            args.benchmark_path,
            split=args.benchmark_split,
            limit=limit,
            db_filter=db_filter,
        )
    else:
        print(f"error: unknown benchmark '{args.benchmark}'", file=sys.stderr)
        sys.exit(1)

    if not examples:
        print("No benchmark examples loaded. Check --benchmark-path and filters.", file=sys.stderr)
        sys.exit(1)

    service = Text2QL(provider=_build_provider(args))
    config = BenchmarkConfig(
        mode=args.benchmark_mode,
        service=service,
    )

    print(f"Running {args.benchmark.upper()} benchmark ({len(examples)} examples, mode={args.benchmark_mode})...", file=sys.stderr)
    report = run_benchmark(examples, config=config)
    print(format_report(report, verbose=args.benchmark_verbose))

    # Also emit machine-readable JSON summary to stdout
    summary = {
        "benchmark": report.benchmark,
        "split": report.split,
        "total": report.total,
        "exact_match_accuracy": report.exact_match_accuracy,
        "structural_accuracy": report.structural_accuracy,
        "execution_accuracy": report.execution_accuracy,
        "errors": report.errors,
        "elapsed_seconds": round(report.elapsed_seconds, 2),
        "accuracy_by_difficulty": report.accuracy_by_difficulty,
    }
    print(json.dumps(summary, indent=2))


def _resolve_generation_schema_mapping(
    args: argparse.Namespace,
    schema: dict[str, Any] | None,
    mapping: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not args.data_file:
        return schema, mapping

    data_payload = _read_json_file(args.data_file)
    root_payload = data_payload.get("portfolio_data", data_payload)
    if not isinstance(root_payload, dict):
        return schema, mapping

    inferred_schema = infer_schema_from_json_payload(root_payload)
    schema_payload = schema or inferred_schema
    hybrid_mapping = generate_hybrid_mapping(
        schema_payload=schema_payload,
        data_payload=root_payload,
        overrides=mapping,
    )
    # Keep caller-provided schema for generation; only use payload-inferred
    # schema as a fallback when no schema was supplied.
    return schema_payload, hybrid_mapping


def _build_prompts_and_metadata(
    args: argparse.Namespace,
    schema: dict[str, Any] | None,
    mapping: dict[str, Any] | None,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    plugins = [token.strip() for token in args.rewrite_plugins.split(",") if token.strip()]
    use_synthetic = bool(plugins or args.domain or max(1, args.variants_per_example) > 1)
    if not use_synthetic:
        seed_meta = {
            "synthetic_domain": args.domain or None,
            "synthetic_rewrite_source": "seed",
        }
        return [args.text], [seed_meta], False
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
    execution_eval_enabled = bool(args.execute_on_payload or expected_query or args.expected_execution_file)
    execution_data_payload = _read_json_file(args.data_file) if args.data_file else None
    return expected_query, expected_execution, execution_eval_enabled, execution_data_payload


def _generate_result_payloads(
    args: argparse.Namespace,
    service: Text2QL,
    rewrite_provider: LLMProvider | None,
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
        rewritten_prompt, rewrite_meta = _rewrite_prompt_if_enabled(
            args=args,
            prompt=prompt,
            rewrite_provider=rewrite_provider,
            schema=schema,
            mapping=mapping,
        )
        result = service.generate(
            text=rewritten_prompt,
            target=args.target,
            schema=schema,
            mapping=mapping,
            context={
                "mode": args.mode,
                "language": args.language,
                "system_context": args.system_context,
            },
        )
        dynamic_meta = dynamic_synthetic_meta(
            base_meta=metadata[idx] if idx < len(metadata) else {},
            seed_prompt=args.text,
            active_prompt=prompt,
            engine_confidence=as_unit_float(result.confidence, default=0.5),
            rewrite_meta=rewrite_meta if (args.llm_rewrite == "on" and rewrite_provider is not None) else None,
        )
        payload = _build_generation_payload(prompt, rewritten_prompt, rewrite_meta, result, dynamic_meta)
        if execution_eval_enabled:
            _apply_execution_evaluation(
                payload=payload,
                result=result,
                target=args.target,
                execution_data_payload=execution_data_payload,
                expected_query=expected_query,
                expected_execution=expected_execution,
            )
            execution_matches, execution_total = _update_execution_counters(
                payload=payload,
                execution_matches=execution_matches,
                execution_total=execution_total,
            )
        results.append(payload)
    return results, execution_matches, execution_total


def _rewrite_prompt_if_enabled(
    args: argparse.Namespace,
    prompt: str,
    rewrite_provider: LLMProvider | None,
    schema: dict[str, Any] | None,
    mapping: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if args.llm_rewrite != "on":
        return prompt, {"applied": False, "reason": "disabled"}
    if rewrite_provider is None:
        return prompt, {"applied": False, "reason": "missing_api_key_or_provider"}
    rewritten_prompt, rewrite_meta = rewrite_user_utterance(
        text=prompt,
        target=args.target,
        schema=schema,
        mapping=mapping,
        provider=rewrite_provider,
        system_context=args.system_context,
    )
    return rewritten_prompt, rewrite_meta


def _build_generation_payload(
    prompt: str,
    rewritten_prompt: str,
    rewrite_meta: dict[str, Any],
    result: QueryResult,
    dynamic_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "rewritten_prompt": rewritten_prompt,
        "rewrite": rewrite_meta,
        "query": result.query,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "synthetic": dynamic_meta,
        "metadata": dynamic_meta,
        "engine_metadata": result.metadata,
    }


def _update_execution_counters(
    payload: dict[str, Any],
    execution_matches: int,
    execution_total: int,
) -> tuple[int, int]:
    if "execution_match" not in payload:
        return execution_matches, execution_total
    execution_total += 1
    if payload["execution_match"]:
        execution_matches += 1
    return execution_matches, execution_total


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
    if str(target).strip().lower() == "sql":
        rows, note = execute_sql_on_json(result.query, execution_data_payload, root_key="portfolio_data")
    else:
        rows, note = execute_query_result_on_json(result, execution_data_payload, root_key="portfolio_data")
    payload["execution_rows"] = rows
    payload["execution_note"] = note

    expected_rows = expected_execution
    expected_note = None
    if expected_rows is None and expected_query:
        if str(target).strip().lower() == "sql":
            expected_rows, expected_note = execute_sql_on_json(
                expected_query, execution_data_payload, root_key="portfolio_data"
            )
        else:
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
    if expected_rows is None:
        return
    payload["execution_match"] = stable_json(rows) == stable_json(expected_rows)


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
    if args.mode not in {"llm", "function_calling"}:
        return RuleBasedProvider()
    if args.llm_provider == "rule-based":
        return RuleBasedProvider()
    api_key = (
        (args.llm_api_key or "").strip()
        or (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("TEXT2QL_API_KEY") or "").strip()
    )
    if not api_key:
        raise ValueError(
            "LLM mode requires an API key. Pass --llm-api-key "
            "or set OPENAI_API_KEY/TEXT2QL_API_KEY."
        )
    return OpenAICompatibleProvider(
        api_key=api_key,
        model=args.llm_model,
        base_url=args.llm_base_url,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff,
        use_structured_output=(args.mode == "function_calling"),
    )


def _build_rewrite_provider(args: argparse.Namespace) -> LLMProvider | None:
    if args.llm_rewrite != "on":
        return None
    if args.llm_provider == "rule-based":
        return RuleBasedProvider()
    api_key = (
        (args.llm_api_key or "").strip()
        or (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("TEXT2QL_API_KEY") or "").strip()
    )
    if not api_key:
        return None
    return OpenAICompatibleProvider(
        api_key=api_key,
        model=args.llm_model,
        base_url=args.llm_base_url,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff,
    )


def _warn_validation_notes(engine_metadata: dict[str, Any]) -> None:
    """Print any validation notes from the engine to stderr as warnings."""
    notes = engine_metadata.get("validation_notes", [])
    if notes:
        for note in notes:
            print(f"warning: {note}", file=sys.stderr)


if __name__ == "__main__":
    main()
