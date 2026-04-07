from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

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
    schema, mapping = _resolve_generation_schema_mapping(args, schema, mapping)
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
    return inferred_schema, hybrid_mapping


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
        rewritten_prompt = prompt
        rewrite_meta: dict[str, Any] = {"applied": False, "reason": "disabled"}
        if args.llm_rewrite == "on" and args.mode == "llm":
            rewritten_prompt, rewrite_meta = rewrite_user_utterance(
                text=prompt,
                target=args.target,
                schema=schema,
                mapping=mapping,
                provider=service.provider,
                system_context=args.system_context,
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
        dynamic_meta = _dynamic_synthetic_meta(
            base_meta=metadata[idx] if idx < len(metadata) else {},
            seed_prompt=args.text,
            active_prompt=prompt,
            engine_confidence=_as_unit_float(result.confidence, default=0.5),
            rewrite_meta=rewrite_meta if (args.llm_rewrite == "on" and args.mode == "llm") else None,
        )
        payload: dict[str, Any] = {
            "prompt": prompt,
            "rewritten_prompt": rewritten_prompt,
            "rewrite": rewrite_meta,
            "query": result.query,
            "synthetic": dynamic_meta,
            "metadata": dynamic_meta,
            "engine_metadata": result.metadata,
        }
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
    if str(target).strip().lower() == "sql":
        rows, note = _execute_sql_on_json(result.query, execution_data_payload, root_key="portfolio_data")
    else:
        rows, note = execute_query_result_on_json(result, execution_data_payload, root_key="portfolio_data")
    payload["execution_rows"] = rows
    payload["execution_note"] = note

    expected_rows = expected_execution
    expected_note = None
    if expected_rows is None and expected_query:
        if str(target).strip().lower() == "sql":
            expected_rows, expected_note = _execute_sql_on_json(
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
        api_key=(args.llm_api_key or "").strip() or None,
        model=args.llm_model,
        base_url=args.llm_base_url,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff,
    )


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", str(text).lower()) if token}


def _compute_novelty(seed_prompt: str, candidate_prompt: str) -> float:
    seed_tokens = _tokenize(seed_prompt)
    cand_tokens = _tokenize(candidate_prompt)
    if not seed_tokens and not cand_tokens:
        return 0.0
    union = seed_tokens | cand_tokens
    if not union:
        return 0.0
    overlap = seed_tokens & cand_tokens
    return max(0.0, min(1.0, 1.0 - (len(overlap) / len(union))))


def _as_unit_float(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _dynamic_synthetic_meta(
    base_meta: dict[str, Any],
    seed_prompt: str,
    active_prompt: str,
    engine_confidence: float,
    rewrite_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(base_meta or {})
    novelty = _compute_novelty(seed_prompt, active_prompt)
    fallback_conf = _as_unit_float(engine_confidence, default=0.5)
    if rewrite_meta and isinstance(rewrite_meta, dict):
        confidence = _as_unit_float(
            rewrite_meta.get("synthetic_rewrite_confidence", rewrite_meta.get("confidence", fallback_conf)),
            default=fallback_conf,
        )
    else:
        confidence = _as_unit_float(meta.get("synthetic_rewrite_confidence", fallback_conf), default=fallback_conf)
    score = _as_unit_float(0.65 * confidence + 0.35 * novelty)
    meta["synthetic_rewrite_confidence"] = confidence
    meta["synthetic_rewrite_novelty"] = novelty
    meta["synthetic_rewrite_score"] = score
    meta.setdefault("synthetic_rewrite_source", "seed" if novelty == 0 else "synthetic")
    return meta


def _collect_entity_rows(node: Any, out: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, list[dict[str, Any]]]:
    if out is None:
        out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict):
                out.setdefault(str(key), []).append(value)
                _collect_entity_rows(value, out)
            elif isinstance(value, list):
                dict_items = [item for item in value if isinstance(item, dict)]
                if dict_items:
                    out.setdefault(str(key), []).extend(dict_items)
                    for item in dict_items:
                        _collect_entity_rows(item, out)
                else:
                    for item in value:
                        _collect_entity_rows(item, out)
    elif isinstance(node, list):
        for item in node:
            _collect_entity_rows(item, out)
    return out


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _to_sql_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, default=str)
    return value


def _execute_sql_on_json(
    query: str,
    data_payload: dict[str, Any],
    root_key: str = "portfolio_data",
) -> tuple[list[dict[str, Any]], str | None]:
    root = data_payload.get(root_key, data_payload)
    if not isinstance(root, dict):
        return [], "SQL execution skipped: payload must be a JSON object."

    entity_rows = _collect_entity_rows(root)
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
                f"CREATE TABLE {_quote_ident(table_name)} ({', '.join(f'{_quote_ident(col)} TEXT' for col in columns)});"
            )
            insert_sql = (
                f"INSERT INTO {_quote_ident(table_name)} ({', '.join(_quote_ident(col) for col in columns)}) "
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


if __name__ == "__main__":
    main()
