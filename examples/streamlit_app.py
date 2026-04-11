#!/usr/bin/env python3
"""Interactive Streamlit playground for text2ql GraphQL/SQL testing."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Logging — attach a handler directly to the text2ql logger so Streamlit's
# root-logger reconfiguration cannot suppress it.
#
# Control via TEXT2QL_LOG_LEVEL env var (default: WARNING).
# Examples:
#   TEXT2QL_LOG_LEVEL=DEBUG   ./venv/bin/python -m streamlit run examples/streamlit_app.py
#   TEXT2QL_LOG_LEVEL=WARNING ./venv/bin/python -m streamlit run examples/streamlit_app.py
#
# Logs appear in the terminal where you launched Streamlit, not the browser.
# ---------------------------------------------------------------------------
_log_level_name = os.getenv("TEXT2QL_LOG_LEVEL", "DEBUG").upper()
_log_level = getattr(logging, _log_level_name, logging.DEBUG)

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

def _import_text2ql() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        from text2ql import (
            DatasetExample,
            QueryResult,
            Text2QL,
            execute_query_result_on_json,
            generate_hybrid_mapping,
            generate_synthetic_examples,
            infer_schema_from_json_payload,
            rewrite_user_utterance,
        )
        from text2ql.evaluate import sql_execution_match
        from text2ql.providers.openai_compatible import OpenAICompatibleProvider

        return (
            DatasetExample,
            QueryResult,
            Text2QL,
            execute_query_result_on_json,
            generate_hybrid_mapping,
            generate_synthetic_examples,
            infer_schema_from_json_payload,
            rewrite_user_utterance,
            sql_execution_match,
            OpenAICompatibleProvider,
        )
    except (ModuleNotFoundError, ImportError):
        repo_root = Path(__file__).resolve().parents[1]
        local_src = repo_root / "src"
        if str(local_src) not in sys.path:
            sys.path.insert(0, str(local_src))
        for module_name in list(sys.modules):
            if module_name == "text2ql" or module_name.startswith("text2ql."):
                del sys.modules[module_name]
        from text2ql import (
            DatasetExample,
            QueryResult,
            Text2QL,
            execute_query_result_on_json,
            generate_hybrid_mapping,
            generate_synthetic_examples,
            infer_schema_from_json_payload,
            rewrite_user_utterance,
        )
        from text2ql.evaluate import sql_execution_match
        from text2ql.providers.openai_compatible import OpenAICompatibleProvider

        return (
            DatasetExample,
            QueryResult,
            Text2QL,
            execute_query_result_on_json,
            generate_hybrid_mapping,
            generate_synthetic_examples,
            infer_schema_from_json_payload,
            rewrite_user_utterance,
            sql_execution_match,
            OpenAICompatibleProvider,
        )


(
    DatasetExample,
    QueryResult,
    Text2QL,
    execute_query_result_on_json,
    generate_hybrid_mapping,
    generate_synthetic_examples,
    infer_schema_from_json_payload,
    rewrite_user_utterance,
    sql_execution_match,
    OpenAICompatibleProvider,
) = _import_text2ql()

# Import shared utilities after _import_text2ql() has resolved the package path.
from text2ql._cli_utils import (  # noqa: E402
    as_unit_float as _as_unit_float,
    dynamic_synthetic_meta as _dynamic_synthetic_meta,
    execute_sql_on_json as _execute_sql_on_json,
    stable_json as _stable_json,
)


PLUGIN_OPTIONS = ["generic", "portfolio", "banking", "ecommerce", "crm", "healthcare"]
DOMAIN_OPTIONS = ["", "portfolio", "banking", "ecommerce", "crm", "healthcare"]


@st.cache_data(show_spinner=False)
def _load_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return payload


def _load_uploaded_json(uploaded: Any) -> dict[str, Any]:
    payload = json.loads(uploaded.getvalue().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected top-level JSON object in uploaded file")
    return payload


def _default_api_key() -> str:
    secret_key = ""
    try:
        secret_key = str(st.secrets.get("OPENAI_API_KEY", st.secrets.get("TEXT2QL_API_KEY", ""))).strip()
    except Exception:
        secret_key = ""
    env_key = (os.getenv("OPENAI_API_KEY") or os.getenv("TEXT2QL_API_KEY") or "").strip()
    return secret_key or env_key


def _build_service(mode: str, llm_model: str, api_key: str) -> Any:
    if mode != "llm":
        return Text2QL()
    return Text2QL(provider=OpenAICompatibleProvider(api_key=api_key or None, model=llm_model))


def _build_prompts(
    prompt: str,
    requested_variants: int,
    plugins: list[str],
    domain: str,
    target: str,
    schema: dict[str, Any],
    mapping: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    prompts_with_meta: list[tuple[str, dict[str, Any]]] = [
        (
            prompt,
            {
                "synthetic_domain": domain or None,
                "synthetic_rewrite_source": "seed",
                "synthetic_rewrite_confidence": 0.35,
                "synthetic_rewrite_novelty": 0.0,
                "synthetic_rewrite_score": 0.35,
            },
        )
    ]
    should_expand = bool(plugins or domain or requested_variants > 1)
    if not should_expand:
        return prompts_with_meta

    seed = DatasetExample(
        text=prompt,
        target=target,
        expected_query="",
        schema=schema,
        mapping=mapping,
    )
    synthetic = generate_synthetic_examples(
        [seed],
        variants_per_example=requested_variants,
        rewrite_plugins=plugins or None,
        domain=domain or None,
    )
    for example in synthetic:
        if example.text.strip().lower() == prompt.strip().lower():
            continue
        prompts_with_meta.append((example.text, example.metadata))
        if len(prompts_with_meta) >= requested_variants:
            break
    return prompts_with_meta


def main() -> None:
    st.set_page_config(page_title="text2ql Playground", layout="wide")
    st.title("text2ql Playground")
    st.caption("v0.2.0 · GraphQL and SQL query generation from natural language · deterministic or LLM mode")

    with st.sidebar:
        st.header("Settings")
        target = st.selectbox("Target", options=["graphql", "sql"], index=0)
        mode = st.selectbox("Mode", options=["deterministic", "llm"], index=0)
        llm_model = st.text_input("LLM Model", value="gpt-4o-mini")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=_default_api_key(),
            type="password",
            help="Optional. Uses this key first, then Streamlit Secrets/env vars.",
        )
        llm_rewrite = st.checkbox(
            "LLM Utterance Rewrite",
            value=False,
            help="Rewrite user utterance with schema-aware LLM before query generation.",
        )
        system_context = st.text_area("System Context", value="", height=90)
        variants_per_example = st.number_input("Variants per Prompt", min_value=1, max_value=20, value=1, step=1)
        rewrite_plugins = st.multiselect("Rewrite Plugins", options=PLUGIN_OPTIONS, default=[])
        domain = st.selectbox("Domain", options=DOMAIN_OPTIONS, index=0)
        st.divider()
        execute_on_payload = st.checkbox(
            "Execute on JSON Payload",
            value=True,
            help="Turn off to only generate queries without running GraphQL/SQL execution.",
        )
        expected_query = st.text_area(
            "Expected Query (optional)",
            value="",
            height=120,
            help="GraphQL: execution compare, SQL: signature compare",
        )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Schema and Data")
        use_bundled = st.checkbox("Use bundled sample JSON files", value=True)

        schema_upload = st.file_uploader("Schema JSON", type=["json"], accept_multiple_files=False)
        data_upload = st.file_uploader("Data JSON", type=["json"], accept_multiple_files=False)
        mapping_override_upload = st.file_uploader("Mapping Overrides JSON (optional)", type=["json"], accept_multiple_files=False)

        prompt = st.text_area("Prompt", value="how many qqq do I own", height=90)
        run_btn = st.button("Run", type="primary", use_container_width=True)

    with right:
        st.subheader("Run Notes")
        st.markdown(
            "**Modes**\n"
            "- `deterministic`: no LLM calls, fully rule-based\n"
            "- `llm`: uses provider then validates against schema\n"
            "- `LLM Utterance Rewrite`: optional pre-generation rewrite step\n\n"
            "**Auto-detected features** (deterministic mode)\n"
            "- Aggregations: COUNT / SUM / AVG / MIN / MAX from natural language\n"
            "- SQL JOIN: when query mentions a related entity declared in `schema.relations`\n"
            "- GraphQL nested: up to 3-hop nested selections via `schema.relations`\n"
            "- Pagination: LIMIT / OFFSET / first / after\n"
            "- Ordering: ORDER BY ASC/DESC\n\n"
            "**Schema keys**\n"
            "- `entities`, `fields`, `relations`, `args`, `introspection`\n"
            "- `keyword_intents` for compound-keyword → entity routing\n\n"
            "**Execution**\n"
            "- `Execute on JSON Payload`: toggle between query-only and query+execution\n"
            "- GraphQL mode executes against JSON payload\n"
            "- SQL mode supports signature match against expected query"
        )

    if not run_btn:
        return

    try:
        if use_bundled:
            base = Path(__file__).resolve().parent
            schema_payload = _load_json(str(base / "sample_schema.json"))
            data_payload = _load_json(str(base / "sample_data.json"))
        else:
            if schema_upload is None or data_upload is None:
                st.error("Upload both schema and data JSON, or enable bundled sample files.")
                return
            schema_payload = _load_uploaded_json(schema_upload)
            data_payload = _load_uploaded_json(data_upload)

        overrides = _load_uploaded_json(mapping_override_upload) if mapping_override_upload is not None else None
        root_payload = data_payload.get("portfolio_data", data_payload)

        inferred_schema = infer_schema_from_json_payload(root_payload)
        mapping = generate_hybrid_mapping(
            schema_payload=schema_payload,
            data_payload=root_payload,
            overrides=overrides,
        )

        api_key = (api_key_input or "").strip() or _default_api_key()
        if (mode == "llm" or llm_rewrite) and not api_key:
            st.warning(
                "LLM mode or LLM rewrite selected but no API key found. Set OpenAI API Key in sidebar "
                "or configure OPENAI_API_KEY/TEXT2QL_API_KEY in Streamlit Secrets."
            )

        service = _build_service(mode, llm_model, api_key=(api_key_input or "").strip())
        rewrite_provider = None
        if llm_rewrite and api_key:
            rewrite_provider = OpenAICompatibleProvider(api_key=api_key, model=llm_model)
        prompts = _build_prompts(
            prompt=prompt,
            requested_variants=int(variants_per_example),
            plugins=rewrite_plugins,
            domain=domain,
            target=target,
            schema=inferred_schema,
            mapping=mapping,
        )

        st.success(f"Prepared {len(prompts)} prompt variant(s).")

        results = []
        for idx, (active_prompt, synth_meta) in enumerate(prompts, start=1):
            started = time.perf_counter()
            gen_start = time.perf_counter()
            rewritten_prompt = active_prompt
            rewrite_meta: dict[str, Any] = {"applied": False, "reason": "disabled"}
            if llm_rewrite and rewrite_provider is not None:
                rewritten_prompt, rewrite_meta = rewrite_user_utterance(
                    text=active_prompt,
                    target=target,
                    schema=inferred_schema,
                    mapping=mapping,
                    provider=rewrite_provider,
                    system_context=system_context,
                )
            result = service.generate(
                text=rewritten_prompt,
                target=target,
                schema=inferred_schema,
                mapping=mapping,
                context={
                    "mode": mode,
                    "language": "english",
                    "system_context": system_context,
                },
            )
            gen_elapsed = time.perf_counter() - gen_start
            total_elapsed = time.perf_counter() - started

            row: dict[str, Any] = {
                "idx": idx,
                "prompt": active_prompt,
                "rewritten_prompt": rewritten_prompt,
                "rewrite_meta": rewrite_meta,
                "query": result.query,
                "metadata": result.metadata,
                "timing_ms": {
                    "total": total_elapsed * 1000,
                    "generate": gen_elapsed * 1000,
                },
            }
            row["synthetic"] = _dynamic_synthetic_meta(
                base_meta=synth_meta,
                seed_prompt=prompt,
                active_prompt=active_prompt,
                engine_confidence=_as_unit_float(result.confidence, default=0.5),
                rewrite_meta=rewrite_meta if (llm_rewrite and rewrite_provider is not None) else None,
            )

            if target == "graphql" and execute_on_payload:
                exec_start = time.perf_counter()
                rows, note = execute_query_result_on_json(result, data_payload, root_key="portfolio_data")
                exec_elapsed = time.perf_counter() - exec_start
                row["execution_rows"] = rows
                row["execution_note"] = note
                row["timing_ms"]["execute"] = exec_elapsed * 1000

                if expected_query.strip():
                    expected_result = QueryResult(
                        query=expected_query.strip(),
                        target="graphql",
                        confidence=1.0,
                        explanation="expected",
                    )
                    expected_rows, expected_note = execute_query_result_on_json(
                        expected_result,
                        data_payload,
                        root_key="portfolio_data",
                    )
                    if expected_note:
                        row["execution_eval_warning"] = expected_note
                    else:
                        row["execution_match"] = _stable_json(rows) == _stable_json(expected_rows)

            if target == "sql" and execute_on_payload and expected_query.strip():
                row["sql_signature_match"] = sql_execution_match(result.query, expected_query.strip())
            if target == "sql" and execute_on_payload:
                exec_start = time.perf_counter()
                sql_rows, sql_note = _execute_sql_on_json(result.query, data_payload, root_key="portfolio_data")
                exec_elapsed = time.perf_counter() - exec_start
                row["sql_execution_rows"] = sql_rows
                row["sql_execution_note"] = sql_note
                row["timing_ms"]["execute"] = exec_elapsed * 1000

            results.append(row)

        st.subheader("Results")
        for row in results:
            with st.expander(f"Variant {row['idx']}: {row['prompt']}", expanded=(row["idx"] == 1)):
                st.caption(
                    f"timing_ms total={row['timing_ms'].get('total', 0):.3f} "
                    f"generate={row['timing_ms'].get('generate', 0):.3f} "
                    f"execute={row['timing_ms'].get('execute', 0):.3f}"
                )
                if llm_rewrite and rewrite_provider is not None:
                    st.markdown("**Rewritten Prompt**")
                    st.code(row.get("rewritten_prompt", ""), language="text")
                    st.markdown("**Rewrite metadata**")
                    st.json(row.get("rewrite_meta", {}), expanded=False)
                # Surface validation notes prominently before the query.
                engine_meta = row.get("metadata", {})
                validation_notes = engine_meta.get("validation_notes", [])
                if validation_notes:
                    for note in validation_notes:
                        st.warning(f"Validation: {note}")

                st.markdown("**Generated Query**")
                st.code(row["query"], language="graphql" if target == "graphql" else "sql")

                # Show aggregations when present.
                aggregations = engine_meta.get("aggregations", [])
                if aggregations:
                    st.markdown("**Aggregations**")
                    st.json(aggregations, expanded=True)

                if target == "graphql":
                    if execute_on_payload:
                        st.markdown("**Execution rows**")
                        st.json(row.get("execution_rows", []), expanded=False)
                        if row.get("execution_note"):
                            st.info(row["execution_note"])
                        if "execution_match" in row:
                            st.write(f"execution_match: `{row['execution_match']}`")
                        if row.get("execution_eval_warning"):
                            st.warning(row["execution_eval_warning"])

                if target == "sql" and "sql_signature_match" in row:
                    st.write(f"sql_signature_match: `{row['sql_signature_match']}`")
                if target == "sql":
                    if execute_on_payload:
                        st.markdown("**Execution rows**")
                        st.json(row.get("sql_execution_rows", []), expanded=False)
                        if row.get("sql_execution_note"):
                            st.info(row["sql_execution_note"])

                st.markdown("**Synthetic Scores**")
                st.json(row.get("synthetic", {}), expanded=False)
                st.markdown("**Engine metadata**")
                st.json(row.get("metadata", {}), expanded=False)

    except Exception as exc:  # noqa: BLE001
        st.exception(exc)


if __name__ == "__main__":
    main()
