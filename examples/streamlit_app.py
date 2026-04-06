#!/usr/bin/env python3
"""Interactive Streamlit playground for text2ql GraphQL/SQL testing."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st


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


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _build_service(mode: str, llm_model: str) -> Any:
    if mode != "llm":
        return Text2QL()
    return Text2QL(provider=OpenAICompatibleProvider(model=llm_model))


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
    st.caption("Test GraphQL and SQL generation in deterministic or LLM mode.")

    with st.sidebar:
        st.header("Settings")
        target = st.selectbox("Target", options=["graphql", "sql"], index=0)
        mode = st.selectbox("Mode", options=["deterministic", "llm"], index=0)
        llm_model = st.text_input("LLM Model", value="gpt-4o-mini")
        llm_rewrite = st.checkbox(
            "LLM Utterance Rewrite",
            value=False,
            help="Rewrite user utterance with schema-aware LLM before query generation (LLM mode only).",
        )
        system_context = st.text_area("System Context", value="", height=90)
        variants_per_example = st.number_input("Variants per Prompt", min_value=1, max_value=20, value=1, step=1)
        rewrite_plugins = st.multiselect("Rewrite Plugins", options=PLUGIN_OPTIONS, default=[])
        domain = st.selectbox("Domain", options=DOMAIN_OPTIONS, index=0)
        st.divider()
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
            "- `deterministic`: no LLM calls, fully rule-based\n"
            "- `llm`: uses provider then validates against schema\n"
            "- `LLM Utterance Rewrite`: optional pre-generation rewrite step\n"
            "- GraphQL mode can execute against JSON payload\n"
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

        if mode == "llm":
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("TEXT2QL_API_KEY")
            if not api_key:
                st.warning("LLM mode selected but OPENAI_API_KEY/TEXT2QL_API_KEY is not set. Request may fail.")

        service = _build_service(mode, llm_model)
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
            if llm_rewrite and mode == "llm":
                rewritten_prompt, rewrite_meta = rewrite_user_utterance(
                    text=active_prompt,
                    target=target,
                    schema=inferred_schema,
                    mapping=mapping,
                    provider=service.provider,
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
                "synthetic": synth_meta,
                "query": result.query,
                "metadata": result.metadata,
                "timing_ms": {
                    "total": total_elapsed * 1000,
                    "generate": gen_elapsed * 1000,
                },
            }

            if target == "graphql":
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

            if target == "sql" and expected_query.strip():
                row["sql_signature_match"] = sql_execution_match(result.query, expected_query.strip())

            results.append(row)

        st.subheader("Results")
        for row in results:
            with st.expander(f"Variant {row['idx']}: {row['prompt']}", expanded=(row["idx"] == 1)):
                st.caption(
                    f"timing_ms total={row['timing_ms'].get('total', 0):.3f} "
                    f"generate={row['timing_ms'].get('generate', 0):.3f} "
                    f"execute={row['timing_ms'].get('execute', 0):.3f}"
                )
                if llm_rewrite and mode == "llm":
                    st.markdown("**Rewritten Prompt**")
                    st.code(row.get("rewritten_prompt", ""), language="text")
                    st.markdown("**Rewrite metadata**")
                    st.json(row.get("rewrite_meta", {}), expanded=False)
                st.json(row.get("synthetic", {}), expanded=False)
                st.code(row["query"], language="graphql" if target == "graphql" else "sql")

                if target == "graphql":
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

                st.markdown("**Engine metadata**")
                st.json(row.get("metadata", {}), expanded=False)

    except Exception as exc:  # noqa: BLE001
        st.exception(exc)


if __name__ == "__main__":
    main()
