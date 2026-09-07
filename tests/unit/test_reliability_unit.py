from __future__ import annotations

import asyncio
import json
import sqlite3
from copy import deepcopy
from unittest.mock import patch

import pytest
from graphql import build_schema, graphql_sync, get_introspection_query

from text2ql import IRFilter, IRSort, QueryIR, QueryRequest, SQLIRRenderer, Text2QL, ValidationError
from text2ql.constrained import ConstrainedOutputError, parse_sql_intent
from text2ql.engines.sql import SQLEngine
from text2ql.evaluate import normalize_query, sql_execution_match, evaluate_examples
from text2ql.providers.base import LLMProvider
from text2ql.providers.openai_compatible import OpenAICompatibleProvider
from text2ql.prompting import GRAPHQL_INTENT_JSON_SCHEMA, SQL_INTENT_JSON_SCHEMA, build_sql_prompts
from text2ql.query_validation import validate_sql
from text2ql.schema_config import normalize_schema_config
from text2ql.sql_executor import SQLAlchemyExecutor, create_sqlite_executor

pytestmark = pytest.mark.unit
SCHEMA = {"entities": ["users"], "fields": {"users": ["id", "age", "status", "code"]}}


class ResponseProvider(LLMProvider):
    def __init__(self, response):
        self.response = response

    def complete(self, system_prompt, user_prompt):
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


def intent_result(**changes):
    payload = {"table": "users", "columns": ["id"], "filters": {}, "joins": [], **changes}
    return Text2QL(provider=ResponseProvider(payload)).generate(
        "show users", target="sql", schema=SCHEMA, context={"mode": "function_calling"})


@pytest.mark.parametrize(("filters", "expected"), [
    ({"id_in": [1, 2]}, '"id" IN (1, 2)'),
    ({"age": None}, '"age" IS NULL'),
    ({"age_ne": None}, '"age" IS NOT NULL'),
    ({"age": {"operator": ">", "value": 60}}, '"age" > 60'),
    ({"code": "00123"}, '"code" = \'00123\''),
    ({"or": [{"status": "active"}, {"status": "pending"}]}, " OR "),
    ([{"field": "age", "operator": ">", "value": 60, "children": []}], '"age" > 60'),
])
def test_structured_filters_preserve_types(filters, expected):
    assert expected in intent_result(filters=filters).query


def test_pure_count_executes_as_one_aggregate():
    result = intent_result(columns=[], aggregations=[{"function": "COUNT", "field": "*"}])
    assert "GROUP BY" not in result.query
    with create_sqlite_executor({"users": [{"id": 1}, {"id": 2}]}) as executor:
        assert list(executor.execute(result)[0].values()) == [2]


@pytest.mark.parametrize("filters", [{"tenant_id": 42}, {"or": [{"status": "active"}, {"tenant_id": 42}]}])
def test_strict_validation_never_drops_constraints(filters):
    with pytest.raises(ValidationError):
        intent_result(filters=filters)


@pytest.mark.parametrize("response", ["DROP TABLE users;", "SELECT id FROM users; DELETE FROM users;", "SELECT id INTO backup FROM users;", "SELECT password FROM users;", "SELECT id FROM secrets;"])
def test_direct_sql_rejects_unsafe_or_unknown_output(response):
    service = Text2QL(provider=ResponseProvider(response))
    with pytest.raises(ValidationError):
        service.generate("show users", target="sql", schema=SCHEMA, context={"mode": "llm"})


def test_strict_engine_rejects_unknown_columns():
    engine = SQLEngine(provider=ResponseProvider({"table": "users", "columns": ["secret"], "filters": {}}), strict_validation=True)
    with pytest.raises(ValidationError):
        engine.generate(QueryRequest("show users", "sql", SCHEMA, context={"mode": "function_calling"}))


def test_unrelated_request_abstains_and_is_not_executable():
    result = Text2QL().generate("what is the weather tomorrow", target="sql", schema=SCHEMA)
    assert result.status == "needs_clarification"
    assert not result.query and not result.executable and result.confidence == 0
    with create_sqlite_executor() as executor, pytest.raises(ValidationError):
        executor.execute(result)


def test_distinct_round_trip_and_projection_words():
    result = Text2QL().generate("show distinct status from users", target="sql", schema=SCHEMA)
    assert "WHERE" not in result.query
    assert result.ir is not None and result.ir.distinct
    assert SQLIRRenderer().render(QueryIR.from_query_result(result)) == result.query
    copy = QueryIR.from_query_result(result)
    copy.fields.append("age")
    assert "age" not in result.ir.fields


def test_explicit_grouping_and_multiple_order_keys():
    ir = QueryIR("users", fields=["status"], group_by=["status"], sort_by=[IRSort("status", "DESC"), IRSort("id")], target="sql")
    sql = SQLIRRenderer().render(ir)
    assert 'GROUP BY "users"."status"' in sql
    assert 'ORDER BY "users"."status" DESC, "users"."id" ASC' in sql


@pytest.mark.parametrize("dialect", ["sqlite", "postgres", "mysql", "tsql"])
def test_dialects_and_parameters(dialect):
    ir = QueryIR("users", fields=["id"], filters=[IRFilter("status", "O'Reilly")], limit=2, target="sql")
    sql, parameters = SQLIRRenderer(dialect).render_parameterized(ir)
    assert "O'Reilly" not in sql and parameters == {"p0": "O'Reilly"}
    validate_sql(sql, dialect=dialect)


def test_identifier_quotes_are_escaped():
    ir = QueryIR('user"records', fields=['display"name'], target="sql")
    conn = sqlite3.connect(":memory:")
    conn.execute('CREATE TABLE "user""records" ("display""name" TEXT)')
    conn.execute('INSERT INTO "user""records" VALUES (?)', ("Alice",))
    assert conn.execute(SQLIRRenderer().render(ir)).fetchall() == [("Alice",)]
    conn.close()


@pytest.mark.parametrize("sql", ["SELECT id FROM users LIMIT 999;", "SELECT id, 'unlimited' AS label FROM users;", "SELECT id FROM users -- LIMIT 1", "SELECT id FROM users WHERE id IN (SELECT id FROM users LIMIT 2);"])
def test_executor_caps_rows_independent_of_query_text(sql):
    with create_sqlite_executor({"users": [{"id": 1}, {"id": 2}, {"id": 3}]}, row_limit=1) as executor:
        assert len(executor.execute(sql)) == 1


def test_executor_rejects_ddl_before_database_mutation():
    with create_sqlite_executor({"users": [{"id": 1}]}) as executor:
        with pytest.raises(ValidationError):
            executor.execute("DROP TABLE users -- LIMIT")
        assert executor.execute("SELECT id FROM users") == [{"id": 1}]


def test_executor_honors_binds_async_shared_memory_and_fixture_types():
    with create_sqlite_executor({"users": [{"id": 2}, {"id": 100}]}) as executor:
        assert executor.execute("SELECT MAX(id) AS maximum FROM users") == [{"maximum": 100}]
        assert asyncio.run(executor.aexecute("SELECT id FROM users WHERE id = :id", {"id": 100})) == [{"id": 100}]
        executor.load_json_data("users", [{"id": 7}], if_exists="append")
        assert executor.execute("SELECT COUNT(*) AS n FROM users") == [{"n": 3}]
        with pytest.raises(ValueError):
            executor.load_json_data("users", [{"id": 9}], if_exists="fail")
        assert executor.execute("SELECT COUNT(*) AS n FROM users") == [{"n": 3}]


def test_executor_policy_and_deadline():
    def deny(sql, parameters):
        raise PermissionError("Policy denied")
    with SQLAlchemyExecutor("sqlite://", before_execute=deny) as executor, pytest.raises(PermissionError):
        executor.execute("SELECT 1")
    with SQLAlchemyExecutor("sqlite://", timeout_seconds=0.005, row_limit=None) as executor:
        with pytest.raises(Exception, match="interrupted"):
            executor.execute("WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<100000000) SELECT SUM(n) FROM c")
        assert executor.execute("SELECT 1 AS value") == [{"value": 1}]


def test_native_strict_schema_contract():
    for schema in (GRAPHQL_INTENT_JSON_SCHEMA, SQL_INTENT_JSON_SCHEMA):
        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    assert node.get("additionalProperties") is False
                    assert set(node.get("required", [])) == set(node.get("properties", {}))
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)
        check(schema)


def test_async_provider_passes_timeout_keyword_reads_and_closes_off_loop():
    import threading
    main_thread = threading.get_ident()
    events = []
    class Response:
        def __enter__(self):
            return self
        def read(self):
            assert threading.get_ident() != main_thread
            return b'{"choices":[{"message":{"content":"SELECT 1"}}]}'
        def __exit__(self, *args):
            events.append("closed")
    def open_request(request, data=None, *, timeout):
        assert data is None and timeout == 7
        assert isinstance(request.data, bytes)
        return Response()
    provider = OpenAICompatibleProvider(api_key="test", timeout_seconds=7)
    with patch("urllib.request.urlopen", open_request):
        assert asyncio.run(provider.acomplete("system", "user")) == "SELECT 1"
    assert events == ["closed"]


def test_native_structured_failure_does_not_downgrade_by_default():
    provider = OpenAICompatibleProvider(api_key="test", use_structured_output=True)
    with patch.object(provider, "_request_with_retries", side_effect=RuntimeError("schema rejected")), patch.object(provider, "complete") as plain:
        with pytest.raises(RuntimeError, match="schema rejected"):
            provider.complete_structured("system", "user", SQL_INTENT_JSON_SCHEMA)
        plain.assert_not_called()


def test_graphql_sdl_introspection_variables_and_operation_validation():
    sdl = "type User { id: ID! } type Query { users(limit: Int): [User!]! } type Mutation { remove: Boolean }"
    document = "query Users($limit: Int) { users(limit: $limit) { ...UserFields } } fragment UserFields on User { id }"
    service = Text2QL(provider=ResponseProvider(document))
    for schema in ({"sdl": sdl}, {"introspection": {"data": graphql_sync(build_schema(sdl), get_introspection_query()).data}}):
        result = service.generate("show users", schema=schema, context={"mode": "llm"})
        assert result.query == document
    for invalid in ("mutation { remove }", "{ users { password } }", "{ users { count } }"):
        service = Text2QL(provider=ResponseProvider(invalid))
        with pytest.raises(ValidationError):
            service.generate("show users", schema={"sdl": sdl}, context={"mode": "llm"})


def test_large_schema_prompt_selects_relevant_table_with_all_columns():
    tables = [f"table_{n}" for n in range(60)]
    schema = {"entities": tables, "fields": {table: [f"column_{n}" for n in range(110)] for table in tables}}
    config = normalize_schema_config(schema)
    system, prompt = build_sql_prompts("show table_59", config)
    assert "table_59" in prompt and "column_109" in prompt
    assert len(config.entities) == 60


@pytest.mark.parametrize(("left", "right"), [("SELECT id FROM users WHERE status='Active'", "SELECT id FROM users WHERE status='active'"), ("{ users(status: \"ACTIVE\") { id } }", "{ users(status: \"active\") { id } }")])
def test_evaluation_preserves_literal_case(left, right):
    assert normalize_query(left) != normalize_query(right)


@pytest.mark.parametrize("suffix", [" LIMIT 1", " GROUP BY id", " WHERE id > 1", " ORDER BY id DESC"])
def test_structural_comparison_preserves_query_semantics(suffix):
    assert not sql_execution_match("SELECT id FROM users", "SELECT id FROM users" + suffix)


@pytest.mark.parametrize("value", ["00123", "false", "null", ["00123", None, 3]])
def test_graphql_structured_values_preserve_types(value):
    key = "code_in" if isinstance(value, list) else "code"
    response = {"entity": "users", "fields": ["id"], "filters": {key: value}}
    result = Text2QL(provider=ResponseProvider(response)).generate(
        "show users", schema=SCHEMA, context={"mode": "function_calling"})
    assert result.metadata["filters"][key] == value


def test_graphql_structured_count_and_alias_survive_compilation():
    response = {"entity": "users", "fields": [], "filters": {},
                "aggregations": [{"function": "COUNT", "field": "*", "alias": "total"}]}
    result = Text2QL(provider=ResponseProvider(response)).generate(
        "count users", schema=SCHEMA, context={"mode": "function_calling"})
    assert "total: count" in result.query
    assert result.ir.fields == []


def test_graphql_does_not_drop_invalid_predicates_from_partially_valid_groups():
    response = {"entity": "users", "fields": ["id"], "filters": {
        "or": [{"code": "00123", "and": [{"status": "active", "tenant_id": 42}]}]}}
    with pytest.raises(ValidationError):
        Text2QL(provider=ResponseProvider(response)).generate(
            "show users", schema=SCHEMA, context={"mode": "function_calling"})


@pytest.mark.parametrize("response", ["{ users(tenant_id: 42) { id } }", "{ users { password } }"])
def test_direct_graphql_validates_custom_schema_fields_and_arguments(response):
    with pytest.raises(ValidationError):
        Text2QL(provider=ResponseProvider(response)).generate(
            "show users", schema=SCHEMA, context={"mode": "llm"})


@pytest.mark.parametrize("changes", [{"distinct": "false"}, {"joins": ["invalid"]},
                                     {"aggregations": [{"field": "id"}]}, {"order_dir": "sideways"}])
def test_malformed_intent_components_fail_instead_of_being_discarded(changes):
    with pytest.raises(ValidationError):
        intent_result(**changes)


def test_aggregate_arithmetic_is_parsed_and_values_are_bound():
    result = intent_result(columns=[], aggregations=[{'function': 'SUM', 'field': 'age + 2'}])
    with create_sqlite_executor({'users': [{'age': 10}, {'age': 20}]}) as executor:
        assert list(executor.execute(result)[0].values()) == [34]
    with pytest.raises(ValueError):
        intent_result(columns=[], aggregations=[{'function': 'SUM', 'field': 'age + (SELECT id FROM users)'}])
