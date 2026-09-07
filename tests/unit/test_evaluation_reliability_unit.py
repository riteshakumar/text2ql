"""Regression tests for evaluation validity and execution boundaries."""
import asyncio
import sqlite3

import pytest

from text2ql import Text2QL, QueryResult, QueryIR, IRFilter, SQLIRRenderer
from text2ql._cli_utils import execute_sql_on_json
from text2ql.benchmarks import BenchmarkConfig, run_benchmark, arun_benchmark
from text2ql.benchmarks.runner import _execute_sql
from text2ql.dataset import DatasetExample
from text2ql.evaluate import evaluate_examples, aevaluate_examples, sql_execution_match
from text2ql.json_execution import execute_query_result_on_json
from text2ql.sql_executor import _maybe_add_limit
from text2ql.types import ValidationError

pytestmark = pytest.mark.unit


class EchoService:
    def generate(self, text, **kwargs):
        if text == 'fail':
            raise ValueError('generation failed')
        return QueryResult(text, 'sql', 0.0, '')

    async def agenerate(self, **kwargs):
        return self.generate(**kwargs)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / 'fixture.sqlite'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE items (id INTEGER, price REAL, name TEXT)')
        conn.executemany('INSERT INTO items VALUES (?, ?, ?)', [(1, 2, 'Alice'), (2, 100, 'alice'), (3, None, 'Alice')])
    return path


@pytest.mark.parametrize('asynchronous', [False, True])
def test_benchmark_failures_remain_in_all_denominators(database, asynchronous):
    metadata = {'db_path': str(database), 'db_id': 'fixture', 'difficulty': 'hard'}
    examples = [DatasetExample(text, 'sql', 'SELECT MAX(price) FROM items', metadata=metadata)
                for text in ('SELECT MAX(price) FROM items', 'fail', 'SELECT missing FROM items')]
    config = BenchmarkConfig(service=EchoService())
    report = asyncio.run(arun_benchmark(examples, config=config)) if asynchronous else run_benchmark(examples, config=config)
    assert report.total == 3 and report.errors == 2
    assert report.execution_accuracy == pytest.approx(1 / 3)
    assert report.accuracy_by_db['fixture']['execution_accuracy'] == pytest.approx(1 / 3)
    assert report.accuracy_by_difficulty['hard']['execution_accuracy'] == pytest.approx(1 / 3)


def test_benchmark_preserves_gold_order_and_does_not_create_missing_db(database, tmp_path):
    ordered = DatasetExample('SELECT price FROM items ORDER BY price DESC', 'sql',
                             'SELECT price FROM items ORDER BY price ASC', metadata={'db_path': str(database)})
    assert run_benchmark([ordered], config=BenchmarkConfig(service=EchoService())).execution_accuracy == 0
    missing = tmp_path / 'missing.sqlite'
    ordered.metadata['db_path'] = str(missing)
    report = run_benchmark([ordered], config=BenchmarkConfig(service=EchoService()))
    assert report.errors == 1 and report.execution_accuracy == 0 and not missing.exists()
    ordered.metadata.clear()
    assert run_benchmark([ordered], config=BenchmarkConfig(service=EchoService())).execution_accuracy == 0


@pytest.mark.parametrize('asynchronous', [False, True])
def test_generic_evaluator_counts_generation_failures(asynchronous):
    examples = [DatasetExample(text, 'sql', 'SELECT 1') for text in ('SELECT 1', 'fail')]
    report = asyncio.run(aevaluate_examples(EchoService(), examples)) if asynchronous else evaluate_examples(EchoService(), examples)
    assert report.total == 2 and report.execution_accuracy == 0.5
    assert report.rows[1].execution_backend_error.startswith('Generation error:')


def test_sqlite_cpu_deadline_and_read_guard_restore_connection(database):
    conn = sqlite3.connect(database)
    slow = 'WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 100000000) SELECT SUM(x) FROM n'
    try:
        with pytest.raises(RuntimeError) as error:
            _execute_sql(conn, slow, 0.001)
        assert 'interrupt' in str(error.value.__cause__).lower()
        assert _execute_sql(conn, 'SELECT MAX(price) FROM items', 1) == [(100.0,)]
        with pytest.raises(ValidationError):
            _execute_sql(conn, 'DELETE FROM items', 1)
        assert conn.execute('PRAGMA query_only').fetchone() == (0,)
    finally:
        conn.close()


def test_json_sql_preserves_numeric_types_caps_rows_and_rejects_writes():
    payload = {'items': [{'price': 2}, {'price': 100}, {'price': None}]}
    assert execute_sql_on_json('SELECT MAX(price) AS price FROM items', payload) == ([{'price': 100}], None)
    assert len(execute_sql_on_json('SELECT price FROM items LIMIT 999', payload, row_limit=1)[0]) == 1
    assert execute_sql_on_json('DROP TABLE items', payload)[1]


def test_json_metadata_execution_orders_before_projection_and_aggregates_before_limit():
    payload = {'items': [{'id': 1, 'price': 2}, {'id': 2, 'price': 100}]}
    result = QueryResult('', 'graphql', 0, '', metadata={'entity': 'items', 'fields': ['id'],
                        'filters': {'orderBy': 'price', 'orderDirection': 'DESC', 'limit': 1}})
    assert execute_query_result_on_json(result, payload) == ([{'id': 2}], '')
    result.metadata['aggregations'] = [{'function': 'sum', 'field': 'price'}]
    assert execute_query_result_on_json(result, payload) == ([{'sum_price': 102}], '')
    result.metadata['filters']['price_gt'] = 1000
    result.metadata['aggregations'] = [{'function': 'count', 'field': '*'}]
    assert execute_query_result_on_json(result, payload) == ([{'count': 0}], '')
    result.status = 'needs_clarification'
    assert execute_query_result_on_json(result, payload)[1].startswith('Execution skipped')


@pytest.mark.parametrize('dialect', ['sqlite', 'postgres', 'mysql', 'tsql'])
def test_sqlalchemy_named_parameters_survive_dialect_rendering_and_row_cap(dialect):
    from sqlalchemy import text
    ir = QueryIR('users', fields=['id'], filters=[IRFilter('name', "O'Reilly")], offset=3, target='sql')
    sql, params = SQLIRRenderer(dialect).render_parameterized(ir)
    assert set(text(sql)._bindparams) == set(params)
    assert set(text(_maybe_add_limit(sql, 10, dialect=dialect))._bindparams) == set(params)
    if dialect != 'sqlite':
        assert 'LIMIT -1' not in sql


@pytest.mark.parametrize('clause', ['DISTINCT', 'GROUP BY name', 'HAVING COUNT(*) > 1', 'ORDER BY name DESC', 'LIMIT 1'])
def test_structural_matching_never_discards_semantic_clauses(clause):
    base = 'SELECT name FROM items'
    changed = base.replace('SELECT', 'SELECT DISTINCT') if clause == 'DISTINCT' else base + ' ' + clause
    assert not sql_execution_match(base, changed)
    assert not sql_execution_match("SELECT 'Alice'", "SELECT 'alice'")


def test_default_entity_does_not_make_unrelated_questions_executable():
    schema = {'entities': ['users'], 'fields': {'users': ['id']}, 'default_entity': 'users'}
    assert not Text2QL().generate('What is the weather tomorrow?', target='sql', schema=schema).executable
    assert Text2QL().generate('show all', target='sql', schema=schema).executable


def test_structural_matching_normalizes_sqlite_qualifiers_and_unused_labels():
    assert sql_execution_match('SELECT "users"."Name" FROM "users"', 'SELECT name FROM users')
    assert sql_execution_match('SELECT COUNT(*) AS total FROM users', 'SELECT count(*) FROM users')
    assert not sql_execution_match('SELECT id AS name FROM users ORDER BY name',
                                   'SELECT id FROM users ORDER BY name')
