# Reliability and migration notes

These changes apply to the unreleased branch after 0.2.7. They correct concrete
query, execution, provider, and evaluation defects; they do not establish
production accuracy on arbitrary questions.

## Changes by review finding

| Finding | Resolution | Regression coverage |
|---|---|---|
| Direct LLM output bypassed validation; execution accepted writes and unbounded results | SQL/GraphQL parsers inspect complete statements/documents. SQL execution rejects writes, multiple statements, locking reads and SELECT INTO, binds IR values, caps returned rows and applies deadlines. | `test_reliability_unit.py`, `test_evaluation_reliability_unit.py` |
| Structured intent lost lists, nulls, OR groups, and pure COUNT semantics | Preserve scalar/list types and recursive filters; reject malformed components; retain empty non-aggregate projections for pure aggregates. | Typed filter and aggregate execution tests |
| Async provider timeout corrupted request bodies; SQLite fixtures disappeared across threads | HTTP open/read/close runs in a worker with a keyword timeout; SQLite helper uses a shared connection protected by a lock. Retry transient 429/5xx failures. | Async HTTP, provider retries, and async SQLite tests |
| Native structured schemas were open/incomplete and errors silently downgraded modes | Close every object, require declared properties, represent filters recursively, and record per-completion provenance. Fallback requires explicit opt-in. | Schema traversal and sync/async fallback tests |
| Evaluation changed literal case, dropped semantic clauses, used text-only fixtures, and excluded failures | Parse comparisons, preserve literals and full clauses, retain numeric fixture types, count failures in every denominator, and preserve gold ordering. SQLite CPU deadlines use a progress handler. | Structural, typed fixture, ordering, failure denominator and interruption tests |
| Heuristic scores implied calibrated certainty; rebuilding IR lost information | Label heuristics, mark direct confidence unavailable, return clarification for unrelated deterministic requests, and retain the compiled IR on QueryResult. | Clarification, exact IR round-trip and DISTINCT tests |
| SQL tests could be skipped in CI; releases were not test-gated | Install SQL dependencies in CI, smoke-test the installed wheel, and run the test workflow at the release commit before building publishable artifacts. | GitHub Actions workflow gates |

## Generation and migration

`Text2QL()` now defaults to `strict_validation=True`. Invalid schema references
or intent components raise `ValidationError`, whose `issues` list gives details.
A request with no recognized schema vocabulary can return a clarification:

```python
from text2ql import Text2QL

result = Text2QL().generate(
    "What is the weather tomorrow?",
    target="sql",
    schema={"entities": ["users"], "fields": {"users": ["id", "name"]}},
)
assert result.status == "needs_clarification"
assert not result.executable
```

`Text2QL(strict_validation=False)` permits some legacy repairs, but marks the
result `needs_review`; the executor rejects it. Low-level engine constructors
retain their previous lenient default. Syntax and statement checks always run.
Use the facade and check `result.executable` before external execution. Passing
only a query string loses the result's status, so prefer passing QueryResult to
the SQL executor.

`confidence` remains for compatibility. `metadata["confidence_kind"]` distinguishes
`heuristic` from `unavailable`. A heuristic of 0.8 is not an 80% probability of a
correct answer. Direct LLM queries now return confidence 0 with kind unavailable.
Provider self-reports remain in `llm_confidence`; they are not calibration data.

LLM errors raise by default. To opt in to deterministic fallback:

```python
result = service.generate(
    "show users", target="sql", schema=schema,
    context={"mode": "function_calling", "allow_llm_fallback": True},
)
```

The CLI equivalent is `--allow-llm-fallback`. Native provider structured output
requires `use_structured_output=True`. The separate provider option
`allow_structured_fallback=True` permits a plain completion after a native
structured failure. `structured_output` and `structured_fallback_reason` describe
the actual path on each result, including concurrent calls. Legacy dictionary
filters remain accepted by the intent parser; native wire schemas use arrays of
`field`, `operator`, `value`, and `children` predicate objects.

## Execution

```python
from text2ql import create_sqlite_executor

with create_sqlite_executor({"users": [{"id": 1, "name": "Alice"}]}) as executor:
    result = Text2QL().generate(
        "show users", target="sql",
        schema={"entities": ["users"], "fields": {"users": ["id", "name"]}},
    )
    rows = executor.execute(result)  # IR literals become named parameters
```

`SQLAlchemyExecutor` defaults to a 10,000-row cap and a 30-second query deadline.
A smaller outer LIMIT is preserved; comments, string literals and subquery limits
do not bypass the fetch cap. `row_limit=None` explicitly disables that cap.
Raw SQL can use SQLAlchemy named binds via `execute(sql, parameters={...})`.
The optional `before_execute(sql, parameters)` callback may reject a query by
raising before the database statement runs.

SQLite uses query-only mode, an authorizer and a progress callback. Its previous
query-only and busy-timeout settings are restored; this helper owns the progress
and authorizer callbacks for the connection while executing. PostgreSQL uses a
read-only transaction and a local statement timeout. MySQL and SQL Server require
caller-configured driver/server limits and `timeout_seconds=None`; no portable
server deadline is claimed for them. Use a database role restricted to intended
reads on every backend. Database functions, resolver authorization, tenant
boundaries, and sensitive columns still require application policy. A SQLite
progress callback cannot interrupt a blocking Python UDF while that UDF runs.

The CLI/playground SQL-on-JSON helper uses typed SQLite fixtures, read controls,
a deadline, and a row cap. The separate metadata-on-JSON helper supports simple
filtering, sorting, projection, pagination and aggregates; it rejects unsupported
nested/join/grouped queries and raw GraphQL documents. It is not a GraphQL server.

## Supported additions and limits

- `QueryResult.ir` retains the compiled intent. `QueryIR.from_query_result` makes
  a copy when it is present. Direct LLM documents have no portable IR;
  `parse_to_ir` fails clearly for that mode.
- `SQLIRRenderer(dialect=...)` supports `sqlite`, `postgres`, `mysql`, and `tsql`.
  Generation accepts `context={"dialect": "postgres"}`; CLI uses `--dialect`.
  `QueryIR.group_by` and `sort_by=[IRSort(...), ...]` support explicit grouping
  and multiple sort keys. This does not add natural-language recognition for
  every SQL construct or extend the native intent schema to all IR features.
- GraphQL accepts `schema={"sdl": "..."}` or a standard introspection response
  (directly or under `introspection`). Full schemas validate document fields,
  argument/variable types and fragments. Named operations and variable
  declarations are retained; `context["operation_name"]` selects an operation
  when the document has several. Values and resolver execution remain the
  GraphQL client's responsibility. The older dictionary schema validates known
  names but cannot provide full GraphQL type validation.
- Large SQL/custom-schema prompts select relevant entities and relationship
  paths while retaining all columns on selected tables. This is lexical
  retrieval, not embedding retrieval or an exact model token-budget guarantee.
- CTEs, windows, arbitrary correlated queries and set operations are not modeled
  by the portable IR. Valid read queries may use them in direct SQL mode. More
  query languages, learned schema retrieval, and measured confidence calibration
  on held-out data remain future work. Domain-specific deterministic rules still
  require maintenance; schema aliases and `keyword_intents` are customization
  points, not a universal language parser.

## Evaluation evidence

Run `python -m pytest` with the `dev,sql` extras installed. Tests exercise the
SQLite execution paths and compile the four supported SQL dialects. Provider
transport tests use mocked HTTP; they do not call a paid model endpoint.
PostgreSQL, MySQL and SQL Server have not been exercised against live servers in
this change.

The deterministic synthetic harness contains 100 small examples. After these
fixes it reports 84% execution agreement on Spider-style fixtures (8 errors out
of 50) and 62% on BIRD-style fixtures (17 errors out of 50). Errors remain failed
cases. Structural accuracy is 54% and 62%, respectively. Structural comparison
is deliberately conservative and may reject equivalent SQL with different
aliasing or query shapes; it never proves execution equivalence.

`evaluate_examples` retains its historical `execution_accuracy` field name;
without a backend, each row explicitly reports `execution_mode="structural"`.
Supply an execution backend for actual execution checks. The benchmark runner
reports structural and execution metrics separately.

Old synthetic 100% results are not a production accuracy claim. Agreement on a
single fixture, especially an empty result, may be coincidental. Use numeric
boundaries, duplicates, nulls, case differences, order-sensitive answers and
multiple populated fixtures to expose semantic errors. Real Spider/BIRD data,
held-out evaluation and a live-provider rerun are separate validation work; no
new official benchmark or calibrated accuracy is claimed here.
