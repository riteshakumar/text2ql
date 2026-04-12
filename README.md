# text2ql

Natural Language to Query Language framework.

`text2ql` is designed as a **pip-installable package** for converting natural language into query languages with a plugin architecture. 

## Project goals

- Build a reusable core abstraction (`Text2QL`) for `text -> query` conversion.
- Support multiple target query languages (GraphQL, SQL first; Cypher/SPARQL/JQ/Jsonata next).
- Keep provider integrations optional (deterministic local mode, LLM mode as adapters).
- Provide a CLI for quick experimentation and batch evaluation.
- Include dataset ingestion, synthetic variant generation, and evaluation hooks for benchmarking.

## Install

Python requirement: `>=3.10`.

### Pip Users

Install from PyPI:

```bash
pip install text2ql
```

With SQLAlchemy backend execution support:

```bash
pip install text2ql[sql]
```

Quick smoke test:

```bash
python -c "from text2ql import Text2QL; print(Text2QL().generate(text='list users').query)"
text2ql --help
```

### Repo / Development Users

Install from source (editable):

```bash
pip install -e .
```

For local development and tests:

```bash
pip install -e ".[dev]"
```

For SQL execution support:

```bash
pip install -e ".[sql]"
```

## Getting Started

Install + verify:

```bash
pip install text2ql
python -c "from text2ql import Text2QL; print(Text2QL().generate(text='list users').query)"
text2ql --help
```

Four copy-paste starter commands:

```bash
# 1) GraphQL deterministic
text2ql "show top 5 client records with mail state enabled" \
  --target graphql \
  --schema '{"entities":["customers"],"fields":["id","email","status"]}' \
  --mapping '{"entities":{"client":"customers"},"fields":{"mail":"email"},"filters":{"state":"status"},"filter_values":{"status":{"enabled":"active"}}}'

# 2) SQL deterministic
text2ql "show customers highest total first 5 offset 10" \
  --target sql \
  --schema '{"entities":["customers"],"fields":{"customers":["id","total","status"]}}'

# 3) LLM mode (GraphQL)
export OPENAI_API_KEY=...
text2ql "show latest 5 orders with status active" \
  --target graphql \
  --mode llm \
  --llm-rewrite on \
  --schema '{"entities":["orders"],"fields":{"orders":["id","status","createdAt"]},"args":{"orders":["status","limit","orderBy","orderDirection"]}}'

# 4) LLM mode (SQL)
export OPENAI_API_KEY=...
text2ql "show latest 5 orders with status active" \
  --target sql \
  --mode llm \
  --llm-rewrite on \
  --schema '{"entities":["orders"],"fields":{"orders":["id","status","createdAt"]}}'
```

## Production Setup

Recommended file layout:

```text
project/
  schema.json
  mapping.json
  data.json
  expected_query.graphql
  expected_rows.json
```

Create minimal files:

```bash
cat > schema.json <<'JSON'
{
  "entities": ["positions"],
  "fields": {"positions": ["symbol", "quantity", "status"]},
  "args": {"positions": ["symbol", "status", "limit", "offset", "orderBy", "orderDirection"]}
}
JSON

cat > mapping.json <<'JSON'
{
  "filters": {"ticker": "symbol"},
  "filter_values": {"symbol": {"qqq": "QQQ"}}
}
JSON

cat > data.json <<'JSON'
{
  "portfolio_data": {
    "positions": [
      {"symbol": "QQQ", "quantity": 100.104, "status": "active"},
      {"symbol": "AAPL", "quantity": 12, "status": "active"}
    ]
  }
}
JSON
```

Run with synthetic variants + execution evaluation:

```bash
text2ql "how many qqq do i own" \
  --target graphql \
  --schema-file ./schema.json \
  --mapping-file ./mapping.json \
  --data-file ./data.json \
  --variants-per-example 3 \
  --rewrite-plugins generic,portfolio \
  --domain portfolio \
  --expected-execution-file ./expected_rows.json
```

Operational notes:

- Set `OPENAI_API_KEY` (or `TEXT2QL_API_KEY`) for `--mode llm`.
- You can also pass `--llm-api-key` directly in CLI instead of env vars.
- `--expected-query` / `--expected-query-file` / `--expected-execution-file` require `--data-file`.
- If expected query execution cannot be derived from payload JSON, CLI emits an eval warning and skips that item from accuracy denominator.
- `--llm-rewrite on` enables schema-aware LLM utterance rewrite independently of generation mode.
- `--mode` controls query generation path; `--llm-rewrite` controls prompt rewrite path.

## Feature Matrix

| Capability | GraphQL | SQL |
|---|---|---|
| Deterministic generation | Yes | Yes |
| Runtime confidence scoring | Yes | Yes |
| LLM mode + constrained parsing | Yes | Yes |
| **Function-calling / structured output mode** | **Yes** | **Yes** |
| Schema/introspection validation | Yes | Yes |
| **Strict validation (`ValidationError` on contradictions)** | **Yes** | **Yes** |
| Enum/type coercion | Yes | Yes |
| Advanced filters (`not`, `!=`, ranges, grouped precedence) | Yes | Yes |
| Order parsing (`latest/highest/lowest`) | Yes | Yes |
| Pagination (`limit`, `offset`, `first`, `after`) | Yes | Yes |
| Nested/relation safety | **Recursive (up to 3 hops, cycle-safe)** | Yes |
| **JOIN ON-clause column validation** | — | **Yes** |
| Async generation (`agenerate`, `agenerate_many`) | Yes | Yes |
| Structural execution match (no backend) | Yes | Yes |
| **SQLAlchemy real backend execution** | — | **Yes** |
| Real backend execution accuracy hook | Yes | Yes (via `evaluate_examples(..., execution_backend=...)`) |
| Concurrent evaluation (`aevaluate_examples`) | Yes | Yes |
| Synthetic rewrite plugins | Yes | Yes (target-agnostic dataset API) |
| **Abstract IR layer (`QueryIR`, `IRRenderer`, `GraphQLIRRenderer`, `SQLIRRenderer`)** | **Yes** | **Yes** |
| **Structured logging (`TEXT2QL_LOG_LEVEL`)** | **Yes** | **Yes** |
| **Spider/BIRD benchmark evaluation** | — | **Yes** |

## CLI-First Workflow

1) Generate hybrid mapping from schema + data:

```bash
text2ql --generate-hybrid-mapping \
  --schema-file ./schema.json \
  --data-file ./data.json \
  --mapping-output-file ./mapping.generated.json
```

2) Generate query (GraphQL):

```bash
text2ql "how many qqq do i own" \
  --target graphql \
  --schema-file ./schema.json \
  --mapping-file ./mapping.generated.json
```

3) Generate query (SQL):

```bash
text2ql "show latest 5 positions by quantity" \
  --target sql \
  --schema-file ./schema.json \
  --mapping-file ./mapping.generated.json
```

4) Batch prompt variants + execution eval in one CLI run:

```bash
text2ql "how many qqq do i own" \
  --target graphql \
  --schema-file ./schema.json \
  --mapping-file ./mapping.generated.json \
  --data-file ./data.json \
  --variants-per-example 5 \
  --rewrite-plugins generic,portfolio \
  --domain portfolio \
  --expected-query-file ./expected_query.graphql
```

5) Enable schema-aware LLM utterance rewrite:

```bash
export OPENAI_API_KEY=...
text2ql "how many qqq do i own" \
  --target graphql \
  --mode llm \
  --llm-api-key "$OPENAI_API_KEY" \
  --llm-rewrite on \
  --schema-file ./schema.json \
  --mapping-file ./mapping.generated.json \
  --data-file ./data.json
```

6) Rewrite with LLM, but keep query generation deterministic:

```bash
export OPENAI_API_KEY=...
text2ql "how many qqq do i own" \
  --target sql \
  --mode deterministic \
  --llm-rewrite on \
  --llm-api-key "$OPENAI_API_KEY" \
  --schema-file ./schema.json \
  --data-file ./data.json
```

CLI JSON output includes:

- `prompt`: original user utterance
- `rewritten_prompt`: prompt after optional LLM rewrite
- `rewrite`: rewrite metadata (`applied`, `reason/source`, `confidence`, `notes`)
- `synthetic`: dynamic synthetic metadata (`synthetic_rewrite_confidence`, `synthetic_rewrite_novelty`, `synthetic_rewrite_score`)
- `engine_metadata`: query-engine metadata returned by `Text2QL`
- `execution_rows` / `execution_note`: execution output for GraphQL and SQL when `--data-file` is provided
- `execution_match`: only included when expected output is provided (`--expected-query*` or `--expected-execution-file`)

## Streamlit Playground

Run an interactive UI for GraphQL + SQL testing:

```bash
pip install -e ".[app]"
streamlit run examples/streamlit_app.py
```

Hosted app:

- https://text2ql.streamlit.app/

What you get:

- Deterministic and LLM modes in one app.
- GraphQL and SQL target switch.
- Bundled sample data (`examples/sample_schema.json`, `examples/sample_data.json`) or uploaded JSON files.
- Sidebar `OpenAI API Key` input (`type=password`) plus fallback to Streamlit secrets/env vars.
- Synthetic rewrite controls (`variants`, `plugins`, `domain`).
- LLM utterance rewrite toggle works independently from generation mode.
- `Execute on JSON Payload` toggle controls query-only vs query+execution mode.
- GraphQL execution on JSON payload + optional expected-query execution match.
- SQL execution on JSON payload, plus optional expected-query signature match.

For Streamlit Community Cloud deployment:

```bash
# from repo root
streamlit run examples/streamlit_app.py
```

- Main file path: `examples/streamlit_app.py`
- Requirements file: `examples/requirements.txt`
- Add secrets in App Settings -> Secrets:

```toml
OPENAI_API_KEY = "sk-your-openai-key-here"
```

## Sample Runners

The `research` workspace includes sample runners that support synthetic variants and LLM rewrite:

GraphQL:

```bash
python3 sample_text2ql_graphql_runner.py \
  --schema-file ./test_schema_definition.json \
  --portfolio-file ./test_customer_portfolio_data.json \
  --prompt "how many qqq do i own" \
  --mode llm \
  --llm-model gpt-4o-mini \
  --llm-rewrite on \
  --verbose
```

SQL:

```bash
python3 sample_text2ql_sql_runner.py \
  --schema-file ./test_schema_definition.json \
  --portfolio-file ./test_customer_portfolio_data.json \
  --prompt "show latest positions" \
  --mode llm \
  --llm-model gpt-4o-mini \
  --llm-rewrite on
```

## Quickstart

```python
from text2ql import Text2QL

service = Text2QL()
result = service.generate(
    text="show top 5 client records with mail state enabled",
    target="graphql",
    schema={
        "entities": [
            {"name": "customers", "aliases": ["client", "clients"]}
        ],
        "fields": [
            {"name": "id"},
            {"name": "email", "aliases": ["mail"]},
            {"name": "status", "aliases": ["state"]}
        ],
        "default_entity": "customers",
        "default_fields": ["id", "email", "status"]
    },
    mapping={
        "filters": {"state": "status"},
        "filter_values": {"status": {"enabled": "active"}}
    },
)

print(result.query)
print(result.explanation)
```

Default mode is deterministic. The `confidence` field on `QueryResult` is computed at runtime — it reflects schema coverage, entity resolution quality, field match ratio, filter richness, and any validation warnings. It is not a hardcoded heuristic.

## Async API

All generation methods have async equivalents that are safe to use inside `asyncio` applications.

### Single request

```python
import asyncio
from text2ql import Text2QL

async def main():
    result = await Text2QL().agenerate(
        text="show active users",
        target="graphql",
        schema={"entities": ["users"], "fields": {"users": ["id", "name", "status"]}},
    )
    print(result.query, result.confidence)

asyncio.run(main())
```

### Concurrent batch

```python
import asyncio
from text2ql import Text2QL

async def main():
    svc = Text2QL()
    results = await svc.agenerate_many(
        [
            {"text": "show active users", "schema": schema},
            {"text": "top 5 orders by total", "target": "sql", "schema": schema},
            {"text": "users where status is pending", "schema": schema},
        ],
        concurrency=5,  # max simultaneous in-flight requests
    )
    for r in results:
        print(r.confidence, r.query[:60])

asyncio.run(main())
```

`concurrency` defaults to `5` — lower this when hitting rate-limited LLM providers.

### Async rewrite

```python
from text2ql import arewrite_user_utterance
from text2ql.providers.openai_compatible import OpenAICompatibleProvider

provider = OpenAICompatibleProvider()
rewritten, meta = await arewrite_user_utterance(
    text="how many appl shares do i have",
    target="graphql",
    schema=schema,
    mapping=mapping,
    provider=provider,
)
```

### LLM provider async

`OpenAICompatibleProvider.acomplete()` is a native async implementation — retry backoff uses `asyncio.sleep()` so no thread is held during waits. Custom providers only need to implement `complete()`; `acomplete()` defaults to offloading it to a thread pool.

## LLM mode (adapter-based)

```python
from text2ql import Text2QL
from text2ql.providers.openai_compatible import OpenAICompatibleProvider

service = Text2QL(
    provider=OpenAICompatibleProvider(model="gpt-4o-mini")
)

result = service.generate(
    text="show top 3 clients with mail state enabled",
    target="graphql",
    schema={"entities": ["customers"], "fields": ["id", "email", "status"]},
    mapping={"entities": {"clients": "customers"}, "fields": {"mail": "email"}},
    context={
        "mode": "llm",
        "language": "english",
        "system_context": "Prefer customer/account semantics and preserve canonical field names.",
    },
)
```

Required env var for `OpenAICompatibleProvider`:

```bash
export OPENAI_API_KEY=...
```

Fallback key name also supported:

```bash
export TEXT2QL_API_KEY=...
```

If LLM output fails constrained JSON validation, `text2ql` falls back to deterministic mode.

## Function-Calling Mode (Structured Output)

For models that support native structured output (e.g. OpenAI `gpt-4o-2024-08-06` and later), enable `use_structured_output` on the provider. The engine sends the intent JSON schema via `response_format: json_schema`, forcing the model to emit valid JSON rather than relying on regex extraction as a fallback.

```python
from text2ql import Text2QL
from text2ql.providers.openai_compatible import OpenAICompatibleProvider

service = Text2QL(
    provider=OpenAICompatibleProvider(
        model="gpt-4o",
        use_structured_output=True,   # sends response_format: json_schema
    )
)

result = service.generate(
    text="show top 5 active orders by total",
    target="sql",
    schema={"entities": ["orders"], "fields": {"orders": ["id", "status", "total", "createdAt"]}},
    context={"mode": "function_calling"},   # use complete_structured() path
)
print(result.query)
```

When the model or endpoint does not support structured output the provider falls back to plain `complete()` automatically — no change needed on the caller side.

You can also use `mode="function_calling"` interchangeably with `mode="llm"` when `use_structured_output=False`; the difference is only visible when the provider supports and enables native structured output.

### Custom providers

`LLMProvider` base now exposes `complete_structured()` and `acomplete_structured()`. Override them in custom providers to use tool-call / function-calling APIs:

```python
from text2ql.providers.base import LLMProvider

class MyProvider(LLMProvider):
    def complete(self, system_prompt, user_prompt):
        ...

    def complete_structured(self, system_prompt, user_prompt, json_schema):
        # Use your API's native structured output / function-calling here.
        return self._call_with_schema(system_prompt, user_prompt, json_schema)
```

## Validation Hardening (Strict Mode)

By default both engines degrade gracefully: invalid filters are dropped and join issues are recorded in `result.metadata["validation_notes"]`. Enable `strict_validation=True` to raise instead.

```python
from text2ql.engines.sql import SQLEngine
from text2ql.types import ValidationError

engine = SQLEngine(strict_validation=True)

try:
    result = engine.generate(request)
except ValidationError as exc:
    print("Validation failed:", exc.issues)
    # exc.issues is a list of individual problem strings
```

`ValidationError` is raised for:

- **Contradictory equality filters** — same field assigned two different plain-equality values (e.g. `{"status": "active"}` and `{"status": "inactive"}` in the same filter dict).
- **Invalid JOIN ON-clause columns** — a JOIN references a column that does not exist in the parent or target table according to the provided schema.
- **Invalid relations** — a JOIN references a relation not declared in `schema["relations"]`.

Both `SQLEngine` and `GraphQLEngine` accept `strict_validation`. In non-strict mode (the default) the same checks run and log at `WARNING` level, but execution continues.

## Real SQL Execution (SQLAlchemy)

`SQLAlchemyExecutor` connects to any SQLAlchemy-compatible database and plugs directly into the evaluation framework as an `execution_backend`.

Install the optional dependency first:

```bash
pip install text2ql[sql]   # or: pip install sqlalchemy>=2.0
```

### Evaluate against a real database

```python
from text2ql import evaluate_examples
from text2ql.sql_executor import SQLAlchemyExecutor

executor = SQLAlchemyExecutor("postgresql+psycopg2://user:pw@host/mydb")
report = evaluate_examples(service, examples, execution_backend=executor)
print(report.execution_accuracy)
```

### In-memory SQLite for tests (no external DB)

```python
from text2ql.sql_executor import create_sqlite_executor

executor = create_sqlite_executor({
    "orders": [
        {"id": 1, "status": "active",   "total": 250.0},
        {"id": 2, "status": "inactive", "total":  80.0},
    ],
    "users": [
        {"id": 1, "name": "Alice", "status": "active"},
    ],
})

rows = executor.execute("SELECT id, total FROM orders WHERE status = 'active';")
# [{"id": 1, "total": 250.0}]
```

`create_sqlite_executor` also accepts `pandas` for bulk-loading if installed; falls back to raw DDL otherwise.

### Execute directly

```python
executor = SQLAlchemyExecutor("sqlite:///sales.db")
rows = executor.execute("SELECT COUNT(*) AS cnt FROM orders;")
# [{"cnt": 42}]
```

Use as a context manager to release the connection pool:

```python
with SQLAlchemyExecutor("sqlite:///sales.db") as executor:
    rows = executor.execute("SELECT * FROM users LIMIT 10;")
```

## Abstract IR Layer

`text2ql` defines a canonical **Intermediate Representation** (`QueryIR`) that sits between natural-language parsing and target-language rendering.  Both engines now build a `QueryIR` internally and call a concrete `IRRenderer` to produce the final query string — the renderer is the **live production path**, not just an inspection helper.

### Core types

| Type | Purpose |
|---|---|
| `QueryIR` | Top-level IR: entity, fields, filters, joins, nested, aggregations, ordering, pagination |
| `IRFilter` | Single predicate — `key`, `value`, `operator` (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `nin`, `is_null`) |
| `IRJoin` | JOIN between two tables — left/right ON-clause refs, fields, filters |
| `IRNested` | GraphQL nested selection — relation, target entity, fields, filters |
| `IRAggregation` | Aggregate expression — `COUNT`, `SUM`, `AVG`, `MIN`, `MAX` |
| `IRRenderer` | Abstract base — implement `render(ir: QueryIR) -> str` to add a new query language |
| `GraphQLIRRenderer` | **Concrete** renderer for GraphQL (used by `GraphQLEngine` in production) |
| `SQLIRRenderer` | **Concrete** renderer for SQL — also emits `GROUP BY` when aggregations are present (used by `SQLEngine` in production) |

### Inspect a result as IR

```python
from text2ql import QueryIR, Text2QL

result = Text2QL().generate(
    text="show active orders over 100 sorted by total",
    target="sql",
    schema={"entities": ["orders"], "fields": {"orders": ["id", "status", "total"]}},
)

ir = QueryIR.from_query_result(result, source_text=result.query)
print(ir.entity)          # "orders"
print(ir.filters)         # [IRFilter(key='status', value='active', operator='eq'), ...]
print(ir.order_by)        # "total"
```

### Build IR from engine components

```python
from text2ql import QueryIR

ir = QueryIR.from_components(
    entity="orders",
    fields=["id", "status", "total"],
    filters={"status": "active", "total_gte": 100},
    order_by="total",
    order_dir="DESC",
    target="sql",
    source_text="show active orders over 100 sorted by total",
)
# ir.filters → [IRFilter(key='status', operator='eq'), IRFilter(key='total', operator='gte')]
```

### Add a new query language

Subclass `IRRenderer` — that's the only requirement:

```python
from text2ql.ir import IRRenderer, QueryIR

class CypherRenderer(IRRenderer):
    def render(self, ir: QueryIR) -> str:
        label = ir.entity.capitalize()
        where = self._build_where(ir.filters)
        fields = ", ".join(f"n.{f}" for f in ir.fields) or "n"
        match = f"MATCH (n:{label})"
        where_clause = f" WHERE {where}" if where else ""
        return f"{match}{where_clause} RETURN {fields}"

    def _build_where(self, filters):
        parts = []
        for f in filters:
            if f.operator == "eq":
                parts.append(f"n.{f.key} = '{f.value}'")
            elif f.operator == "gte":
                parts.append(f"n.{f.key} >= {f.value}")
        return " AND ".join(parts)

renderer = CypherRenderer()
print(renderer.render(ir))
# MATCH (n:Orders) WHERE n.status = 'active' AND n.total >= 100 RETURN n.id, n.status, n.total
```

### SQL aggregations

`SQLIRRenderer` emits `COUNT(*)`, `SUM()`, `AVG()`, `MIN()`, `MAX()` automatically when the SQL engine detects aggregate intent words:

```python
result = Text2QL().generate(
    "count all orders with status active",
    target="sql",
    schema={"entities": ["orders"], "fields": {"orders": ["id", "status"]}},
)
print(result.query)
# SELECT orders.status, COUNT(*) AS count FROM orders
# WHERE orders.status = 'active' GROUP BY orders.status;
```

### Domain-specific entity routing via `keyword_intents`

Instead of hardcoding domain keywords inside the engine, declare routing rules in your schema config.  The engine evaluates these before falling back to alias and semantic matching:

```python
result = Text2QL().generate(
    "show my dividend history",
    target="graphql",
    schema={
        "entities": ["transactions", "positions"],
        "fields": {"transactions": ["id", "amount", "type", "date"]},
        "keyword_intents": [
            {
                "keywords": ["dividend"],
                "find_entity_by_name": "transactions"
            },
            {
                "keywords": ["net", "worth"],
                "find_entity_with_fields": ["netWorth", "regulatoryNetWorth"]
            },
            {
                "keywords": ["buying", "power"],
                "find_entity_with_fields": ["cash", "margin"],
                "preferred_entity_names": ["buyingPowerDetail"]
            }
        ],
    },
)
print(result.metadata["entity"])  # "transactions"
```

Each intent requires **all** listed `keywords` to appear in the query.  Use `find_entity_by_name` for an exact entity name match, or `find_entity_with_fields` to pick the entity whose schema contains the most candidate fields (with optional `preferred_entity_names` as a tiebreaker).

## Structured Logging

All engines and providers emit structured log records via Python's standard `logging` module under the `text2ql` namespace. No handlers are installed by default — configure them in your application.

### Basic setup

```python
import logging
logging.getLogger("text2ql").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
```

### Log levels

| Level | What fires |
|---|---|
| `DEBUG` | Every LLM provider call, SQL queries executed, IR traces |
| `WARNING` | LLM fallbacks to deterministic mode, retry attempts (429), contradictory filter detections, dropped invalid JOINs |
| `ERROR` | Network failures after all retries exhausted |

### Streamlit

Logs go to the terminal where you launched the app, not the browser. Control via the `TEXT2QL_LOG_LEVEL` environment variable:

```bash
# See fallbacks and validation warnings
TEXT2QL_LOG_LEVEL=WARNING ./venv/bin/python -m streamlit run examples/streamlit_app.py

# See every provider call and SQL query
TEXT2QL_LOG_LEVEL=DEBUG ./venv/bin/python -m streamlit run examples/streamlit_app.py
```

The Streamlit app wires a `StreamHandler` directly to the `text2ql` logger with `propagate=False` so Streamlit's root-logger reconfiguration cannot suppress it.

## CLI

```bash
text2ql "show top 5 client records with mail state enabled" \
  --target graphql \
  --schema '{"entities":["customers"],"fields":["id","email","status"]}' \
  --mapping '{"entities":{"client":"customers"},"fields":{"mail":"email"},"filters":{"state":"status"},"filter_values":{"status":{"enabled":"active"}}}'
```

SQL target via CLI:

```bash
text2ql "show customers highest total first 5 offset 10" \
  --target sql \
  --schema '{"entities":["customers"],"fields":{"customers":["id","total","status"]}}'
```

You can also load JSON files:

```bash
text2ql "show top 5 client records with mail state enabled" \
  --schema-file ./schema.json \
  --mapping-file ./mapping.json
```

LLM mode via CLI:

```bash
export OPENAI_API_KEY=...
text2ql "show top 3 clients with mail state enabled" \
  --mode llm \
  --llm-api-key "$OPENAI_API_KEY" \
  --language english \
  --system-context "Prefer customer/account semantics and preserve canonical field names." \
  --llm-provider openai-compatible \
  --llm-model gpt-4o-mini \
  --llm-max-retries 4 \
  --llm-retry-backoff 2.0
```

SQL in LLM mode:

```bash
export OPENAI_API_KEY=...
text2ql "show latest 5 orders with status active" \
  --target sql \
  --mode llm \
  --llm-api-key "$OPENAI_API_KEY" \
  --schema '{"entities":["orders"],"fields":{"orders":["id","status","createdAt"]}}'
```

If you do not pass `--mode llm`, CLI runs deterministic mode.
`--system-context` is consumed in LLM mode and ignored in deterministic mode.
`--llm-rewrite on` can still apply LLM rewrite even when `--mode deterministic` is used.

CLI parity for synthetic variants + execution evaluation:

```bash
text2ql "how many qqq do i own" \
  --schema-file ./schema.json \
  --mapping-file ./mapping.json \
  --data-file ./portfolio.json \
  --execute-on-payload \
  --variants-per-example 3 \
  --rewrite-plugins generic,portfolio \
  --domain portfolio \
  --expected-query-file ./expected.graphql
```

Execution-eval notes:

- `--expected-query` / `--expected-query-file` / `--expected-execution-file` require `--data-file`.
- `--data-file` should be the execution payload JSON used for query evaluation.
- `--execute-on-payload` enables execution without requiring expected-query comparison.
- SQL target executes generated SQL in an in-memory SQLite database built from `--data-file`.
- If expected query execution cannot be derived from the payload, CLI reports an eval warning and skips that sample from accuracy denominator.

Minimal file-based CLI example:

```bash
cat > schema.json <<'JSON'
{
  "entities": ["positions"],
  "fields": {"positions": ["symbol", "quantity"]},
  "args": {"positions": ["symbol", "limit", "orderBy", "orderDirection"]}
}
JSON

cat > mapping.json <<'JSON'
{
  "filters": {"ticker": "symbol"},
  "filter_values": {"symbol": {"qqq": "QQQ"}}
}
JSON

cat > data.json <<'JSON'
{
  "portfolio_data": {
    "positions": [
      {"symbol": "QQQ", "quantity": 100.104},
      {"symbol": "AAPL", "quantity": 12}
    ]
  }
}
JSON

text2ql "how many qqq do i own" \
  --schema-file ./schema.json \
  --mapping-file ./mapping.json
```

## Prompt templates

LLM mode supports prompt template override through request context:

```python
result = service.generate(
    text="list users",
    context={
        "mode": "llm",
        "prompt_template": "Convert request to JSON intent. Request: {text}\\nEntities: {entities}\\nFields: {fields}"
    },
)
```

Template placeholders:

- `{text}`
- `{entities}`
- `{fields}`
- `{field_aliases}`
- `{filter_aliases}`

Language support:

- `english` (default)
- `en` (alias)

## Arbitrary JSON support

`text2ql` can infer schema config from arbitrary nested JSON payloads, generate hybrid mappings (auto + overrides), and execute generated metadata against JSON data:

```python
from text2ql import (
    Text2QL,
    execute_query_result_on_json,
    generate_hybrid_mapping,
    infer_schema_from_json_payload,
)

schema = infer_schema_from_json_payload(raw_json_payload)
mapping = generate_hybrid_mapping(schema_payload=schema, data_payload=raw_json_payload)
service = Text2QL()
result = service.generate("how many qqq do I own", schema=schema, mapping=mapping)

rows, note = execute_query_result_on_json(result, raw_json_payload)
print(result.query)
print(rows, note)
```

Deterministic parsing includes a built-in holdings pattern for prompts like:

- `how many <asset> do I own`

## Nested GraphQL + schema validation

You can define relation-aware schema config for nested query generation and strict validation.

Example:

```python
result = service.generate(
    text="show customers with latest order total",
    target="graphql",
    schema={
        "entities": ["customers"],
        "fields": {"customers": ["id", "email"]},
        "args": {"customers": ["limit", "status"]},
        "relations": {
            "customers": {
                "orders": {
                    "target": "orders",
                    "fields": ["id", "total", "createdAt"],
                    "args": ["limit"],
                    "aliases": ["order"]
                }
            }
        }
    },
)
```

Behavior:

- Detects nested intents (e.g. `latest order total`) and emits nested selections **recursively up to 3 relation hops** (configurable via `max_depth`).
- Cycle-safe: if the schema has back-edges (e.g. `user → posts → user`), the traversal stops at the repeated entity — no infinite loops.
- Validates entity, fields, and args against schema before returning query.
- Drops invalid fields/args and records notes in `result.metadata["validation_notes"]`.

Additional GraphQL intent support:

- Aggregations: `count`, `sum`, `avg`, `min`, `max`
- Advanced filters:
  - range: `price between 10 and 20` -> `price_gte`, `price_lte`
  - in-list: `category in retail, wholesale` -> `category_in`
  - grouped filters: `AND`/`OR` groups
- Post-generation introspection validation:
  - supply `schema["introspection"]` with `query` and `types`
  - engine validates generated entity/args/fields against introspection metadata

## SQL support

SQL engine supports deterministic generation with:

- strict schema/table/column validation
- sort/order parsing (`latest`, `highest`, `lowest`) -> `ORDER BY ... ASC|DESC`
- pagination (`limit`, `offset`, `first`, `after`)
- robust filter parsing:
  - negation (`not`, `!=`, `is not`)
  - comparative operators (`>`, `<`, `>=`, `<=`)
  - ranges (`between`, date ranges)
  - grouped precedence (`AND`/`OR` groups)
- type coercion (`int`, `float`, `bool`, `null`, date-like literals, enums via introspection)
- relation-safe joins from schema `relations` with local relation filter extraction

## Dataset + evaluation hooks

```python
from text2ql import Text2QL, ingest_dataset, generate_synthetic_examples, evaluate_examples

examples = ingest_dataset("examples.jsonl")
synthetic = generate_synthetic_examples(examples, variants_per_example=2)
report = evaluate_examples(Text2QL(), synthetic)
print(report.exact_match_accuracy, report.execution_accuracy)
```

### Concurrent evaluation (async)

`aevaluate_examples` runs all examples concurrently — on 100 examples at 500ms LLM latency this is ~50× faster than the serial version:

```python
import asyncio
from text2ql import Text2QL, aevaluate_examples

async def main():
    report = await aevaluate_examples(
        Text2QL(),
        examples,
        concurrency=10,  # simultaneous in-flight requests
    )
    print(report.exact_match_accuracy, report.execution_accuracy)

asyncio.run(main())
```

Both sync and async `execution_backend` callables are supported in `aevaluate_examples`.

Execution accuracy against a real backend:

```python
from text2ql import Text2QL, evaluate_examples

def backend_executor(query: str, example):
    # Replace with your real backend call (GraphQL/SQL/etc).
    # Return normalized execution payload (rows/object).
    return run_query_against_backend(query)

report = evaluate_examples(
    Text2QL(),
    examples,
    execution_backend=backend_executor,
)
print(report.execution_accuracy)
```

Domain-aware synthetic rewrites via plugins:

```python
synthetic = generate_synthetic_examples(
    examples,
    variants_per_example=3,
    rewrite_plugins=["generic", "portfolio"],
    domain="portfolio",
)
```

More domain examples:

```python
crm_synthetic = generate_synthetic_examples(
    examples,
    variants_per_example=3,
    rewrite_plugins=["generic", "crm"],
    domain="crm",
)

healthcare_synthetic = generate_synthetic_examples(
    examples,
    variants_per_example=3,
    rewrite_plugins=["generic", "healthcare"],
    domain="healthcare",
)
```

Template slot generation and schema-aware lexicalization:

```python
from text2ql.dataset import DatasetExample

seed = [DatasetExample(
    text="show sales pipeline",
    target="graphql",
    expected_query="{ opportunities { amount } }",
    schema={
        "entities": ["opportunities"],
        "fields": {"opportunities": ["amount", "createdAt", "stage"]},
        "args": {"opportunities": ["stage"]},
    },
    mapping={"filter_values": {"stage": {"open": "Open"}}},
)]

synthetic = generate_synthetic_examples(
    seed,
    variants_per_example=4,
    rewrite_plugins=["generic", "crm"],
    domain="crm",
)
```

Generation behavior:

- Uses per-domain slot templates filled from schema/mapping (`entity`, `metric`, `date`, `filter`, `value`).
- Applies schema-aware lexicalization so synthetic prompts prefer schema/mapping terms.
- Scores candidates (`source`, `confidence`, `novelty`) and keeps highest-scoring variants first.
- Adds metadata on each synthetic example:
  - `synthetic_rewrite_source`
  - `synthetic_rewrite_confidence`
  - `synthetic_rewrite_novelty`
  - `synthetic_rewrite_score`

Built-in rewrite plugins:

- `generic`: neutral paraphrases (`show`/`list`, `top`/`first`, etc.).
- `portfolio`: holdings/asset phrasing rewrites (`how many qqq do i own` -> quantity/share variants).
- `banking`: account and money-flow paraphrases (`balance`, `transfer`, `deposit`, `withdraw`, `statement`).
- `crm`: sales workflow paraphrases (`leads`, `opportunities`, `pipeline`, `contacts`, `deals`).
- `healthcare`: clinical data paraphrases (`patients`, `encounters`, `diagnosis`, `medications`, `labs`, `claims`).
- `ecommerce`: shopping/order paraphrases (`orders`, `products`, `cart`, `inventory`, `refund`, `shipment`).

### Dataset format

Supported file types:

- `.jsonl`: one JSON object per line
- `.json`: JSON array of objects

Required fields per example:

- `text` (string)
- `expected_query` (string)

Optional fields:

- `target` (default: `"graphql"`)
- `schema` (object)
- `mapping` (object)
- `context` (object)
- `metadata` (object)

Example `.jsonl` row:

```json
{"text":"list users","target":"graphql","expected_query":"{ user { id name } }"}
```

### Evaluation metrics

- `exact_match_accuracy`: normalized string match (whitespace-insensitive).
- `execution_accuracy`:
  - with `execution_backend`: compares real backend execution outputs for predicted vs expected queries.
  - without `execution_backend`: structural signature match (`graphql` + `sql` supported).

For precomputed gold execution output, set `example.metadata["expected_execution_result"]`.

## Troubleshooting

- `Missing API key...`: set `OPENAI_API_KEY` (or `TEXT2QL_API_KEY`) before LLM mode.
- `JSON file ... must contain an object`: schema/mapping files must be top-level JSON objects.
- `Inline JSON value must contain an object`: `--schema`/`--mapping` payloads must be JSON objects.
- `HTTP Error 429: Too Many Requests`: provider retries with backoff; if retries fail, engine falls back to deterministic mode.
- Provider/network errors in LLM mode: verify API URL/model/key and retry.

## Testing

```bash
python3 -m pytest -m unit
python3 -m pytest -m e2e
python3 -m pytest
```

## Publish to PyPI

Release workflow file:

- `.github/workflows/release.yml`

Publishing modes:

1. `workflow_dispatch` -> TestPyPI (`publish_target=testpypi`)
2. GitHub Release `published` -> PyPI
3. `workflow_dispatch` -> PyPI (`publish_target=pypi`)

Setup checklist:

1. Replace placeholder URLs in `pyproject.toml` (`project.urls`).
2. In PyPI and TestPyPI, configure Trusted Publishing for this GitHub repo/workflow.
3. In GitHub repo settings, create environments `testpypi` and `pypi`.
4. Protect `pypi` environment with required reviewers if desired.

Local preflight before release:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Current architecture

- `text2ql.core.Text2QL`: orchestrator/facade — `generate()`, `agenerate()`, `agenerate_many()`.
- `text2ql.types`: request/result schemas (`QueryRequest`, `QueryResult`, `ValidationError`).
- `text2ql.ir`: abstract IR layer — `QueryIR`, `IRFilter`, `IRJoin`, `IRNested`, `IRAggregation`, `IRRenderer`.
- `text2ql.renderers`: concrete renderers — `GraphQLIRRenderer` and `SQLIRRenderer`.  Both engines build a `QueryIR` internally and call `renderer.render(ir)` to produce the final query string.  `SQLIRRenderer` also handles `SELECT COUNT(*)/SUM()/AVG()` + `GROUP BY` when aggregations are detected.
- `text2ql.filters`: shared compiled regex singletons and stateless helpers (`detect_comparison_filters`, `detect_negation_filters`, `detect_between_filters`, `detect_in_filters`, `detect_date_range_filters`) — imported by both engines to eliminate regex duplication.
- `text2ql.engines.*`: per-target query generators — each exposes both `generate()` and `agenerate()`.
  - `strict_validation=True` raises `ValidationError` on contradictory filters or invalid JOIN ON-clause columns.
  - Entity detection is fully schema-driven: alias → name → semantic field match → `schema.keyword_intents` → first schema entity → generic text extraction (no hardcoded domain lists).
  - GraphQL nested detection is recursive (up to `max_depth=3` hops) with cycle prevention.
- `text2ql.engines.base.compute_deterministic_confidence`: runtime confidence scoring (schema, entity, fields, filters, validation).
- `text2ql.providers.*`: pluggable LLM provider adapters — implement `complete()`, get `acomplete()` for free (or override for native async).
  - `complete_structured()` / `acomplete_structured()` for function-calling / structured output mode.
  - `OpenAICompatibleProvider(use_structured_output=True)` uses `response_format: json_schema`.
- `text2ql.prompting`: prompt templates + `GRAPHQL_INTENT_JSON_SCHEMA` / `SQL_INTENT_JSON_SCHEMA` for structured output.
- `text2ql.sql_executor`: `SQLAlchemyExecutor` and `create_sqlite_executor()` for real SQL execution.
- `text2ql.evaluate`: `evaluate_examples()` (serial) + `aevaluate_examples()` (concurrent async).
- `text2ql.rewrite`: `rewrite_user_utterance()` + `arewrite_user_utterance()` (async).
- `text2ql.benchmarks`: Spider and BIRD benchmark adapters — `load_spider()`, `load_bird()`, `run_benchmark()`, `arun_benchmark()`, `format_report()`.

## Benchmarking (Spider & BIRD)

text2ql includes built-in adapters for the two standard text-to-SQL benchmarks: [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/). Run them to measure accuracy against published baselines and track improvements over time.

### Quick start

```bash
pip install text2ql[sql]   # SQLAlchemy needed for execution-accuracy mode
```

### Python API

```python
from text2ql.benchmarks import load_spider, load_bird, run_benchmark, format_report
from text2ql.benchmarks.runner import BenchmarkConfig

# Load Spider dev split (1034 examples)
examples = load_spider("/path/to/spider", split="dev")

# Run with structural matching (no database needed)
report = run_benchmark(examples, config=BenchmarkConfig(mode="structural"))
print(format_report(report))

# Run with execution accuracy (requires SQLite databases)
report = run_benchmark(examples, config=BenchmarkConfig(mode="execution"))
print(format_report(report, verbose=True))
```

BIRD mini-dev (500 examples with SQLite databases):

```python
examples = load_bird("/path/to/bird-minidev", split="dev")
report = run_benchmark(examples, config=BenchmarkConfig(mode="execution"))
print(format_report(report))
```

### LLM mode benchmarking

```python
from text2ql.core import Text2QL
from text2ql.providers.openai_compatible import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(model="gpt-4o-mini")
service = Text2QL(provider=provider)

# Set LLM mode on all examples
for ex in examples:
    ex.context["mode"] = "llm"

config = BenchmarkConfig(mode="execution", service=service)
report = run_benchmark(examples, config=config)
print(format_report(report))
```

### CLI

```bash
# Spider — structural evaluation
text2ql --benchmark spider \
  --benchmark-path /path/to/spider \
  --benchmark-mode structural

# BIRD — execution accuracy with SQLite
text2ql --benchmark bird \
  --benchmark-path /path/to/bird-minidev \
  --benchmark-mode execution

# With LLM provider
export OPENAI_API_KEY=...
text2ql --benchmark spider \
  --benchmark-path /path/to/spider \
  --benchmark-mode structural \
  --mode llm \
  --llm-model gpt-4o-mini

# Limit examples for quick smoke test
text2ql --benchmark spider \
  --benchmark-path /path/to/spider \
  --benchmark-limit 50

# Filter to a single database
text2ql --benchmark bird \
  --benchmark-path /path/to/bird-minidev \
  --benchmark-db california_schools

# Verbose output (shows individual failures)
text2ql --benchmark spider \
  --benchmark-path /path/to/spider \
  --benchmark-verbose
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmark` | — | Benchmark name: `spider` or `bird` |
| `--benchmark-path` | — | Path to dataset root directory |
| `--benchmark-split` | `dev` | Split to evaluate (`dev`, `train`) |
| `--benchmark-limit` | `0` (all) | Cap the number of examples |
| `--benchmark-db` | — | Only evaluate this `db_id` |
| `--benchmark-mode` | `execution` | `exact`, `structural`, or `execution` |
| `--benchmark-verbose` | `false` | Show per-example failure details |

### Evaluation modes

| Mode | Requires DB | What it checks |
|------|-------------|----------------|
| `exact` | No | Normalized string equality |
| `structural` | No | Parsed SQL signature match (table, columns, filters, ordering) |
| `execution` | Yes (SQLite) | Run both gold and predicted SQL, compare result sets |

### Report output

Reports include:
- Overall exact-match, structural-match, and execution accuracy
- Breakdown by difficulty level (easy/medium/hard/extra for Spider; simple/moderate/challenging for BIRD)
- Breakdown by database
- Bottom-5 databases by accuracy
- Per-example failure details (verbose mode)
- Machine-readable JSON summary

### Dataset setup

**Spider**: clone the [taoyds/spider](https://github.com/taoyds/spider) repo. The `evaluation_examples/examples/` directory contains `tables.json` and `dev.json`. For execution accuracy, download the database files from the [official site](https://yale-lily.github.io/spider).

**BIRD mini-dev**: download from [bird-bench/mini_dev](https://github.com/bird-bench/mini_dev). The `MINIDEV/` directory contains `mini_dev_sqlite.json` and `dev_databases/` with SQLite files.

## Roadmap

1. Add `Cypher`, `Jsonata`, `Jq` and `SPARQL` engines using the `IRRenderer` base class. *(IR layer is ready — only `render()` needs implementing per language.)*
2. Expand prompts and constraints per target language.
3. Add richer synthetic generation using domain-specific rewrite plugins.
4. ~~Add execution evaluation hooks for real backends~~ *(done — `SQLAlchemyExecutor` + `create_sqlite_executor`)*.
5. ~~Add function-calling / structured output support~~ *(done — `use_structured_output`, `mode="function_calling"`, `complete_structured()`)*.
6. ~~Add Spider/BIRD benchmark evaluation~~ *(done — `load_spider`, `load_bird`, `run_benchmark`, CLI `--benchmark`)*.
7. Add few-shot example support in LLM mode with dynamic example retrieval based on schema/mapping similarity.
8. Multilingual prompt support (currently English only).

