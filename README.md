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
| Schema/introspection validation | Yes | Yes |
| Enum/type coercion | Yes | Yes |
| Advanced filters (`not`, `!=`, ranges, grouped precedence) | Yes | Yes |
| Order parsing (`latest/highest/lowest`) | Yes | Yes |
| Pagination (`limit`, `offset`, `first`, `after`) | Yes | Yes |
| Nested/relation safety | Yes | Yes |
| Async generation (`agenerate`, `agenerate_many`) | Yes | Yes |
| Structural execution match (no backend) | Yes | Yes |
| Real backend execution accuracy hook | Yes | Yes (via `evaluate_examples(..., execution_backend=...)`) |
| Concurrent evaluation (`aevaluate_examples`) | Yes | Yes |
| Synthetic rewrite plugins | Yes | Yes (target-agnostic dataset API) |

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
  --variants-per-example 3 \
  --rewrite-plugins generic,portfolio \
  --domain portfolio \
  --expected-query-file ./expected.graphql
```

Execution-eval notes:

- `--expected-query` / `--expected-query-file` / `--expected-execution-file` require `--data-file`.
- `--data-file` should be the execution payload JSON used for query evaluation.
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

- Detects nested intents (e.g. `latest order total`) and emits nested selections.
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
    expected_query="query GeneratedQuery { opportunities { amount } }",
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
{"text":"list users","target":"graphql","expected_query":"query GeneratedQuery { user { id name } }"}
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
- `text2ql.types`: request/result schemas (`QueryRequest`, `QueryResult`).
- `text2ql.engines.*`: per-target query generators — each exposes both `generate()` and `agenerate()`.
- `text2ql.engines.base.compute_deterministic_confidence`: runtime confidence scoring (schema, entity, fields, filters, validation).
- `text2ql.providers.*`: pluggable LLM provider adapters — implement `complete()`, get `acomplete()` for free (or override for native async).
- `text2ql.evaluate`: `evaluate_examples()` (serial) + `aevaluate_examples()` (concurrent async).
- `text2ql.rewrite`: `rewrite_user_utterance()` + `arewrite_user_utterance()` (async).

## Roadmap

1. Add `Cypher`, `Jsonata`, `Jq` and `SPARQL` engines.  
2. Expand prompts and constraints per target language.
3. Add richer synthetic generation using domain-specific rewrite plugins.
4. Add execution evaluation hooks for real backends (e.g. GraphQL endpoint, SQL database).
5. Add more provider adapters and support for provider-specific features (e.g. function calling).
6. Add few-shot example support in LLM mode with dynamic example retrieval based on schema/mapping similarity.
7. Add a playground interface for interactive experimentation and debugging.
    
