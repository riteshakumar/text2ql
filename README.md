# text2ql
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/text2ql?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/text2ql)

Natural-language to query-language toolkit for **GraphQL** and **SQL**.

`text2ql` is built for practical usage: deterministic generation, LLM-assisted generation, schema/mapping normalization, evaluation utilities, and benchmark adapters (Spider/BIRD).
Current targets are GraphQL and SQL. SQL IR rendering supports SQLite, PostgreSQL, MySQL, and SQL Server. Additional query languages remain on the roadmap.

## Why teams use it

- Fast path from text -> query for GraphQL and SQL.
- Three generation modes: `deterministic`, `llm`, `function_calling`.
- Schema + mapping aware generation, parsed validation, and explicitly labeled heuristic scores.
- Built-in dataset ingestion, synthetic rewrites, and evaluation hooks.
- Built-in Spider/BIRD benchmark loaders and runner.

## Install

Python: `>=3.10`. Core query validation uses SQLGlot and graphql-core; generation can still run offline in deterministic mode.

```bash
pip install text2ql
```

Optional extras:

```bash
# Streamlit playground
pip install "text2ql[app]"

# SQL execution backend support (SQLAlchemy)
pip install "text2ql[sql]"

# Local development tools
pip install -e ".[dev,sql]"
```

## Quick Start (CLI)

1) Deterministic GraphQL:

```bash
text2ql "show top 5 client records with mail state enabled" \
  --target graphql \
  --schema '{"entities":["customers"],"fields":{"customers":["id","email","status"]}}' \
  --mapping '{"entities":{"client":"customers"},"fields":{"mail":"email"},"filters":{"state":"status"},"filter_values":{"status":{"enabled":"active"}}}'
```

2) Deterministic SQL:

```bash
text2ql "show customers highest total first 5 offset 10" \
  --target sql \
  --schema '{"entities":["customers"],"fields":{"customers":["id","total","status"]}}'
```

3) LLM mode:

```bash
export OPENAI_API_KEY=...
text2ql "show latest 5 orders with status active" \
  --target sql \
  --mode llm \
  --llm-model gpt-4o-mini \
  --schema '{"entities":["orders"],"fields":{"orders":["id","status","createdAt"]}}'
```

4) Function-calling / structured output mode:

```bash
export OPENAI_API_KEY=...
text2ql "show latest 5 orders with status active" \
  --target graphql \
  --mode function_calling \
  --llm-model gpt-4o-mini \
  --schema '{"entities":["orders"],"fields":{"orders":["id","status","createdAt"]},"args":{"orders":["status","limit","orderBy","orderDirection"]}}'
```

## Quick Start (Python API)

```python
from text2ql import Text2QL

service = Text2QL()
result = service.generate(
    text="list active customers",
    target="graphql",
    schema={"entities": ["customers"], "fields": {"customers": ["id", "status", "email"]}},
    mapping={"filters": {"state": "status"}, "filter_values": {"status": {"active": "active"}}},
)

print(result.query)
print(result.status)  # ok, needs_review, or needs_clarification
print(result.metadata["confidence_kind"])  # heuristic, not a probability
if result.executable:
    print(result.ir)  # exact compiled intent; direct LLM queries have no IR
```

LLM provider wiring:

```python
from text2ql import Text2QL
from text2ql.providers.openai_compatible import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    api_key="...",  # or use OPENAI_API_KEY / TEXT2QL_API_KEY
    model="gpt-4o-mini",
    use_structured_output=True,
)
service = Text2QL(provider=provider)
```

## Streamlit Playground

Use hosted app: https://text2ql.streamlit.app/
For local/private data: `pip install -e ".[app]"`.
Run locally: `python -m streamlit run examples/streamlit_app.py`.

## Modes at a glance

| Mode | What it does | Best for |
|---|---|---|
| `deterministic` | Rule/schema-driven generation | Known query patterns covered by your own tests |
| `llm` | LLM writes a query, then parsers validate it | Queries beyond the current IR compiler |
| `function_calling` | Typed intent compiled through the IR | Preserving supported filters, grouping, and selections |

## Production Setup (brief)

Recommended project files:

```text
project/
  schema.json
  mapping.json
  data.json                # optional; used for payload execution checks
  expected_query.sql       # optional
  expected_rows.json       # optional
```

`Text2QL()` validates strictly by default. Invalid fields, relations, or generated statements raise `ValidationError`; unsupported requests can return `needs_clarification`. Always check `result.executable` before execution. Explicit lenient mode (`Text2QL(strict_validation=False)`) marks repaired results `needs_review`.

LLM failures propagate by default. To deliberately allow deterministic fallback, use `context={"allow_llm_fallback": True}` or `--allow-llm-fallback`. Native structured requests require `use_structured_output=True`; falling back from native structured output to plain JSON requires the separate provider option `allow_structured_fallback=True`. Results record the actual mode and fallback reason.

Run generated SQL with a database role restricted to the intended reads. `SQLAlchemyExecutor` parses every statement, binds compiled IR values, caps returned rows, and applies SQLite/PostgreSQL read controls and query deadlines. Its `before_execute` hook can enforce application policy. SQL parsing does not enforce tenant boundaries or make arbitrary database functions safe. GraphQL resolvers must enforce their own authorization.

See [reliability and migration notes](docs/reliability.md) for execution settings, schema formats, behavior changes, and remaining limitations.

Useful CLI operations:

```bash
# Generate hybrid mapping (auto baseline + optional overrides)
text2ql --generate-hybrid-mapping \
  --schema-file ./schema.json \
  --data-file ./data.json \
  --mapping-output-file ./mapping.generated.json

# Execute generated query against payload JSON and compare expected output
text2ql "how many qqq do i own" \
  --target graphql \
  --schema-file ./schema.json \
  --mapping-file ./mapping.json \
  --data-file ./data.json \
  --expected-execution-file ./expected_rows.json
```

## Benchmarking (Spider & BIRD)

`text2ql` ships with benchmark adapters and runner APIs:

- `load_spider(...)`
- `load_bird(...)`
- `run_benchmark(...)`, `arun_benchmark(...)`
- `format_report(...)`

CLI examples:

```bash
# Spider
text2ql --benchmark spider --benchmark-path /path/to/spider --benchmark-mode structural

# BIRD
text2ql --benchmark bird --benchmark-path /path/to/bird-minidev --benchmark-mode execution
```

### Synthetic regression results

The local harness contains 50 Spider-style and 50 BIRD-style examples. These are small synthetic fixtures, not official benchmark results. Numeric columns now retain their types, execution errors remain in the denominator, and gold `ORDER BY` clauses preserve row order during comparison. Structural matching compares parsed clauses conservatively and is separate from execution.

Deterministic run after the reliability fixes:

| Fixture set | Exact | Structural | Execution | Errors |
|---|---:|---:|---:|---:|
| Spider-style (50) | 0.0% | 54.0% | 84.0% | 8/50 |
| BIRD-style (50) | 0.0% | 62.0% | 62.0% | 17/50 |

```bash
python run_deterministic_benchmark.py
```

The earlier 100% deterministic and 84%/90% LLM snapshots used the previous evaluation harness and are withdrawn as reliability evidence. The LLM harness has not been rerun with a live provider after these fixes. To run it with your configured provider credentials:

```bash
python run_llm_benchmark.py
```

Execution agreement on one small fixture can be accidental, especially when both queries return no rows. Use diverse database fixtures and held-out questions for deployment decisions. Download the real datasets separately and use their official evaluators when reporting Spider/BIRD benchmark scores.

## Evaluation + Synthetic Data Utilities

Public helpers:

- `ingest_dataset(...)`
- `generate_synthetic_examples(...)`
- `evaluate_examples(...)`
- `aevaluate_examples(...)`
- `rewrite_user_utterance(...)`

Built-in rewrite plugins include:

- `generic`, `portfolio`, `banking`, `crm`, `healthcare`, `ecommerce`

## Repo layout (high level)

```text
src/text2ql/
  core.py                 # Text2QL facade
  cli.py                  # CLI entrypoint
  engines/                # GraphQL/SQL engines + stage modules
  renderers.py            # GraphQLIRRenderer / SQLIRRenderer
  evaluate.py             # exact/structural/backend evaluation
  benchmarks/             # Spider/BIRD loaders + runner
  mapping.py              # hybrid mapping generation
  dataset.py              # ingestion + synthetic variants
  sql_executor.py         # SQLAlchemy-backed execution backend
```

## Testing

```bash
python3 -m pytest -m unit
python3 -m pytest -m e2e
python3 -m pytest
```

## License

Apache-2.0
