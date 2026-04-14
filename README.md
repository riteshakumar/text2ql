# text2ql

Natural-language to query-language toolkit for **GraphQL** and **SQL**.

`text2ql` is built for practical usage: deterministic generation, LLM-assisted generation, schema/mapping normalization, evaluation utilities, and benchmark adapters (Spider/BIRD).
Current first-class targets are GraphQL and SQL; planned future targets include JSONata, jq-style queries, Cypher, and dialect-focused SQL paths (PostgreSQL, MySQL, and more).

## Why teams use it

- Fast path from text -> query for GraphQL and SQL.
- Three generation modes: `deterministic`, `llm`, `function_calling`.
- Schema + mapping aware generation with validation and confidence scoring.
- Built-in dataset ingestion, synthetic rewrites, and evaluation hooks.
- Built-in Spider/BIRD benchmark loaders and runner.

## Install

Python: `>=3.10`

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
pip install -e ".[dev]"
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
print(result.confidence)
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
| `deterministic` | Rule/schema-driven generation | Fast, predictable production defaults |
| `llm` | Direct LLM-assisted generation | Broad language coverage |
| `function_calling` | Schema-constrained structured intent path | Better output shape control |

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

Production checklist:

1. Keep `schema.json` and `mapping.json` in source control.
2. Start with `--mode deterministic` for baseline reliability.
3. Enable `--mode llm` or `--mode function_calling` after baseline tests pass.
4. Use retry settings for provider calls (`--llm-max-retries`, `--llm-retry-backoff`).
5. Run evaluation/benchmark checks in CI before releases.
6. Use `-v` during rollout/debugging for detailed logs.

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

### Latest LLM benchmark snapshot (synthetic harness)

Using `run_llm_benchmark.py` (50 Spider-style + 50 BIRD-style synthetic examples, `mode=llm`, `gpt-4o-mini`):

Raw summary:

| Benchmark | Exact | Structural | Execution | Errors |
|---|---:|---:|---:|---:|
| Spider (50) | 62.0% | 64.0% | 84.0% | 0/50 |
| BIRD (50) | 70.0% | 78.0% | 90.0% | 0/50 |

Reproduce:

```bash
OPENAI_API_KEY=... ./venv/bin/python run_llm_benchmark.py
# or
OPENAI_API_KEY=... python3 run_llm_benchmark.py
```

### Latest deterministic benchmark snapshot (synthetic harness)

Using `run_deterministic_benchmark.py` (50 Spider-style + 50 BIRD-style synthetic examples, `mode=deterministic`):

Raw summary:

| Benchmark | Exact | Structural | Execution | Errors |
|---|---:|---:|---:|---:|
| Spider (50) | 0.0% | 40.0% | 100.0% | 0/50 |
| BIRD (50) | 0.0% | 52.0% | 100.0% | 0/50 |

Reproduce:

```bash
./venv/bin/python run_deterministic_benchmark.py
# or
python3 run_deterministic_benchmark.py
```

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
