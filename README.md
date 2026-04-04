# text2ql

Natural Language to Query Language framework.

`text2ql` is designed as a **pip-installable package** for converting natural language into query languages with a plugin architecture. The first implemented target is **GraphQL**.

## Project goals

- Build a reusable core abstraction (`Text2QL`) for `text -> query` conversion.
- Support multiple target query languages (GraphQL first; SQL/Cypher/SPARQL next).
- Keep provider integrations optional (deterministic local mode, LLM mode as adapters).
- Encourage benchmark/data generation workflows inspired by graph-centric datasets.

## Install

```bash
pip install -e .
```

For local development and tests:

```bash
pip install -e ".[dev]"
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

Default mode is deterministic.

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
    context={"mode": "llm"},
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
  --llm-provider openai-compatible \
  --llm-model gpt-4o-mini \
  --llm-max-retries 4 \
  --llm-retry-backoff 2.0
```

If you do not pass `--mode llm`, CLI runs deterministic mode.

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

## Dataset + evaluation hooks

```python
from text2ql import Text2QL, ingest_dataset, generate_synthetic_examples, evaluate_examples

examples = ingest_dataset("examples.jsonl")
synthetic = generate_synthetic_examples(examples, variants_per_example=2)
report = evaluate_examples(Text2QL(), synthetic)
print(report.exact_match_accuracy, report.execution_accuracy)
```

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
- `execution_accuracy`: structural GraphQL signature match (entity + filters + selected fields).

Current execution accuracy is a static structural approximation, not live backend execution.

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

## Current architecture

- `text2ql.core.Text2QL`: orchestrator/facade.
- `text2ql.types`: request/result schemas.
- `text2ql.engines.*`: per-target query generators.
- `text2ql.providers.*`: pluggable LLM provider adapters.

## Roadmap

1. Add `SQL`, `Cypher`, and `SPARQL` engines.
2. Expand prompts and constraints per target language.
3. Add richer synthetic generation using domain-specific rewrite plugins.
4. Add execution accuracy against real backends.
5. Publish package to PyPI and add CI release workflow.
