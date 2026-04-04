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
  --llm-model gpt-4o-mini
```

## Dataset + evaluation hooks

```python
from text2ql import Text2QL, ingest_dataset, generate_synthetic_examples, evaluate_examples

examples = ingest_dataset("examples.jsonl")
synthetic = generate_synthetic_examples(examples, variants_per_example=2)
report = evaluate_examples(Text2QL(), synthetic)
print(report.exact_match_accuracy, report.execution_accuracy)
```

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
