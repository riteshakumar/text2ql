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
    text="show top 5 customers with email status active",
    target="graphql",
    schema={
        "entities": ["customers"],
        "fields": ["id", "email", "status"]
    }
)

print(result.query)
print(result.explanation)
```

## CLI

```bash
text2ql "show top 5 customers with email status active" --target graphql --schema '{"entities":["customers"],"fields":["id","email","status"]}'
```

## Current architecture

- `text2ql.core.Text2QL`: orchestrator/facade.
- `text2ql.types`: request/result schemas.
- `text2ql.engines.*`: per-target query generators.
- `text2ql.providers.*`: pluggable LLM provider adapters.

## Roadmap

1. Add `SQL`, `Cypher`, and `SPARQL` engines.
2. Introduce prompt templates and constrained output validation.
3. Add dataset ingestion and synthetic training data hooks.
4. Add evaluation harness for exact match + execution accuracy.
5. Publish package to PyPI and add CI release workflow.
