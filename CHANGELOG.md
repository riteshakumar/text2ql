# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.2.0] — 2026-04-11

### Added

- **Intermediate Representation (IR) layer** — `QueryIR`, `IRFilter`, `IRJoin`,
  `IRNested`, `IRAggregation`, and the `IRRenderer` abstract base class are now
  exported from the top-level package.  Engines build a `QueryIR` and hand it to
  a renderer; new query languages only need to subclass `IRRenderer`.
- **`IRNested.children`** — `IRNested` now carries a `children: list[IRNested]`
  field for multi-hop nested relations.  `_nested_from_dict()` recurses into the
  `"nested"` key automatically, and `GraphQLIRRenderer._render_nested()` renders
  the full tree.
- **Concrete IR renderers** — `GraphQLIRRenderer` and `SQLIRRenderer` (in
  `text2ql.renderers`) translate a `QueryIR` into the target query string.
  Both are exported from the package root.
- **SQL aggregations** — `COUNT(*)`, `SUM`, `AVG`, `MIN`, `MAX` are detected in
  deterministic mode by `_detect_aggregations()` and rendered via
  `SQLIRRenderer` with correct `GROUP BY` generation.
- **SQL aggregations in LLM mode** — `_build_llm_result()` now calls
  `_detect_aggregations()` and passes the result to `_build_sql()` and the
  returned `QueryResult.metadata`.
- **Schema-driven `keyword_intents`** — `NormalizedSchemaConfig` now accepts a
  `keyword_intents` list so domain-specific compound-keyword routing rules
  (previously hardcoded financial keywords) can be supplied via the schema
  config.
- **Recursive nested detection** — `GraphQLEngine._detect_nested()` recurses
  up to `max_depth=3` hops and uses a `frozenset` visited-set for cycle safety.
  Detected child nodes are stored in `IRNested.children`.
- **Generic entity text extraction** — `QueryEngine._extract_entity_from_text()`
  (now in the base class) applies stop-word filtering and basic singularisation
  to extract an entity name from free text when no schema entities are declared.
  Both `GraphQLEngine` and `SQLEngine` inherit and use it.
- **SQL entity text fallback** — `SQLEngine._detect_table()` now calls
  `_extract_entity_from_text()` instead of returning the hardcoded string
  `"items"` when no schema entities are found.
- **Shared filter regex module** — `text2ql.filters` provides compiled regex
  constants and helper functions (`detect_comparison_filters`,
  `detect_negation_filters`, `detect_between_filters`, `detect_in_filters`,
  `detect_date_range_filters`) shared by both engines.
- **`exact_filter_keys` guard** — `_split_filters()` and
  `QueryIR.from_components()` accept a `frozenset[str]` of schema column names
  whose suffixes must not be stripped as comparison operators (e.g. a column
  literally named `shipped_ne`).
- **SQLAlchemy executor** — `SQLAlchemyExecutor` and `create_sqlite_executor`
  execute generated SQL queries against a live database.
- **Async API** — `agenerate`, `aevaluate_examples`, `arewrite_user_utterance`
  use `asyncio.to_thread` for non-blocking operation.
- **Runtime confidence scoring** — `compute_deterministic_confidence()` in
  `engines/base.py` scores each deterministic result based on schema coverage,
  entity resolution, field coverage, filters, and extra signals.
- **E2E test coverage** — new e2e tests for SQL JOIN, SQL COUNT/SUM
  aggregation, GraphQL nested relations (single and multi-hop),
  GraphQL COUNT, and `IRRenderer` round-trips.

### Changed

- **GraphQL coercion alignment** — `GraphQLEngine._coerce_filter_value()` now
  applies enum validation to list items (not only scalars), matching the SQL
  engine's behaviour.  A variable-shadowing bug in the canonical enum lookup was
  also fixed (`value` → `enum_val`).
- **`_resolve_special_entity()` generalised** — uses the schema-supplied
  `keyword_intents` list instead of hardcoded financial-domain keyword sets.
- **`_detect_filters()` refactored** — split into `_apply_alias_filters()` and
  `_apply_advanced_filters()` with a thin orchestrator, reducing method length
  and improving testability.
- Version bumped from **0.1.13 → 0.2.0**.

### Fixed

- `GraphQLEngine._coerce_filter_value()`: shadowed `value` variable in
  `next()` generator expression caused wrong canonical enum name to be returned.
- `SQLEngine._detect_table()`: hardcoded `"items"` fallback replaced with
  heuristic text extraction.
- `IRNested` rendered with empty `child_nodes` in `GraphQLIRRenderer` even when
  children were detected — fixed by populating from `IRNested.children`.

---

## [0.1.13] and earlier

See the git log for a summary of earlier changes.
