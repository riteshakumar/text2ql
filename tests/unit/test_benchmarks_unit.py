"""Unit tests for the benchmarks module.

These tests use synthetic fixture data (no actual Spider/BIRD downloads
required) to verify the adapters, schema converters, and runner logic.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from text2ql.benchmarks.spider import load_spider, spider_schema_to_text2ql
from text2ql.benchmarks.bird import load_bird, bird_schema_to_text2ql
from text2ql.benchmarks.runner import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkRow,
    run_benchmark,
    format_report,
)
from text2ql.core import Text2QL
from text2ql.dataset import DatasetExample


# ---------------------------------------------------------------------------
# Fixtures — minimal Spider-format data
# ---------------------------------------------------------------------------

SPIDER_TABLES = [
    {
        "db_id": "test_db",
        "table_names_original": ["users", "orders"],
        "table_names": ["Users", "Orders"],
        "column_names_original": [
            [-1, "*"],
            [0, "id"],
            [0, "name"],
            [0, "email"],
            [1, "id"],
            [1, "user_id"],
            [1, "amount"],
            [1, "status"],
        ],
        "column_types": ["text", "number", "text", "text", "number", "number", "number", "text"],
        "primary_keys": [1, 4],
        "foreign_keys": [[5, 1]],  # orders.user_id -> users.id
    }
]

SPIDER_DEV = [
    {
        "db_id": "test_db",
        "question": "Show all users",
        "query": "SELECT * FROM users",
        "difficulty": "easy",
    },
    {
        "db_id": "test_db",
        "question": "How many orders are there?",
        "query": "SELECT COUNT(*) FROM orders",
        "difficulty": "medium",
    },
    {
        "db_id": "test_db",
        "question": "List users with their order amounts",
        "query": "SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.user_id",
        "difficulty": "hard",
    },
]


def _create_spider_fixture(tmp: Path) -> Path:
    """Create a minimal Spider dataset directory."""
    root = tmp / "spider"
    root.mkdir()
    (root / "tables.json").write_text(json.dumps(SPIDER_TABLES))
    (root / "dev.json").write_text(json.dumps(SPIDER_DEV))

    # Create a SQLite database
    db_dir = root / "database" / "test_db"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "test_db.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@example.com')")
    conn.execute("INSERT INTO orders VALUES (1, 1, 100.0, 'active')")
    conn.execute("INSERT INTO orders VALUES (2, 2, 200.0, 'pending')")
    conn.commit()
    conn.close()

    return root


def _create_bird_fixture(tmp: Path) -> Path:
    """Create a minimal BIRD dataset directory."""
    root = tmp / "bird"
    root.mkdir()

    dev_examples = [
        {
            "db_id": "test_db",
            "question": "Show all users",
            "SQL": "SELECT * FROM users",
            "evidence": "",
            "difficulty": "simple",
        },
        {
            "db_id": "test_db",
            "question": "Count the orders",
            "SQL": "SELECT COUNT(*) FROM orders",
            "evidence": "Count means total number",
            "difficulty": "simple",
        },
    ]
    (root / "dev.json").write_text(json.dumps(dev_examples))

    # Create database directory
    db_dir = root / "dev_databases" / "test_db"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "test_db.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@example.com')")
    conn.execute("INSERT INTO orders VALUES (1, 1, 100.0, 'active')")
    conn.execute("INSERT INTO orders VALUES (2, 2, 200.0, 'pending')")
    conn.commit()
    conn.close()

    return root


# ---------------------------------------------------------------------------
# Spider adapter tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpiderAdapter:
    def test_load_spider_basic(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev")
        assert len(examples) == 3
        assert all(ex.target == "sql" for ex in examples)
        assert examples[0].text == "Show all users"
        assert examples[0].expected_query == "SELECT * FROM users"

    def test_load_spider_with_limit(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev", limit=1)
        assert len(examples) == 1

    def test_load_spider_with_db_filter(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev", db_filter="nonexistent")
        assert len(examples) == 0

    def test_load_spider_metadata(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev")
        meta = examples[0].metadata
        assert meta["benchmark"] == "spider"
        assert meta["split"] == "dev"
        assert meta["db_id"] == "test_db"
        assert meta["difficulty"] == "easy"
        assert meta["db_path"] is not None

    def test_load_spider_missing_tables_json(self, tmp_path: Path) -> None:
        root = tmp_path / "empty_spider"
        root.mkdir()
        with pytest.raises(FileNotFoundError, match="tables.json"):
            load_spider(root)

    def test_load_spider_missing_split_file(self, tmp_path: Path) -> None:
        root = tmp_path / "spider_no_dev"
        root.mkdir()
        (root / "tables.json").write_text(json.dumps([]))
        with pytest.raises(FileNotFoundError, match="split"):
            load_spider(root, split="dev")


@pytest.mark.unit
class TestSpiderSchemaConversion:
    def test_basic_conversion(self) -> None:
        schema = spider_schema_to_text2ql(SPIDER_TABLES[0])
        assert "entities" in schema
        assert "fields" in schema
        assert "relations" in schema
        assert "args" in schema

    def test_entities_extracted(self) -> None:
        schema = spider_schema_to_text2ql(SPIDER_TABLES[0])
        entity_names = [e["name"] if isinstance(e, dict) else e for e in schema["entities"]]
        assert "users" in entity_names
        assert "orders" in entity_names

    def test_fields_per_entity(self) -> None:
        schema = spider_schema_to_text2ql(SPIDER_TABLES[0])
        assert "users" in schema["fields"]
        assert "id" in schema["fields"]["users"]
        assert "name" in schema["fields"]["users"]
        assert "email" in schema["fields"]["users"]
        assert "orders" in schema["fields"]
        assert "amount" in schema["fields"]["orders"]

    def test_relations_from_foreign_keys(self) -> None:
        schema = spider_schema_to_text2ql(SPIDER_TABLES[0])
        # orders.user_id -> users.id should create relations in both directions
        assert "orders" in schema["relations"]
        order_rels = schema["relations"]["orders"]
        target_names = [r["target"] for r in order_rels]
        assert "users" in target_names

    def test_args_match_fields(self) -> None:
        schema = spider_schema_to_text2ql(SPIDER_TABLES[0])
        for table in ["users", "orders"]:
            assert schema["args"][table] == schema["fields"][table]

    def test_metadata_preserved(self) -> None:
        schema = spider_schema_to_text2ql(SPIDER_TABLES[0])
        assert schema["metadata"]["source"] == "spider"
        assert schema["metadata"]["db_id"] == "test_db"


# ---------------------------------------------------------------------------
# BIRD adapter tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBirdAdapter:
    def test_load_bird_basic(self, tmp_path: Path) -> None:
        root = _create_bird_fixture(tmp_path)
        examples = load_bird(root, split="dev")
        assert len(examples) == 2
        assert all(ex.target == "sql" for ex in examples)

    def test_load_bird_with_limit(self, tmp_path: Path) -> None:
        root = _create_bird_fixture(tmp_path)
        examples = load_bird(root, split="dev", limit=1)
        assert len(examples) == 1

    def test_load_bird_metadata(self, tmp_path: Path) -> None:
        root = _create_bird_fixture(tmp_path)
        examples = load_bird(root, split="dev")
        meta = examples[0].metadata
        assert meta["benchmark"] == "bird"
        assert meta["db_id"] == "test_db"

    def test_load_bird_evidence_in_context(self, tmp_path: Path) -> None:
        root = _create_bird_fixture(tmp_path)
        examples = load_bird(root, split="dev")
        # Second example has evidence
        assert examples[1].context.get("evidence") == "Count means total number"

    def test_load_bird_schema_introspection(self, tmp_path: Path) -> None:
        root = _create_bird_fixture(tmp_path)
        examples = load_bird(root, split="dev")
        schema = examples[0].schema
        assert schema is not None
        entity_names = [e["name"] if isinstance(e, dict) else e for e in schema["entities"]]
        assert "users" in entity_names
        assert "orders" in entity_names


@pytest.mark.unit
class TestBirdSchemaConversion:
    def test_basic_conversion(self) -> None:
        raw_schema = {
            "tables": ["products", "reviews"],
            "columns_by_table": {
                "products": [{"name": "id", "type": "INTEGER"}, {"name": "name", "type": "TEXT"}],
                "reviews": [{"name": "id", "type": "INTEGER"}, {"name": "product_id", "type": "INTEGER"}],
            },
            "foreign_keys": [
                {"from_table": "reviews", "to_table": "products", "from_column": "product_id", "to_column": "id"}
            ],
        }
        schema = bird_schema_to_text2ql(raw_schema)
        assert "products" in schema["fields"]
        assert "reviews" in schema["fields"]
        assert "reviews" in schema["relations"]


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkRunner:
    def test_run_benchmark_structural_mode(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev", limit=2)
        config = BenchmarkConfig(mode="structural")
        report = run_benchmark(examples, config=config)
        assert isinstance(report, BenchmarkReport)
        assert report.total == 2
        assert report.benchmark == "spider"
        assert report.split == "dev"
        assert 0 <= report.exact_match_accuracy <= 1
        assert 0 <= report.structural_accuracy <= 1
        assert report.execution_accuracy is None  # structural mode doesn't run SQL

    def test_run_benchmark_execution_mode(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev", limit=1)
        config = BenchmarkConfig(mode="execution")
        report = run_benchmark(examples, config=config)
        assert report.total == 1
        # execution_accuracy should be set when db_path exists
        assert report.execution_accuracy is not None or report.errors > 0

    def test_report_has_difficulty_breakdown(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev")
        config = BenchmarkConfig(mode="structural")
        report = run_benchmark(examples, config=config)
        assert "easy" in report.accuracy_by_difficulty
        assert "medium" in report.accuracy_by_difficulty

    def test_report_has_db_breakdown(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev")
        config = BenchmarkConfig(mode="structural")
        report = run_benchmark(examples, config=config)
        assert "test_db" in report.accuracy_by_db

    def test_empty_examples(self) -> None:
        report = run_benchmark([], config=BenchmarkConfig(mode="structural"))
        assert report.total == 0

    def test_format_report_produces_string(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev", limit=2)
        config = BenchmarkConfig(mode="structural")
        report = run_benchmark(examples, config=config)
        output = format_report(report)
        assert "text2ql Benchmark Report" in output
        assert "SPIDER" in output
        assert "Exact Match" in output
        assert "Structural Match" in output

    def test_format_report_verbose(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev", limit=2)
        config = BenchmarkConfig(mode="execution")
        report = run_benchmark(examples, config=config)
        output = format_report(report, verbose=True)
        assert isinstance(output, str)

    def test_custom_service(self, tmp_path: Path) -> None:
        root = _create_spider_fixture(tmp_path)
        examples = load_spider(root, split="dev", limit=1)
        service = Text2QL()  # explicit default
        config = BenchmarkConfig(mode="structural", service=service)
        report = run_benchmark(examples, config=config)
        assert report.total == 1


@pytest.mark.unit
class TestBenchmarkRow:
    def test_row_fields(self) -> None:
        row = BenchmarkRow(
            question="test?",
            db_id="db",
            gold_sql="SELECT 1",
            predicted_sql="SELECT 1",
            exact_match=True,
            structural_match=True,
            execution_match=True,
            difficulty="easy",
        )
        assert row.exact_match is True
        assert row.error is None
        assert row.latency_ms == 0.0


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkCLI:
    def test_benchmark_flag_in_parser(self) -> None:
        from text2ql.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--benchmark", "spider", "--benchmark-path", "/tmp/spider"])
        assert args.benchmark == "spider"
        assert args.benchmark_path == "/tmp/spider"
        assert args.benchmark_split == "dev"
        assert args.benchmark_limit == 0
        assert args.benchmark_mode == "execution"

    def test_benchmark_bird_flag(self) -> None:
        from text2ql.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--benchmark", "bird",
            "--benchmark-path", "/tmp/bird",
            "--benchmark-split", "train",
            "--benchmark-limit", "50",
            "--benchmark-mode", "structural",
        ])
        assert args.benchmark == "bird"
        assert args.benchmark_split == "train"
        assert args.benchmark_limit == 50
        assert args.benchmark_mode == "structural"
