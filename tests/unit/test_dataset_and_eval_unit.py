import json
from pathlib import Path

import pytest

from text2ql import Text2QL
from text2ql.dataset import DatasetExample, generate_synthetic_examples, ingest_dataset
from text2ql.evaluate import evaluate_examples

pytestmark = pytest.mark.unit


def test_ingest_dataset_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "text": "list users",
                        "target": "graphql",
                        "expected_query": "query GeneratedQuery { user { id name } }",
                    }
                ),
                json.dumps(
                    {
                        "text": "list customers",
                        "expected_query": "query GeneratedQuery { customer { id name } }",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    examples = ingest_dataset(dataset_path)

    assert len(examples) == 2
    assert examples[0].target == "graphql"


def test_generate_synthetic_examples_marks_variants() -> None:
    seeds = [
        DatasetExample(
            text="show top 3 customers",
            target="graphql",
            expected_query="query GeneratedQuery { customers(limit: 3) { id } }",
        )
    ]

    synthetic = generate_synthetic_examples(seeds, variants_per_example=2)

    assert len(synthetic) == 2
    assert all(example.context.get("synthetic") for example in synthetic)
    assert synthetic[0].metadata["synthetic_variant"] == 1


def test_evaluate_examples_reports_exact_and_execution_accuracy() -> None:
    service = Text2QL()
    examples = [
        DatasetExample(
            text="list users",
            target="graphql",
            expected_query=(
                "query GeneratedQuery {\n"
                "  user {\n"
                "    id\n"
                "    name\n"
                "  }\n"
                "}"
            ),
        )
    ]

    report = evaluate_examples(service, examples)

    assert report.total == 1
    assert report.exact_match_accuracy == pytest.approx(1.0)
    assert report.execution_accuracy == pytest.approx(1.0)


def test_ingest_dataset_rejects_non_object_example(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(json.dumps(["bad"]), encoding="utf-8")

    with pytest.raises(ValueError, match="must be an object"):
        ingest_dataset(dataset_path)
