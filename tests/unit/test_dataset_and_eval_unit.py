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
                        "expected_query": "{ user { id name } }",
                    }
                ),
                json.dumps(
                    {
                        "text": "list customers",
                        "expected_query": "{ customer { id name } }",
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
            expected_query="{ customers(limit: 3) { id } }",
        )
    ]

    synthetic = generate_synthetic_examples(seeds, variants_per_example=2)

    assert len(synthetic) == 2
    assert all(example.context.get("synthetic") for example in synthetic)
    assert synthetic[0].metadata["synthetic_variant"] == 1


def test_generate_synthetic_examples_with_portfolio_plugin() -> None:
    seeds = [
        DatasetExample(
            text="how many qqq do i own",
            target="graphql",
            expected_query='{ positions(symbol: "QQQ") { quantity } }',
        )
    ]

    synthetic = generate_synthetic_examples(
        seeds,
        variants_per_example=3,
        rewrite_plugins=["generic", "portfolio"],
    )

    assert len(synthetic) == 3
    texts = [example.text.lower() for example in synthetic]
    assert any("shares of qqq" in text for text in texts)


def test_generate_synthetic_examples_auto_domain_plugin() -> None:
    seeds = [
        DatasetExample(
            text="how many bac do i own",
            target="graphql",
            expected_query='{ positions(symbol: "BAC") { quantity } }',
            schema={"entities": ["positions"], "fields": {"positions": ["symbol", "quantity"]}},
        )
    ]

    synthetic = generate_synthetic_examples(seeds, variants_per_example=2, domain="portfolio")

    assert len(synthetic) == 2
    assert all(example.metadata.get("synthetic_domain") == "portfolio" for example in synthetic)


def test_generate_synthetic_examples_with_banking_plugin() -> None:
    seeds = [
        DatasetExample(
            text="show account balance",
            target="graphql",
            expected_query="{ accounts { balance } }",
        )
    ]

    synthetic = generate_synthetic_examples(
        seeds,
        variants_per_example=3,
        rewrite_plugins=["generic", "banking"],
    )

    assert len(synthetic) == 3
    texts = [example.text.lower() for example in synthetic]
    assert any("available balance" in text for text in texts)


def test_generate_synthetic_examples_with_ecommerce_plugin() -> None:
    seeds = [
        DatasetExample(
            text="show recent orders",
            target="graphql",
            expected_query="{ orders { id } }",
        )
    ]

    synthetic = generate_synthetic_examples(
        seeds,
        variants_per_example=3,
        rewrite_plugins=["generic", "ecommerce"],
    )

    assert len(synthetic) == 3
    texts = [example.text.lower() for example in synthetic]
    assert any("customer orders" in text for text in texts)


def test_generate_synthetic_examples_with_crm_plugin() -> None:
    seeds = [
        DatasetExample(
            text="show opportunity pipeline",
            target="graphql",
            expected_query="{ opportunities { id } }",
        )
    ]

    synthetic = generate_synthetic_examples(
        seeds,
        variants_per_example=3,
        rewrite_plugins=["generic", "crm"],
    )

    assert len(synthetic) == 3
    texts = [example.text.lower() for example in synthetic]
    assert any("sales opportunities in pipeline" in text for text in texts)


def test_generate_synthetic_examples_with_healthcare_plugin() -> None:
    seeds = [
        DatasetExample(
            text="show patient medications",
            target="graphql",
            expected_query="{ medications { id } }",
        )
    ]

    synthetic = generate_synthetic_examples(
        seeds,
        variants_per_example=3,
        rewrite_plugins=["generic", "healthcare"],
    )

    assert len(synthetic) == 3
    texts = [example.text.lower() for example in synthetic]
    assert any("patient records" in text for text in texts)


def test_generate_synthetic_examples_auto_domain_plugin_banking() -> None:
    seeds = [
        DatasetExample(
            text="show transfer transactions",
            target="graphql",
            expected_query="{ transactions { id } }",
            schema={"entities": ["accounts"], "fields": {"accounts": ["balance", "accountNumber"]}},
        )
    ]

    synthetic = generate_synthetic_examples(seeds, variants_per_example=2, domain="banking")

    assert len(synthetic) == 2
    assert all(example.metadata.get("synthetic_domain") == "banking" for example in synthetic)


def test_generate_synthetic_examples_auto_domain_plugin_ecommerce() -> None:
    seeds = [
        DatasetExample(
            text="show inventory records",
            target="graphql",
            expected_query="{ inventory { sku } }",
            schema={"entities": ["products"], "fields": {"products": ["sku", "inventory"]}},
        )
    ]

    synthetic = generate_synthetic_examples(seeds, variants_per_example=2, domain="ecommerce")

    assert len(synthetic) == 2
    assert all(example.metadata.get("synthetic_domain") == "ecommerce" for example in synthetic)


def test_generate_synthetic_examples_auto_domain_plugin_crm() -> None:
    seeds = [
        DatasetExample(
            text="show lead pipeline",
            target="graphql",
            expected_query="{ leads { id } }",
            schema={"entities": ["opportunities"], "fields": {"opportunities": ["stage", "amount"]}},
        )
    ]

    synthetic = generate_synthetic_examples(seeds, variants_per_example=2, domain="crm")

    assert len(synthetic) == 2
    assert all(example.metadata.get("synthetic_domain") == "crm" for example in synthetic)


def test_generate_synthetic_examples_auto_domain_plugin_healthcare() -> None:
    seeds = [
        DatasetExample(
            text="show patient encounters",
            target="graphql",
            expected_query="{ encounters { id } }",
            schema={"entities": ["patients"], "fields": {"patients": ["patientId", "name"]}},
        )
    ]

    synthetic = generate_synthetic_examples(seeds, variants_per_example=2, domain="healthcare")

    assert len(synthetic) == 2
    assert all(example.metadata.get("synthetic_domain") == "healthcare" for example in synthetic)


def test_generate_synthetic_examples_rejects_unknown_plugin() -> None:
    seeds = [
        DatasetExample(
            text="list users",
            target="graphql",
            expected_query="{ users { id } }",
        )
    ]
    with pytest.raises(ValueError, match="Unknown rewrite plugin"):
        generate_synthetic_examples(seeds, variants_per_example=1, rewrite_plugins=["unknown-domain"])


def test_generate_synthetic_examples_domain_templates_use_schema_slots() -> None:
    seeds = [
        DatasetExample(
            text="show sales pipeline",
            target="graphql",
            expected_query="{ opportunities { amount } }",
            schema={
                "entities": ["opportunities"],
                "fields": {"opportunities": ["amount", "createdAt", "stage"]},
                "args": {"opportunities": ["stage"]},
            },
            mapping={"filter_values": {"stage": {"open": "Open"}}},
        )
    ]

    synthetic = generate_synthetic_examples(
        seeds,
        variants_per_example=4,
        rewrite_plugins=["generic", "crm"],
        domain="crm",
    )

    texts = [example.text for example in synthetic]
    template_rows = [
        example
        for example in synthetic
        if str(example.metadata.get("synthetic_rewrite_source", "")).endswith("-template")
    ]
    assert template_rows
    assert any("opportunit" in text.lower() for text in texts)
    assert any(("amount" in text.lower()) or ("stage" in text.lower()) for text in texts)
    assert any("open" in text.lower() for text in texts)


def test_generate_synthetic_examples_schema_aware_lexicalization_filters_unknown_terms() -> None:
    def _noisy_plugin(_: DatasetExample) -> list[str]:
        return ["show foobar bazqux now"]

    seeds = [
        DatasetExample(
            text="show opportunities",
            target="graphql",
            expected_query="{ opportunities { id } }",
            schema={"entities": ["opportunities"], "fields": {"opportunities": ["id", "stage"]}},
        )
    ]

    synthetic = generate_synthetic_examples(
        seeds,
        variants_per_example=1,
        rewrite_plugins=[_noisy_plugin],
    )

    assert synthetic[0].text == "show opportunities"


def test_generate_synthetic_examples_adds_scoring_metadata() -> None:
    seeds = [
        DatasetExample(
            text="show recent orders",
            target="graphql",
            expected_query="{ orders { id } }",
            schema={"entities": ["orders"], "fields": {"orders": ["id", "createdAt", "status"]}},
            mapping={"filter_values": {"status": {"new": "new"}}},
        )
    ]

    synthetic = generate_synthetic_examples(
        seeds,
        variants_per_example=3,
        rewrite_plugins=["generic", "ecommerce"],
        domain="ecommerce",
    )

    for item in synthetic:
        assert "synthetic_rewrite_source" in item.metadata
        assert "synthetic_rewrite_confidence" in item.metadata
        assert "synthetic_rewrite_novelty" in item.metadata
        assert "synthetic_rewrite_score" in item.metadata
    assert synthetic[0].metadata["synthetic_rewrite_score"] >= synthetic[-1].metadata["synthetic_rewrite_score"]


def test_generate_synthetic_examples_templates_scope_filters_to_entity_fields() -> None:
    seeds = [
        DatasetExample(
            text="show account summary",
            target="graphql",
            expected_query="{ accountSummary { isPartialBalance } }",
            schema={
                "entities": ["positions", "accountSummary"],
                "fields": {
                    "positions": ["symbol", "status", "quantity"],
                    "accountSummary": ["isPartialBalance", "acctNum"],
                },
                "args": {
                    "positions": ["status", "symbol"],
                    "accountSummary": ["acctNum"],
                },
            },
            mapping={
                "filter_values": {
                    "status": {"open": "open"},
                    "acctNum": {"x21985452": "X21985452"},
                }
            },
        )
    ]

    synthetic = generate_synthetic_examples(
        seeds,
        variants_per_example=6,
        rewrite_plugins=["generic", "portfolio"],
        domain="portfolio",
    )

    texts = [example.text.lower() for example in synthetic]
    assert any("accountsummary where acctnum is x21985452" in text for text in texts)
    assert not any("accountsummary where status is" in text for text in texts)


def test_evaluate_examples_reports_exact_and_execution_accuracy() -> None:
    service = Text2QL()
    examples = [
        DatasetExample(
            text="list users",
            target="graphql",
            expected_query=(
                "{\n"
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


def test_evaluate_examples_with_backend_execution() -> None:
    service = Text2QL()
    examples = [
        DatasetExample(
            text="list users",
            target="graphql",
            expected_query="{ user { id name } }",
            metadata={"expected_execution_result": [{"id": "1", "name": "Ada"}]},
        )
    ]

    def _backend_executor(query: str, _: DatasetExample) -> list[dict[str, str]]:
        if "user" in query:
            return [{"id": "1", "name": "Ada"}]
        return []

    report = evaluate_examples(service, examples, execution_backend=_backend_executor)

    assert report.total == 1
    assert report.rows[0].execution_mode == "backend"
    assert report.rows[0].execution_backend_error is None
    assert report.execution_accuracy == pytest.approx(1.0)


def test_evaluate_examples_with_backend_error_marks_failure() -> None:
    service = Text2QL()
    examples = [
        DatasetExample(
            text="list users",
            target="graphql",
            expected_query="{ user { id name } }",
        )
    ]

    def _backend_executor(_: str, __: DatasetExample) -> list[dict[str, str]]:
        raise RuntimeError("backend unavailable")

    report = evaluate_examples(service, examples, execution_backend=_backend_executor)

    assert report.total == 1
    assert report.rows[0].execution_mode == "backend"
    assert report.rows[0].execution_match is False
    assert "RuntimeError" in str(report.rows[0].execution_backend_error)


def test_ingest_dataset_rejects_non_object_example(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(json.dumps(["bad"]), encoding="utf-8")

    with pytest.raises(ValueError, match="must be an object"):
        ingest_dataset(dataset_path)


def test_evaluate_examples_structural_execution_for_sql() -> None:
    service = Text2QL()
    examples = [
        DatasetExample(
            text="show customers highest total first 5",
            target="sql",
            expected_query='SELECT "customers"."total" FROM "customers" ORDER BY "customers"."total" DESC LIMIT 5;',
            schema={"entities": ["customers"], "fields": {"customers": ["id", "total"]}},
        )
    ]

    report = evaluate_examples(service, examples)

    assert report.total == 1
    assert report.execution_accuracy == pytest.approx(1.0)
