import pytest

from text2ql import Text2QL

pytestmark = pytest.mark.e2e


def test_text2ql_supports_multiple_targets_e2e() -> None:
    service = Text2QL()

    graphql_result = service.generate(
        text="list users",
        target="graphql",
        schema={"entities": ["users"], "fields": {"users": ["id", "name"]}},
    )
    sql_result = service.generate(
        text="list users",
        target="sql",
        schema={"entities": ["users"], "fields": {"users": ["id", "name"]}},
    )

    assert graphql_result.target == "graphql"
    assert graphql_result.query.strip().startswith("{")
    assert sql_result.target == "sql"
    assert sql_result.query.startswith("SELECT")


def test_text2ql_generate_unsupported_target_raises() -> None:
    service = Text2QL()

    with pytest.raises(ValueError, match="Unsupported target"):
        service.generate("show customers", target="sparql")
