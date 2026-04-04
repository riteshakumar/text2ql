from text2ql import Text2QL


def test_generates_graphql_query_with_defaults() -> None:
    service = Text2QL()
    result = service.generate("list users")

    assert result.target == "graphql"
    assert "query GeneratedQuery" in result.query
    assert "user" in result.query


def test_generates_graphql_query_with_schema_and_limit() -> None:
    service = Text2QL()
    schema = {"entities": ["customers"], "fields": ["id", "email", "status"]}

    result = service.generate("show top 5 customers with email status active", schema=schema)

    assert "customers(limit: 5, status: \"active\")" in result.query
    assert "email" in result.query
