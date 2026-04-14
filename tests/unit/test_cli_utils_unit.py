import pytest

from text2ql._cli_utils import execute_sql_on_json

pytestmark = pytest.mark.unit


def test_execute_sql_on_json_flattens_nested_dict_leaves_into_columns() -> None:
    payload = {
        "portfolio_data": {
            "transactions": [
                {
                    "quantity": 10,
                    "dateDetail": {"postedDate": 1700000000},
                    "securityDetail": {"symbol": "QQQ", "securityDescription": "INVESCO QQQ"},
                }
            ]
        }
    }
    query = (
        'SELECT "transactions"."quantity", "transactions"."symbol" '
        'FROM "transactions" ORDER BY "transactions"."postedDate" DESC LIMIT 1;'
    )

    rows, note = execute_sql_on_json(query, payload, root_key="portfolio_data")

    assert note is None
    assert rows == [{"quantity": "10", "symbol": "QQQ"}]


def test_execute_sql_on_json_preserves_top_level_field_on_name_collision() -> None:
    payload = {
        "portfolio_data": {
            "transactions": [
                {
                    "symbol": "TOP",
                    "securityDetail": {"symbol": "NESTED"},
                }
            ]
        }
    }
    query = 'SELECT "transactions"."symbol" FROM "transactions";'

    rows, note = execute_sql_on_json(query, payload, root_key="portfolio_data")

    assert note is None
    assert rows == [{"symbol": "TOP"}]
