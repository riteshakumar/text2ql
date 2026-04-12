"""text2ql package."""

from .core import Text2QL
from .dataset import DatasetExample, generate_synthetic_examples, ingest_dataset
from .evaluate import EvaluationReport, aevaluate_examples, evaluate_examples
from .ir import IRAggregation, IRFilter, IRJoin, IRNested, IRRenderer, QueryIR
from .json_execution import execute_query_result_on_json
from .mapping import generate_hybrid_mapping
from .renderers import GraphQLIRRenderer, SQLIRRenderer
from .rewrite import arewrite_user_utterance, rewrite_user_utterance
from .schema_config import infer_schema_from_json_payload
from .sql_executor import SQLAlchemyExecutor, create_sqlite_executor
from .types import QueryRequest, QueryResult, ValidationError

# Lazy-loaded benchmark subpackage — import from text2ql.benchmarks directly
# to avoid pulling in sqlite3/json at import time for non-benchmark users.

__all__ = [
    "DatasetExample",
    "EvaluationReport",
    "GraphQLIRRenderer",
    "IRAggregation",
    "IRFilter",
    "IRJoin",
    "IRNested",
    "IRRenderer",
    "QueryIR",
    "QueryRequest",
    "QueryResult",
    "SQLAlchemyExecutor",
    "SQLIRRenderer",
    "Text2QL",
    "ValidationError",
    "aevaluate_examples",
    "arewrite_user_utterance",
    "create_sqlite_executor",
    "execute_query_result_on_json",
    "evaluate_examples",
    "generate_hybrid_mapping",
    "generate_synthetic_examples",
    "infer_schema_from_json_payload",
    "ingest_dataset",
    "rewrite_user_utterance",
]
