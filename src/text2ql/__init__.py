"""text2ql package."""

from .core import Text2QL
from .dataset import DatasetExample, generate_synthetic_examples, ingest_dataset
from .evaluate import EvaluationReport, evaluate_examples
from .json_execution import execute_query_result_on_json
from .mapping import generate_hybrid_mapping
from .rewrite import rewrite_user_utterance
from .schema_config import infer_schema_from_json_payload
from .types import QueryRequest, QueryResult

__all__ = [
    "DatasetExample",
    "EvaluationReport",
    "QueryRequest",
    "QueryResult",
    "Text2QL",
    "execute_query_result_on_json",
    "evaluate_examples",
    "generate_hybrid_mapping",
    "generate_synthetic_examples",
    "infer_schema_from_json_payload",
    "ingest_dataset",
    "rewrite_user_utterance",
]
