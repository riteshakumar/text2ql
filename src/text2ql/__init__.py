"""text2ql package."""

from .core import Text2QL
from .dataset import DatasetExample, generate_synthetic_examples, ingest_dataset
from .evaluate import EvaluationReport, evaluate_examples
from .types import QueryRequest, QueryResult

__all__ = [
    "DatasetExample",
    "EvaluationReport",
    "QueryRequest",
    "QueryResult",
    "Text2QL",
    "evaluate_examples",
    "generate_synthetic_examples",
    "ingest_dataset",
]
