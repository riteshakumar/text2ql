"""Spider & BIRD benchmark adapters for text2ql.

Usage
-----
.. code-block:: python

    from text2ql.benchmarks import load_spider, load_bird, run_benchmark

    examples = load_spider("/path/to/spider", split="dev")
    report = run_benchmark(examples, mode="execution")
"""

from text2ql.benchmarks.spider import load_spider, spider_schema_to_text2ql
from text2ql.benchmarks.bird import load_bird, bird_schema_to_text2ql
from text2ql.benchmarks.runner import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkRow,
    run_benchmark,
    arun_benchmark,
    format_report,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkRow",
    "arun_benchmark",
    "bird_schema_to_text2ql",
    "format_report",
    "load_bird",
    "load_spider",
    "run_benchmark",
    "spider_schema_to_text2ql",
]
