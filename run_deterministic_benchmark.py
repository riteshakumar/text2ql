"""
Synthetic Spider + BIRD benchmark — deterministic mode.

Creates 50-example synthetic datasets in Spider/BIRD file format, runs
them through text2ql's native benchmark runner, and prints a full report.

Usage:
    python run_deterministic_benchmark.py
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from typing import Any

# Reuse the same synthetic dataset builders as the LLM benchmark script.
from run_llm_benchmark import build_bird_dataset, build_spider_dataset


def main() -> None:
    repo_src = str(Path(__file__).parent / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)

    benchmarks_mod = importlib.import_module("text2ql.benchmarks")
    core_mod = importlib.import_module("text2ql.core")
    dataset_mod = importlib.import_module("text2ql.dataset")

    load_spider = benchmarks_mod.load_spider
    load_bird = benchmarks_mod.load_bird
    run_benchmark = benchmarks_mod.run_benchmark
    format_report = benchmarks_mod.format_report
    benchmark_config_cls = benchmarks_mod.BenchmarkConfig
    text2ql_cls = core_mod.Text2QL
    dataset_example_cls = dataset_mod.DatasetExample

    service = text2ql_cls()

    with tempfile.TemporaryDirectory() as tmpdir:
        spider_root = Path(tmpdir) / "spider"
        bird_root = Path(tmpdir) / "bird"

        print("Building synthetic Spider dataset (50 examples)...")
        build_spider_dataset(spider_root)
        print("Building synthetic BIRD dataset (50 examples)...")
        build_bird_dataset(bird_root)

        spider_examples = load_spider(spider_root, split="dev", limit=50)
        bird_examples = load_bird(bird_root, split="dev", limit=50)

        def with_deterministic_mode(examples: list[Any]) -> list[Any]:
            patched = []
            for ex in examples:
                ctx = dict(ex.context)
                ctx["mode"] = "deterministic"
                patched.append(
                    dataset_example_cls(
                        text=ex.text,
                        target=ex.target,
                        expected_query=ex.expected_query,
                        schema=ex.schema,
                        mapping=ex.mapping,
                        context=ctx,
                        metadata=ex.metadata,
                    )
                )
            return patched

        spider_examples = with_deterministic_mode(spider_examples)
        bird_examples = with_deterministic_mode(bird_examples)

        cfg = benchmark_config_cls(mode="execution", service=service, concurrency=1)

        print(f"\nRunning Spider benchmark ({len(spider_examples)} examples, mode=deterministic)...")
        spider_report = run_benchmark(spider_examples, config=cfg)
        print(format_report(spider_report, verbose=True))

        print(f"\nRunning BIRD benchmark ({len(bird_examples)} examples, mode=deterministic)...")
        bird_report = run_benchmark(bird_examples, config=cfg)
        print(format_report(bird_report, verbose=True))

        print("\n" + "=" * 68)
        print("  SUMMARY")
        print("=" * 68)
        for label, report in [("Spider", spider_report), ("BIRD", bird_report)]:
            print(
                f"  {label:<8}  Exact={report.exact_match_accuracy:.1%}  "
                f"Structural={report.structural_accuracy:.1%}  "
                f"Execution={report.execution_accuracy:.1%}  "
                f"Errors={report.errors}/{report.total}"
            )
        print("=" * 68)


if __name__ == "__main__":
    main()
