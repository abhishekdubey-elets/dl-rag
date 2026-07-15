"""Evaluation CLI: ``python -m dl_rag.evaluation.run --dataset eval/questions.json``.

Runs each case through the live retrieval (+ optionally generation + LLM judge)
pipeline and prints a metric summary. Requires the datastores to be up and an
ingested index; the LLM key is only needed for generation/judge tiers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dl_rag.api.deps import build_container
from dl_rag.config import get_settings
from dl_rag.evaluation.evaluator import EvalCase, Evaluator
from dl_rag.logging_config import configure_logging


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    p.add_argument("--dataset", type=Path, default=Path("eval/questions.json"),
                   help="JSON file: [{query, expected_query_type?, expected_urls?, expected_keywords?}]")
    p.add_argument("--retrieval-only", action="store_true",
                   help="Skip generation + judging (no LLM calls).")
    p.add_argument("--no-judge", action="store_true",
                   help="Generate answers but skip LLM-as-judge scoring.")
    p.add_argument("--out", type=Path, default=None, help="Write the full JSON report here.")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    container = build_container(settings)

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = [EvalCase.model_validate(item) for item in raw]
    print(f"Loaded {len(cases)} eval cases from {args.dataset}")

    evaluator = Evaluator(
        analyzer=container.analyzer,
        retriever=container.retriever,
        generator=None if args.retrieval_only else container.generator,
        judge=None if (args.retrieval_only or args.no_judge) else container.llm,
    )

    try:
        report = await evaluator.evaluate(cases)
    finally:
        await container.db.dispose()
        await container.cache.close()

    print("\n=== Per-case ===")
    for c in report.cases:
        print(f"  [{c.query_type:12s}] retrieved={c.retrieved:2d} "
              f"recall={_fmt(c.top_k_recall)} cite_prec={_fmt(c.citation_precision)} "
              f"latency={c.latency_ms}ms :: {c.query[:60]}")

    print("\n=== Summary ===")
    for key, value in report.summary.items():
        print(f"  {key:26s}: {_fmt(value)}")

    if args.out:
        args.out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.out}")


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
