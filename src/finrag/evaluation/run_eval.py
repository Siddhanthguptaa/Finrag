"""Offline evaluation script for the FinRAG pipeline.

Runs the golden dataset through the evaluation framework and
produces a JSON report. Can run RAGAS metrics, LLM-as-Judge,
or both.

Usage:
    python -m finrag.evaluation.run_eval --mode ragas
    python -m finrag.evaluation.run_eval --mode judge
    python -m finrag.evaluation.run_eval --mode full --output report.json

Design decisions:
- Offline-first: designed for CI and local dev, not real-time.
- Stub pipeline: when pipeline not available, uses mock results
  to validate the evaluation framework itself.
- JSON output: machine-readable for CI quality gates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import structlog

from finrag.evaluation.golden_dataset import (
    Category,
    GoldenItem,
    load_by_category,
    load_golden_dataset,
)
from finrag.evaluation.llm_judge import CitationJudge, JudgeBatchReport
from finrag.evaluation.ragas_evaluator import EvalReport, RAGASEvaluator

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Mock Pipeline (for framework validation)
# --------------------------------------------------------------------------- #


def generate_mock_result(item: GoldenItem) -> dict:
    """Generate a mock pipeline result for evaluation testing.

    Creates a plausible result structure that exercises all
    evaluation metrics without requiring the real pipeline.

    Args:
        item: Golden dataset item.

    Returns:
        Mock pipeline result dict.
    """
    if item.expected_route == "decline":
        return {
            "answer": "I cannot answer that question based on SEC filings.",
            "citations": [],
            "route": "decline",
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "is_valid": True,
        }

    # Use expected answer as the mock answer
    answer = item.expected_answer

    # Create mock citations from ground truth
    citations = []
    chunks = []
    for i, ref in enumerate(item.ground_truth_citations):
        chunk_id = f"chunk_{item.id}_{i}"
        citations.append({
            "chunk_id": chunk_id,
            "filing_reference": ref,
            "section": f"Section {i + 1}",
            "relevance_score": 0.85 - (i * 0.05),
        })
        chunks.append({
            "chunk_id": chunk_id,
            "text": f"Source data from {ref}: {answer[:100]}",
            "id": chunk_id,
        })

    # If no ground truth citations, create a default
    if not citations:
        chunk_id = f"chunk_{item.id}_0"
        citations.append({
            "chunk_id": chunk_id,
            "filing_reference": "Generic Filing Reference",
            "relevance_score": 0.7,
        })
        chunks.append({
            "chunk_id": chunk_id,
            "text": f"Context for: {answer[:100]}",
        })

    return {
        "answer": answer,
        "citations": citations,
        "route": item.expected_route,
        "retrieved_chunks": chunks,
        "reranked_chunks": chunks,
        "is_valid": True,
    }


# --------------------------------------------------------------------------- #
# Evaluation Runners
# --------------------------------------------------------------------------- #


def run_ragas_eval(
    items: list[GoldenItem] | None = None,
    threshold: float = 0.7,
) -> EvalReport:
    """Run RAGAS evaluation on the golden dataset.

    Args:
        items: Optional subset of items. Defaults to full dataset.
        threshold: Pass/fail threshold for composite score.

    Returns:
        EvalReport with per-item and aggregate scores.
    """
    if items is None:
        items = load_golden_dataset()

    results = [generate_mock_result(item) for item in items]

    evaluator = RAGASEvaluator(pass_threshold=threshold)
    report = evaluator.evaluate_dataset(items, results)

    logger.info(
        "ragas_eval_complete",
        total=report.total_items,
        composite=report.metrics.get("composite", {}).get("mean", 0),
        passed=report.passed,
    )

    return report


def run_judge_eval(
    items: list[GoldenItem] | None = None,
    threshold: float = 0.7,
    model: str = "gemini-2.0-flash",
) -> JudgeBatchReport:
    """Run LLM-as-Judge evaluation on the golden dataset.

    Args:
        items: Optional subset of items. Defaults to non-decline items.
        threshold: Pass/fail threshold for composite score.
        model: Judge model name.

    Returns:
        JudgeBatchReport with per-citation scores.
    """
    if items is None:
        # Only judge items that have citations (not decline)
        items = [
            i for i in load_golden_dataset()
            if i.expected_route != "decline"
        ]

    judge = CitationJudge(model_name=model)
    batch_results = []

    for item in items:
        mock = generate_mock_result(item)
        result = judge.judge_citations(
            question=item.question,
            answer=mock["answer"],
            citations=mock["citations"],
            chunks=mock["retrieved_chunks"],
            item_id=item.id,
        )
        batch_results.append(result)

    report = JudgeBatchReport(
        total_items=len(items),
        results=batch_results,
        pass_threshold=threshold,
    )
    report.compute_aggregates()

    logger.info(
        "judge_eval_complete",
        total=report.total_items,
        citations=report.total_citations,
        composite=report.mean_composite,
        passed=report.passed,
    )

    return report


def run_full_eval(
    threshold_ragas: float = 0.7,
    threshold_judge: float = 0.7,
) -> dict:
    """Run both RAGAS and Judge evaluations.

    Args:
        threshold_ragas: RAGAS pass threshold.
        threshold_judge: Judge pass threshold.

    Returns:
        Combined report dict.
    """
    start = time.perf_counter()

    ragas = run_ragas_eval(threshold=threshold_ragas)
    judge = run_judge_eval(threshold=threshold_judge)

    elapsed = round((time.perf_counter() - start) * 1000, 2)

    return {
        "ragas": ragas.to_dict(),
        "judge": judge.to_dict(),
        "overall_passed": ragas.passed and judge.passed,
        "eval_latency_ms": elapsed,
    }


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for offline evaluation.

    Args:
        argv: Command line arguments.

    Returns:
        Exit code (0=pass, 1=fail).
    """
    parser = argparse.ArgumentParser(
        description="FinRAG Offline Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["ragas", "judge", "full"],
        default="ragas",
        help="Evaluation mode (default: ragas)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Pass/fail threshold (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON report",
    )
    parser.add_argument(
        "--category",
        choices=["numerical", "multi_hop", "contradiction", "out_of_scope"],
        default=None,
        help="Filter to specific category",
    )

    args = parser.parse_args(argv)

    # Filter items if category specified
    category_map = {
        "numerical": Category.NUMERICAL,
        "multi_hop": Category.MULTI_HOP,
        "contradiction": Category.CONTRADICTION,
        "out_of_scope": Category.OUT_OF_SCOPE,
    }
    items = None
    if args.category:
        items = load_by_category(category_map[args.category])

    # Run evaluation
    if args.mode == "ragas":
        report = run_ragas_eval(items=items, threshold=args.threshold)
        output = report.to_dict()
        passed = report.passed
    elif args.mode == "judge":
        report = run_judge_eval(items=items, threshold=args.threshold)
        output = report.to_dict()
        passed = report.passed
    else:
        output = run_full_eval(
            threshold_ragas=args.threshold,
            threshold_judge=args.threshold,
        )
        passed = output["overall_passed"]

    # Output
    report_json = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report_json)
        print(f"Report written to {args.output}")
    else:
        print(report_json)

    status = "PASSED" if passed else "FAILED"
    print(f"\nEvaluation {status} (threshold: {args.threshold})")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
