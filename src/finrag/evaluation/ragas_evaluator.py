"""RAGAS-style evaluation metrics for the FinRAG pipeline.

Implements three core metrics without requiring the ragas library:
1. Faithfulness: is the answer grounded in retrieved context?
2. Answer Relevancy: does the answer address the question?
3. Context Precision: are retrieved chunks relevant to the question?

Plus FinRAG-specific metrics:
4. Citation Coverage: % of answer claims backed by citations
5. Route Accuracy: did the pipeline take the expected route?
6. Decline Precision: were out-of-scope questions correctly declined?

Design decisions:
- Self-contained: no ragas dependency. Metrics computed via
  string overlap, embedding similarity stubs, and heuristics.
  LLM-based scoring deferred to Day 14 (LLM-as-Judge).
- Deterministic first: heuristic metrics run fast and cheaply.
  LLM-based metrics layered on top for precision.
- Per-item + aggregate: each item scored individually, then
  aggregated with mean/median/min across the dataset.
"""

import re
from dataclasses import dataclass, field

import structlog

from finrag.evaluation.golden_dataset import GoldenItem

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Evaluation Result Models
# --------------------------------------------------------------------------- #


@dataclass
class ItemResult:
    """Evaluation result for a single golden dataset item.

    Attributes:
        item_id: Golden item identifier.
        category: Question category.
        faithfulness: Answer grounded in context (0-1).
        answer_relevancy: Answer addresses question (0-1).
        context_precision: Retrieved chunks relevant (0-1).
        citation_coverage: Claims backed by citations (0-1).
        route_correct: Pipeline took expected route.
        answer_generated: The pipeline's actual answer.
        error: Error message if evaluation failed.
    """
    item_id: str = ""
    category: str = ""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    citation_coverage: float = 0.0
    route_correct: bool = False
    answer_generated: str = ""
    error: str | None = None

    def composite_score(self) -> float:
        """Weighted composite of all metrics.

        Returns:
            Weighted average: faithfulness(0.3) + relevancy(0.3) +
            context_precision(0.2) + citation_coverage(0.2).
        """
        return (
            self.faithfulness * 0.3
            + self.answer_relevancy * 0.3
            + self.context_precision * 0.2
            + self.citation_coverage * 0.2
        )


@dataclass
class EvalReport:
    """Aggregate evaluation report across the dataset.

    Attributes:
        total_items: Number of items evaluated.
        results: Per-item results.
        metrics: Aggregate metric summaries.
        category_metrics: Per-category breakdowns.
        pass_threshold: Minimum composite score to pass.
        passed: Whether the evaluation passed the threshold.
    """
    total_items: int = 0
    results: list[ItemResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    category_metrics: dict = field(default_factory=dict)
    pass_threshold: float = 0.7
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "total_items": self.total_items,
            "passed": self.passed,
            "pass_threshold": self.pass_threshold,
            "metrics": self.metrics,
            "category_metrics": self.category_metrics,
        }


# --------------------------------------------------------------------------- #
# Metric Functions
# --------------------------------------------------------------------------- #


def compute_faithfulness(
    answer: str,
    context_chunks: list[str],
) -> float:
    """Measure how well the answer is grounded in context.

    Splits the answer into sentences and checks how many
    have significant word overlap with at least one context chunk.

    Args:
        answer: Generated answer text.
        context_chunks: List of retrieved context texts.

    Returns:
        Faithfulness score (0-1).
    """
    if not answer or not context_chunks:
        return 0.0

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    context_text = " ".join(context_chunks).lower()
    context_words = set(context_text.split())

    grounded = 0
    for sent in sentences:
        sent_words = set(sent.lower().split())
        # Filter out stopwords for meaningful overlap
        meaningful = sent_words - _STOPWORDS
        if not meaningful:
            grounded += 1  # Trivial sentence, count as grounded
            continue
        overlap = meaningful & context_words
        if len(overlap) / max(len(meaningful), 1) >= 0.3:
            grounded += 1

    return grounded / len(sentences)


def compute_answer_relevancy(
    question: str,
    answer: str,
) -> float:
    """Measure how well the answer addresses the question.

    Uses keyword overlap between question and answer as a proxy.
    LLM-based relevancy scoring added in Day 14.

    Args:
        question: The original question.
        answer: Generated answer text.

    Returns:
        Answer relevancy score (0-1).
    """
    if not answer or not question:
        return 0.0

    q_words = set(question.lower().split()) - _STOPWORDS
    a_words = set(answer.lower().split()) - _STOPWORDS

    if not q_words:
        return 0.5

    overlap = q_words & a_words
    # Relevancy = fraction of question keywords appearing in answer
    return min(len(overlap) / max(len(q_words), 1), 1.0)


def compute_context_precision(
    question: str,
    context_chunks: list[str],
) -> float:
    """Measure relevance of retrieved chunks to the question.

    For each chunk, compute keyword overlap with the question.
    Average across all chunks.

    Args:
        question: The original question.
        context_chunks: List of retrieved context texts.

    Returns:
        Context precision score (0-1).
    """
    if not context_chunks or not question:
        return 0.0

    q_words = set(question.lower().split()) - _STOPWORDS
    if not q_words:
        return 0.5

    scores = []
    for chunk in context_chunks:
        chunk_words = set(chunk.lower().split()) - _STOPWORDS
        overlap = q_words & chunk_words
        scores.append(len(overlap) / max(len(q_words), 1))

    return min(sum(scores) / len(scores), 1.0)


def compute_citation_coverage(
    answer: str,
    citations: list[dict],
) -> float:
    """Measure citation coverage of the answer.

    Checks that citations exist and reference valid chunks.
    Returns ratio of sentences with at least one citation nearby.

    Args:
        answer: Generated answer text.
        citations: List of citation dicts with chunk_id.

    Returns:
        Citation coverage score (0-1).
    """
    if not answer:
        return 0.0

    # If no citations but answer exists, coverage is 0
    if not citations:
        return 0.0

    valid_citations = [c for c in citations if c.get("chunk_id")]
    sentences = _split_sentences(answer)
    if not sentences:
        return 1.0 if valid_citations else 0.0

    # Simple heuristic: coverage = min(citations / sentences, 1.0)
    return min(len(valid_citations) / max(len(sentences), 1), 1.0)


def compute_route_accuracy(
    expected_route: str,
    actual_route: str,
) -> bool:
    """Check if the pipeline took the expected route.

    Args:
        expected_route: Expected route from golden item.
        actual_route: Actual route from pipeline result.

    Returns:
        True if routes match.
    """
    return expected_route.lower() == actual_route.lower()


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


class RAGASEvaluator:
    """Evaluates pipeline results against the golden dataset.

    Runs all metrics per item, then aggregates into a report.

    Attributes:
        pass_threshold: Minimum composite score to pass.
    """

    def __init__(self, pass_threshold: float = 0.7) -> None:
        self.pass_threshold = pass_threshold

    def evaluate_item(
        self,
        item: GoldenItem,
        pipeline_result: dict,
    ) -> ItemResult:
        """Evaluate a single pipeline result against a golden item.

        Args:
            item: The golden dataset item.
            pipeline_result: Dict with answer, citations, route,
                retrieved_chunks, reranked_chunks.

        Returns:
            ItemResult with all metric scores.
        """
        answer = pipeline_result.get("answer", "")
        citations = pipeline_result.get("citations", [])
        route = pipeline_result.get("route", "unknown")
        chunks = pipeline_result.get("retrieved_chunks", [])

        # Extract text from chunks
        chunk_texts = []
        for c in chunks:
            if isinstance(c, dict):
                chunk_texts.append(c.get("text", c.get("content", "")))
            elif isinstance(c, str):
                chunk_texts.append(c)

        # For decline items, check route correctness
        if item.expected_route == "decline":
            return ItemResult(
                item_id=item.id,
                category=item.category.value,
                faithfulness=1.0 if route == "decline" else 0.0,
                answer_relevancy=1.0 if route == "decline" else 0.0,
                context_precision=1.0,  # N/A for declines
                citation_coverage=1.0,  # N/A for declines
                route_correct=route == "decline",
                answer_generated=answer[:200],
            )

        faithfulness = compute_faithfulness(answer, chunk_texts)
        relevancy = compute_answer_relevancy(item.question, answer)
        precision = compute_context_precision(item.question, chunk_texts)
        coverage = compute_citation_coverage(answer, citations)
        route_ok = compute_route_accuracy(item.expected_route, route)

        return ItemResult(
            item_id=item.id,
            category=item.category.value,
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(relevancy, 4),
            context_precision=round(precision, 4),
            citation_coverage=round(coverage, 4),
            route_correct=route_ok,
            answer_generated=answer[:200],
        )

    def evaluate_dataset(
        self,
        items: list[GoldenItem],
        results: list[dict],
    ) -> EvalReport:
        """Evaluate all items and produce an aggregate report.

        Args:
            items: Golden dataset items.
            results: Pipeline results in same order as items.

        Returns:
            EvalReport with per-item and aggregate metrics.
        """
        if len(items) != len(results):
            raise ValueError(
                f"Mismatch: {len(items)} items vs {len(results)} results"
            )

        item_results = []
        for item, result in zip(items, results):
            try:
                ir = self.evaluate_item(item, result)
            except Exception as e:
                ir = ItemResult(
                    item_id=item.id,
                    category=item.category.value,
                    error=str(e),
                )
            item_results.append(ir)

        # Aggregate
        valid = [r for r in item_results if r.error is None]
        if not valid:
            return EvalReport(
                total_items=len(items),
                results=item_results,
                passed=False,
                pass_threshold=self.pass_threshold,
            )

        metrics = {
            "faithfulness": _agg([r.faithfulness for r in valid]),
            "answer_relevancy": _agg([r.answer_relevancy for r in valid]),
            "context_precision": _agg([r.context_precision for r in valid]),
            "citation_coverage": _agg([r.citation_coverage for r in valid]),
            "route_accuracy": sum(r.route_correct for r in valid) / len(valid),
            "composite": _agg([r.composite_score() for r in valid]),
            "error_count": len(item_results) - len(valid),
        }

        # Per-category
        cat_metrics = {}
        for cat in set(r.category for r in valid):
            cat_items = [r for r in valid if r.category == cat]
            cat_metrics[cat] = {
                "count": len(cat_items),
                "faithfulness": _mean([r.faithfulness for r in cat_items]),
                "answer_relevancy": _mean([r.answer_relevancy for r in cat_items]),
                "context_precision": _mean([r.context_precision for r in cat_items]),
                "citation_coverage": _mean([r.citation_coverage for r in cat_items]),
                "composite": _mean([r.composite_score() for r in cat_items]),
                "route_accuracy": sum(r.route_correct for r in cat_items) / len(cat_items),
            }

        composite_mean = metrics["composite"]["mean"]
        passed = composite_mean >= self.pass_threshold

        report = EvalReport(
            total_items=len(items),
            results=item_results,
            metrics=metrics,
            category_metrics=cat_metrics,
            pass_threshold=self.pass_threshold,
            passed=passed,
        )

        logger.info(
            "eval_complete",
            total=len(items),
            composite=round(composite_mean, 4),
            passed=passed,
        )

        return report


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just",
    "about", "also", "how", "what", "which", "who", "whom", "this",
    "that", "these", "those", "it", "its", "i", "we", "you", "he",
    "she", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their",
}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Args:
        text: Input text.

    Returns:
        List of sentence strings.
    """
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _agg(values: list[float]) -> dict:
    """Aggregate a list of values into summary stats."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "mean": round(sum(sorted_v) / n, 4),
        "min": round(sorted_v[0], 4),
        "max": round(sorted_v[-1], 4),
        "median": round(sorted_v[n // 2], 4),
    }
