"""Tests for Day 13: Golden dataset + RAGAS evaluation.

Covers:
- Golden dataset loading, filtering, summary
- Individual RAGAS metrics (faithfulness, relevancy, precision, coverage)
- Route accuracy for decline items
- RAGASEvaluator per-item and dataset-level evaluation
- EvalReport composite scoring and pass/fail threshold
"""

import pytest

from finrag.evaluation.golden_dataset import (
    Category,
    Difficulty,
    GoldenItem,
    dataset_summary,
    load_by_category,
    load_by_difficulty,
    load_golden_dataset,
)
from finrag.evaluation.ragas_evaluator import (
    EvalReport,
    ItemResult,
    RAGASEvaluator,
    compute_answer_relevancy,
    compute_citation_coverage,
    compute_context_precision,
    compute_faithfulness,
    compute_route_accuracy,
)


# --------------------------------------------------------------------------- #
# Golden Dataset Tests
# --------------------------------------------------------------------------- #


class TestGoldenDataset:
    """Test golden dataset loading and structure."""

    def test_load_returns_50_items(self):
        dataset = load_golden_dataset()
        assert len(dataset) == 50

    def test_all_items_have_required_fields(self):
        for item in load_golden_dataset():
            assert item.id
            assert item.question
            assert item.expected_answer
            assert item.category in Category
            assert item.difficulty in Difficulty

    def test_unique_ids(self):
        ids = [item.id for item in load_golden_dataset()]
        assert len(ids) == len(set(ids))

    def test_category_counts(self):
        summary = dataset_summary()
        assert summary["by_category"]["numerical_extraction"] == 15
        assert summary["by_category"]["multi_hop_comparison"] == 12
        assert summary["by_category"]["contradiction_detection"] == 11
        assert summary["by_category"]["out_of_scope"] == 12

    def test_filter_by_category(self):
        numerical = load_by_category(Category.NUMERICAL)
        assert len(numerical) == 15
        assert all(i.category == Category.NUMERICAL for i in numerical)

    def test_filter_by_difficulty(self):
        easy = load_by_difficulty(Difficulty.EASY)
        assert len(easy) > 0
        assert all(i.difficulty == Difficulty.EASY for i in easy)

    def test_out_of_scope_expect_decline(self):
        oos = load_by_category(Category.OUT_OF_SCOPE)
        for item in oos:
            assert item.expected_route == "decline"

    def test_numerical_expect_retrieve(self):
        numerical = load_by_category(Category.NUMERICAL)
        for item in numerical:
            assert item.expected_route == "retrieve"

    def test_item_to_dict(self):
        item = load_golden_dataset()[0]
        d = item.to_dict()
        assert "id" in d
        assert "question" in d
        assert isinstance(d["category"], str)
        assert isinstance(d["difficulty"], str)

    def test_dataset_summary(self):
        summary = dataset_summary()
        assert summary["total"] == 50
        assert "by_category" in summary
        assert "by_difficulty" in summary


# --------------------------------------------------------------------------- #
# Faithfulness Tests
# --------------------------------------------------------------------------- #


class TestFaithfulness:
    """Test faithfulness metric."""

    def test_grounded_answer(self):
        answer = "Apple revenue was $391 billion in FY2024."
        chunks = ["Apple reported total net revenue of $391 billion for fiscal year 2024."]
        score = compute_faithfulness(answer, chunks)
        assert score >= 0.5

    def test_ungrounded_answer(self):
        answer = "The quantum flux capacitor generated 1.21 gigawatts."
        chunks = ["Apple reported revenue of $391 billion."]
        score = compute_faithfulness(answer, chunks)
        assert score < 0.5

    def test_empty_answer(self):
        assert compute_faithfulness("", ["some context"]) == 0.0

    def test_empty_context(self):
        assert compute_faithfulness("Some answer.", []) == 0.0

    def test_partial_grounding(self):
        answer = "Revenue was $100B. The moon is made of cheese."
        chunks = ["Total revenue reached $100 billion in the fiscal year."]
        score = compute_faithfulness(answer, chunks)
        assert 0.0 < score < 1.0


# --------------------------------------------------------------------------- #
# Answer Relevancy Tests
# --------------------------------------------------------------------------- #


class TestAnswerRelevancy:
    """Test answer relevancy metric."""

    def test_relevant_answer(self):
        q = "What was Apple's revenue in FY2024?"
        a = "Apple's total revenue in FY2024 was $391 billion."
        score = compute_answer_relevancy(q, a)
        assert score >= 0.4

    def test_irrelevant_answer(self):
        q = "What was Apple's revenue?"
        a = "The weather in Tokyo is sunny today."
        score = compute_answer_relevancy(q, a)
        assert score < 0.3

    def test_empty_inputs(self):
        assert compute_answer_relevancy("", "answer") == 0.0
        assert compute_answer_relevancy("question", "") == 0.0


# --------------------------------------------------------------------------- #
# Context Precision Tests
# --------------------------------------------------------------------------- #


class TestContextPrecision:
    """Test context precision metric."""

    def test_relevant_chunks(self):
        q = "What was Apple total net revenue fiscal year?"
        chunks = ["Apple reported total net revenue of $391B in fiscal year 2024."]
        score = compute_context_precision(q, chunks)
        assert score > 0.0

    def test_irrelevant_chunks(self):
        q = "What was Apple's revenue?"
        chunks = ["The weather forecast shows sunny skies."]
        score = compute_context_precision(q, chunks)
        assert score < 0.3

    def test_empty_chunks(self):
        assert compute_context_precision("question", []) == 0.0


# --------------------------------------------------------------------------- #
# Citation Coverage Tests
# --------------------------------------------------------------------------- #


class TestCitationCoverage:
    """Test citation coverage metric."""

    def test_full_coverage(self):
        answer = "Revenue was high."
        citations = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        score = compute_citation_coverage(answer, citations)
        assert score >= 0.5

    def test_no_citations(self):
        score = compute_citation_coverage("Some answer.", [])
        assert score == 0.0

    def test_empty_answer(self):
        assert compute_citation_coverage("", [{"chunk_id": "c1"}]) == 0.0

    def test_invalid_citations_filtered(self):
        answer = "Revenue was $100B."
        citations = [{"chunk_id": ""}, {}]
        score = compute_citation_coverage(answer, citations)
        assert score == 0.0


# --------------------------------------------------------------------------- #
# Route Accuracy Tests
# --------------------------------------------------------------------------- #


class TestRouteAccuracy:
    """Test route matching."""

    def test_matching_routes(self):
        assert compute_route_accuracy("retrieve", "retrieve") is True
        assert compute_route_accuracy("decline", "decline") is True

    def test_mismatched_routes(self):
        assert compute_route_accuracy("retrieve", "decline") is False

    def test_case_insensitive(self):
        assert compute_route_accuracy("Retrieve", "retrieve") is True


# --------------------------------------------------------------------------- #
# ItemResult Tests
# --------------------------------------------------------------------------- #


class TestItemResult:
    """Test ItemResult composite scoring."""

    def test_composite_score(self):
        ir = ItemResult(
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_precision=1.0,
            citation_coverage=1.0,
        )
        assert ir.composite_score() == 1.0

    def test_composite_zero(self):
        ir = ItemResult()
        assert ir.composite_score() == 0.0

    def test_composite_weighted(self):
        ir = ItemResult(
            faithfulness=0.8,
            answer_relevancy=0.6,
            context_precision=0.4,
            citation_coverage=0.2,
        )
        expected = 0.8 * 0.3 + 0.6 * 0.3 + 0.4 * 0.2 + 0.2 * 0.2
        assert abs(ir.composite_score() - expected) < 1e-6


# --------------------------------------------------------------------------- #
# RAGASEvaluator Tests
# --------------------------------------------------------------------------- #


class TestRAGASEvaluator:
    """Test the evaluator end-to-end."""

    def test_evaluate_retrieve_item(self):
        evaluator = RAGASEvaluator()
        item = GoldenItem(
            id="TEST-001",
            question="What was revenue?",
            expected_answer="Revenue was $100B.",
            category=Category.NUMERICAL,
            difficulty=Difficulty.EASY,
            expected_route="retrieve",
        )
        result = {
            "answer": "Revenue was approximately $100 billion in the fiscal year.",
            "citations": [{"chunk_id": "c1", "filing_reference": "10-K"}],
            "route": "retrieve",
            "retrieved_chunks": [
                {"text": "Total revenue reached $100 billion in fiscal year 2024."}
            ],
        }
        ir = evaluator.evaluate_item(item, result)
        assert ir.item_id == "TEST-001"
        assert ir.route_correct is True
        assert ir.faithfulness > 0.0
        assert ir.citation_coverage > 0.0

    def test_evaluate_decline_item(self):
        evaluator = RAGASEvaluator()
        item = GoldenItem(
            id="TEST-OOS",
            question="What will the stock price be?",
            expected_answer="Cannot predict.",
            category=Category.OUT_OF_SCOPE,
            difficulty=Difficulty.EASY,
            expected_route="decline",
        )
        result = {"answer": "I cannot predict stock prices.", "route": "decline", "citations": []}
        ir = evaluator.evaluate_item(item, result)
        assert ir.route_correct is True
        assert ir.faithfulness == 1.0
        assert ir.composite_score() == 1.0

    def test_evaluate_decline_wrong_route(self):
        evaluator = RAGASEvaluator()
        item = GoldenItem(
            id="TEST-OOS-2",
            question="Buy NVDA?",
            expected_answer="Cannot advise.",
            category=Category.OUT_OF_SCOPE,
            difficulty=Difficulty.EASY,
            expected_route="decline",
        )
        result = {"answer": "Yes buy NVDA!", "route": "retrieve", "citations": []}
        ir = evaluator.evaluate_item(item, result)
        assert ir.route_correct is False
        # Wrong-route decline still gets N/A scores for context+citation
        assert ir.composite_score() < 1.0

    def test_evaluate_dataset(self):
        evaluator = RAGASEvaluator(pass_threshold=0.3)
        items = [
            GoldenItem(
                id="D-1", question="Revenue?", expected_answer="$100B",
                category=Category.NUMERICAL, difficulty=Difficulty.EASY,
            ),
            GoldenItem(
                id="D-2", question="Stock price?", expected_answer="Cannot say.",
                category=Category.OUT_OF_SCOPE, difficulty=Difficulty.EASY,
                expected_route="decline",
            ),
        ]
        results = [
            {
                "answer": "Revenue was $100 billion.",
                "citations": [{"chunk_id": "c1"}],
                "route": "retrieve",
                "retrieved_chunks": [{"text": "Revenue $100 billion reported."}],
            },
            {"answer": "Cannot predict.", "route": "decline", "citations": []},
        ]
        report = evaluator.evaluate_dataset(items, results)
        assert report.total_items == 2
        assert len(report.results) == 2
        assert "faithfulness" in report.metrics
        assert "composite" in report.metrics
        assert len(report.category_metrics) == 2

    def test_dataset_length_mismatch_raises(self):
        evaluator = RAGASEvaluator()
        with pytest.raises(ValueError, match="Mismatch"):
            evaluator.evaluate_dataset([GoldenItem(
                id="x", question="q", expected_answer="a",
                category=Category.NUMERICAL, difficulty=Difficulty.EASY,
            )], [])

    def test_pass_threshold(self):
        evaluator = RAGASEvaluator(pass_threshold=0.0)
        items = [GoldenItem(
            id="P-1", question="Q?", expected_answer="A",
            category=Category.OUT_OF_SCOPE, difficulty=Difficulty.EASY,
            expected_route="decline",
        )]
        results = [{"answer": "Declined.", "route": "decline", "citations": []}]
        report = evaluator.evaluate_dataset(items, results)
        assert report.passed is True

    def test_report_to_dict(self):
        report = EvalReport(total_items=5, passed=True, pass_threshold=0.7)
        d = report.to_dict()
        assert d["total_items"] == 5
        assert d["passed"] is True
