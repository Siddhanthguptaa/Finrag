"""Tests for Day 14: LLM-as-Judge citation scorer + offline eval.

Covers:
- CitationScore composite scoring
- JudgeResult means computation
- CitationJudge heuristic fallback
- JudgeBatchReport aggregation
- Judge prompt building
- Mock pipeline result generation
- Offline eval runners (ragas, judge, full)
- CLI entry point
"""

import json

import pytest

from finrag.evaluation.golden_dataset import (
    Category,
    Difficulty,
    GoldenItem,
    load_golden_dataset,
)
from finrag.evaluation.llm_judge import (
    CitationJudge,
    CitationScore,
    JudgeBatchReport,
    JudgeResult,
    _build_judge_prompt,
    _clamp,
)
from finrag.evaluation.run_eval import (
    generate_mock_result,
    main,
    run_full_eval,
    run_judge_eval,
    run_ragas_eval,
)


# --------------------------------------------------------------------------- #
# CitationScore Tests
# --------------------------------------------------------------------------- #


class TestCitationScore:
    """Test citation score model."""

    def test_compute_composite(self):
        cs = CitationScore(relevance=1.0, support=1.0, accuracy=1.0)
        assert cs.compute_composite() == 1.0

    def test_composite_weights(self):
        cs = CitationScore(relevance=0.5, support=0.8, accuracy=0.6)
        expected = 0.5 * 0.2 + 0.8 * 0.4 + 0.6 * 0.4
        assert abs(cs.compute_composite() - expected) < 1e-4

    def test_zero_composite(self):
        cs = CitationScore()
        assert cs.compute_composite() == 0.0


# --------------------------------------------------------------------------- #
# JudgeResult Tests
# --------------------------------------------------------------------------- #


class TestJudgeResult:
    """Test judge result aggregation."""

    def test_compute_means(self):
        jr = JudgeResult(
            citation_scores=[
                CitationScore(relevance=0.8, support=0.6, accuracy=0.9, composite=0.75),
                CitationScore(relevance=0.4, support=0.3, accuracy=0.5, composite=0.38),
            ]
        )
        jr.compute_means()
        assert jr.mean_relevance == 0.6
        assert abs(jr.mean_support - 0.45) < 1e-4
        assert jr.mean_accuracy == 0.7

    def test_empty_means(self):
        jr = JudgeResult()
        jr.compute_means()
        assert jr.mean_relevance == 0.0

    def test_to_dict(self):
        jr = JudgeResult(item_id="TEST", question="Q?", judge_model="test")
        d = jr.to_dict()
        assert d["item_id"] == "TEST"
        assert d["citation_count"] == 0
        assert "mean_composite" in d


# --------------------------------------------------------------------------- #
# CitationJudge (Heuristic Mode) Tests
# --------------------------------------------------------------------------- #


class TestCitationJudge:
    """Test citation judge in heuristic mode."""

    def test_judge_empty_citations(self):
        judge = CitationJudge()
        result = judge.judge_citations(
            question="What was revenue?",
            answer="Revenue was $100B.",
            citations=[],
            chunks=[],
        )
        assert len(result.citation_scores) == 0
        assert result.judge_model == "gemini-2.0-flash"

    def test_judge_heuristic_with_matching_content(self):
        judge = CitationJudge()
        result = judge.judge_citations(
            question="What was Apple revenue?",
            answer="Apple revenue was $391 billion.",
            citations=[{"chunk_id": "c1", "filing_reference": "AAPL 10-K"}],
            chunks=[{"chunk_id": "c1", "text": "Apple reported revenue of $391 billion in FY2024."}],
            item_id="TEST-001",
        )
        assert len(result.citation_scores) == 1
        assert result.citation_scores[0].chunk_id == "c1"
        assert result.citation_scores[0].relevance > 0.0
        assert result.citation_scores[0].support > 0.0
        assert result.citation_scores[0].accuracy == 0.5  # heuristic default
        assert result.judge_model == "heuristic"

    def test_judge_heuristic_no_overlap(self):
        judge = CitationJudge()
        result = judge.judge_citations(
            question="What was revenue?",
            answer="Revenue was high.",
            citations=[{"chunk_id": "c1"}],
            chunks=[{"chunk_id": "c1", "text": "The weather is sunny today."}],
        )
        assert result.citation_scores[0].support < 0.5

    def test_judge_missing_chunk_text(self):
        judge = CitationJudge()
        result = judge.judge_citations(
            question="Q?",
            answer="A.",
            citations=[{"chunk_id": "unknown"}],
            chunks=[],
        )
        assert result.citation_scores[0].accuracy == 0.0

    def test_judge_multiple_citations(self):
        judge = CitationJudge()
        result = judge.judge_citations(
            question="Compare A and B",
            answer="A was higher than B.",
            citations=[
                {"chunk_id": "c1"},
                {"chunk_id": "c2"},
            ],
            chunks=[
                {"chunk_id": "c1", "text": "Company A reported higher numbers."},
                {"chunk_id": "c2", "text": "Company B reported lower numbers."},
            ],
        )
        assert len(result.citation_scores) == 2
        result.compute_means()
        assert result.mean_composite > 0.0


# --------------------------------------------------------------------------- #
# JudgeBatchReport Tests
# --------------------------------------------------------------------------- #


class TestJudgeBatchReport:
    """Test batch report aggregation."""

    def test_compute_aggregates(self):
        report = JudgeBatchReport(
            total_items=2,
            results=[
                JudgeResult(citation_scores=[
                    CitationScore(relevance=0.8, support=0.7, accuracy=0.9, composite=0.8),
                ]),
                JudgeResult(citation_scores=[
                    CitationScore(relevance=0.6, support=0.5, accuracy=0.7, composite=0.6),
                ]),
            ],
            pass_threshold=0.5,
        )
        report.compute_aggregates()
        assert report.total_citations == 2
        assert report.mean_relevance == 0.7
        assert report.passed is True

    def test_empty_report(self):
        report = JudgeBatchReport()
        report.compute_aggregates()
        assert report.total_citations == 0
        assert report.passed is False

    def test_to_dict(self):
        report = JudgeBatchReport(total_items=5, passed=True)
        d = report.to_dict()
        assert d["total_items"] == 5
        assert "mean_composite" in d


# --------------------------------------------------------------------------- #
# Prompt Building Tests
# --------------------------------------------------------------------------- #


class TestPromptBuilding:
    """Test judge prompt construction."""

    def test_build_prompt(self):
        prompt = _build_judge_prompt(
            question="What was revenue?",
            answer="Revenue was $100B.",
            citations=[{"chunk_id": "c1", "filing_reference": "10-K"}],
            chunks=[{"chunk_id": "c1", "text": "Revenue data here."}],
        )
        assert "What was revenue?" in prompt
        assert "Revenue was $100B" in prompt
        assert "c1" in prompt
        assert "Revenue data here" in prompt

    def test_build_prompt_missing_chunk(self):
        prompt = _build_judge_prompt(
            question="Q?",
            answer="A.",
            citations=[{"chunk_id": "missing"}],
            chunks=[],
        )
        assert "not available" in prompt


# --------------------------------------------------------------------------- #
# Utility Tests
# --------------------------------------------------------------------------- #


class TestUtilities:
    """Test helper functions."""

    def test_clamp(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(-0.1) == 0.0
        assert _clamp(1.5) == 1.0
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0


# --------------------------------------------------------------------------- #
# Mock Pipeline Tests
# --------------------------------------------------------------------------- #


class TestMockPipeline:
    """Test mock result generation."""

    def test_mock_retrieve_item(self):
        item = GoldenItem(
            id="M-1", question="Q?", expected_answer="A.",
            category=Category.NUMERICAL, difficulty=Difficulty.EASY,
            ground_truth_citations=["AAPL 10-K FY2024"],
        )
        result = generate_mock_result(item)
        assert result["answer"] == "A."
        assert result["route"] == "retrieve"
        assert len(result["citations"]) == 1
        assert len(result["retrieved_chunks"]) == 1

    def test_mock_decline_item(self):
        item = GoldenItem(
            id="M-2", question="Stock price?", expected_answer="Cannot.",
            category=Category.OUT_OF_SCOPE, difficulty=Difficulty.EASY,
            expected_route="decline",
        )
        result = generate_mock_result(item)
        assert result["route"] == "decline"
        assert len(result["citations"]) == 0

    def test_mock_no_citations_fallback(self):
        item = GoldenItem(
            id="M-3", question="Q?", expected_answer="A.",
            category=Category.NUMERICAL, difficulty=Difficulty.EASY,
            ground_truth_citations=[],
        )
        result = generate_mock_result(item)
        assert len(result["citations"]) >= 1  # fallback citation


# --------------------------------------------------------------------------- #
# Eval Runner Tests
# --------------------------------------------------------------------------- #


class TestEvalRunners:
    """Test evaluation runners."""

    def test_run_ragas_eval(self):
        report = run_ragas_eval(threshold=0.1)
        assert report.total_items == 50
        assert report.passed is True
        assert "faithfulness" in report.metrics

    def test_run_ragas_category_filter(self):
        from finrag.evaluation.golden_dataset import load_by_category
        items = load_by_category(Category.OUT_OF_SCOPE)
        report = run_ragas_eval(items=items, threshold=0.1)
        assert report.total_items == 12

    def test_run_judge_eval(self):
        report = run_judge_eval(threshold=0.1)
        assert report.total_items > 0
        assert report.total_citations > 0

    def test_run_full_eval(self):
        result = run_full_eval(threshold_ragas=0.1, threshold_judge=0.1)
        assert "ragas" in result
        assert "judge" in result
        assert "overall_passed" in result
        assert result["overall_passed"] is True


# --------------------------------------------------------------------------- #
# CLI Tests
# --------------------------------------------------------------------------- #


class TestCLI:
    """Test CLI entry point."""

    def test_main_ragas(self, capsys):
        exit_code = main(["--mode", "ragas", "--threshold", "0.1"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "PASSED" in captured.out

    def test_main_judge(self, capsys):
        exit_code = main(["--mode", "judge", "--threshold", "0.1"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "PASSED" in captured.out

    def test_main_full(self, capsys):
        exit_code = main(["--mode", "full", "--threshold", "0.1"])
        assert exit_code == 0

    def test_main_category_filter(self, capsys):
        exit_code = main(["--mode", "ragas", "--category", "numerical", "--threshold", "0.1"])
        assert exit_code == 0

    def test_main_output_file(self, tmp_path, capsys):
        output = tmp_path / "report.json"
        exit_code = main(["--mode", "ragas", "--threshold", "0.1", "--output", str(output)])
        assert exit_code == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert "total_items" in data
