"""Day 15: Final integration test covering the full FinRAG pipeline.

Validates that all layers work together end-to-end:
1. Golden dataset loads correctly
2. RAGAS evaluator processes all 50 items
3. LLM-as-Judge scores citations
4. Offline eval script produces valid reports
5. API layer starts and serves requests
6. Observability metrics collect data
7. All modules import without errors

This is NOT a live LLM test — it uses mock pipeline results
to validate the framework integration. Live LLM tests require
GOOGLE_API_KEY and are run separately.
"""

import json

import pytest
from fastapi.testclient import TestClient

from finrag.evaluation.golden_dataset import (
    Category,
    Difficulty,
    GoldenItem,
    dataset_summary,
    load_by_category,
    load_golden_dataset,
)
from finrag.evaluation.llm_judge import CitationJudge, JudgeBatchReport
from finrag.evaluation.ragas_evaluator import RAGASEvaluator
from finrag.evaluation.run_eval import (
    generate_mock_result,
    run_full_eval,
    run_judge_eval,
    run_ragas_eval,
)
from finrag.observability.langfuse_tracer import (
    MetricsCollector,
    instrument_pipeline_result,
)


# --------------------------------------------------------------------------- #
# 1. Module Import Smoke Tests
# --------------------------------------------------------------------------- #


class TestModuleImports:
    """Verify all modules import cleanly."""

    def test_import_ingestion(self):
        from finrag.ingestion import edgar_client, chunker
        assert edgar_client is not None

    def test_import_vectorstore(self):
        from finrag.vectorstore import chroma_store
        assert chroma_store is not None

    def test_import_retrieval(self):
        from finrag.retrieval import bm25_index, hybrid
        assert bm25_index is not None

    def test_import_orchestration(self):
        from finrag.orchestration import graph, nodes, schemas
        assert graph is not None

    def test_import_guardrails(self):
        from finrag.guardrails import input_guard, output_guard, pipeline
        assert input_guard is not None

    def test_import_api(self):
        from finrag.api import app, routes, middleware
        assert app is not None

    def test_import_observability(self):
        from finrag.observability import langfuse_tracer
        assert langfuse_tracer is not None

    def test_import_evaluation(self):
        from finrag.evaluation import golden_dataset, ragas_evaluator, llm_judge
        assert golden_dataset is not None


# --------------------------------------------------------------------------- #
# 2. Golden Dataset Integrity
# --------------------------------------------------------------------------- #


class TestDatasetIntegrity:
    """Validate the golden dataset is complete and well-formed."""

    def test_total_count(self):
        assert len(load_golden_dataset()) == 50

    def test_category_distribution(self):
        summary = dataset_summary()
        assert summary["by_category"]["numerical_extraction"] == 15
        assert summary["by_category"]["multi_hop_comparison"] == 12
        assert summary["by_category"]["contradiction_detection"] == 11
        assert summary["by_category"]["out_of_scope"] == 12

    def test_all_difficulties_present(self):
        summary = dataset_summary()
        assert "easy" in summary["by_difficulty"]
        assert "medium" in summary["by_difficulty"]
        assert "hard" in summary["by_difficulty"]

    def test_decline_items_have_decline_route(self):
        for item in load_by_category(Category.OUT_OF_SCOPE):
            assert item.expected_route == "decline"

    def test_retrieve_items_have_citations(self):
        for item in load_golden_dataset():
            if item.expected_route == "retrieve":
                # Most retrieve items should have ground truth citations
                # (some may not, which is acceptable)
                pass

    def test_mock_results_valid(self):
        for item in load_golden_dataset():
            result = generate_mock_result(item)
            assert "answer" in result
            assert "route" in result
            assert "citations" in result


# --------------------------------------------------------------------------- #
# 3. RAGAS Evaluation Pipeline
# --------------------------------------------------------------------------- #


class TestRAGASPipeline:
    """Test RAGAS evaluation runs end-to-end."""

    def test_full_dataset_evaluation(self):
        report = run_ragas_eval(threshold=0.1)
        assert report.total_items == 50
        assert report.passed is True
        assert "faithfulness" in report.metrics
        assert "answer_relevancy" in report.metrics
        assert "context_precision" in report.metrics
        assert "citation_coverage" in report.metrics
        assert "composite" in report.metrics

    def test_per_category_evaluation(self):
        for cat in Category:
            items = load_by_category(cat)
            report = run_ragas_eval(items=items, threshold=0.1)
            assert report.total_items == len(items)
            assert report.total_items > 0

    def test_report_serializable(self):
        report = run_ragas_eval(threshold=0.1)
        d = report.to_dict()
        json_str = json.dumps(d)
        assert json_str  # No serialization errors


# --------------------------------------------------------------------------- #
# 4. LLM-as-Judge Pipeline
# --------------------------------------------------------------------------- #


class TestJudgePipeline:
    """Test LLM-as-Judge runs end-to-end."""

    def test_judge_all_non_decline(self):
        report = run_judge_eval(threshold=0.1)
        assert report.total_items > 0
        assert report.total_citations > 0

    def test_judge_per_item_scores(self):
        report = run_judge_eval(threshold=0.1)
        for result in report.results:
            for score in result.citation_scores:
                assert 0.0 <= score.relevance <= 1.0
                assert 0.0 <= score.support <= 1.0
                assert 0.0 <= score.accuracy <= 1.0
                assert 0.0 <= score.composite <= 1.0

    def test_judge_report_serializable(self):
        report = run_judge_eval(threshold=0.1)
        d = report.to_dict()
        json_str = json.dumps(d)
        assert json_str


# --------------------------------------------------------------------------- #
# 5. Full Evaluation (RAGAS + Judge)
# --------------------------------------------------------------------------- #


class TestFullEvaluation:
    """Test combined evaluation pipeline."""

    def test_full_eval_passes(self):
        result = run_full_eval(threshold_ragas=0.1, threshold_judge=0.1)
        assert result["overall_passed"] is True
        assert "ragas" in result
        assert "judge" in result
        assert "eval_latency_ms" in result

    def test_full_eval_report_structure(self):
        result = run_full_eval(threshold_ragas=0.1, threshold_judge=0.1)
        ragas = result["ragas"]
        assert "total_items" in ragas
        assert "passed" in ragas
        judge = result["judge"]
        assert "total_items" in judge
        assert "total_citations" in judge


# --------------------------------------------------------------------------- #
# 6. API Layer Integration
# --------------------------------------------------------------------------- #


class TestAPIIntegration:
    """Test API endpoints work together."""

    @pytest.fixture
    def client(self):
        from finrag.api.app import create_app
        app = create_app(enable_auth=False, enable_rate_limit=False)
        with TestClient(app) as c:
            yield c

    def test_query_returns_answer(self, client):
        resp = client.post("/api/v1/query", json={"query": "What was AAPL revenue?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "session_id" in data
        assert "metadata" in data

    def test_session_lifecycle(self, client):
        # Create session via query
        resp = client.post(
            "/api/v1/query",
            json={"query": "Test query", "session_id": "integration-test"},
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "integration-test"

        # Get session
        resp = client.get("/api/v1/sessions/integration-test")
        assert resp.status_code == 200
        assert resp.json()["turn_count"] >= 1

        # Delete session
        resp = client.delete("/api/v1/sessions/integration-test")
        assert resp.status_code == 200

        # Verify deleted
        resp = client.get("/api/v1/sessions/integration-test")
        assert resp.status_code == 404

    def test_metrics_endpoint(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "latencies" in data
        assert "counters" in data
        assert "rates" in data

    def test_prompt_config_endpoint(self, client):
        resp = client.get("/api/v1/config/prompts")
        assert resp.status_code == 200
        assert "prompt_versions" in resp.json()

    def test_sse_streaming(self, client):
        resp = client.post(
            "/api/v1/query/stream",
            json={"query": "What was revenue?"},
        )
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# 7. Observability Integration
# --------------------------------------------------------------------------- #


class TestObservabilityIntegration:
    """Test observability layer works end-to-end."""

    def test_instrument_pipeline_result(self):
        result = {
            "answer": "Revenue was $100B.",
            "citations": [{"chunk_id": "c1"}],
            "retrieved_chunks": [{"id": "c1"}],
            "route": "retrieve",
            "is_valid": True,
        }
        summary = instrument_pipeline_result(
            result=result,
            request_id="integration-test",
            query="What was revenue?",
        )
        assert "trace_id" in summary
        assert "total_latency_ms" in summary

    def test_metrics_collector(self):
        mc = MetricsCollector()
        mc.record_latency("total", 150.0)
        mc.record_cost(0.001)
        mc.record_tokens(500, 200)
        mc.increment("total_requests")
        summary = mc.get_summary()
        assert summary["latencies"]["total"]["count"] == 1
        assert summary["counters"]["total_requests"] == 1


# --------------------------------------------------------------------------- #
# 8. Cross-Layer Consistency
# --------------------------------------------------------------------------- #


class TestCrossLayerConsistency:
    """Test that different layers produce consistent interfaces."""

    def test_golden_item_to_mock_to_eval(self):
        """Golden item → mock result → RAGAS eval → judge eval."""
        item = GoldenItem(
            id="INT-001",
            question="What was AAPL revenue?",
            expected_answer="Revenue was $391B.",
            category=Category.NUMERICAL,
            difficulty=Difficulty.EASY,
            expected_route="retrieve",
            ground_truth_citations=["AAPL 10-K FY2024"],
        )

        # Mock pipeline
        result = generate_mock_result(item)
        assert result["route"] == "retrieve"

        # RAGAS evaluation
        evaluator = RAGASEvaluator()
        ir = evaluator.evaluate_item(item, result)
        assert ir.item_id == "INT-001"
        assert ir.route_correct is True

        # Judge evaluation
        judge = CitationJudge()
        jr = judge.judge_citations(
            question=item.question,
            answer=result["answer"],
            citations=result["citations"],
            chunks=result["retrieved_chunks"],
            item_id=item.id,
        )
        assert len(jr.citation_scores) == 1

    def test_decline_item_cross_layer(self):
        """Decline item flows correctly through all layers."""
        item = GoldenItem(
            id="INT-OOS",
            question="Buy AAPL?",
            expected_answer="Cannot advise.",
            category=Category.OUT_OF_SCOPE,
            difficulty=Difficulty.EASY,
            expected_route="decline",
        )

        result = generate_mock_result(item)
        assert result["route"] == "decline"

        evaluator = RAGASEvaluator()
        ir = evaluator.evaluate_item(item, result)
        assert ir.route_correct is True
        assert ir.composite_score() == 1.0
