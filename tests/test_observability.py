"""Tests for Day 12: Langfuse instrumentation and observability.

Covers:
- Token cost estimation
- Failure classification
- MetricBucket statistics (mean, percentiles, min/max)
- MetricsCollector (latency, cost, tokens, counters, rates)
- FinRAGTracer (no-op when Langfuse not configured)
- instrument_pipeline_result helper
- /metrics endpoint integration
"""

import pytest

from finrag.observability.langfuse_tracer import (
    FailureType,
    FinRAGTracer,
    MetricBucket,
    MetricsCollector,
    classify_failure,
    estimate_cost,
    instrument_pipeline_result,
    metrics,
    tracer,
)


# --------------------------------------------------------------------------- #
# Token Cost Estimation
# --------------------------------------------------------------------------- #


class TestCostEstimation:
    """Test token cost calculation."""

    def test_known_model_cost(self):
        cost = estimate_cost("gemini-2.0-flash", input_tokens=1000, output_tokens=500)
        # 1000 * 0.0001/1000 + 500 * 0.0004/1000
        expected = (1000 * 0.0001 / 1000) + (500 * 0.0004 / 1000)
        assert abs(cost - expected) < 1e-8

    def test_unknown_model_uses_default(self):
        cost = estimate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        assert cost > 0

    def test_zero_tokens_zero_cost(self):
        cost = estimate_cost("gemini-2.0-flash", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_free_model(self):
        cost = estimate_cost(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            input_tokens=5000,
            output_tokens=0,
        )
        assert cost == 0.0


# --------------------------------------------------------------------------- #
# Failure Classification
# --------------------------------------------------------------------------- #


class TestFailureClassification:
    """Test pipeline failure classification."""

    def test_no_failure(self):
        result = {
            "answer": "Revenue was $100B",
            "citations": [{}],
            "is_valid": True,
            "retrieved_chunks": [{"id": 1}],
        }

    def test_input_guard_block(self):
        result = {"input_guard_blocked": True}
        assert classify_failure(result) == FailureType.INPUT_GUARD_BLOCK

    def test_output_guard_block(self):
        result = {"output_guard_blocked": True}
        assert classify_failure(result) == FailureType.OUTPUT_GUARD_BLOCK

    def test_timeout(self):
        result = {"error": "Request timeout after 30s"}
        assert classify_failure(result) == FailureType.TIMEOUT

    def test_generation_error(self):
        result = {"error": "LLM generation failed"}
        assert classify_failure(result) == FailureType.GENERATION_ERROR

    def test_unknown_error(self):
        result = {"error": "Something broke"}
        assert classify_failure(result) == FailureType.UNKNOWN

    def test_decline_route(self):
        result = {"route": "decline"}
        assert classify_failure(result) == FailureType.RERANKER_LOW_SCORE

    def test_retrieval_empty(self):
        result = {"retrieved_chunks": []}
        assert classify_failure(result) == FailureType.RETRIEVAL_EMPTY

    def test_citation_enforcement_fail(self):
        result = {
            "is_valid": False,
            "retrieved_chunks": [{"id": 1}],
            "validation_errors": ["citation missing for claim 1"],
        }
        assert classify_failure(result) == FailureType.CITATION_ENFORCEMENT

    def test_validation_fail_non_citation(self):
        result = {
            "is_valid": False,
            "retrieved_chunks": [{"id": 1}],
            "validation_errors": ["answer too short"],
        }
        assert classify_failure(result) == FailureType.VALIDATION_FAIL


# --------------------------------------------------------------------------- #
# MetricBucket
# --------------------------------------------------------------------------- #


class TestMetricBucket:
    """Test MetricBucket statistics."""

    def test_empty_bucket(self):
        b = MetricBucket()
        assert b.count == 0
        assert b.mean == 0.0
        assert b.percentile(50) == 0.0

    def test_single_value(self):
        b = MetricBucket()
        b.record(42.0)
        assert b.count == 1
        assert b.mean == 42.0
        assert b.min_val == 42.0
        assert b.max_val == 42.0

    def test_multiple_values(self):
        b = MetricBucket()
        for v in [10, 20, 30, 40, 50]:
            b.record(v)
        assert b.count == 5
        assert b.mean == 30.0
        assert b.min_val == 10
        assert b.max_val == 50

    def test_percentiles(self):
        b = MetricBucket()
        for v in range(1, 101):
            b.record(float(v))
        p50 = b.percentile(50)
        p95 = b.percentile(95)
        assert 49 <= p50 <= 51
        assert 94 <= p95 <= 96

    def test_to_dict(self):
        b = MetricBucket()
        b.record(10.0)
        b.record(20.0)
        d = b.to_dict()
        assert d["count"] == 2
        assert "mean" in d
        assert "p50" in d
        assert "p95" in d
        assert "p99" in d

    def test_rolling_window(self):
        b = MetricBucket()
        # Record more than 1000 values
        for i in range(1100):
            b.record(float(i))
        assert b.count == 1100
        assert len(b.values) == 1000  # Capped at 1000


# --------------------------------------------------------------------------- #
# MetricsCollector
# --------------------------------------------------------------------------- #


class TestMetricsCollector:
    """Test MetricsCollector aggregation."""

    def test_record_latency(self):
        mc = MetricsCollector()
        mc.record_latency("retrieval", 50.0)
        mc.record_latency("retrieval", 100.0)
        summary = mc.get_summary()
        assert summary["latencies"]["retrieval"]["count"] == 2
        assert summary["latencies"]["retrieval"]["mean"] == 75.0

    def test_record_cost(self):
        mc = MetricsCollector()
        mc.record_cost(0.001)
        mc.record_cost(0.002)
        summary = mc.get_summary()
        assert summary["costs"]["count"] == 2

    def test_record_tokens(self):
        mc = MetricsCollector()
        mc.record_tokens(500, 200)
        summary = mc.get_summary()
        assert summary["tokens"]["input"]["count"] == 1
        assert summary["tokens"]["output"]["count"] == 1

    def test_counters(self):
        mc = MetricsCollector()
        mc.increment("total_requests")
        mc.increment("total_requests")
        mc.increment("declines")
        summary = mc.get_summary()
        assert summary["counters"]["total_requests"] == 2
        assert summary["counters"]["declines"] == 1

    def test_rates_empty(self):
        mc = MetricsCollector()
        rates = mc.get_rates()
        assert rates["decline_rate"] == 0.0
        assert rates["citation_coverage"] == 0.0

    def test_rates_with_data(self):
        mc = MetricsCollector()
        for _ in range(10):
            mc.increment("total_requests")
        for _ in range(3):
            mc.increment("declines")
        for _ in range(7):
            mc.increment("cited_responses")
        rates = mc.get_rates()
        assert rates["decline_rate"] == 0.3
        assert rates["citation_coverage"] == 0.7

    def test_reset(self):
        mc = MetricsCollector()
        mc.record_latency("test", 100.0)
        mc.increment("total_requests")
        mc.reset()
        summary = mc.get_summary()
        assert len(summary["latencies"]) == 0
        assert len(summary["counters"]) == 0


# --------------------------------------------------------------------------- #
# FinRAGTracer (no-op mode)
# --------------------------------------------------------------------------- #


class TestFinRAGTracer:
    """Test tracer in no-op mode (no Langfuse keys)."""

    def test_start_trace_returns_context(self):
        t = FinRAGTracer()
        ctx = t.start_trace(query="What was revenue?")
        assert "trace_id" in ctx
        assert "start_time" in ctx
        assert "query" in ctx
        assert ctx["query"] == "What was revenue?"

    def test_start_and_end_span(self):
        t = FinRAGTracer()
        ctx = t.start_trace(query="test")
        span = t.start_span(ctx, "retrieval")
        latency = t.end_span(span, output={"chunks": 5})
        assert latency >= 0

    def test_record_generation_noop(self):
        t = FinRAGTracer()
        ctx = t.start_trace(query="test")
        # Should not raise
        t.record_generation(
            ctx,
            model="gemini-2.0-flash",
            prompt="test prompt",
            completion="test answer",
            input_tokens=100,
            output_tokens=50,
        )

    def test_score_trace_noop(self):
        t = FinRAGTracer()
        ctx = t.start_trace(query="test")
        t.score_trace(ctx, name="faithfulness", value=0.95)

    def test_end_trace_returns_summary(self):
        t = FinRAGTracer()
        ctx = t.start_trace(query="test", request_id="req-123")
        summary = t.end_trace(ctx, result={"route": "retrieve", "is_valid": True})
        assert summary["trace_id"] == "req-123"
        assert summary["route"] == "retrieve"
        assert "total_latency_ms" in summary

    def test_end_trace_with_failure(self):
        t = FinRAGTracer()
        ctx = t.start_trace(query="test")
        summary = t.end_trace(
            ctx,
            result={"route": "decline", "input_guard_blocked": True},
        )
        assert summary["failure_type"] == "input_guard_block"

    def test_flush_noop(self):
        t = FinRAGTracer()
        t.flush()


# --------------------------------------------------------------------------- #
# Pipeline Instrumentation Helper
# --------------------------------------------------------------------------- #


class TestInstrumentPipelineResult:
    """Test the one-shot instrumentation helper."""

    def setup_method(self):
        """Reset global metrics before each test."""
        metrics.reset()

    def test_instrument_success(self):
        result = {
            "answer": "Revenue was $100B",
            "citations": [{"chunk_id": "c1"}],
            "reranked_chunks": [{"id": 1}, {"id": 2}],
            "retrieved_chunks": [{"id": 1}, {"id": 2}],
            "route": "retrieve",
            "is_valid": True,
        }
        summary = instrument_pipeline_result(
            result=result,
            request_id="test-req",
            query="What was revenue?",
        )
        assert summary["trace_id"] == "test-req"
        assert summary["route"] == "retrieve"
        assert "failure_type" not in summary

    def test_instrument_decline(self):
        result = {"route": "decline", "answer": "", "citations": []}
        summary = instrument_pipeline_result(result=result, query="test")
        assert summary["failure_type"] == "reranker_low_score"

    def test_instrument_increments_counters(self):
        result = {"route": "retrieve", "is_valid": True, "citations": [{"id": 1}]}
        instrument_pipeline_result(result=result, query="test")
        summary = metrics.get_summary()
        assert summary["counters"]["total_requests"] >= 1
        assert summary["counters"]["cited_responses"] >= 1


# --------------------------------------------------------------------------- #
# /metrics Endpoint Integration
# --------------------------------------------------------------------------- #


class TestMetricsEndpoint:
    """Test GET /api/v1/metrics endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from finrag.api.app import create_app

        app = create_app(enable_auth=False, enable_rate_limit=False)
        with TestClient(app) as c:
            yield c

    def test_metrics_endpoint_returns_200(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "latencies" in data
        assert "costs" in data
        assert "tokens" in data
        assert "counters" in data
        assert "rates" in data

    def test_metrics_after_query(self, client):
        # Make a query first
        client.post(
            "/api/v1/query",
            json={"query": "What was AAPL revenue?"},
        )

        resp = client.get("/api/v1/metrics")
        data = resp.json()
        assert data["counters"].get("total_requests", 0) >= 1
        assert "total" in data["latencies"]
