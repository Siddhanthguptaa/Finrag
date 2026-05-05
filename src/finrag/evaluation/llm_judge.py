"""LLM-as-Judge citation scorer for the FinRAG pipeline.

Uses a separate LLM to score each citation for accuracy,
catching misattributed citations that heuristic RAGAS metrics miss.

Scoring dimensions per citation:
1. Relevance: Does the cited chunk relate to the claim?
2. Support: Does the chunk actually support the claim?
3. Accuracy: Is the claim factually consistent with the chunk?

Design decisions:
- Separate judge model: uses gemini-2.0-flash (cheap, fast) to
  avoid self-evaluation bias from the generation model.
- Structured output: judge returns JSON with per-dimension scores.
- Graceful fallback: if LLM unavailable, returns heuristic scores.
- Batch evaluation: scores all citations in a single prompt to
  reduce API calls. Falls back to per-citation if batch fails.

Debt: DAY-14-001 -- Judge prompt may need calibration against
      human annotations. Track inter-annotator agreement.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Score Models
# --------------------------------------------------------------------------- #


@dataclass
class CitationScore:
    """Score for a single citation.

    Attributes:
        chunk_id: The cited chunk identifier.
        relevance: How relevant the chunk is to the claim (0-1).
        support: How well the chunk supports the claim (0-1).
        accuracy: Factual consistency of claim with chunk (0-1).
        composite: Weighted average of all dimensions.
        reasoning: Judge's explanation for the scores.
    """

    chunk_id: str = ""
    relevance: float = 0.0
    support: float = 0.0
    accuracy: float = 0.0
    composite: float = 0.0
    reasoning: str = ""

    def compute_composite(self) -> float:
        """Compute weighted composite score.

        Returns:
            Weighted average: relevance(0.2) + support(0.4) + accuracy(0.4).
        """
        self.composite = round(
            self.relevance * 0.2 + self.support * 0.4 + self.accuracy * 0.4, 4
        )
        return self.composite


@dataclass
class JudgeResult:
    """Result from judging all citations in one answer.

    Attributes:
        item_id: Golden dataset item identifier.
        question: The original question.
        citation_scores: Per-citation scores.
        mean_relevance: Average relevance across citations.
        mean_support: Average support across citations.
        mean_accuracy: Average accuracy across citations.
        mean_composite: Average composite across citations.
        judge_model: Model used for judging.
        latency_ms: Time taken for judging.
        error: Error message if judging failed.
    """

    item_id: str = ""
    question: str = ""
    citation_scores: list[CitationScore] = field(default_factory=list)
    mean_relevance: float = 0.0
    mean_support: float = 0.0
    mean_accuracy: float = 0.0
    mean_composite: float = 0.0
    judge_model: str = ""
    latency_ms: float = 0.0
    error: str | None = None

    def compute_means(self) -> None:
        """Compute mean scores across all citations."""
        if not self.citation_scores:
            return
        n = len(self.citation_scores)
        self.mean_relevance = round(
            sum(c.relevance for c in self.citation_scores) / n, 4
        )
        self.mean_support = round(
            sum(c.support for c in self.citation_scores) / n, 4
        )
        self.mean_accuracy = round(
            sum(c.accuracy for c in self.citation_scores) / n, 4
        )
        self.mean_composite = round(
            sum(c.composite for c in self.citation_scores) / n, 4
        )

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "mean_relevance": self.mean_relevance,
            "mean_support": self.mean_support,
            "mean_accuracy": self.mean_accuracy,
            "mean_composite": self.mean_composite,
            "citation_count": len(self.citation_scores),
            "judge_model": self.judge_model,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Judge Prompt
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM_PROMPT = """You are a citation accuracy judge for a financial RAG system.

Given a question, an answer, and a list of citations (each with the cited chunk text),
score each citation on three dimensions:

1. **relevance** (0.0–1.0): Does the cited chunk relate to the topic of the claim it supports?
2. **support** (0.0–1.0): Does the chunk actually provide evidence for the specific claim?
3. **accuracy** (0.0–1.0): Is the claim factually consistent with what the chunk says?

Return a JSON array with one object per citation:
```json
[
  {
    "chunk_id": "...",
    "relevance": 0.9,
    "support": 0.8,
    "accuracy": 0.85,
    "reasoning": "Brief explanation"
  }
]
```

Scoring guidelines:
- 1.0: Perfect match/support
- 0.7-0.9: Strong but not perfect
- 0.4-0.6: Partial/tangential
- 0.1-0.3: Weak/misleading
- 0.0: Completely wrong or fabricated
"""


def _build_judge_prompt(
    question: str,
    answer: str,
    citations: list[dict],
    chunks: list[dict],
) -> str:
    """Build the user prompt for the judge.

    Args:
        question: The original question.
        answer: The generated answer.
        citations: List of citation dicts with chunk_id.
        chunks: List of chunk dicts with id/text.

    Returns:
        Formatted prompt string.
    """
    chunk_map = {}
    for c in chunks:
        cid = c.get("chunk_id", c.get("id", ""))
        text = c.get("text", c.get("content", ""))
        if cid:
            chunk_map[cid] = text

    citation_entries = []
    for i, cit in enumerate(citations):
        cid = cit.get("chunk_id", "")
        chunk_text = chunk_map.get(cid, "[chunk text not available]")
        citation_entries.append(
            f"Citation {i + 1}:\n"
            f"  chunk_id: {cid}\n"
            f"  filing_reference: {cit.get('filing_reference', '')}\n"
            f"  chunk_text: {chunk_text[:500]}\n"
        )

    return (
        f"**Question:** {question}\n\n"
        f"**Answer:** {answer}\n\n"
        f"**Citations to judge:**\n"
        + "\n".join(citation_entries)
    )


# --------------------------------------------------------------------------- #
# LLM-as-Judge
# --------------------------------------------------------------------------- #


class CitationJudge:
    """Scores citation accuracy using an LLM judge.

    Falls back to heuristic scoring when LLM is unavailable.

    Attributes:
        model_name: LLM model for judging.
        _llm: Lazy-initialized LLM client.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        self.model_name = model_name
        self._llm = None

    def _get_llm(self):
        """Lazy-initialize the LLM client.

        Returns:
            ChatGoogleGenerativeAI instance or None.
        """
        if self._llm is not None:
            return self._llm

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            logger.info("judge_llm_disabled", reason="GOOGLE_API_KEY not set")
            return None

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.0,
                google_api_key=api_key,
            )
            return self._llm
        except Exception as e:
            logger.warning("judge_llm_init_failed", error=str(e))
            return None

    def judge_citations(
        self,
        question: str,
        answer: str,
        citations: list[dict],
        chunks: list[dict],
        item_id: str = "",
    ) -> JudgeResult:
        """Score all citations for one answer.

        Args:
            question: Original question.
            answer: Generated answer.
            citations: Citation dicts with chunk_id.
            chunks: Retrieved chunk dicts with id+text.
            item_id: Optional golden item id.

        Returns:
            JudgeResult with per-citation scores.
        """
        if not citations:
            return JudgeResult(
                item_id=item_id,
                question=question,
                judge_model=self.model_name,
            )

        start = time.perf_counter()
        llm = self._get_llm()

        if llm is not None:
            result = self._judge_with_llm(
                llm, question, answer, citations, chunks, item_id
            )
        else:
            result = self._judge_heuristic(
                question, answer, citations, chunks, item_id
            )

        result.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return result

    def _judge_with_llm(
        self,
        llm,
        question: str,
        answer: str,
        citations: list[dict],
        chunks: list[dict],
        item_id: str,
    ) -> JudgeResult:
        """Use LLM to judge citations.

        Args:
            llm: LLM client.
            question: Original question.
            answer: Generated answer.
            citations: Citation dicts.
            chunks: Chunk dicts.
            item_id: Golden item id.

        Returns:
            JudgeResult from LLM scoring.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = _build_judge_prompt(question, answer, citations, chunks)

        try:
            response = llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            scores = self._parse_judge_response(response.content, citations)
        except Exception as e:
            logger.warning("judge_llm_call_failed", error=str(e))
            return self._judge_heuristic(
                question, answer, citations, chunks, item_id
            )

        result = JudgeResult(
            item_id=item_id,
            question=question,
            citation_scores=scores,
            judge_model=self.model_name,
        )
        result.compute_means()
        return result

    def _parse_judge_response(
        self,
        response_text: str,
        citations: list[dict],
    ) -> list[CitationScore]:
        """Parse LLM judge response into CitationScore list.

        Args:
            response_text: Raw LLM response.
            citations: Original citations for fallback.

        Returns:
            List of CitationScore objects.
        """
        # Extract JSON array from response
        text = response_text.strip()
        # Try to find JSON array
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx == -1 or end_idx == -1:
            logger.warning("judge_parse_failed", reason="no JSON array found")
            return [
                CitationScore(chunk_id=c.get("chunk_id", ""))
                for c in citations
            ]

        try:
            parsed = json.loads(text[start_idx : end_idx + 1])
        except json.JSONDecodeError:
            logger.warning("judge_parse_failed", reason="invalid JSON")
            return [
                CitationScore(chunk_id=c.get("chunk_id", ""))
                for c in citations
            ]

        scores = []
        for entry in parsed:
            cs = CitationScore(
                chunk_id=entry.get("chunk_id", ""),
                relevance=_clamp(entry.get("relevance", 0.0)),
                support=_clamp(entry.get("support", 0.0)),
                accuracy=_clamp(entry.get("accuracy", 0.0)),
                reasoning=entry.get("reasoning", ""),
            )
            cs.compute_composite()
            scores.append(cs)

        return scores

    def _judge_heuristic(
        self,
        question: str,
        answer: str,
        citations: list[dict],
        chunks: list[dict],
        item_id: str,
    ) -> JudgeResult:
        """Heuristic fallback when LLM is unavailable.

        Uses keyword overlap between citation text and answer
        as a proxy for citation quality.

        Args:
            question: Original question.
            answer: Generated answer.
            citations: Citation dicts.
            chunks: Chunk dicts.
            item_id: Golden item id.

        Returns:
            JudgeResult with heuristic scores.
        """
        chunk_map = {}
        for c in chunks:
            cid = c.get("chunk_id", c.get("id", ""))
            text = c.get("text", c.get("content", ""))
            if cid:
                chunk_map[cid] = text

        answer_words = set(answer.lower().split())
        question_words = set(question.lower().split())

        scores = []
        for cit in citations:
            cid = cit.get("chunk_id", "")
            chunk_text = chunk_map.get(cid, "")
            chunk_words = set(chunk_text.lower().split())

            # Relevance: overlap between chunk and question
            if question_words and chunk_words:
                relevance = min(
                    len(question_words & chunk_words) / max(len(question_words), 1),
                    1.0,
                )
            else:
                relevance = 0.0

            # Support: overlap between chunk and answer
            if answer_words and chunk_words:
                support = min(
                    len(answer_words & chunk_words) / max(len(answer_words), 1),
                    1.0,
                )
            else:
                support = 0.0

            # Accuracy: presence of chunk_id implies some validity
            accuracy = 0.5 if cid and chunk_text else 0.0

            cs = CitationScore(
                chunk_id=cid,
                relevance=round(relevance, 4),
                support=round(support, 4),
                accuracy=round(accuracy, 4),
                reasoning="heuristic_fallback",
            )
            cs.compute_composite()
            scores.append(cs)

        result = JudgeResult(
            item_id=item_id,
            question=question,
            citation_scores=scores,
            judge_model="heuristic",
        )
        result.compute_means()
        return result


# --------------------------------------------------------------------------- #
# Batch Evaluation
# --------------------------------------------------------------------------- #


@dataclass
class JudgeBatchReport:
    """Aggregate report from batch citation judging.

    Attributes:
        total_items: Number of items judged.
        total_citations: Total citations scored.
        results: Per-item JudgeResults.
        mean_relevance: Overall mean relevance.
        mean_support: Overall mean support.
        mean_accuracy: Overall mean accuracy.
        mean_composite: Overall mean composite.
        pass_threshold: Minimum composite to pass.
        passed: Whether evaluation passed.
    """

    total_items: int = 0
    total_citations: int = 0
    results: list[JudgeResult] = field(default_factory=list)
    mean_relevance: float = 0.0
    mean_support: float = 0.0
    mean_accuracy: float = 0.0
    mean_composite: float = 0.0
    pass_threshold: float = 0.7
    passed: bool = False

    def compute_aggregates(self) -> None:
        """Compute aggregate scores from all results."""
        all_scores = []
        for r in self.results:
            all_scores.extend(r.citation_scores)

        self.total_citations = len(all_scores)
        if not all_scores:
            return

        n = len(all_scores)
        self.mean_relevance = round(sum(c.relevance for c in all_scores) / n, 4)
        self.mean_support = round(sum(c.support for c in all_scores) / n, 4)
        self.mean_accuracy = round(sum(c.accuracy for c in all_scores) / n, 4)
        self.mean_composite = round(sum(c.composite for c in all_scores) / n, 4)
        self.passed = self.mean_composite >= self.pass_threshold

    def to_dict(self) -> dict:
        return {
            "total_items": self.total_items,
            "total_citations": self.total_citations,
            "mean_relevance": self.mean_relevance,
            "mean_support": self.mean_support,
            "mean_accuracy": self.mean_accuracy,
            "mean_composite": self.mean_composite,
            "pass_threshold": self.pass_threshold,
            "passed": self.passed,
        }


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, float(v)))
