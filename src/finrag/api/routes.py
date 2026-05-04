"""API routes for the FinRAG pipeline.

Endpoints:
    POST /api/v1/query         -- Synchronous JSON response
    POST /api/v1/query/stream  -- SSE streaming response
    GET  /api/v1/sessions/{id} -- Session state inspection
    DELETE /api/v1/sessions/{id} -- Clear a session
    GET  /api/v1/config/prompts -- Active prompt versions

Design decisions:
- SSE over WebSocket: simpler, works through proxies, sufficient
  for server-to-client streaming. Client never streams back.
- Session ID auto-generation: if client omits session_id, we
  generate one. Simplifies client while enabling multi-turn.
- Streaming granularity: structured events at each pipeline stage
  so clients can show progressive UI. Token-level streaming needs
  LangChain streaming callbacks (tracked as DAY-11-002).

Debt: DAY-11-002 -- SSE simulates chunking of final answer. True
      token streaming requires LangChain callbacks. Add Day 12.
"""

import asyncio
import json
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from finrag.orchestration.memory import SessionStore
from finrag.orchestration.prompt_config import get_active_prompt_version

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Request/Response Models
# --------------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    """Request body for /query endpoint.

    Attributes:
        query: Natural language financial question.
        session_id: Optional session ID for multi-turn context.
        metadata_filter: Optional retrieval filter.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Financial research question",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for multi-turn. Auto-generated if missing.",
    )
    metadata_filter: dict | None = Field(
        default=None,
        description="Metadata filter for retrieval, e.g. {ticker: AAPL}",
    )


class CitationResponse(BaseModel):
    """Single citation in the response.

    Attributes:
        chunk_id: Source chunk identifier.
        filing_reference: Human-readable filing reference.
        section: Filing section.
        page: Page number in filing.
        relevance_score: Reranker score.
    """

    chunk_id: str = ""
    filing_reference: str = ""
    section: str = ""
    page: int | None = None
    relevance_score: float = 0.0


class QueryResponse(BaseModel):
    """Response body for /query endpoint.

    Attributes:
        answer: Generated answer text.
        citations: Supporting citations.
        session_id: Session ID used.
        confidence: Overall confidence score.
        route: Pipeline route taken.
        prompt_version: Prompt config version used.
        metadata: Pipeline metadata.
    """

    answer: str = ""
    citations: list[CitationResponse] = []
    session_id: str = ""
    confidence: float = 0.0
    route: str = ""
    prompt_version: str = ""
    metadata: dict = {}


class SessionResponse(BaseModel):
    """Response body for session inspection.

    Attributes:
        session_id: The session identifier.
        turn_count: Conversation turns so far.
        entities: Entities discussed across turns.
        filings: Filing types referenced.
        periods: Time periods mentioned.
        cited_chunks: Unique chunks cited count.
    """

    session_id: str
    turn_count: int = 0
    entities: list[str] = []
    filings: list[str] = []
    periods: list[str] = []
    cited_chunks: int = 0


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/api/v1", tags=["FinRAG"])


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


def get_session_store(request: Request) -> SessionStore:
    """Get SessionStore from app state.

    Args:
        request: Incoming request.

    Returns:
        Shared SessionStore instance.
    """
    return request.app.state.session_store


def get_compiled_graph(request: Request):
    """Get compiled RAG graph from app state.

    Args:
        request: Incoming request.

    Returns:
        Compiled LangGraph pipeline or None.
    """
    return getattr(request.app.state, "compiled_graph", None)


# --------------------------------------------------------------------------- #
# POST /query
# --------------------------------------------------------------------------- #


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    request: Request,
    session_store: SessionStore = Depends(get_session_store),
    compiled_graph=Depends(get_compiled_graph),
) -> QueryResponse:
    """Run a financial research query through the RAG pipeline.

    Executes the full pipeline synchronously. If session_id provided,
    conversation history is injected for multi-turn context.

    Args:
        body: Query request body.
        request: HTTP request.
        session_store: Shared session store.
        compiled_graph: Compiled RAG graph.

    Returns:
        QueryResponse with answer, citations, metadata.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    session_id = body.session_id or str(uuid.uuid4())

    logger.info(
        "query_received",
        request_id=request_id,
        query_preview=body.query[:80],
        session_id=session_id,
        has_filter=body.metadata_filter is not None,
    )

    session = session_store.get_or_create(session_id)
    resolved_query = session.resolve_references(body.query)
    conversation_history = session.get_conversation_history(max_turns=5)

    if compiled_graph is not None:
        try:
            result = await asyncio.to_thread(
                compiled_graph.invoke,
                {
                    "query": resolved_query,
                    "metadata_filter": body.metadata_filter,
                    "conversation_history": conversation_history,
                    "step_count": 0,
                    "max_steps": 15,
                    "messages": [],
                },
            )
        except Exception as e:
            logger.error("pipeline_error", error=str(e), request_id=request_id)
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e!s}") from e
    else:
        result = {
            "answer": "Pipeline not initialized. This is a stub response.",
            "citations": [],
            "route": "stub",
            "is_valid": True,
        }

    answer = result.get("answer", "")
    citations = result.get("citations", [])
    session.add_turn(
        query=body.query,
        answer=answer,
        citations=citations if isinstance(citations, list) else [],
        metadata_filter=body.metadata_filter,
    )

    citation_responses = []
    for c in citations:
        if isinstance(c, dict):
            citation_responses.append(
                CitationResponse(
                    chunk_id=c.get("chunk_id", ""),
                    filing_reference=c.get("filing_reference", ""),
                    section=c.get("section", ""),
                    page=c.get("page"),
                    relevance_score=c.get("relevance_score", 0.0),
                )
            )

    prompt_versions = get_active_prompt_version()

    return QueryResponse(
        answer=answer,
        citations=citation_responses,
        session_id=session_id,
        confidence=result.get("route_confidence", 0.0),
        route=result.get("route", "unknown"),
        prompt_version=prompt_versions.get("generation", "unknown"),
        metadata={
            "request_id": request_id,
            "step_count": result.get("step_count", 0),
            "is_valid": result.get("is_valid", False),
            "input_blocked": result.get("input_guard_blocked", False),
            "output_blocked": result.get("output_guard_blocked", False),
        },
    )


# --------------------------------------------------------------------------- #
# POST /query/stream -- SSE
# --------------------------------------------------------------------------- #


@router.post("/query/stream")
async def query_stream_endpoint(
    body: QueryRequest,
    request: Request,
    session_store: SessionStore = Depends(get_session_store),
    compiled_graph=Depends(get_compiled_graph),
):
    """Stream a query response via Server-Sent Events.

    Emits structured events at each pipeline stage:
    retrieval_start, chunks_found, rerank_done,
    generation_start, answer_chunk, citation, done.

    Args:
        body: Query request body.
        request: HTTP request.
        session_store: Shared session store.
        compiled_graph: Compiled RAG graph.

    Returns:
        EventSourceResponse with SSE event stream.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    session_id = body.session_id or str(uuid.uuid4())

    logger.info(
        "stream_query_received",
        request_id=request_id,
        query_preview=body.query[:80],
        session_id=session_id,
    )

    async def event_generator():
        """Generate SSE events for the query pipeline."""
        session = session_store.get_or_create(session_id)
        resolved_query = session.resolve_references(body.query)
        conversation_history = session.get_conversation_history(max_turns=5)

        yield {
            "event": "retrieval_start",
            "data": json.dumps({"query": body.query, "session_id": session_id}),
        }

        if compiled_graph is not None:
            try:
                result = await asyncio.to_thread(
                    compiled_graph.invoke,
                    {
                        "query": resolved_query,
                        "metadata_filter": body.metadata_filter,
                        "conversation_history": conversation_history,
                        "step_count": 0,
                        "max_steps": 15,
                        "messages": [],
                    },
                )
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}),
                }
                return
        else:
            result = {
                "answer": "Pipeline not initialized. Stub response.",
                "citations": [],
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "route": "stub",
            }

        retrieved = result.get("retrieved_chunks", [])
        yield {
            "event": "chunks_found",
            "data": json.dumps({"count": len(retrieved)}),
        }

        reranked = result.get("reranked_chunks", [])
        yield {
            "event": "rerank_done",
            "data": json.dumps({"count": len(reranked)}),
        }

        yield {
            "event": "generation_start",
            "data": json.dumps({"route": result.get("route", "unknown")}),
        }

        answer = result.get("answer", "")
        chunk_size = 80
        for i in range(0, max(len(answer), 1), chunk_size):
            chunk = answer[i : i + chunk_size]
            yield {
                "event": "answer_chunk",
                "data": json.dumps({"text": chunk, "index": i // chunk_size}),
            }
            await asyncio.sleep(0.02)

        citations = result.get("citations", [])
        for c in citations:
            if isinstance(c, dict):
                yield {"event": "citation", "data": json.dumps(c)}

        session.add_turn(
            query=body.query,
            answer=answer,
            citations=citations if isinstance(citations, list) else [],
            metadata_filter=body.metadata_filter,
        )

        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "session_id": session_id,
                    "route": result.get("route", "unknown"),
                    "is_valid": result.get("is_valid", False),
                    "total_citations": len(citations),
                    "request_id": request_id,
                }
            ),
        }

    return EventSourceResponse(event_generator())


# --------------------------------------------------------------------------- #
# GET /sessions/{session_id}
# --------------------------------------------------------------------------- #


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> SessionResponse:
    """Get the state of a conversation session.

    Args:
        session_id: The session identifier.
        session_store: Shared session store.

    Returns:
        SessionResponse with session state.

    Raises:
        HTTPException: 404 if session not found.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    state = session.to_dict()
    return SessionResponse(
        session_id=state["session_id"],
        turn_count=state["turn_count"],
        entities=state["entities"],
        filings=state["filings"],
        periods=state["periods"],
        cited_chunks=state["cited_chunks"],
    )


# --------------------------------------------------------------------------- #
# DELETE /sessions/{session_id}
# --------------------------------------------------------------------------- #


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    """Delete a conversation session.

    Args:
        session_id: The session identifier.
        session_store: Shared session store.

    Returns:
        Confirmation dict.

    Raises:
        HTTPException: 404 if session not found.
    """
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return {"detail": f"Session '{session_id}' deleted", "session_id": session_id}


# --------------------------------------------------------------------------- #
# GET /config/prompts
# --------------------------------------------------------------------------- #


@router.get("/config/prompts")
async def get_prompt_config() -> dict:
    """Return active prompt configuration versions.

    Returns:
        Dict with generation and retrieval prompt versions.
    """
    versions = get_active_prompt_version()
    return {
        "prompt_versions": versions,
        "status": "loaded" if versions.get("generation") != "not_loaded" else "not_loaded",
    }
