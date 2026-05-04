"""MCP (Model Context Protocol) tool server for FinRAG.

Exposes the RAG pipeline as callable tools for LLM agents.
External agents can call FinRAG to answer financial questions
with citations.

Tool catalog:
    query_financial_data   -- Ask a question, get a cited answer
    get_session_context    -- Get accumulated entities/filings for a session
    list_available_tickers -- List tickers with indexed filings

Design decisions:
- JSON-RPC style interface: each tool accepts a dict and returns a dict.
  Maps directly to MCP tool calling convention.
- Mounted as FastAPI sub-router: shares app instance, session store,
  compiled graph. No separate process.
- Schema descriptions follow MCP conventions: name, description,
  input_schema for automatic discovery.
"""

import uuid

import structlog
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from finrag.orchestration.memory import SessionStore

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Tool Definitions (MCP Schema)
# --------------------------------------------------------------------------- #

TOOL_DEFINITIONS = [
    {
        "name": "query_financial_data",
        "description": (
            "Ask a question about SEC filings (10-K, 10-Q, 8-K) and earnings "
            "call transcripts. Returns a citation-grounded answer with references "
            "to specific filing sections and pages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The financial research question to answer",
                },
                "ticker": {
                    "type": "string",
                    "description": "Optional stock ticker to filter by (e.g. AAPL)",
                },
                "filing_type": {
                    "type": "string",
                    "description": "Optional filing type filter (10-K, 10-Q, 8-K)",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID for multi-turn context",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_session_context",
        "description": (
            "Get the accumulated context for a conversation session: "
            "discussed entities, filing types, time periods, turn count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to inspect",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "list_available_tickers",
        "description": (
            "List stock tickers with indexed filings available for querying."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# --------------------------------------------------------------------------- #
# Request/Response Models
# --------------------------------------------------------------------------- #


class MCPToolCallRequest(BaseModel):
    """MCP tool call request.

    Attributes:
        name: Tool name to invoke.
        arguments: Tool-specific arguments dict.
        call_id: Optional call identifier for correlation.
    """

    name: str = Field(..., description="Tool name to call")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")
    call_id: str = Field(default="", description="Call ID for correlation")


class MCPToolCallResponse(BaseModel):
    """MCP tool call response.

    Attributes:
        call_id: Correlates with request call_id.
        result: Tool execution result.
        error: Error message if tool failed.
    """

    call_id: str = ""
    result: dict = {}
    error: str | None = None


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

mcp_router = APIRouter(prefix="/mcp", tags=["MCP Tools"])


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
        Compiled graph or None.
    """
    return getattr(request.app.state, "compiled_graph", None)


# --------------------------------------------------------------------------- #
# GET /mcp/tools
# --------------------------------------------------------------------------- #


@mcp_router.get("/tools")
async def list_tools() -> dict:
    """List all available MCP tools with schemas.

    Returns:
        Dict with tool definitions following MCP convention.
    """
    return {"tools": TOOL_DEFINITIONS}


# --------------------------------------------------------------------------- #
# POST /mcp/call
# --------------------------------------------------------------------------- #


@mcp_router.post("/call", response_model=MCPToolCallResponse)
async def call_tool(
    body: MCPToolCallRequest,
    request: Request,
    session_store: SessionStore = Depends(get_session_store),
    compiled_graph=Depends(get_compiled_graph),
) -> MCPToolCallResponse:
    """Execute an MCP tool call.

    Dispatches to the appropriate handler based on tool name.

    Args:
        body: Tool call request with name and arguments.
        request: HTTP request.
        session_store: Shared session store.
        compiled_graph: Compiled RAG graph.

    Returns:
        MCPToolCallResponse with result or error.
    """
    call_id = body.call_id or str(uuid.uuid4())

    logger.info(
        "mcp_tool_call",
        tool=body.name,
        call_id=call_id,
        has_args=bool(body.arguments),
    )

    try:
        if body.name == "query_financial_data":
            result = await _tool_query_financial_data(
                body.arguments, session_store, compiled_graph
            )
        elif body.name == "get_session_context":
            result = _tool_get_session_context(body.arguments, session_store)
        elif body.name == "list_available_tickers":
            result = _tool_list_available_tickers()
        else:
            return MCPToolCallResponse(
                call_id=call_id,
                error=f"Unknown tool: {body.name}. Available: "
                f"{[t['name'] for t in TOOL_DEFINITIONS]}",
            )

        return MCPToolCallResponse(call_id=call_id, result=result)

    except Exception as e:
        logger.error("mcp_tool_error", tool=body.name, error=str(e))
        return MCPToolCallResponse(call_id=call_id, error=str(e))


# --------------------------------------------------------------------------- #
# Tool Implementations
# --------------------------------------------------------------------------- #


async def _tool_query_financial_data(
    args: dict,
    session_store: SessionStore,
    compiled_graph,
) -> dict:
    """Execute the query_financial_data tool.

    Args:
        args: Tool arguments (query, ticker, filing_type, session_id).
        session_store: Session store for multi-turn context.
        compiled_graph: The compiled RAG pipeline.

    Returns:
        Dict with answer, citations, session info.
    """
    import asyncio

    query = args.get("query", "")
    if not query:
        return {"error": "Missing required argument: query"}

    session_id = args.get("session_id", str(uuid.uuid4()))
    session = session_store.get_or_create(session_id)

    metadata_filter: dict | None = None
    ticker = args.get("ticker")
    filing_type = args.get("filing_type")
    if ticker or filing_type:
        metadata_filter = {}
        if ticker:
            metadata_filter["ticker"] = ticker.upper()
        if filing_type:
            metadata_filter["filing_type"] = filing_type.upper()

    resolved_query = session.resolve_references(query)
    conversation_history = session.get_conversation_history(max_turns=3)

    if compiled_graph is not None:
        result = await asyncio.to_thread(
            compiled_graph.invoke,
            {
                "query": resolved_query,
                "metadata_filter": metadata_filter,
                "conversation_history": conversation_history,
                "step_count": 0,
                "max_steps": 15,
                "messages": [],
            },
        )
    else:
        result = {
            "answer": "Pipeline not initialized. Stub response for tool call.",
            "citations": [],
            "route": "stub",
        }

    answer = result.get("answer", "")
    citations = result.get("citations", [])

    session.add_turn(
        query=query,
        answer=answer,
        citations=citations if isinstance(citations, list) else [],
        metadata_filter=metadata_filter,
    )

    return {
        "answer": answer,
        "citations": citations,
        "session_id": session_id,
        "route": result.get("route", "unknown"),
    }


def _tool_get_session_context(args: dict, session_store: SessionStore) -> dict:
    """Execute the get_session_context tool.

    Args:
        args: Tool arguments (session_id).
        session_store: Session store.

    Returns:
        Dict with session context.
    """
    session_id = args.get("session_id", "")
    if not session_id:
        return {"error": "Missing required argument: session_id"}

    session = session_store.get(session_id)
    if session is None:
        return {"error": f"Session '{session_id}' not found"}

    return session.to_dict()


def _tool_list_available_tickers() -> dict:
    """Execute the list_available_tickers tool.

    Returns placeholder list. Production would query ChromaDB.

    Returns:
        Dict with available tickers.
    """
    return {
        "tickers": [],
        "note": "Connect to vector store for actual ticker list.",
    }
