"""FastAPI application factory for the FinRAG pipeline.

Creates and configures the FastAPI app with middleware, routes,
and shared resources. Factory pattern supports different configs
for test vs production.

Startup sequence:
    1. Initialize SessionStore for conversation memory
    2. Load versioned prompt configs from YAML
    3. (Optional) Initialize HybridRetriever, Reranker, RAGGenerator
    4. Compile the LangGraph pipeline
    5. Store everything in app.state for route access

Heavy initialization (embeddings, indexes, LLM clients) only
happens when FINRAG_INIT_PIPELINE=true. In test mode, we skip
pipeline init and use stub responses.

Usage:
    uvicorn finrag.api.app:create_app --factory --reload
"""

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finrag.api.middleware import (
    AuthMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
)
from finrag.api.mcp_server import mcp_router
from finrag.api.routes import router as api_router
from finrag.orchestration.memory import SessionStore
from finrag.orchestration.prompt_config import load_generation_config, load_retrieval_config

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Application Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic.

    Startup:
        - Create SessionStore
        - Load prompt configs
        - Optionally initialize the full RAG pipeline

    Shutdown:
        - Log shutdown

    Args:
        app: The FastAPI application instance.
    """
    logger.info("finrag_api_starting")

    # --- Session Store ---
    max_sessions = int(os.environ.get("FINRAG_MAX_SESSIONS", "1000"))
    app.state.session_store = SessionStore(max_sessions=max_sessions)

    # --- Prompt Configs ---
    prompt_version = os.environ.get("FINRAG_PROMPT_VERSION", "v1")
    try:
        gen_config = load_generation_config(version=prompt_version)
        ret_config = load_retrieval_config(version=prompt_version)
        logger.info(
            "prompt_configs_loaded",
            generation_version=gen_config.version,
            retrieval_version=ret_config.version,
        )
    except Exception as e:
        logger.warning("prompt_config_load_failed", error=str(e))

    # --- Pipeline Initialization (optional) ---
    init_pipeline = os.environ.get("FINRAG_INIT_PIPELINE", "false").lower() == "true"

    if init_pipeline:
        try:
            from finrag.orchestration.generator import RAGGenerator
            from finrag.orchestration.graph import compile_rag_graph
            from finrag.retrieval.hybrid import HybridRetriever
            from finrag.retrieval.reranker import CrossEncoderReranker

            hybrid_retriever = HybridRetriever()
            reranker = CrossEncoderReranker()
            rag_generator = RAGGenerator()

            app.state.compiled_graph = compile_rag_graph(
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
                rag_generator=rag_generator,
            )
            logger.info("rag_pipeline_initialized")

        except Exception as e:
            logger.error("pipeline_init_failed", error=str(e))
            app.state.compiled_graph = None
    else:
        app.state.compiled_graph = None
        logger.info("pipeline_init_skipped", reason="FINRAG_INIT_PIPELINE != true")

    logger.info(
        "finrag_api_ready",
        pipeline_active=app.state.compiled_graph is not None,
        max_sessions=max_sessions,
        prompt_version=prompt_version,
    )

    yield

    logger.info("finrag_api_shutdown")


# --------------------------------------------------------------------------- #
# Application Factory
# --------------------------------------------------------------------------- #


def create_app(
    api_key: str | None = None,
    max_requests_per_minute: int = 60,
    enable_auth: bool = True,
    enable_rate_limit: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        api_key: API key for auth. Reads FINRAG_API_KEY if None.
        max_requests_per_minute: Rate limit per client IP.
        enable_auth: Enable auth middleware.
        enable_rate_limit: Enable rate limiting.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="FinRAG",
        description=(
            "Citation-enforced financial research assistant over SEC filings. "
            "Every answer is grounded in specific paragraphs from specific filings."
        ),
        version="0.11.0",
        lifespan=lifespan,
    )

    # --- CORS ---
    allowed_origins = os.environ.get("FINRAG_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Middleware Stack ---
    # Applied in reverse: last add_middleware is outermost.
    if enable_rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            max_requests=max_requests_per_minute,
            window_seconds=60,
        )

    if enable_auth:
        app.add_middleware(AuthMiddleware, api_key=api_key)

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)

    # --- Routes ---
    app.include_router(api_router)
    app.include_router(mcp_router)

    # --- Health Check ---
    @app.get("/healthz", tags=["Health"])
    async def health_check() -> dict:
        """Health check for load balancers.

        Returns:
            Status dict with pipeline state.
        """
        pipeline_active = (
            hasattr(app.state, "compiled_graph") and app.state.compiled_graph is not None
        )
        session_count = (
            app.state.session_store.active_count
            if hasattr(app.state, "session_store")
            else 0
        )
        return {
            "status": "healthy",
            "pipeline_active": pipeline_active,
            "active_sessions": session_count,
            "version": "0.11.0",
        }

    logger.info(
        "fastapi_app_created",
        auth_enabled=enable_auth,
        rate_limit_enabled=enable_rate_limit,
        max_rpm=max_requests_per_minute,
    )

    return app
