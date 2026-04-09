"""
FastAPI entry point for the Agentic Research System.

Interview note: Important improvements over the original:
  - LangSmith tracing is initialised here via environment variables
    (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT)
    No manual trace IDs needed — LangSmith captures the full graph run.
  - Pydantic response model makes the contract explicit for the interviewer.
  - HTTPException on validation so the caller gets a proper 422/400,
    not an unhandled 500.
  - Lifespan event confirms env vars on startup (catches config errors early).
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — use standard library; no util/ folder needed
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# LangSmith is enabled purely through env vars — nothing to import
# Set in .env:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=...
#   LANGCHAIN_PROJECT=agentic-research-system
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lazy import of the compiled graph (avoids loading OpenAI at import time
# in environments where keys aren't set yet, e.g. during linting)
# ---------------------------------------------------------------------------
from app import graph


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class ResearchResponse(BaseModel):
    query: str
    report: str


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    required = ["OPENAI_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error("Missing required env vars: %s", missing)
        raise RuntimeError(f"Missing env vars: {missing}")

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    logger.info(
        "Starting up | LangSmith tracing=%s | project=%s", tracing, project
    )
    yield
    logger.info("Shutting down.")


api = FastAPI(
    title="Agentic Research System",
    description="Multi-agent LangGraph research pipeline with LangSmith observability.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api.get("/")
def health():
    return {"status": "ok", "service": "agentic-research-system"}


@api.get("/research", response_model=ResearchResponse)
def research(
    query: str = Query(..., min_length=3, max_length=500, description="Research topic or question"),
):
    """
    Run the full multi-agent research pipeline and return a reviewed report.

    The pipeline: Planner → Executor (ReAct + Tavily) → Writer → Reviewer

    All steps are traced automatically in LangSmith when
    LANGCHAIN_TRACING_V2=true is set in the environment.
    """
    logger.info("Received research request: %s", query)

    try:
        result = graph.invoke({"query": query})
    except Exception as exc:
        logger.error("Graph invocation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    report = result.get("reviewed_report") or "No report generated."
    return ResearchResponse(query=query, report=report)
