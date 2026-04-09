"""
LangGraph workflow — wires the four agents into a directed graph.
"""

import logging

from langgraph.graph import StateGraph, END

from app.state import ResearchState
from app.agents import planner_node, executor_node, writer_node, reviewer_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge: skip to reviewer if an error occurred upstream
# ---------------------------------------------------------------------------

def route_after_executor(state: ResearchState) -> str:
    """
    If the executor (or planner) set an error, jump straight to the reviewer
    so the API always gets a response — never a 500.
    """
    if state.get("error"):
        logger.warning("Error detected — routing directly to reviewer.")
        return "reviewer"
    return "writer"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    workflow = StateGraph(ResearchState)

    # Register nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    # Entry point
    workflow.set_entry_point("planner")

    # Edges
    workflow.add_edge("planner", "executor")

    # Conditional: on error skip writer and go straight to reviewer
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "writer": "writer",
            "reviewer": "reviewer",
        },
    )

    workflow.add_edge("writer", "reviewer")
    workflow.add_edge("reviewer", END)

    return workflow


# Compile once at import time — reused across all requests
graph = build_graph().compile()
