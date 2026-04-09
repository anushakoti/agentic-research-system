"""
agents package — individual LangGraph node functions.

Each node is a plain function:
    (state: ResearchState) -> dict

Import them here so graph.py only needs:
    from app.agents import planner_node, executor_node, writer_node, reviewer_node
"""

from app.agents.planner import planner_node
from app.agents.executor import executor_node
from app.agents.writer import writer_node
from app.agents.reviewer import reviewer_node

__all__ = [
    "planner_node",
    "executor_node",
    "writer_node",
    "reviewer_node",
]
