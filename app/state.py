"""
Shared state for the LangGraph multi-agent research workflow.

Interview note: Using TypedDict keeps the state schema explicit and
easy to reason about — every agent reads/writes clearly named fields.
"""

from typing import List, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    """
    The single source of truth passed between all graph nodes.

    Fields:
        query           - The original user research question.
        plan            - Planner's step-by-step research plan.
        search_results  - Raw results collected by the Executor agent.
        draft_report    - Initial report written by the Writer agent.
        reviewed_report - Final polished output from the Reviewer agent.
        error           - Optional error message for graceful failure paths.
    """

    query: str
    plan: Optional[str]
    search_results: Optional[List[str]]
    draft_report: Optional[str]
    reviewed_report: Optional[str]
    error: Optional[str]
