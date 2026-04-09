"""
Reviewer Agent — quality-gates the draft and returns the final report.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.state import ResearchState

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPT = """You are a senior research editor and fact-checker.
Your job is to review a draft research report and improve it:

1. Fix any factual inconsistencies or unsupported claims.
2. Improve clarity and flow.
3. Ensure the report directly answers the original research query.
4. Add a short 'Quality Review Note' at the end summarising what was changed.

Return the complete, polished final report."""


def reviewer_node(state: ResearchState) -> dict:
    """
    LangGraph node: reviews and finalises the draft report.
    Returns: {'reviewed_report': str}
    """
    if state.get("error"):
        # Return a graceful error report instead of crashing
        error_msg = state["error"]
        logger.warning("[Reviewer] Skipping review due to upstream error: %s", error_msg)
        return {
            "reviewed_report": (
                f"Research could not be completed due to an error: {error_msg}\n"
                "Please try again or refine your query."
            )
        }

    draft = state.get("draft_report", "")
    if not draft:
        return {"reviewed_report": "No draft was produced to review."}

    logger.info("[Reviewer] Reviewing draft (%d chars)", len(draft))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original research query: {state['query']}\n\n"
                f"Draft report to review:\n{draft}"
            )
        ),
    ]

    try:
        response = _llm.invoke(messages)
        reviewed = response.content
        logger.info("[Reviewer] Final report ready (%d chars)", len(reviewed))
        return {"reviewed_report": reviewed}
    except Exception as exc:
        logger.error("[Reviewer] Failed: %s", exc)
        return {
            "reviewed_report": (
                f"Reviewer failed ({exc}). Returning unreviewed draft:\n\n{draft}"
            )
        }
