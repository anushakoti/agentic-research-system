"""
Planner Agent — breaks a research query into an ordered plan.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.state import ResearchState

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPT = """You are a senior research strategist.
Given a research query, produce a concise numbered plan (max 5 steps)
that a research agent should follow to gather comprehensive, accurate information.
Be specific about what to search for in each step."""


def planner_node(state: ResearchState) -> dict:
    """
    LangGraph node: generates a research plan from the user query.
    Returns: {'plan': str}
    """
    logger.info("[Planner] Building plan for: %s", state["query"])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research query: {state['query']}"),
    ]

    try:
        response = _llm.invoke(messages)
        plan = response.content
        logger.info("[Planner] Plan created (%d chars)", len(plan))
        return {"plan": plan}
    except Exception as exc:
        logger.error("[Planner] Failed: %s", exc)
        return {"error": f"Planner failed: {exc}"}
