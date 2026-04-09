"""
Writer Agent — synthesises search results into a structured draft report.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.state import ResearchState

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

SYSTEM_PROMPT = """You are an expert research writer.
Your task is to synthesise raw research findings into a well-structured report.

Report structure:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points, grouped by theme)
3. Detailed Analysis (2-3 paragraphs)
4. Conclusions & Recommendations
5. Sources Used

Write in a clear, professional tone. Do not hallucinate — only use the
information provided in the research findings."""


def writer_node(state: ResearchState) -> dict:
    """
    LangGraph node: drafts a research report from search_results.
    Returns: {'draft_report': str}
    """
    if state.get("error"):
        return {}

    results = state.get("search_results") or []
    combined = "\n\n---\n\n".join(results) if results else "No search results available."

    logger.info("[Writer] Drafting report from %d result chunk(s)", len(results))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research query: {state['query']}\n\n"
                f"Research plan followed:\n{state.get('plan', 'N/A')}\n\n"
                f"Gathered findings:\n{combined}"
            )
        ),
    ]

    try:
        response = _llm.invoke(messages)
        draft = response.content
        logger.info("[Writer] Draft created (%d chars)", len(draft))
        return {"draft_report": draft}
    except Exception as exc:
        logger.error("[Writer] Failed: %s", exc)
        return {"error": f"Writer failed: {exc}"}
