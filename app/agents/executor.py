"""
Executor Agent — runs web searches according to the plan and stores results.
"""

import logging
from typing import List

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from app.state import ResearchState
from app.tools import web_search, memory_search, store_in_memory

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_TOOLS = [web_search, memory_search]

# ReAct prompt template — must contain {tools}, {tool_names}, {input}, {agent_scratchpad}
_PROMPT = PromptTemplate.from_template(
    """You are a thorough research executor. Follow the research plan step-by-step.
Use the available tools to gather information. Check memory before searching the web.
Stop once you have enough information to write a comprehensive report.

Available tools: {tools}
Tool names: {tool_names}

Research plan and query:
{input}

{agent_scratchpad}"""
)

_agent = create_react_agent(llm=_llm, tools=_TOOLS, prompt=_PROMPT)
_executor = AgentExecutor(
    agent=_agent,
    tools=_TOOLS,
    max_iterations=8,
    handle_parsing_errors=True,
    verbose=False,
)


def executor_node(state: ResearchState) -> dict:
    """
    LangGraph node: executes the plan via a ReAct agent and collects results.
    Returns: {'search_results': List[str]}
    """
    if state.get("error"):
        return {}  # propagate error, skip execution

    plan = state.get("plan", "No plan provided.")
    query = state["query"]

    logger.info("[Executor] Running ReAct agent for query: %s", query)

    try:
        result = _executor.invoke(
            {"input": f"Query: {query}\n\nPlan:\n{plan}"}
        )
        raw_output = result.get("output", "")

        # Store gathered info in vector memory for the session
        search_results: List[str] = [raw_output] if raw_output else []
        if search_results:
            store_in_memory(search_results)

        logger.info("[Executor] Collected %d result(s)", len(search_results))
        return {"search_results": search_results}

    except Exception as exc:
        logger.error("[Executor] Failed: %s", exc)
        return {"error": f"Executor failed: {exc}"}
