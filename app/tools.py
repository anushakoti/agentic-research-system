"""
Tool definitions used by the Executor agent.

Interview note: Using @tool decorator from LangChain is the idiomatic way
to define tools — it auto-generates the schema the LLM uses for tool-calling.
Tavily is the recommended search tool for LangChain agentic apps.
"""

import logging
import os
from typing import List

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Web Search Tool
# ---------------------------------------------------------------------------

_tavily = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
)


@tool
def web_search(query: str) -> List[str]:
    """
    Search the web for up-to-date information on a given query.
    Returns a list of result snippets ranked by relevance.
    """
    try:
        results = _tavily.invoke(query)
        # Each result is a dict with 'content' and 'url'
        return [f"{r['content']} (source: {r['url']})" for r in results]
    except Exception as exc:
        logger.error("web_search failed: %s", exc)
        return [f"Search failed: {exc}"]


# ---------------------------------------------------------------------------
# FAISS In-Memory Vector Store (for context retrieval within a session)
# ---------------------------------------------------------------------------

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

_embeddings = OpenAIEmbeddings()
_vector_store: FAISS | None = None


def store_in_memory(texts: List[str]) -> None:
    """Upsert text chunks into the in-process FAISS vector store."""
    global _vector_store
    docs = [Document(page_content=t) for t in texts]
    if _vector_store is None:
        _vector_store = FAISS.from_documents(docs, _embeddings)
    else:
        _vector_store.add_documents(docs)


@tool
def memory_search(query: str, k: int = 3) -> List[str]:
    """
    Retrieve the most relevant passages from previously gathered research.
    Use this before doing a fresh web search to avoid duplicate calls.
    """
    if _vector_store is None:
        return ["No prior research in memory yet."]
    results = _vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
