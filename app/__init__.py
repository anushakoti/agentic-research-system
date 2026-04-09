"""
app package — Agentic Research System core modules.

Exposes the compiled LangGraph instance so api.py can do:
    from app import graph
"""

from app.graph import graph

__all__ = ["graph"]
