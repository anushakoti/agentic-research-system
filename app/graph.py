from langgraph.graph import StateGraph
from app.state import AgentState

from app.agents.planner import create_plan
from app.agents.executor import execute_plan
from app.agents.writer import generate_report
from app.agents.reviewer import review_report


def planner_node(state):
    return {"plan": create_plan(state["query"])}

def executor_node(state):
    return {"data": execute_plan(state["plan"])}

def writer_node(state):
    return {"report": generate_report(state["query"], state["data"])}

def reviewer_node(state):
    return {"reviewed_report": review_report(state["report"])}


workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "writer")
workflow.add_edge("writer", "reviewer")

workflow.set_finish_point("reviewer")

app = workflow.compile()