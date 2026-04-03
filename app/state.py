from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    plan: str
    data: List[str]
    report: str
    reviewed_report: str