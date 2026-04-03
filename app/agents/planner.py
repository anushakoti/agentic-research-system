from langchain.chat_models import ChatOpenAI
from app.utils.logger import get_logger
from app.utils.tracer import trace_step

logger = get_logger("PLANNER")
llm = ChatOpenAI()

@trace_step("Planner Agent")
def create_plan(query: str):
    logger.info(f"Query: {query}")

    prompt = f"""
    Break the task into steps:
    {query}
    Include search steps.
    """

    plan = llm.predict(prompt)
    logger.info(f"Plan:\n{plan}")

    return plan