from langchain.chat_models import ChatOpenAI
from app.utils.logger import get_logger
from app.utils.tracer import trace_step

logger = get_logger("REVIEWER")
llm = ChatOpenAI()

@trace_step("Reviewer Agent")
def review_report(report: str):
    prompt = f"""
    Improve this report:
    - Fix inaccuracies
    - Improve clarity

    Report:
    {report}
    """

    reviewed = llm.predict(prompt)

    logger.info("Review complete")
    return reviewed