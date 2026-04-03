from langchain.chat_models import ChatOpenAI
from app.utils.logger import get_logger
from app.utils.tracer import trace_step

logger = get_logger("WRITER")
llm = ChatOpenAI()

@trace_step("Writer Agent")
def generate_report(query, data):
    context = "\n".join(data)

    prompt = f"""
    Write a structured report on:
    {query}

    Data:
    {context}

    Include:
    - Introduction
    - Key Insights
    - Conclusion
    """

    report = llm.predict(prompt)

    logger.info("Report generated")
    return report