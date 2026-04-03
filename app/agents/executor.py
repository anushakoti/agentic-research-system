from app.tools.search import search_web
from app.memory.vector_store import store_docs
from app.utils.logger import get_logger
from app.utils.tracer import trace_step

logger = get_logger("EXECUTOR")

MAX_STEPS = 5

@trace_step("Executor Agent")
def execute_plan(plan: str):
    steps = plan.split("\n")
    collected_data = []

    for i, step in enumerate(steps):
        if i >= MAX_STEPS:
            break

        logger.info(f"Step: {step}")

        if "search" in step.lower():
            try:
                results = search_web(step)
                collected_data.extend(results)
                store_docs(results)

                logger.info(f"Results: {results[:2]}")

            except Exception as e:
                logger.error(f"Error: {e}")

    return collected_data