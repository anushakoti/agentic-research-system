import time
import uuid
from app.utils.logger import get_logger

logger = get_logger("TRACE")

def trace_step(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())[:8]
            start = time.time()

            logger.info(f"[START] {step_name} | trace_id={trace_id}")

            try:
                result = func(*args, **kwargs)
                duration = round(time.time() - start, 2)

                logger.info(
                    f"[END] {step_name} | trace_id={trace_id} | duration={duration}s"
                )

                return result

            except Exception as e:
                logger.error(
                    f"[ERROR] {step_name} | trace_id={trace_id} | error={str(e)}"
                )
                raise

        return wrapper
    return decorator