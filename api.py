from fastapi import FastAPI
from app.graph import app
from app.utils.logger import get_logger

logger = get_logger("API")

api = FastAPI()

@api.get("/")
def home():
    return {"message": "Agentic AI system running"}

@api.get("/research")
def research(query: str):
    logger.info(f"Request: {query}")

    result = app.invoke({"query": query})

    return {"output": result["reviewed_report"]}