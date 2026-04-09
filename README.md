# 🔬 Agentic Research System

A production-grade **multi-agent research pipeline** built with **LangGraph** + **LangChain**, fully observable via **LangSmith**. Given any research query, four specialized AI agents collaborate to plan, search, write, and review a comprehensive report.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────┐     ┌──────────┐     ┌────────┐     ┌──────────┐
│ Planner │────▶│ Executor │────▶│ Writer │────▶│ Reviewer │
└─────────┘     └──────────┘     └────────┘     └──────────┘
   │ Breaks         │ ReAct          │ Drafts        │ Quality-
   │ query into     │ agent loop     │ structured    │ gates &
   │ a step-by-     │ (Tavily +      │ report        │ polishes
   │ step plan      │ FAISS memory)  │               │ final output
                                                      │
                                                      ▼
                                               Final Report (API)
```

**Conditional edge:** If the Executor fails (e.g. API rate limit), the graph bypasses the Writer and routes directly to the Reviewer, which returns a clean error message. The API **never returns a 500**.

---

## 📁 Project Structure

```
agentic-research-system/
│
├── api.py                        # FastAPI entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── README.md                     # This file
│
└── app/
    ├── __init__.py
    ├── state.py                  # Shared TypedDict state for LangGraph
    ├── tools.py                  # LangChain @tool definitions (Tavily + FAISS)
    ├── graph.py                  # LangGraph StateGraph with conditional edges
    │
    └── agents/
        ├── __init__.py
        ├── planner.py            # Breaks query into research plan
        ├── executor.py           # ReAct agent — runs web searches
        ├── writer.py             # Synthesises findings into draft report
        └── reviewer.py           # Quality-gates and finalises the report
```

---

## ⚙️ Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/agentic-research-system.git
cd agentic-research-system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# LangSmith observability (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=agentic-research-system
```

| Variable | Where to get it |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) — free tier available |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) — free tier available |

### 3. Run the server

```bash
uvicorn api:api --reload
```

Server starts at `http://localhost:8000`.

---

## 🚀 API Usage

### Health check

```bash
curl http://localhost:8000/
# {"status": "ok", "service": "agentic-research-system"}
```

### Run a research query

```bash
curl "http://localhost:8000/research?query=What+are+the+latest+advances+in+quantum+computing"
```

**Response:**

```json
{
  "query": "What are the latest advances in quantum computing",
  "report": "## Executive Summary\n...\n## Key Findings\n...\n## Detailed Analysis\n..."
}
```

### Interactive API docs

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

---

## 🔭 LangSmith Observability

When `LANGCHAIN_TRACING_V2=true` is set, **every run is automatically traced** — no extra code required. LangChain/LangGraph auto-instruments:

- Each graph node (Planner → Executor → Writer → Reviewer)
- Every LLM call (inputs, outputs, token usage, latency)
- Every tool call (Tavily search queries and results)
- The full ReAct agent scratchpad

View traces at [smith.langchain.com](https://smith.langchain.com) under your project.

> **Why LangSmith over a custom logger?**
> LangSmith captures structured, hierarchical traces that are searchable and filterable. A custom `print`/`logging` logger only gives flat text — unusable for debugging complex multi-agent runs.

---

## 🧠 Design Decisions

### TypedDict State
`ResearchState` in `app/state.py` is a `TypedDict` with explicitly typed fields. Every agent reads and writes known keys — no silent `KeyError` bugs from typos.

### ReAct Executor
The Executor uses `create_react_agent` (Reason + Act loop) instead of a hardcoded for-loop over search queries. This means the agent self-determines how many searches are needed and when it has enough information to stop — that's what makes it truly *agentic*.

### Conditional Edges for Resilience
`app/graph.py` adds a conditional edge after the Executor node. If any upstream error is detected, the graph skips the Writer and sends an error summary directly to the Reviewer. The API always returns a structured response.

### No utils/ Folder
Logging is handled with Python's standard `logging` module configured in `api.py`. LangSmith handles observability. A `utils/` folder with custom loggers adds complexity without value in a LangChain project.

### Single-Responsibility Agents
Each agent does exactly one thing:

| Agent | Responsibility |
|---|---|
| Planner | Decompose the query into steps |
| Executor | Execute searches and gather raw data |
| Writer | Synthesise data into a structured draft |
| Reviewer | Quality-gate and polish the final report |

---

## 🔄 Pipeline Flow

```python
# Simplified — see app/graph.py for full implementation

workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")

# Key: conditional routing on error
workflow.add_conditional_edges(
    "executor",
    route_after_executor,       # checks state["error"]
    {"writer": "writer", "reviewer": "reviewer"},
)

workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", END)
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Multi-agent graph orchestration |
| `langchain` | Agent framework, prompts, tool definitions |
| `langchain-openai` | OpenAI LLM + embedding wrappers |
| `langchain-community` | Tavily search tool, FAISS vector store |
| `langsmith` | Tracing and observability |
| `openai` | Underlying OpenAI SDK |
| `faiss-cpu` | In-process vector similarity search |
| `fastapi` + `uvicorn` | REST API server |
| `tavily-python` | Web search API client |
| `python-dotenv` | `.env` file loading |
| `pydantic` | Response schema validation |

---

## 🛠️ Extending the System

**Add a new agent:**
1. Create `app/agents/your_agent.py` with a `your_agent_node(state: ResearchState) -> dict` function
2. Register it in `app/graph.py` with `workflow.add_node("your_agent", your_agent_node)`
3. Wire edges to/from it

**Add a new tool:**
1. Add a `@tool`-decorated function in `app/tools.py`
2. Import it in `app/agents/executor.py` and add it to `_TOOLS`

**Switch LLM provider:**
Replace `ChatOpenAI(...)` with any LangChain-compatible chat model, e.g. `ChatAnthropic`, `ChatGoogleGenerativeAI`, etc.

---

## 📄 License

MIT
