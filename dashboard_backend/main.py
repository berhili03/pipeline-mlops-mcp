import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mcp_client.client import MCPAgentClient  # noqa: E402

agent = MCPAgentClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.connect()
    yield
    await agent.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


async def call_tool_json(name: str, args: dict | None = None):
    result = await agent.session.call_tool(name, args or {})
    texts = [getattr(block, "text", "") for block in result.content]

    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return {"raw": texts[0]}

    # Plusieurs blocs (ex: query_predictions renvoie une liste, un bloc par élément)
    items = []
    for t in texts:
        try:
            items.append(json.loads(t))
        except json.JSONDecodeError:
            items.append(t)
    return items


@app.get("/api/kpis")
async def kpis():
    return await call_tool_json("get_monitoring_stats")


@app.get("/api/drift")
async def drift():
    return await call_tool_json("get_drift_report")


@app.get("/api/production")
async def production():
    return await call_tool_json("get_production_data_summary")


@app.get("/api/model-info")
async def model_info():
    return await call_tool_json("get_model_info")


@app.get("/api/predictions")
async def predictions(limit: int = 200, only_errors: bool = False):
    return await call_tool_json(
        "query_predictions", {"limit": limit, "only_errors": only_errors}
    )


@app.get("/api/health-status")
async def health_status():
    kpi_data = await call_tool_json("get_monitoring_stats")
    drift_data = await call_tool_json("get_drift_report")

    error_rate = kpi_data.get("error_rate", 0) or 0
    drift_share = drift_data.get("drift_share", 0) or 0

    if error_rate > 0.10 or drift_share > 0.6:
        status = "Critical"
    elif error_rate > 0.02 or drift_share > 0.3:
        status = "Warning"
    else:
        status = "Healthy"

    return {
        "status": status,
        "error_rate": error_rate,
        "drift_share": drift_share,
    }


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
async def ask(payload: AskRequest):
    answer = await agent.ask(payload.question)
    return {"answer": answer}