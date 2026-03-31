"""FastAPI server exposing the OpenEnv spec endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .engine import CloudFinOpsEngine
from .models import Action, Observation

# ---------------------------------------------------------------------------
# App & Engine
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CloudFinOps-Env",
    description="RL environment combining cloud cost-optimization with SLA incident management.",
    version="1.0.0",
)

engine = CloudFinOpsEngine()


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
async def reset(req: ResetRequest) -> Observation:
    """Reset the environment for the given task and return the initial observation."""
    try:
        obs = engine.reset(req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step", response_model=StepResponse)
async def step(action: Action) -> StepResponse:
    """Advance the engine by one tick and return the new observation + reward."""
    obs, reward, done, info = engine.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=Observation)
async def state() -> Observation:
    """Return the current observation without advancing the engine."""
    return engine.state()
