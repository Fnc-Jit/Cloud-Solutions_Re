"""FastAPI server exposing the OpenEnv spec endpoints."""

from __future__ import annotations

import time
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .engine import CloudFinOpsEngine
from .models import Action, Observation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cloudfinops")

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
# Request / Response Models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Startup / Shutdown Events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ☁️  CloudFinOps-Env  v1.0.0                                ║
    ║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                               ║
    ║                                                              ║
    ║   Cloud Cost-Optimization + SLA Incident Management          ║
    ║   OpenEnv-compatible RL Environment                          ║
    ║                                                              ║
    ║   Endpoints:                                                 ║
    ║     POST /reset   → Reset environment for a task             ║
    ║     POST /step    → Advance engine by 1 tick                 ║
    ║     GET  /state   → Current observation (read-only)          ║
    ║                                                              ║
    ║   Tasks:                                                     ║
    ║     • easy   → Zombie Cleanup (terminate idle servers)       ║
    ║     • medium → CTO Budget Squeeze (cut costs 50%)            ║
    ║     • hard   → Black Friday Chaos (handle traffic spike)     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    log.info("Server started successfully")
    log.info("Listening on http://0.0.0.0:8000")
    log.info("Available tasks: easy, medium, hard")
    log.info("Ready to accept connections ✓")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    log.info("Server shutting down... Goodbye! 👋")


# ---------------------------------------------------------------------------
# Middleware — Request Logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    log.info(
        "%s %s → %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
async def reset(req: ResetRequest) -> Observation:
    """Reset the environment for the given task and return the initial observation."""
    try:
        obs = engine.reset(req.task_id)
    except ValueError as exc:
        log.error("Reset failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    servers = len(obs.servers)
    running = sum(1 for s in obs.servers if s.status == "running")
    log.info(
        "🔄 RESET task=%s │ servers=%d │ running=%d │ budget=$%.2f │ traffic=%.0f%%",
        req.task_id,
        servers,
        running,
        obs.budget_remaining,
        obs.traffic_load,
    )
    return obs


@app.post("/step", response_model=StepResponse)
async def step(action: Action) -> StepResponse:
    """Advance the engine by one tick and return the new observation + reward."""
    obs, reward, done, info = engine.step(action)

    status = "🏁 DONE" if done else f"⏩ Step {obs.time_step}"
    score_str = ""
    if done and "grader_score" in info:
        score_str = f" │ SCORE={info['grader_score']:.4f}"

    log.info(
        "%s │ action=%s target=%s │ reward=%+.1f │ budget=$%.2f%s",
        status,
        action.command,
        action.target_id or "—",
        reward,
        obs.budget_remaining,
        score_str,
    )
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=Observation)
async def state() -> Observation:
    """Return the current observation without advancing the engine."""
    obs = engine.state()
    log.info(
        "👁  STATE │ step=%d │ servers=%d │ budget=$%.2f",
        obs.time_step,
        len(obs.servers),
        obs.budget_remaining,
    )
    return obs
