"""Pydantic Space Definitions for CloudFinOps RL Environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ServerState(BaseModel):
    """Represents a single cloud server instance."""

    id: str = Field(..., description="Unique server identifier")
    type: str = Field(..., description="Server type, e.g. 'web', 'db', 'batch'")
    cpu_util: float = Field(0.0, ge=0.0, description="CPU utilisation %")
    memory_util: float = Field(0.0, ge=0.0, description="Memory utilisation %")
    cost_per_hour: float = Field(0.0, ge=0.0, description="Hourly cost in USD")
    status: str = Field("running", description="running | terminated | pending_scale")


class Observation(BaseModel):
    """Full environment observation returned to the agent."""

    servers: List[ServerState]
    traffic_load: float = Field(0.0, description="Global traffic load 0-100")
    spike_detected: bool = Field(False)
    incidents: List[Dict[str, Any]] = Field(default_factory=list)
    budget_remaining: float = Field(100.0)
    time_step: int = Field(0)
    inbox: List[str] = Field(default_factory=list, description="Text messages from humans")


class Action(BaseModel):
    """Agent action submitted each step."""

    command: Literal["UPSCALE", "DOWNSCALE", "TERMINATE", "REDISTRIBUTE_LOAD", "IGNORE"] = "IGNORE"
    target_id: Optional[str] = Field(None, description="Server ID to act on")
    reply: str = Field("", description="Text reply to the inbox")


class RewardInfo(BaseModel):
    """Reward signal returned after each step."""

    score: float = Field(0.0, ge=0.0, le=1.0)
    is_done: bool = False
    feedback: str = ""
