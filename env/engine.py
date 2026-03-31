"""CloudFinOps Physics Simulator & Grading Engine."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from .models import Action, Observation, RewardInfo, ServerState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COST_PER_HOUR_WEB = 0.50
COST_PER_HOUR_DB = 1.20
COST_PER_HOUR_BATCH = 0.30
MAX_STEPS = 15
SLA_CPU_LIMIT = 100.0  # CPU >= this => SLA breach


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Task Blueprints
# ---------------------------------------------------------------------------

def _easy_servers() -> List[ServerState]:
    """10 servers, 3 idle (0% CPU) + 7 active."""
    servers: List[ServerState] = []
    for i in range(7):
        servers.append(ServerState(
            id=f"web-{i}",
            type="web",
            cpu_util=round(25.0 + i * 5.0, 1),
            memory_util=round(20.0 + i * 3.0, 1),
            cost_per_hour=COST_PER_HOUR_WEB,
            status="running",
        ))
    for i in range(3):
        servers.append(ServerState(
            id=f"idle-{i}",
            type="web",
            cpu_util=0.0,
            memory_util=0.0,
            cost_per_hour=COST_PER_HOUR_WEB,
            status="running",
        ))
    return servers


def _medium_servers() -> List[ServerState]:
    """12 over‑provisioned servers at very low CPU."""
    servers: List[ServerState] = []
    types = [("web", COST_PER_HOUR_WEB), ("db", COST_PER_HOUR_DB), ("batch", COST_PER_HOUR_BATCH)]
    for i in range(12):
        t, cost = types[i % 3]
        servers.append(ServerState(
            id=f"{t}-{i}",
            type=t,
            cpu_util=round(3.0 + (i % 4) * 1.5, 1),
            memory_util=round(5.0 + i * 0.8, 1),
            cost_per_hour=cost,
            status="running",
        ))
    return servers


def _hard_servers() -> List[ServerState]:
    """8 servers: mix of DB (high load), web (medium), batch (low)."""
    return [
        ServerState(id="db-0", type="db", cpu_util=85.0, memory_util=70.0, cost_per_hour=COST_PER_HOUR_DB, status="running"),
        ServerState(id="db-1", type="db", cpu_util=78.0, memory_util=65.0, cost_per_hour=COST_PER_HOUR_DB, status="running"),
        ServerState(id="web-0", type="web", cpu_util=55.0, memory_util=40.0, cost_per_hour=COST_PER_HOUR_WEB, status="running"),
        ServerState(id="web-1", type="web", cpu_util=50.0, memory_util=35.0, cost_per_hour=COST_PER_HOUR_WEB, status="running"),
        ServerState(id="web-2", type="web", cpu_util=60.0, memory_util=45.0, cost_per_hour=COST_PER_HOUR_WEB, status="running"),
        ServerState(id="batch-0", type="batch", cpu_util=20.0, memory_util=15.0, cost_per_hour=COST_PER_HOUR_BATCH, status="running"),
        ServerState(id="batch-1", type="batch", cpu_util=15.0, memory_util=10.0, cost_per_hour=COST_PER_HOUR_BATCH, status="running"),
        ServerState(id="batch-2", type="batch", cpu_util=10.0, memory_util=8.0, cost_per_hour=COST_PER_HOUR_BATCH, status="running"),
    ]


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "servers_fn": _easy_servers,
        "budget": 50.0,
        "traffic_load": 30.0,
        "spike": False,
        "inbox": ["Ops Team: Please terminate unused servers to save costs."],
    },
    "medium": {
        "servers_fn": _medium_servers,
        "budget": 100.0,
        "traffic_load": 20.0,
        "spike": False,
        "inbox": ["CTO: Cut costs by 50% immediately. No excuses."],
    },
    "hard": {
        "servers_fn": _hard_servers,
        "budget": 40.0,
        "traffic_load": 75.0,
        "spike": True,
        "inbox": ["Marketing: Massive ad campaign going live RIGHT NOW! Brace for traffic."],
    },
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CloudFinOpsEngine:
    """Deterministic physics engine for the CloudFinOps RL environment."""

    def __init__(self) -> None:
        self.servers: List[ServerState] = []
        self.task_id: str = "easy"
        self.time_step: int = 0
        self.budget_remaining: float = 0.0
        self.initial_budget: float = 0.0
        self.traffic_load: float = 0.0
        self.spike_detected: bool = False
        self.incidents: List[Dict[str, Any]] = []
        self.inbox: List[str] = []
        self.done: bool = False
        self.sla_breached: bool = False
        self.total_cost_spent: float = 0.0
        self.terminated_ids: List[str] = []
        self.upscaled_ids: List[str] = []
        self.pending_scales: Dict[str, str] = {}  # id -> command queued
        self._reward_accum: float = 0.0

    # ---- public API --------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        cfg = TASK_CONFIGS.get(task_id)
        if cfg is None:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASK_CONFIGS)}")

        self.task_id = task_id
        self.servers = cfg["servers_fn"]()
        self.budget_remaining = cfg["budget"]
        self.initial_budget = cfg["budget"]
        self.traffic_load = cfg["traffic_load"]
        self.spike_detected = cfg["spike"]
        self.inbox = list(cfg["inbox"])
        self.incidents = []
        self.time_step = 0
        self.done = False
        self.sla_breached = False
        self.total_cost_spent = 0.0
        self.terminated_ids = []
        self.upscaled_ids = []
        self.pending_scales = {}
        self._reward_accum = 0.0
        return self._obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            return self._obs(), 0.0, True, {"message": "Episode already done."}

        reward = 0.0
        self.time_step += 1

        # 1. Apply pending scales from PREVIOUS step (delayed consequence)
        self._apply_pending_scales()

        # 2. Process current action
        reward += self._process_action(action)

        # 3. Simulate traffic drift (hard task ramps up)
        self._simulate_traffic()

        # 4. Redistribute load across running servers
        self._redistribute_load()

        # 5. Charge costs for running servers
        step_cost = sum(s.cost_per_hour for s in self.servers if s.status == "running")
        self.budget_remaining -= step_cost
        self.total_cost_spent += step_cost

        # 6. Check SLA breaches
        for s in self.servers:
            if s.status == "running" and s.cpu_util >= SLA_CPU_LIMIT:
                self.sla_breached = True
                self.incidents.append({"type": "SLA_BREACH", "server": s.id, "step": self.time_step})
                reward -= 100.0

        # 7. Budget overrun penalty
        if self.budget_remaining < 0:
            reward -= 20.0

        # 8. Check termination
        if self.time_step >= MAX_STEPS or self.sla_breached or self.budget_remaining <= 0:
            self.done = True

        self._reward_accum += reward

        # Normalise step reward to a rough 0‑1 hint (grader is authoritative)
        info: Dict[str, Any] = {}
        if self.done:
            final_score = self.grade()
            info["grader_score"] = final_score

        return self._obs(), reward, self.done, info

    def state(self) -> Observation:
        return self._obs()

    def grade(self) -> float:
        """Return a 0.0‑1.0 score for the current task."""
        if self.task_id == "easy":
            return self._grade_easy()
        elif self.task_id == "medium":
            return self._grade_medium()
        else:
            return self._grade_hard()

    # ---- graders -----------------------------------------------------------

    def _grade_easy(self) -> float:
        # Goal: terminate the 3 idle servers, don't touch active ones.
        idle_ids = {f"idle-{i}" for i in range(3)}
        terminated_idle = len(idle_ids & set(self.terminated_ids))
        active_terminated = len(set(self.terminated_ids) - idle_ids)
        score = (terminated_idle / 3.0) - (active_terminated * 0.25)
        if self.sla_breached:
            score -= 0.5
        return _clamp(score, 0.0, 1.0)

    def _grade_medium(self) -> float:
        # Goal: save >= 50 % budget with 0 crashes
        cost_saved_pct = 1.0 - (self.total_cost_spent / self.initial_budget) if self.initial_budget > 0 else 0.0
        target = 0.50
        efficiency = min(cost_saved_pct / target, 1.0) if target > 0 else 0.0
        crash_penalty = 0.5 if self.sla_breached else 0.0
        return _clamp(efficiency - crash_penalty, 0.0, 1.0)

    def _grade_hard(self) -> float:
        # Multi‑objective: uptime 60 % + cost efficiency 40 %
        uptime_score = 0.0 if self.sla_breached else 1.0
        cost_saved_pct = 1.0 - (self.total_cost_spent / self.initial_budget) if self.initial_budget > 0 else 0.0
        cost_efficiency = _clamp(cost_saved_pct, 0.0, 1.0)
        return round(_clamp(uptime_score * 0.6 + cost_efficiency * 0.4, 0.0, 1.0), 4)

    # ---- internal helpers --------------------------------------------------

    def _obs(self) -> Observation:
        return Observation(
            servers=[s.model_copy() for s in self.servers],
            traffic_load=round(self.traffic_load, 2),
            spike_detected=self.spike_detected,
            incidents=list(self.incidents),
            budget_remaining=round(self.budget_remaining, 2),
            time_step=self.time_step,
            inbox=list(self.inbox),
        )

    def _find_server(self, server_id: Optional[str]) -> Optional[ServerState]:
        if server_id is None:
            return None
        for s in self.servers:
            if s.id == server_id:
                return s
        return None

    def _process_action(self, action: Action) -> float:
        reward = 0.0
        server = self._find_server(action.target_id)

        if action.command == "IGNORE":
            return 0.0

        if server is None:
            return -2.0  # invalid target

        if server.status == "terminated":
            return -2.0  # acting on dead server

        if action.command == "TERMINATE":
            server.status = "terminated"
            server.cpu_util = 0.0
            server.memory_util = 0.0
            self.terminated_ids.append(server.id)
            reward += 10.0  # cost saving reward

        elif action.command == "UPSCALE":
            # Queued – takes effect NEXT step (delayed consequence)
            self.pending_scales[server.id] = "UPSCALE"
            self.upscaled_ids.append(server.id)
            reward -= 5.0  # scaling cost

        elif action.command == "DOWNSCALE":
            # Immediate: halve specs, save money, but CPU load doubles on this server
            server.cost_per_hour = round(server.cost_per_hour * 0.5, 2)
            server.cpu_util = _clamp(server.cpu_util * 1.8)  # load pressure increases
            server.memory_util = _clamp(server.memory_util * 1.3)
            reward += 5.0

        elif action.command == "REDISTRIBUTE_LOAD":
            # Spread load evenly across running servers
            running = [s for s in self.servers if s.status == "running"]
            if len(running) > 1:
                avg_cpu = sum(s.cpu_util for s in running) / len(running)
                avg_mem = sum(s.memory_util for s in running) / len(running)
                for s in running:
                    s.cpu_util = round(_clamp(avg_cpu), 1)
                    s.memory_util = round(_clamp(avg_mem), 1)
                reward += 3.0

        # Clear inbox after agent replies
        if action.reply:
            self.inbox = []

        return reward

    def _apply_pending_scales(self) -> None:
        for sid, cmd in list(self.pending_scales.items()):
            server = self._find_server(sid)
            if server and server.status == "running" and cmd == "UPSCALE":
                server.cost_per_hour = round(server.cost_per_hour * 2.0, 2)
                server.cpu_util = _clamp(server.cpu_util * 0.5)  # doubled capacity -> halved util
                server.memory_util = _clamp(server.memory_util * 0.6)
        self.pending_scales.clear()

    def _simulate_traffic(self) -> None:
        """Ramp up traffic on Hard task; slight drift otherwise."""
        if self.task_id == "hard":
            # Exponential ramp simulating ad campaign traffic
            self.traffic_load = _clamp(self.traffic_load + 5.0 * math.log1p(self.time_step))
            self.spike_detected = True
            # Push DB servers harder
            for s in self.servers:
                if s.status == "running" and s.type == "db":
                    s.cpu_util = _clamp(s.cpu_util + 4.0 * math.log1p(self.time_step))
        else:
            # Slight random‑free drift
            self.traffic_load = _clamp(self.traffic_load + 0.5)

    def _redistribute_load(self) -> None:
        """After terminations, remaining running servers absorb orphaned load."""
        running = [s for s in self.servers if s.status == "running"]
        terminated_this = [s for s in self.servers if s.status == "terminated"]
        if not running:
            return
        # Spread orphaned CPU proportionally
        orphan_cpu = sum(s.cpu_util for s in terminated_this)  # already 0 after terminate, but first‑step residual
        if orphan_cpu > 0 and len(running) > 0:
            per_server = orphan_cpu / len(running)
            for s in running:
                s.cpu_util = round(_clamp(s.cpu_util + per_server), 1)
