"""CloudFinOps Physics Simulator & Grading Engine.

Simulates a realistic cloud cost-optimization scenario using AWS-style
instance types, real-world pricing, stochastic metric noise, and
multi-objective grading.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

from .models import Action, Observation, RewardInfo, ServerState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Realistic AWS-style hourly pricing (on-demand, approximate)
INSTANCE_CATALOG: Dict[str, Dict[str, Any]] = {
    "t3.micro":    {"vcpu": 2,  "mem_gb": 1,   "cost": 0.0104, "category": "web"},
    "t3.medium":   {"vcpu": 2,  "mem_gb": 4,   "cost": 0.0416, "category": "web"},
    "t3.large":    {"vcpu": 2,  "mem_gb": 8,   "cost": 0.0832, "category": "web"},
    "c5.large":    {"vcpu": 2,  "mem_gb": 4,   "cost": 0.0850, "category": "compute"},
    "c5.xlarge":   {"vcpu": 4,  "mem_gb": 8,   "cost": 0.1700, "category": "compute"},
    "r6g.medium":  {"vcpu": 1,  "mem_gb": 8,   "cost": 0.0504, "category": "db"},
    "r6g.large":   {"vcpu": 2,  "mem_gb": 16,  "cost": 0.1008, "category": "db"},
    "r6g.xlarge":  {"vcpu": 4,  "mem_gb": 32,  "cost": 0.2016, "category": "db"},
    "m5.large":    {"vcpu": 2,  "mem_gb": 8,   "cost": 0.0960, "category": "batch"},
    "m5.xlarge":   {"vcpu": 4,  "mem_gb": 16,  "cost": 0.1920, "category": "batch"},
}

# Upscale path — each instance can only be upgraded through these tiers
UPSCALE_PATH: Dict[str, str] = {
    "t3.micro":   "t3.medium",
    "t3.medium":  "t3.large",
    "c5.large":   "c5.xlarge",
    "r6g.medium": "r6g.large",
    "r6g.large":  "r6g.xlarge",
    "m5.large":   "m5.xlarge",
}

MAX_STEPS = 15
SLA_CPU_LIMIT = 100.0  # CPU >= this => SLA breach


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


def _deterministic_noise(seed_str: str, amplitude: float = 2.0) -> float:
    """Return a deterministic pseudo-random noise value in [-amplitude, +amplitude].

    Uses a hash of the seed string so the environment remains fully
    reproducible (same inputs → same outputs) while still exhibiting
    realistic metric jitter.
    """
    h = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    # Map to [-1.0, 1.0]
    normalised = (h / 0xFFFFFFFF) * 2.0 - 1.0
    return normalised * amplitude


# ---------------------------------------------------------------------------
# Task Blueprints
# ---------------------------------------------------------------------------

def _easy_servers() -> List[ServerState]:
    """10 servers: 7 active web instances + 3 completely idle instances."""
    servers: List[ServerState] = []
    for i in range(4):
        servers.append(ServerState(
            id=f"web-{i}",
            type="t3.medium",
            cpu_util=round(25.0 + i * 7.5, 1),
            memory_util=round(20.0 + i * 5.0, 1),
            cost_per_hour=INSTANCE_CATALOG["t3.medium"]["cost"],
            status="running",
        ))
    for i in range(3):
        servers.append(ServerState(
            id=f"compute-{i}",
            type="c5.large",
            cpu_util=round(30.0 + i * 10.0, 1),
            memory_util=round(25.0 + i * 6.0, 1),
            cost_per_hour=INSTANCE_CATALOG["c5.large"]["cost"],
            status="running",
        ))
    # 3 completely idle servers — the zombies
    for i in range(3):
        servers.append(ServerState(
            id=f"idle-{i}",
            type="t3.micro",
            cpu_util=0.0,
            memory_util=0.0,
            cost_per_hour=INSTANCE_CATALOG["t3.micro"]["cost"],
            status="running",
        ))
    return servers


def _medium_servers() -> List[ServerState]:
    """12 over-provisioned servers at very low CPU across web/db/batch types."""
    servers: List[ServerState] = []
    configs = [
        ("t3.medium", "web"),
        ("r6g.large", "db"),
        ("m5.large", "batch"),
    ]
    for i in range(12):
        inst_type, _ = configs[i % 3]
        cat = INSTANCE_CATALOG[inst_type]
        servers.append(ServerState(
            id=f"{cat['category']}-{i}",
            type=inst_type,
            cpu_util=round(3.0 + (i % 4) * 1.5, 1),
            memory_util=round(5.0 + i * 0.8, 1),
            cost_per_hour=cat["cost"],
            status="running",
        ))
    return servers


def _hard_servers() -> List[ServerState]:
    """8 servers: DB (high load), web (medium), batch (low)."""
    return [
        ServerState(id="db-0",    type="r6g.large",  cpu_util=85.0, memory_util=70.0, cost_per_hour=INSTANCE_CATALOG["r6g.large"]["cost"],  status="running"),
        ServerState(id="db-1",    type="r6g.large",  cpu_util=78.0, memory_util=65.0, cost_per_hour=INSTANCE_CATALOG["r6g.large"]["cost"],  status="running"),
        ServerState(id="web-0",   type="t3.medium",  cpu_util=55.0, memory_util=40.0, cost_per_hour=INSTANCE_CATALOG["t3.medium"]["cost"],  status="running"),
        ServerState(id="web-1",   type="t3.medium",  cpu_util=50.0, memory_util=35.0, cost_per_hour=INSTANCE_CATALOG["t3.medium"]["cost"],  status="running"),
        ServerState(id="web-2",   type="c5.large",   cpu_util=60.0, memory_util=45.0, cost_per_hour=INSTANCE_CATALOG["c5.large"]["cost"],   status="running"),
        ServerState(id="batch-0", type="m5.large",   cpu_util=20.0, memory_util=15.0, cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],   status="running"),
        ServerState(id="batch-1", type="m5.large",   cpu_util=15.0, memory_util=10.0, cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],   status="running"),
        ServerState(id="batch-2", type="m5.large",   cpu_util=10.0, memory_util=8.0,  cost_per_hour=INSTANCE_CATALOG["m5.large"]["cost"],   status="running"),
    ]


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "servers_fn": _easy_servers,
        "budget": 5.0,
        "traffic_load": 30.0,
        "spike": False,
        "inbox": [
            "Ops Team: We're seeing 3 zombie instances racking up charges. Please terminate unused servers to save costs.",
        ],
    },
    "medium": {
        "servers_fn": _medium_servers,
        "budget": 10.0,
        "traffic_load": 20.0,
        "spike": False,
        "inbox": [
            "CTO: Our cloud bill is through the roof. Cut costs by at least 50% immediately — no excuses.",
            "Finance: Q3 budget review in 2 days. We need measurable savings by then.",
        ],
    },
    "hard": {
        "servers_fn": _hard_servers,
        "budget": 4.0,
        "traffic_load": 75.0,
        "spike": True,
        "inbox": [
            "Marketing: Massive ad campaign going live RIGHT NOW! Expect 10× normal traffic.",
            "SRE On-Call: DB-0 is approaching capacity. Consider upscaling before it breaches SLA.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CloudFinOpsEngine:
    """Deterministic physics engine for the CloudFinOps RL environment.

    Key mechanics:
    - Realistic AWS instance types with real-world pricing
    - Delayed scaling (UPSCALE takes effect next step)
    - Bounded deterministic noise on server metrics
    - Upscale cap: each server can only be upgraded through a fixed tier path
    - Load redistribution after terminations
    - Continuous reward shaping over full trajectory
    """

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
        self.upscale_counts: Dict[str, int] = {}  # track how many times each server was upscaled
        self.pending_scales: Dict[str, str] = {}   # id -> queued instance type
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
        self.upscale_counts = {}
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

        # 4. Apply deterministic metric noise
        self._apply_noise()

        # 5. Redistribute load across running servers
        self._redistribute_load()

        # 6. Charge costs for running servers
        step_cost = sum(s.cost_per_hour for s in self.servers if s.status == "running")
        self.budget_remaining -= step_cost
        self.total_cost_spent += step_cost

        # 7. Check SLA breaches
        for s in self.servers:
            if s.status == "running" and s.cpu_util >= SLA_CPU_LIMIT:
                self.sla_breached = True
                self.incidents.append({
                    "type": "SLA_BREACH",
                    "server": s.id,
                    "instance_type": s.type,
                    "cpu_at_breach": round(s.cpu_util, 1),
                    "step": self.time_step,
                })
                reward -= 100.0

        # 8. Per-step cost pressure (continuous signal over trajectory)
        if step_cost > 0.50:
            reward -= 1.0  # penalise high ongoing costs each step

        # 9. Budget overrun penalty
        if self.budget_remaining < 0:
            reward -= 20.0

        # 10. Check termination
        if self.time_step >= MAX_STEPS or self.sla_breached or self.budget_remaining <= 0:
            self.done = True

        self._reward_accum += reward

        info: Dict[str, Any] = {"step_reward": reward, "cumulative_reward": self._reward_accum}
        if self.done:
            final_score = self.grade()
            info["grader_score"] = final_score

        return self._obs(), reward, self.done, info

    def state(self) -> Observation:
        return self._obs()

    def grade(self) -> float:
        """Return a 0.0–1.0 score for the current task."""
        if self.task_id == "easy":
            return self._grade_easy()
        elif self.task_id == "medium":
            return self._grade_medium()
        else:
            return self._grade_hard()

    # ---- graders -----------------------------------------------------------

    def _grade_easy(self) -> float:
        """Goal: terminate the 3 idle servers, don't touch active ones.

        Scoring:
        - +1/3 per correctly terminated idle server
        - -0.25 per wrongly terminated active server
        - -0.5 for any SLA breach
        """
        idle_ids = {f"idle-{i}" for i in range(3)}
        terminated_idle = len(idle_ids & set(self.terminated_ids))
        active_terminated = len(set(self.terminated_ids) - idle_ids)
        score = (terminated_idle / 3.0) - (active_terminated * 0.25)
        if self.sla_breached:
            score -= 0.5
        return _clamp(score, 0.0, 1.0)

    def _grade_medium(self) -> float:
        """Goal: save >= 50% budget with 0 SLA breaches.

        Scoring:
        - Proportional to cost_saved_pct / 50% (capped at 1.0)
        - -0.5 for any SLA breach
        """
        cost_saved_pct = 1.0 - (self.total_cost_spent / self.initial_budget) if self.initial_budget > 0 else 0.0
        target = 0.50
        efficiency = min(cost_saved_pct / target, 1.0) if target > 0 else 0.0
        crash_penalty = 0.5 if self.sla_breached else 0.0
        return _clamp(efficiency - crash_penalty, 0.0, 1.0)

    def _grade_hard(self) -> float:
        """Multi-objective: uptime 60% + cost efficiency 40%.

        Scoring:
        - uptime_score: 1.0 if no SLA breach, 0.0 if breached
        - cost_efficiency: proportional to budget saved
        - Bonus: +0.1 if agent replied to inbox messages
        """
        uptime_score = 0.0 if self.sla_breached else 1.0
        cost_saved_pct = 1.0 - (self.total_cost_spent / self.initial_budget) if self.initial_budget > 0 else 0.0
        cost_efficiency = _clamp(cost_saved_pct, 0.0, 1.0)

        base = uptime_score * 0.6 + cost_efficiency * 0.4

        # Bonus for engaging with stakeholders (human-in-the-loop reward)
        inbox_bonus = 0.1 if not self.inbox else 0.0  # inbox cleared = agent replied
        return round(_clamp(base + inbox_bonus, 0.0, 1.0), 4)

    # ---- internal helpers --------------------------------------------------

    def _obs(self) -> Observation:
        return Observation(
            servers=[s.model_copy() for s in self.servers],
            traffic_load=round(self.traffic_load, 2),
            spike_detected=self.spike_detected,
            incidents=list(self.incidents),
            budget_remaining=round(self.budget_remaining, 4),
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
            # Check if this server can be upscaled further
            next_type = UPSCALE_PATH.get(server.type)
            if next_type is None:
                # Already at maximum tier — cannot upscale further
                reward -= 1.0  # small penalty for wasted action
            else:
                count = self.upscale_counts.get(server.id, 0)
                if count >= 2:
                    # Hit the cap — max 2 upgrades per server
                    reward -= 1.0
                else:
                    # Queue the upgrade — takes effect NEXT step (delayed consequence)
                    self.pending_scales[server.id] = next_type
                    self.upscaled_ids.append(server.id)
                    self.upscale_counts[server.id] = count + 1
                    reward -= 5.0  # scaling cost

        elif action.command == "DOWNSCALE":
            # Immediate: halve specs, save money, but CPU load doubles on this server
            server.cost_per_hour = round(server.cost_per_hour * 0.5, 4)
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

        # Reward for responding to inbox (human-in-the-loop engagement)
        if action.reply and self.inbox:
            reward += 2.0
            self.inbox = []

        return reward

    def _apply_pending_scales(self) -> None:
        for sid, new_type in list(self.pending_scales.items()):
            server = self._find_server(sid)
            if server and server.status == "running":
                old_cost = server.cost_per_hour
                new_cost = INSTANCE_CATALOG[new_type]["cost"]
                server.type = new_type
                server.cost_per_hour = new_cost
                # Doubled capacity -> halved utilisation
                server.cpu_util = _clamp(server.cpu_util * 0.5)
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
                if s.status == "running" and s.type.startswith("r6g"):
                    s.cpu_util = _clamp(s.cpu_util + 4.0 * math.log1p(self.time_step))
        else:
            # Slight deterministic drift
            self.traffic_load = _clamp(self.traffic_load + 0.5)

    def _apply_noise(self) -> None:
        """Add bounded deterministic noise to CPU and memory metrics.

        The noise is seeded by (task_id, server_id, time_step) so the
        environment is fully reproducible while still exhibiting realistic
        metric jitter that agents must handle.
        """
        for s in self.servers:
            if s.status != "running":
                continue
            seed = f"{self.task_id}:{s.id}:{self.time_step}"
            cpu_noise = _deterministic_noise(seed + ":cpu", amplitude=2.5)
            mem_noise = _deterministic_noise(seed + ":mem", amplitude=1.5)
            s.cpu_util = round(_clamp(s.cpu_util + cpu_noise), 1)
            s.memory_util = round(_clamp(s.memory_util + mem_noise), 1)

    def _redistribute_load(self) -> None:
        """After terminations, remaining running servers absorb orphaned load."""
        running = [s for s in self.servers if s.status == "running"]
        terminated_this = [s for s in self.servers if s.status == "terminated"]
        if not running:
            return
        # Spread orphaned CPU proportionally
        orphan_cpu = sum(s.cpu_util for s in terminated_this)  # already 0 after terminate, but first-step residual
        if orphan_cpu > 0 and len(running) > 0:
            per_server = orphan_cpu / len(running)
            for s in running:
                s.cpu_util = round(_clamp(s.cpu_util + per_server), 1)
