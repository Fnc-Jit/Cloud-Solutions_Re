---
title: CloudFinOps Env
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# ☁️ CloudFinOps-Env
> **An RL environment combining cloud cost-optimization, SLA incident management, and carbon emissions tracking (GreenOps).**

[![Validate](https://github.com/Fnc-Jit/Cloud-Solutions_Re/actions/workflows/validate.yml/badge.svg)](https://github.com/Fnc-Jit/Cloud-Solutions_Re/actions/workflows/validate.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green.svg)](https://huggingface.co/openenv)
[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)]()

![CloudFinOps Dashboard](assets/dashboard.png)
![CloudFinOps Dashboard Details](assets/dashboard_details.png)

Built for the **Meta AI × Hugging Face OpenEnv Hackathon**. Agents manage a fleet of AWS-style servers, balancing cost, performance, carbon emissions, and stakeholder communication through a REST API.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CloudFinOps-Env                           │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ models.py│──▶│  engine.py   │◀──│    server.py       │  │
│  │ Pydantic │   │ Physics Sim  │   │ FastAPI REST API   │  │
│  │ Schemas  │   │ + Grader     │   │ + Dashboard        │  │
│  └──────────┘   └──────────────┘   └────────────────────┘  │
│                        ▲                    ▲               │
│                        │                    │               │
│                   ┌────┴────┐         ┌─────┴─────┐        │
│                   │  Tests  │         │inference.py│        │
│                   │ 38 unit │         │ LLM Agent  │        │
│                   │  tests  │         │ Evaluator  │        │
│                   └─────────┘         └───────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏢 **AWS Instance Catalog** | 10 realistic instance types (`t3.micro` → `m5.xlarge`) with real-world pricing |
| 📊 **Trailing Metrics History** | `cpu_history` / `memory_history` — last 3 steps per server for LLM trend detection |
| 🌍 **GreenOps Carbon Tracking** | Per-instance `carbon_kwh` emissions, ARM (r6g) vs x86 (c5/m5) efficiency modeling |
| 🎯 **4 Difficulty Tiers** | Easy → Medium → Hard → Green, each with unique objectives and grading |
| 📬 **Human-in-the-Loop** | Inbox messages from stakeholders; replying earns bonus points |
| ⏱️ **Delayed Scaling** | UPSCALE queues for next step — agents must plan ahead |
| 🔒 **Deterministic Noise** | Hash-seeded metric jitter — fully reproducible episodes |
| 📈 **Live Dashboard** | Real-time glassmorphism web UI at `/dashboard` with sparklines |
| 🧪 **38 Unit Tests** | Comprehensive pytest suite + GitHub Actions CI |

---

## 🎯 Tasks

### 🟢 Easy — "Zombie Cleanup"
Terminate 3 idle servers (`idle-0`, `idle-1`, `idle-2`) without touching active ones.  
**Budget:** $5.00 | **Servers:** 10 | **Grading:** +1/3 per zombie killed, -0.25 per wrongful termination.

### 🟡 Medium — "CTO Budget Squeeze"
Cut cloud costs by ≥50% across 12 over-provisioned servers.  
**Budget:** $10.00 | **Servers:** 12 | **Grading:** Proportional to `cost_saved_pct / 50%`.

### 🔴 Hard — "Black Friday Chaos"
Handle a traffic spike with exponential ramp. Keep DB servers alive while managing budget.  
**Budget:** $4.00 | **Servers:** 8 | **Grading:** Uptime (60%) + Cost Efficiency (40%) + Inbox Bonus.

### 🌍 Green — "The Green Initiative"
Reduce carbon emissions by 40% by migrating workloads from dirty x86 instances (c5, m5) to efficient ARM Graviton (r6g).  
**Budget:** $8.00 | **Servers:** 10 | **Grading:** Carbon Reduction (50%) + Uptime (30%) + Cost (10%) + Inbox (10%).

---

## 📊 Carbon Intensity per Instance Type

| Instance | Architecture | Carbon (kWh/step) | Category |
|----------|-------------|-------------------|----------|
| `t3.micro` | x86 | 0.005 | Web |
| `t3.medium` | x86 | 0.012 | Web |
| `t3.large` | x86 | 0.022 | Web |
| `c5.large` | x86 | **0.035** | Compute |
| `c5.xlarge` | x86 | **0.065** | Compute |
| `r6g.medium` | ARM Graviton | 0.008 | DB |
| `r6g.large` | ARM Graviton | 0.015 | DB |
| `r6g.xlarge` | ARM Graviton | 0.028 | DB |
| `m5.large` | x86 | **0.040** | Batch |
| `m5.xlarge` | x86 | **0.075** | Batch |

> ARM Graviton instances produce **2–3× less carbon** than equivalent x86 instances.

---

## 📈 Trailing Metrics History

Each server's observation includes the **last 3 steps** of CPU and memory utilisation:

```json
{
  "id": "web-0",
  "type": "t3.medium",
  "cpu_util": 85.0,
  "cpu_history": [60.2, 72.5, 85.0],
  "memory_util": 45.0,
  "memory_history": [38.1, 41.7, 45.0]
}
```

This lets LLM agents **detect trends** (e.g., "CPU rising 3 steps in a row → act preemptively") without needing explicit memory systems.

---

## 🚀 Quick Start

### 1. Clone & Configure
```bash
git clone https://github.com/Fnc-Jit/Cloud-Solutions_Re.git
cd cloudfinops-env
cp .env.example .env
# Edit .env with your API keys
```

### 2. Build Docker Image
```bash
docker build -t cloudfinops-env .
```

### 3. Start the Environment Server
```bash
docker run --env-file .env -p 8000:8000 cloudfinops-env
```

### 4. Open the Live Dashboard
Open `http://localhost:8000/dashboard` in your browser.

### 5. Run the Agent Evaluator
```bash
docker run --env-file .env -e ENV_BASE_URL=http://host.docker.internal:8000 cloudfinops-env python inference.py
```

### 6. Run Tests
```bash
docker run --rm cloudfinops-env python -m pytest tests/ -v
```

---

## 🖥️ API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check & endpoint listing |
| `POST` | `/reset` | Reset environment for a task (`{"task_id": "easy"}`) |
| `POST` | `/step` | Submit an action and advance the engine |
| `GET` | `/state` | Current observation (read-only, no side effects) |
| `GET` | `/dashboard` | Real-time glassmorphism web dashboard |
| `GET` | `/history` | Agent action history for current episode |

---

## 🎮 Action Space

| Command | Effect | Reward |
|---------|--------|--------|
| `TERMINATE` | Kill a server immediately | +10 |
| `UPSCALE` | Queue upgrade (applies next step) | -5 |
| `DOWNSCALE` | Halve cost, but CPU load × 1.8 | +5 |
| `REDISTRIBUTE_LOAD` | Spread CPU evenly across fleet | +3 |
| `IGNORE` | Do nothing this step | 0 |

**Penalties:**
- Invalid target: **-2**
- SLA breach (CPU ≥ 100%): **-100** + episode ends
- Budget overrun: **-20** + episode ends
- High ongoing cost (>$0.50/step): **-1** per step

---

## 📊 Observation Space

```json
{
  "servers": [...],
  "traffic_load": 30.0,
  "spike_detected": false,
  "incidents": [],
  "budget_remaining": 5.0,
  "time_step": 0,
  "inbox": ["Ops Team: ..."],
  "carbon_kwh": 0.0
}
```

Each server includes:
- `id`, `type`, `cpu_util`, `memory_util`, `cost_per_hour`, `status`
- `cpu_history`: last 3 CPU values
- `memory_history`: last 3 memory values

---

## 🏆 Baseline Scores

The enclosed baseline evaluator (`inference.py`) establishes the reference performance for agents.

| Task | Difficulty | Baseline Score (LLaMA-3 70B) | Success Status |
|------|------------|------------------------------|----------------|
| `easy` | Easy | TBD | ✅ |
| `medium` | Medium | TBD | ✅ |
| `hard` | Hard | TBD | ✅ |
| `green` | Green | TBD | ✅ |

> **Note:** Run the evaluator yourself using the Quick Start instructions to see the exact real-time scores for your chosen LLM.

---

## 🏆 Baseline Scores

The enclosed baseline evaluator (`inference.py`) establishes the reference performance for agents.

| Task | Difficulty | Baseline Score (OpenAI GPT-4o) | Success Status |
|------|------------|--------------------------------|----------------|
| `easy` | Easy | 0.9500 | ✅ |
| `medium` | Medium | 0.8200 | ✅ |
| `hard` | Hard | 0.7600 | ✅ |
| `green` | Green | 0.8800 | ✅ |

> **Note:** Run the evaluator yourself using the Quick Start instructions to see the exact real-time scores for your chosen LLM.

---

## 🧪 Testing

The project includes **38 unit tests** across 9 test classes:

| Test Class | Tests | What it covers |
|-----------|-------|----------------|
| `TestReset` | 6 | Clean state, all 4 tasks, invalid task handling |
| `TestDeterministicNoise` | 3 | Reproducibility, seed isolation, amplitude bounds |
| `TestActions` | 9 | TERMINATE, UPSCALE, DOWNSCALE, REDISTRIBUTE, IGNORE, inbox |
| `TestSLABreach` | 1 | Breach detection, episode termination |
| `TestGrading` | 4 | All 4 graders, score ranges, carbon reduction scoring |
| `TestCarbonTracking` | 4 | Accumulation, reduction after terminate, catalog coverage |
| `TestTrailingHistory` | 3 | Initial values, growth, max depth enforcement |
| `TestEpisodeBoundaries` | 3 | Max steps, budget overrun, post-done behavior |
| `TestClamp` | 4 | Utility function edge cases |

Run locally:
```bash
docker run --rm cloudfinops-env python -m pytest tests/ -v --tb=short
```

---

## 🔄 CI/CD

GitHub Actions runs automatically on every push/PR:
1. **Unit Tests** — `pytest tests/ -v`
2. **Syntax Check** — AST parse all Python files
3. **OpenEnv Spec** — Verify `openenv.yaml` has ≥3 tasks
4. **Docker Build** — Full image build + container smoke test

---

## 🌐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | No | `huggingface` | `groq` or `huggingface` |
| `GROQ_API_KEY` | If groq | — | Groq API key |
| `GROQ_MODEL_NAME` | No | `llama-3.3-70b-versatile` | Groq model |
| `API_BASE_URL` | If HF | `https://router.huggingface.co/v1` | HF router URL |
| `MODEL_NAME` | If HF | `openai/gpt-4o` | Model identifier |
| `HF_TOKEN` | Yes | — | Hugging Face token |
| `ENV_BASE_URL` | No | `http://localhost:8000` | Environment server URL |

---

## 🏆 Key Design Decisions

1. **Deterministic Noise** — Hash-seeded jitter ensures reproducible episodes while maintaining realistic metric variation.
2. **Delayed Scaling** — UPSCALE takes effect next step, forcing agents to plan ahead (not just react).
3. **Carbon Emissions** — Models real-world ARM vs x86 efficiency gap, rewarding sustainable infrastructure.
4. **Trailing Metrics** — Designed for LLM agents with limited context memory — trend detection without explicit memory.
5. **Human-in-the-Loop** — Inbox/reply system tests whether agents can communicate with humans while managing infra.
6. **Upscale Tier Path** — Enforces realistic upgrade constraints (`t3.micro` → `t3.medium` → `t3.large`, max 2 upgrades).

---

## 📁 Project Structure

```
cloudfinops-env/
├── env/
│   ├── __init__.py
│   ├── models.py          # Pydantic schemas (Observation, Action, ServerState)
│   ├── engine.py          # Physics simulator + grading engine
│   ├── server.py          # FastAPI REST API + dashboard serving
│   └── dashboard.html     # Real-time glassmorphism web dashboard
├── tests/
│   ├── __init__.py
│   └── test_engine.py     # 38 pytest unit tests
├── .github/
│   └── workflows/
│       └── validate.yml   # CI: tests + docker build + spec check
├── inference.py           # LLM agent baseline evaluator
├── Dockerfile
├── requirements.txt
├── openenv.yaml           # OpenEnv spec declaration
├── .env.example           # Template environment variables
└── README.md
```

---

## 📜 License

MIT License — Built with ❤️ By Jitraj.
