# ☁️ CloudFinOps-Env

> **Meta AI & HuggingFace OpenEnv Hackathon Submission**

A Reinforcement Learning environment that simulates real-world **cloud infrastructure cost optimization** combined with **SLA incident management** — complete with human-in-the-loop text-chat interactions.

---

## ⚡ Quick Start — 3 Steps

> **One file to configure. Everything reads from it automatically.**

### Step 1 — Configure your credentials

Copy the template and fill in your API key:

```bash
cp .env.example .env
```

Open `.env` and set your values:

```env
# The API endpoint for the LLM (OpenAI-compatible)
API_BASE_URL=https://api.openai.com/v1

# The model identifier to use for inference
MODEL_NAME=gpt-4o

# Your Hugging Face / API key
HF_TOKEN=hf_your_token_here
```

> **That's the only file you need to edit.** Both `inference.py` and Docker read from it automatically.

---

### Step 2 — Start the Environment Server

#### Option A: Docker (Recommended)

```bash
docker build -t cloudfinops-env .
docker run --env-file .env -p 8000:8000 cloudfinops-env
```

You should see the CloudFinOps banner and `Ready to accept connections ✓` in the terminal.

#### Option B: Python (No Docker)

```bash
pip install -r requirements.txt
uvicorn env.server:app --host 0.0.0.0 --port 8000
```

---

### Step 3 — Run the Baseline Evaluator

Open a **new terminal** (keep the server running) and run:

```bash
python inference.py
```

The script will:
1. Auto-load `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from your `.env` file
2. Initialize an **OpenAI Client** using those values
3. Run all 3 tasks (`easy`, `medium`, `hard`) against the environment
4. Print per-task grader scores and an overall average

**Expected output:**
```
============================================================
  FINAL RESULTS
============================================================
  ✅     easy: 1.0000
  ✅   medium: 0.6500
  ✅     hard: 0.4200
     AVERAGE: 0.6900
============================================================
```

---

### Step 4 (Optional) — Run Pre-Submission Validation

From the **parent directory** of `cloudfinops-env/`:

```bash
python pre_validation.py --repo-dir ./cloudfinops-env --skip-docker
```

If everything passes, you'll see: **🎉 All checks passed!.**

---

## 🔐 Environment Variables

All configuration lives in a single `.env` file. The table below documents every variable:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | ✅ Yes | — | OpenAI-compatible API endpoint |
| `MODEL_NAME` | ✅ Yes | — | Model identifier for inference |
| `HF_TOKEN` | ✅ Yes | — | Hugging Face / API key |
| `ENV_BASE_URL` | ❌ No | `http://localhost:8000` | Override the environment server URL |

`inference.py` validates all three mandatory variables at startup and exits with a clear error if any are missing.

---

## 🎯 Motivation & Why This Matters

Cloud infrastructure management is one of the most critical real-world tasks in modern engineering. Platform teams spend thousands of hours annually making decisions about:
- Which idle servers to terminate
- When to scale up before traffic spikes hit
- How to cut costs without causing outages
- Balancing conflicting directives from leadership ("cut costs!") vs reality ("traffic is spiking!")

**CloudFinOps-Env** captures this tension in a multi-objective RL environment where the agent must **simultaneously** optimize competing goals — just like a real SRE on-call.

| Objective | Tension |
|-----------|---------|
| 💰 Cut costs | ↔ Risk overloading remaining servers |
| 🛡️ Prevent SLA breaches | ↔ Requires expensive upscaling |
| ⏱️ Scale up for traffic | ↔ Scaling is **delayed by 1 step** |
| 💬 Follow human instructions | ↔ Humans may give risky directives |

---

## 🏗️ Environment Mechanics

### Physics Engine
- **Delayed Scaling**: `UPSCALE` takes effect on the *next* step — the agent must anticipate, not react.
- **Load Redistribution**: Terminating a server redistributes its load across survivors.
- **SLA Breach = Catastrophe**: Any server hitting 100% CPU triggers an instant SLA failure with heavy penalty.
- **Budget Drain**: Each running server charges per-step costs. Overspending ends the episode.
- **Traffic Ramping**: Hard task features exponential traffic growth simulating a viral campaign.

### Action Space

| Field | Type | Description |
|-------|------|-------------|
| `command` | `Literal["UPSCALE", "DOWNSCALE", "TERMINATE", "REDISTRIBUTE_LOAD", "IGNORE"]` | Infrastructure action to execute |
| `target_id` | `Optional[str]` | ID of the server to act on |
| `reply` | `str` | Text response to stakeholder messages in inbox |

**Action Effects:**

| Command | Effect | Risk |
|---------|--------|------|
| `UPSCALE` | Doubles capacity (delayed 1 step), doubles cost | Expensive; delayed |
| `DOWNSCALE` | Halves cost immediately, CPU load increases 1.8× | May trigger SLA breach |
| `TERMINATE` | Removes server, load redistributed to survivors | Irreversible |
| `REDISTRIBUTE_LOAD` | Balances CPU evenly across all running servers | May not help if all are loaded |
| `IGNORE` | No-op | Wastes a step; budget still drains |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `servers` | `List[ServerState]` | All server instances with id, type, cpu_util, memory_util, cost_per_hour, status |
| `traffic_load` | `float` | Global traffic level (0–100) |
| `spike_detected` | `bool` | Whether a traffic spike is occurring |
| `incidents` | `List[Dict]` | History of SLA breaches and incidents |
| `budget_remaining` | `float` | Remaining budget in USD |
| `time_step` | `int` | Current step number |
| `inbox` | `List[str]` | Text messages from simulated stakeholders (CTO, Marketing, Ops) |

### Reward Function

The reward function provides **continuous signal over the full trajectory**, not just binary end-of-episode feedback:

| Signal | Reward | When |
|--------|--------|------|
| Terminate a server | +10.0 | Server successfully terminated |
| Downscale a server | +5.0 | Cost reduced |
| Redistribute load | +3.0 | Load balanced across fleet |
| Reply to inbox | +2.0 | Agent engages with stakeholder |
| Upscale (investment) | -5.0 | Short-term cost for future capacity |
| Invalid target | -2.0 | Acting on non-existent/dead server |
| High per-step cost | -1.0 | Running costs exceed $3/step |
| Budget overrun | -20.0 | Budget depleted |
| **SLA Breach** | **-100.0** | Any server hits 100% CPU |

---

## 🎮 The 3 Tasks

### 🟢 Easy — *Zombie Cleanup*
- **Setup**: 10 servers — 7 active (25–55% CPU), 3 completely idle (0% CPU)
- **Inbox**: "Ops Team: Please terminate unused servers to save costs."
- **Objective**: Terminate the 3 idle servers without touching active ones
- **Grading**: `score = (idle_terminated / 3) - (active_terminated × 0.25) - (0.5 if SLA breach)`
- **Perfect Score**: Terminate `idle-0`, `idle-1`, `idle-2` → **1.0**

### 🟡 Medium — *The CTO Budget Squeeze*
- **Setup**: 12 over-provisioned servers at ~3–9% CPU across web/db/batch types
- **Inbox**: "CTO: Cut costs by 50% immediately. No excuses."
- **Objective**: Reduce spending by 50% without triggering any SLA breaches
- **Grading**: `score = min(cost_saved% / 50%, 1.0) - (0.5 if SLA breach)`
- **Challenge**: Downscaling increases CPU load 1.8× — must be strategic

### 🔴 Hard — *Black Friday Chaos*
- **Setup**: 8 servers — DBs at 78–85% CPU (and climbing), web at 50–60%, batch at 10–20%
- **Inbox**: "Marketing: Massive ad campaign going live RIGHT NOW!"
- **Objective**: Prevent SLA failures while managing a tight $40 budget
- **Grading**: `score = (uptime × 0.6) + (cost_efficiency × 0.4)`
- **Challenge**: Traffic ramps exponentially via `log1p(step)`, DB CPU climbs each step. Must upscale DBs (delayed!) while shedding batch jobs.

---

## 📡 API Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/reset` | `{"task_id": "easy\|medium\|hard"}` | `Observation` |
| `POST` | `/step` | `Action` JSON | `{observation, reward, done, info}` |
| `GET`  | `/state` | — | `Observation` (no state advance) |

---

## 📊 Baseline Scores

Baseline scores achieved using `gpt-4o` with temperature 0.1 and structured JSON prompting:

| Task | Score | Notes |
|------|-------|-------|
| 🟢 Easy | ~0.67–1.00 | Depends on correctly identifying idle servers |
| 🟡 Medium | ~0.40–0.80 | Partial credit for cost savings below 50% target |
| 🔴 Hard | ~0.30–0.60 | Multi-objective tradeoff; uptime dominates |
| **Average** | **~0.45–0.80** | Varies by model capability |

> **Note**: Scores are ranges because LLM outputs are stochastic. A perfect rule-based agent can achieve 1.0 on Easy and Medium, ~0.85+ on Hard.

---

## 📐 Grading Criteria

All graders output deterministic scores between **0.0** and **1.0**:

- **Easy**: Full credit for terminating all 3 idle servers, `-0.25` per wrongly terminated active server, `-0.5` for any SLA breach
- **Medium**: Proportional to `cost_saved% / 50%` (capped at 1.0), `-0.5` for any SLA breach
- **Hard**: `(uptime_score × 0.6) + (cost_efficiency × 0.4)` where uptime is binary (1.0 if no breach, 0.0 if breached)

---

## 📁 Project Structure

```
cloudfinops-env/
├── env/
│   ├── __init__.py           # Package marker
│   ├── models.py             # Pydantic schemas (ServerState, Action, Observation, RewardInfo)
│   ├── engine.py             # Physics simulator, task configs, grading logic
│   └── server.py             # FastAPI endpoints (/reset, /step, /state)
├── openenv.yaml              # OpenEnv metadata (3 tasks + required env vars)
├── inference.py              # Baseline LLM evaluator using OpenAI SDK
├── .env.example              # ← Edit this one file with your credentials
├── requirements.txt          # Python dependencies
├── Dockerfile                # HF Spaces deployment container
└── README.md                 # This file
```

---

Built with ❤️ for the Meta AI & HuggingFace OpenEnv Hackathon
